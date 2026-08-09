//! End-to-end tests for the segment format.
//!
//! These are the tests that decide whether the format is trustworthy:
//!
//! * round trips over many schemas, sizes and row-group shapes,
//! * corruption detection that names the failing byte,
//! * zone-map correctness in the presence of nulls,
//! * the bloom filter's one hard guarantee (no false negatives), as a property,
//! * projection and predicate pushdown, **proved by counting the bytes read**,
//! * a large segment written and read with bounded memory.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use basalt_common::{CompressionLevel, Config};
use basalt_storage::{
    CountingSource, FileSource, Predicate, ScanOptions, SegmentMetadata, SegmentReader,
    SegmentWriter, ZoneMap, BLOCK_HEADER_LEN,
};
use basalt_types::{
    ColumnBuilder, ColumnVector, DataType, Field, PrimitiveArray, RecordBatch, Schema, StringArray,
    TimeUnit, Value,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn write_segment(
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
    config: Config,
) -> (Vec<u8>, SegmentMetadata) {
    let mut bytes = Vec::new();
    let mut writer =
        SegmentWriter::with_name(&mut bytes, Arc::clone(schema), config, "test.bslt").unwrap();
    for batch in batches {
        writer.write_batch(batch).unwrap();
    }
    let metadata = writer.finish().unwrap();
    (bytes, metadata)
}

fn build_column(data_type: &DataType, values: &[Option<Value>]) -> ColumnVector {
    let mut builder = ColumnBuilder::with_capacity(data_type, values.len()).unwrap();
    for value in values {
        match value {
            Some(v) => builder.append_value(v).unwrap(),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

/// Deterministic splitmix64, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A schema covering every physical layout the format has to handle.
fn wide_schema() -> Arc<Schema> {
    Arc::new(
        Schema::new(vec![
            Field::new("f_bool", DataType::Boolean, true),
            Field::new("f_i8", DataType::Int8, true),
            Field::new("f_i16", DataType::Int16, false),
            Field::new("f_i32", DataType::Int32, true),
            Field::new("f_i64", DataType::Int64, false),
            Field::new("f_u8", DataType::UInt8, true),
            Field::new("f_u16", DataType::UInt16, false),
            Field::new("f_u32", DataType::UInt32, true),
            Field::new("f_u64", DataType::UInt64, false),
            Field::new("f_f32", DataType::Float32, true),
            Field::new("f_f64", DataType::Float64, false),
            Field::new("f_utf8", DataType::Utf8, true),
            Field::new("f_binary", DataType::Binary, false),
            Field::new("f_date", DataType::Date32, true),
            Field::new("f_ts", DataType::Timestamp(TimeUnit::Microsecond), false),
            Field::new("f_dec", DataType::Decimal128(20, 4), true),
            Field::new("f_null", DataType::Null, true),
        ])
        .unwrap(),
    )
}

fn wide_batch(schema: &Arc<Schema>, rows: usize, seed: u64) -> RecordBatch {
    let mut rng = Rng::new(seed);
    let mut columns = Vec::new();
    for field in schema.fields() {
        let values: Vec<Option<Value>> = (0..rows)
            .map(|i| {
                let r = rng.next();
                // Nullable fields get roughly one null in seven.
                if field.is_nullable() && r % 7 == 0 {
                    return None;
                }
                Some(match field.data_type() {
                    DataType::Null => return None,
                    DataType::Boolean => Value::Boolean(r % 2 == 0),
                    DataType::Int8 => Value::Int8(r as i8),
                    DataType::Int16 => Value::Int16(r as i16),
                    DataType::Int32 => Value::Int32(r as i32),
                    DataType::Int64 => Value::Int64(r as i64),
                    DataType::UInt8 => Value::UInt8(r as u8),
                    DataType::UInt16 => Value::UInt16(r as u16),
                    DataType::UInt32 => Value::UInt32(r as u32),
                    DataType::UInt64 => Value::UInt64(r),
                    DataType::Float32 => Value::Float32(r as f32 / 7.0),
                    DataType::Float64 => Value::Float64(r as f64 / 13.0),
                    DataType::Utf8 => Value::Utf8(format!("city-{}", r % 40)),
                    DataType::Binary => Value::Binary(vec![r as u8; (r % 5) as usize]),
                    DataType::Date32 => Value::Date32(19_000 + (i % 400) as i32),
                    DataType::Timestamp(unit) => {
                        Value::Timestamp(1_700_000_000_000_000 + (r % 1_000_000) as i64, *unit)
                    }
                    DataType::Decimal128(p, s) => {
                        Value::Decimal128(i128::from(r % 1_000_000_000), *p, *s)
                    }
                })
            })
            .collect();
        columns.push(build_column(field.data_type(), &values));
    }
    RecordBatch::try_new(Arc::clone(schema), columns).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Round trips
// ---------------------------------------------------------------------------

#[test]
fn round_trip_over_many_shapes() {
    let schema = wide_schema();
    for rows in [0usize, 1, 2, 63, 64, 65, 1000, 5000] {
        for row_group_rows in [64usize, 1000, 65_536] {
            for level in [
                CompressionLevel::NONE,
                CompressionLevel::BALANCED,
                CompressionLevel::MAX,
            ] {
                let config = Config::default()
                    .with_batch_size(64)
                    .with_row_group_rows(row_group_rows)
                    .with_compression_level(level);
                let batch = wide_batch(&schema, rows, (rows * 31 + row_group_rows) as u64);
                let (bytes, metadata) =
                    write_segment(&schema, std::slice::from_ref(&batch), config);

                let reader = SegmentReader::open(bytes).unwrap();
                assert_eq!(reader.schema().as_ref(), schema.as_ref());
                assert_eq!(reader.num_rows(), rows as u64);
                assert_eq!(reader.metadata(), &metadata);

                let read_back = reader.read_all().unwrap();
                let total: usize = read_back.iter().map(RecordBatch::num_rows).sum();
                assert_eq!(total, rows, "rows={rows} rg={row_group_rows}");

                let mut offset = 0;
                for produced in &read_back {
                    let expected = batch.slice(offset, produced.num_rows());
                    for column in 0..schema.len() {
                        assert_eq!(
                            produced.column(column),
                            expected.column(column),
                            "column `{}` at row {offset} (rows={rows} rg={row_group_rows} level={level:?})",
                            schema.field(column).name()
                        );
                    }
                    offset += produced.num_rows();
                }
            }
        }
    }
}

#[test]
fn round_trip_across_many_batches() {
    let schema = wide_schema();
    let config = Config::default()
        .with_batch_size(64)
        .with_row_group_rows(256);
    // Batch sizes that do not divide the row-group size, so every boundary case
    // (batch spans a boundary, batch smaller than the remainder) is hit.
    let batches: Vec<RecordBatch> = (0..12)
        .map(|i| wide_batch(&schema, 37 + i * 5, i as u64 + 1))
        .collect();
    let expected_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let (bytes, metadata) = write_segment(&schema, &batches, config);
    assert_eq!(metadata.total_rows, expected_rows as u64);

    let reader = SegmentReader::open(bytes).unwrap();
    let read_back = reader.read_all().unwrap();
    assert_eq!(
        read_back.iter().map(RecordBatch::num_rows).sum::<usize>(),
        expected_rows
    );
    // Every row group except the last is exactly full.
    for group in &metadata.row_groups[..metadata.row_groups.len() - 1] {
        assert_eq!(group.num_rows, 256);
    }

    // Compare value by value against the concatenated input.
    let mut expected: Vec<Value> = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            expected.push(batch.column(4).value(row));
        }
    }
    let mut produced: Vec<Value> = Vec::new();
    for batch in &read_back {
        for row in 0..batch.num_rows() {
            produced.push(batch.column(4).value(row));
        }
    }
    assert_eq!(produced, expected);
}

#[test]
fn file_backed_round_trip() {
    let schema = wide_schema();
    let batch = wide_batch(&schema, 3000, 99);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("orders.bslt");

    let file = std::fs::File::create(&path).unwrap();
    let config = Config::default()
        .with_batch_size(256)
        .with_row_group_rows(1024);
    let mut writer = SegmentWriter::with_name(
        file,
        Arc::clone(&schema),
        config,
        path.display().to_string(),
    )
    .unwrap();
    writer.write_batch(&batch).unwrap();
    let metadata = writer.finish().unwrap();
    assert_eq!(metadata.num_row_groups(), 3);

    let source = FileSource::open(&path).unwrap();
    assert!(source.name().ends_with("orders.bslt"));
    let reader = SegmentReader::open_named(source, path.display().to_string()).unwrap();
    assert_eq!(reader.num_rows(), 3000);
    let read_back = reader.read_all().unwrap();
    assert_eq!(
        read_back.iter().map(RecordBatch::num_rows).sum::<usize>(),
        3000
    );
    assert_eq!(read_back[0].column(4).value(0), batch.column(4).value(0));
}

#[test]
fn empty_segment_is_valid() {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
    let (bytes, metadata) = write_segment(&schema, &[], Config::default());
    assert_eq!(metadata.total_rows, 0);
    assert_eq!(metadata.num_row_groups(), 0);
    let reader = SegmentReader::open(bytes).unwrap();
    assert_eq!(reader.num_rows(), 0);
    assert!(reader.read_all().unwrap().is_empty());
}

#[test]
fn writer_rejects_a_mismatched_batch() {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
    let other = Arc::new(Schema::new(vec![Field::new("b", DataType::Int64, false)]).unwrap());
    let batch = RecordBatch::try_new(
        Arc::clone(&other),
        vec![ColumnVector::Int64(PrimitiveArray::from_vec(vec![1]))],
    )
    .unwrap();
    let mut bytes = Vec::new();
    let mut writer = SegmentWriter::new(&mut bytes, schema, Config::default()).unwrap();
    let err = writer.write_batch(&batch).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match the segment schema"),
        "{err}"
    );
    writer.finish().unwrap();
}

// ---------------------------------------------------------------------------
// 2. Corruption detection
// ---------------------------------------------------------------------------

#[test]
fn a_flipped_bit_in_a_data_block_is_reported_with_its_offset() {
    let schema = wide_schema();
    let batch = wide_batch(&schema, 2000, 5);
    let (bytes, metadata) = write_segment(&schema, &[batch], Config::default());

    let chunk = &metadata.row_groups[0].columns[4];
    let payload_offset = chunk.data.offset + BLOCK_HEADER_LEN;
    // Flip a bit in the middle of the chunk's payload.
    let target = (payload_offset + chunk.data.length / 2) as usize;
    let mut damaged = bytes.clone();
    damaged[target] ^= 0b0010_0000;

    let reader = SegmentReader::open(damaged).unwrap();
    let err = reader.read_row_group(0, Some(&[4])).unwrap_err();
    assert!(err.is_corruption(), "{err}");
    assert_eq!(
        err.corruption_offset(),
        Some(payload_offset),
        "the error must name the start of the damaged block: {err}"
    );
    let message = err.to_string();
    assert!(message.contains("crc32c mismatch"), "{message}");
    assert!(
        message.contains("test.bslt") || message.contains("segment"),
        "{message}"
    );

    // A projection that avoids the damaged column still succeeds: corruption is
    // localised to the block that carries it.
    let ok = reader.read_row_group(0, Some(&[0])).unwrap();
    assert_eq!(ok.num_rows(), 2000);
}

#[test]
fn every_byte_of_the_footer_and_trailer_is_protected() {
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("city", DataType::Utf8, true),
        ])
        .unwrap(),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            ColumnVector::Int64(PrimitiveArray::from_vec((0..100).collect())),
            ColumnVector::Utf8(StringArray::from_strs(
                &(0..100).map(|i| format!("c{i}")).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let (bytes, _metadata) = write_segment(&schema, &[batch], Config::default());
    // The trailer's first 8 bytes are the footer block's offset.
    let mut offset_bytes = [0u8; 8];
    offset_bytes.copy_from_slice(&bytes[bytes.len() - 16..bytes.len() - 8]);
    let footer_start = u64::from_le_bytes(offset_bytes) as usize;

    // Damage anywhere from the footer block to the end of the file must be
    // caught at open() time: the footer is CRC-protected and the trailer is
    // structurally validated.
    let footer_region = footer_start..bytes.len();
    assert!(
        footer_region.len() > 40,
        "the footer should be more than a few bytes"
    );
    let mut caught = 0;
    for index in footer_region.clone() {
        let mut damaged = bytes.clone();
        damaged[index] ^= 0xFF;
        if SegmentReader::open(damaged).is_err() {
            caught += 1;
        }
    }
    assert_eq!(
        caught,
        footer_region.len(),
        "some footer/trailer damage went undetected"
    );
}

#[test]
fn truncated_and_bogus_files_are_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]).unwrap());
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![ColumnVector::Int64(PrimitiveArray::from_vec(
            (0..500).collect(),
        ))],
    )
    .unwrap();
    let (bytes, _) = write_segment(&schema, &[batch], Config::default());

    assert!(SegmentReader::open(Vec::new()).is_err());
    assert!(SegmentReader::open(vec![0u8; 8]).is_err());
    assert!(
        SegmentReader::open(vec![0u8; 64]).is_err(),
        "a file of zeros is not a segment"
    );
    for cut in [16, bytes.len() / 2, bytes.len() - 1] {
        let err = SegmentReader::open(bytes[..cut].to_vec()).unwrap_err();
        assert!(err.is_corruption(), "truncation at {cut}: {err}");
    }
    // Right length, wrong magic.
    let mut wrong = bytes.clone();
    wrong[0] = b'X';
    let err = SegmentReader::open(wrong).unwrap_err();
    assert!(err.to_string().contains("bad magic"), "{err}");
    assert_eq!(err.corruption_offset(), Some(0));
}

// ---------------------------------------------------------------------------
// 3. Zone maps under nulls
// ---------------------------------------------------------------------------

#[test]
fn zone_maps_match_a_reference_scan() {
    let schema = wide_schema();
    let config = Config::default()
        .with_batch_size(64)
        .with_row_group_rows(500);
    let batch = wide_batch(&schema, 2000, 1234);
    let (_, metadata) = write_segment(&schema, std::slice::from_ref(&batch), config);

    for (group_index, group) in metadata.row_groups.iter().enumerate() {
        let offset = group_index * 500;
        let slice = batch.slice(offset, group.num_rows as usize);
        for (column_index, chunk) in group.columns.iter().enumerate() {
            let column = slice.column(column_index);
            let reference = ZoneMap::from_column(column);
            assert_eq!(chunk.zone_map.min, reference.min, "column {column_index}");
            assert_eq!(chunk.zone_map.max, reference.max, "column {column_index}");
            assert_eq!(chunk.zone_map.null_count, reference.null_count);
            assert_eq!(chunk.zone_map.row_count, group.num_rows);

            // And the reference itself agrees with a naive scan.
            let mut naive_min: Option<Value> = None;
            let mut naive_max: Option<Value> = None;
            let mut nulls = 0u64;
            for row in 0..column.len() {
                if column.is_null(row) {
                    nulls += 1;
                    continue;
                }
                let value = column.value(row);
                if naive_min.as_ref().is_none_or(|m| value < *m) {
                    naive_min = Some(value.clone());
                }
                if naive_max.as_ref().is_none_or(|m| value > *m) {
                    naive_max = Some(value);
                }
            }
            assert_eq!(chunk.zone_map.min, naive_min, "column {column_index}");
            assert_eq!(chunk.zone_map.max, naive_max, "column {column_index}");
            assert_eq!(chunk.zone_map.null_count, nulls, "column {column_index}");
        }
    }
}

#[test]
fn all_null_columns_prune_correctly() {
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("maybe", DataType::Int64, true),
        ])
        .unwrap(),
    );
    let config = Config::default()
        .with_batch_size(50)
        .with_row_group_rows(100);
    let mut batches = Vec::new();
    for group in 0..4i64 {
        let ids: Vec<i64> = (0..100).map(|i| group * 100 + i).collect();
        // Row group 2 is entirely null in `maybe`.
        let maybe: Vec<Option<i64>> = (0..100)
            .map(|i| if group == 2 { None } else { Some(i) })
            .collect();
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    ColumnVector::Int64(PrimitiveArray::from_vec(ids)),
                    ColumnVector::Int64(PrimitiveArray::from_opt_vec(maybe)),
                ],
            )
            .unwrap(),
        );
    }
    let (bytes, metadata) = write_segment(&schema, &batches, config);
    assert!(metadata.row_groups[2].columns[1].zone_map.is_all_null());

    let reader = SegmentReader::open(bytes).unwrap();
    // IS NULL keeps only the all-null group.
    let keep = reader
        .prune_row_groups(Some(&Predicate::IsNull(1)))
        .unwrap();
    assert_eq!(keep, vec![2]);
    // IS NOT NULL drops it.
    let keep = reader
        .prune_row_groups(Some(&Predicate::IsNotNull(1)))
        .unwrap();
    assert_eq!(keep, vec![0, 1, 3]);
    // A range predicate also drops it: every comparison against NULL is NULL.
    let keep = reader
        .prune_row_groups(Some(&Predicate::gt(1, Value::Int64(-1))))
        .unwrap();
    assert_eq!(keep, vec![0, 1, 3]);
}

// ---------------------------------------------------------------------------
// 4. Pushdown, measured
// ---------------------------------------------------------------------------

/// Build a segment with 16 row groups of 2048 rows and two very different
/// columns, so the byte accounting is unambiguous.
fn pushdown_segment() -> (Arc<Schema>, Vec<u8>, SegmentMetadata) {
    let schema = Arc::new(
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("bulk", DataType::Binary, false),
        ])
        .unwrap(),
    );
    let config = Config::default()
        .with_batch_size(512)
        .with_row_group_rows(2048);
    let mut rng = Rng::new(4242);
    let mut batches = Vec::new();
    for group in 0..16i64 {
        let ids: Vec<i64> = (0..2048).map(|i| group * 2048 + i).collect();
        // Incompressible payload so the `bulk` column really is large.
        let blobs: Vec<Vec<u8>> = (0..2048)
            .map(|_| (0..64).map(|_| rng.next() as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    ColumnVector::Int64(PrimitiveArray::from_vec(ids)),
                    ColumnVector::Binary(basalt_types::BinaryArray::from_slices(&refs)),
                ],
            )
            .unwrap(),
        );
    }
    let (bytes, metadata) = write_segment(&schema, &batches, config);
    (schema, bytes, metadata)
}

#[test]
fn projection_pushdown_reads_only_the_projected_column() {
    let (_, bytes, metadata) = pushdown_segment();
    let total_bytes = bytes.len() as u64;
    let id_bytes: u64 = metadata
        .row_groups
        .iter()
        .map(|group| group.columns[0].data.length)
        .sum();
    let bulk_bytes: u64 = metadata
        .row_groups
        .iter()
        .map(|group| group.columns[1].data.length)
        .sum();
    assert!(
        bulk_bytes > id_bytes * 20,
        "the test needs a lopsided segment"
    );

    let reader = SegmentReader::open(CountingSource::new(bytes)).unwrap();

    // Full scan.
    reader.source().reset();
    let all: Vec<RecordBatch> = reader
        .scan(&ScanOptions::new())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let full_scan = reader.source().bytes_read().bytes();
    assert_eq!(
        all.iter().map(RecordBatch::num_rows).sum::<usize>(),
        16 * 2048
    );
    assert!(full_scan >= id_bytes + bulk_bytes);

    // Projected scan: `bulk` must never be touched.
    reader.source().reset();
    let options = ScanOptions::new().with_projection(vec![0]);
    let projected: Vec<RecordBatch> = reader
        .scan(&options)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let projected_scan = reader.source().bytes_read().bytes();
    assert_eq!(
        projected.iter().map(RecordBatch::num_rows).sum::<usize>(),
        16 * 2048
    );
    assert_eq!(projected[0].num_columns(), 1);

    assert!(
        projected_scan < full_scan / 10,
        "projection read {projected_scan} of {full_scan} bytes; pushdown is not working"
    );
    assert!(
        projected_scan >= id_bytes,
        "projection must still read the column it was asked for"
    );
    assert!(
        projected_scan < id_bytes + bulk_bytes / 100,
        "projection leaked into the `bulk` column: {projected_scan} vs id={id_bytes}"
    );
    assert!(full_scan <= total_bytes + 1024);
}

#[test]
fn predicate_pushdown_skips_row_groups_and_the_bytes_show_it() {
    let (_, bytes, metadata) = pushdown_segment();
    let reader = SegmentReader::open(CountingSource::new(bytes)).unwrap();

    // `id = 5000` lives in row group 2 only (2 * 2048 = 4096 .. 6143).
    let predicate = Predicate::eq(0, Value::Int64(5000));
    let keep = reader.prune_row_groups(Some(&predicate)).unwrap();
    assert_eq!(keep, vec![2], "zone maps should isolate a single row group");

    // Baseline: the same query without the predicate.
    reader.source().reset();
    let all: Vec<RecordBatch> = reader
        .scan(&ScanOptions::new())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    let full_bytes = reader.source().bytes_read().bytes();
    assert_eq!(all.len(), 16);

    reader.source().reset();
    let options = ScanOptions::new().with_predicate(predicate);
    let batches: Vec<RecordBatch> = reader
        .scan(&options)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2048);
    let pruned_bytes = reader.source().bytes_read().bytes();

    assert!(
        pruned_bytes * 8 < full_bytes,
        "pruned scan read {pruned_bytes} bytes against {full_bytes} for the full scan"
    );

    // The pruned scan reads exactly: the two chunks of row group 2, plus one
    // bloom filter block. Nothing else in the file is touched.
    let group = &metadata.row_groups[2];
    let chunk_bytes: u64 = group.columns.iter().map(|c| c.data.length).sum();
    let bloom_bytes: u64 = group
        .columns
        .iter()
        .filter_map(|c| c.bloom)
        .map(|b| b.length)
        .sum();
    assert!(
        pruned_bytes <= chunk_bytes + bloom_bytes,
        "read {pruned_bytes} bytes but the row group is only {chunk_bytes} + {bloom_bytes} bloom"
    );
    // Worth noting: the `id` column is delta-encoded down to ~50 bytes per
    // chunk, so its bloom filter is *larger* than the data it describes. Bloom
    // filters pay off when the column they guard is expensive to read, which is
    // why the reader only consults them after zone maps have already narrowed
    // the search.
    assert!(bloom_bytes > 0);

    // A predicate no row group can satisfy reads nothing at all.
    reader.source().reset();
    let impossible = ScanOptions::new().with_predicate(Predicate::gt(0, Value::Int64(1_000_000)));
    let batches: Vec<RecordBatch> = reader
        .scan(&impossible)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert!(batches.is_empty());
    assert_eq!(
        reader.source().bytes_read().bytes(),
        0,
        "pruning everything must read zero bytes"
    );
}

#[test]
fn bloom_filters_are_consulted_only_when_they_can_help() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]).unwrap());
    let config = Config::default()
        .with_batch_size(250)
        .with_row_group_rows(1000);
    // Values are spread so zone maps overlap: [0, 990], [1, 991], ...
    let mut batches = Vec::new();
    for group in 0..8i64 {
        let keys: Vec<i64> = (0..1000).map(|i| (i * 7 + group) % 991).collect();
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![ColumnVector::Int64(PrimitiveArray::from_vec(keys))],
            )
            .unwrap(),
        );
    }
    let (bytes, metadata) = write_segment(&schema, &batches, config);
    assert!(
        metadata.row_groups[0].columns[0].bloom.is_some(),
        "a bloom filter should be written"
    );

    let reader = SegmentReader::open(CountingSource::new(bytes)).unwrap();

    // A value present in every group: zone maps cannot prune, blooms cannot
    // either, so everything is read.
    let present = Predicate::eq(0, Value::Int64(0));
    assert_eq!(reader.prune_row_groups(Some(&present)).unwrap().len(), 8);

    // A value inside the zone-map range but absent from the data: only the
    // bloom filter can rule it out. `(i*7 + g) % 991` never produces 990 for
    // g in 0..8 unless 7i = 990 - g (mod 991) has a solution below 1000, which
    // it does; instead probe a value that is genuinely absent everywhere.
    let absent = (0..991i64).find(|candidate| {
        (0..8).all(|group| !(0..1000).any(|i| (i * 7 + group) % 991 == *candidate))
    });
    if let Some(absent) = absent {
        let predicate = Predicate::eq(0, Value::Int64(absent));
        let keep = reader.prune_row_groups(Some(&predicate)).unwrap();
        assert!(
            keep.len() < 8,
            "bloom filters should have pruned something for {absent}"
        );
    }

    // A range predicate never reads a bloom filter.
    reader.source().reset();
    let _ = reader
        .prune_row_groups(Some(&Predicate::gt(0, Value::Int64(500))))
        .unwrap();
    assert_eq!(
        reader.source().bytes_read().bytes(),
        0,
        "range pruning must not read bloom filter blocks"
    );
}

// ---------------------------------------------------------------------------
// 5. Bounded memory on a large segment
// ---------------------------------------------------------------------------

#[test]
fn large_segment_streams_with_bounded_memory() {
    const ROW_GROUP: usize = 8192;
    const BATCH: usize = 997; // deliberately not a divisor of ROW_GROUP
    const TOTAL: usize = 250_000;

    let schema = Arc::new(
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ])
        .unwrap(),
    );
    let config = Config::default()
        .with_batch_size(1024)
        .with_row_group_rows(ROW_GROUP);

    let mut bytes = Vec::new();
    let mut writer = SegmentWriter::new(&mut bytes, Arc::clone(&schema), config).unwrap();
    let mut written = 0usize;
    while written < TOTAL {
        let rows = BATCH.min(TOTAL - written);
        let ids: Vec<i64> = (0..rows).map(|i| (written + i) as i64).collect();
        let labels: Vec<String> = ids.iter().map(|i| format!("row-{}", i % 64)).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                ColumnVector::Int64(PrimitiveArray::from_vec(ids)),
                ColumnVector::Utf8(StringArray::from_strs(&labels)),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).unwrap();
        written += rows;
        // The invariant that makes the writer streaming: it never holds more
        // than one row group's worth of rows, whatever the caller does.
        assert!(
            writer.buffered_rows() < ROW_GROUP,
            "writer buffered {} rows, above the {ROW_GROUP} row-group size",
            writer.buffered_rows()
        );
    }
    let metadata = writer.finish().unwrap();
    assert_eq!(metadata.total_rows, TOTAL as u64);
    assert_eq!(metadata.num_row_groups(), TOTAL.div_ceil(ROW_GROUP));

    // 250 000 rows of (i64, short string) compress hard: the labels have 64
    // distinct values and the ids are monotonic.
    let ratio =
        metadata.uncompressed_size().bytes() as f64 / metadata.compressed_size().bytes() as f64;
    assert!(ratio > 4.0, "expected real compression, got {ratio:.1}x");

    // Reading is streaming too: one row group at a time.
    let reader = SegmentReader::open(bytes).unwrap();
    let mut seen = 0usize;
    let mut peak_batch_rows = 0usize;
    for batch in reader.scan(&ScanOptions::new()).unwrap() {
        let batch = batch.unwrap();
        peak_batch_rows = peak_batch_rows.max(batch.num_rows());
        for row in 0..batch.num_rows() {
            if row % 512 == 0 {
                assert_eq!(
                    batch.column(0).value(row),
                    Value::Int64((seen + row) as i64)
                );
            }
        }
        seen += batch.num_rows();
    }
    assert_eq!(seen, TOTAL);
    assert_eq!(
        peak_batch_rows, ROW_GROUP,
        "the reader must yield one row group at a time, not the whole segment"
    );
}

// ---------------------------------------------------------------------------
// 6. Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// **The bloom filter's one guarantee.** For every value actually present in
    /// a row group, pruning must not remove that row group. A false negative
    /// here would silently drop rows from query results, which is the worst
    /// possible bug in a database.
    #[test]
    fn bloom_filters_never_produce_false_negatives(
        values in prop::collection::vec(any::<i64>(), 1..600),
        group_size in 32usize..200,
    ) {
        let schema = Arc::new(
            Schema::new(vec![Field::new("k", DataType::Int64, false)]).unwrap(),
        );
        let config = Config::default()
            .with_batch_size(32)
            .with_row_group_rows(group_size)
            .with_bloom_filters(true);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![ColumnVector::Int64(PrimitiveArray::from_vec(values.clone()))],
        )
        .unwrap();
        let (bytes, metadata) = write_segment(&schema, &[batch], config);
        let reader = SegmentReader::open(bytes).unwrap();

        for (index, group) in metadata.row_groups.iter().enumerate() {
            let start: usize = metadata.row_groups[..index]
                .iter()
                .map(|g| g.num_rows as usize)
                .sum();
            let present = &values[start..start + group.num_rows as usize];
            for value in present {
                let keep = reader
                    .prune_row_groups(Some(&Predicate::eq(0, Value::Int64(*value))))
                    .unwrap();
                prop_assert!(
                    keep.contains(&index),
                    "row group {index} holds {value} but pruning removed it",
                );
            }
        }
    }

    /// Whatever the data, a full scan returns exactly what was written.
    #[test]
    fn round_trip_preserves_every_value(
        values in prop::collection::vec(prop::option::weighted(0.8, any::<i64>()), 0..400),
        group_size in 16usize..128,
        level in 0u8..=9,
    ) {
        let schema = Arc::new(
            Schema::new(vec![Field::new("v", DataType::Int64, true)]).unwrap(),
        );
        let config = Config::default()
            .with_batch_size(16)
            .with_row_group_rows(group_size)
            .with_compression_level(CompressionLevel::new(level).unwrap_or(CompressionLevel::BALANCED));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![ColumnVector::Int64(PrimitiveArray::from_opt_vec(values.clone()))],
        )
        .unwrap();
        let (bytes, _) = write_segment(&schema, &[batch], config);
        let reader = SegmentReader::open(bytes).unwrap();
        let mut produced = Vec::new();
        for batch in reader.scan(&ScanOptions::new()).unwrap() {
            let batch = batch.unwrap();
            for row in 0..batch.num_rows() {
                produced.push(match batch.column(0).value(row) {
                    Value::Null => None,
                    Value::Int64(v) => Some(v),
                    other => panic!("unexpected value {other:?}"),
                });
            }
        }
        prop_assert_eq!(produced, values);
    }

    /// Pruning must never remove a row group that contains a matching row.
    #[test]
    fn pruning_never_loses_a_matching_row(
        values in prop::collection::vec(-50i64..50, 1..400),
        threshold in -60i64..60,
        group_size in 16usize..96,
    ) {
        let schema = Arc::new(
            Schema::new(vec![Field::new("v", DataType::Int64, false)]).unwrap(),
        );
        let config = Config::default().with_batch_size(16).with_row_group_rows(group_size);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![ColumnVector::Int64(PrimitiveArray::from_vec(values.clone()))],
        )
        .unwrap();
        let (bytes, metadata) = write_segment(&schema, &[batch], config);
        let reader = SegmentReader::open(bytes).unwrap();

        for predicate in [
            Predicate::gt(0, Value::Int64(threshold)),
            Predicate::lt(0, Value::Int64(threshold)),
            Predicate::eq(0, Value::Int64(threshold)),
            Predicate::between(0, Value::Int64(threshold), Value::Int64(threshold + 10)),
        ] {
            let keep = reader.prune_row_groups(Some(&predicate)).unwrap();
            for (index, group) in metadata.row_groups.iter().enumerate() {
                let start: usize =
                    metadata.row_groups[..index].iter().map(|g| g.num_rows as usize).sum();
                let rows = &values[start..start + group.num_rows as usize];
                let matches = rows.iter().any(|v| match &predicate {
                    Predicate::Gt(_, Value::Int64(t)) => v > t,
                    Predicate::Lt(_, Value::Int64(t)) => v < t,
                    Predicate::Eq(_, Value::Int64(t)) => v == t,
                    Predicate::Between(_, Value::Int64(lo), Value::Int64(hi)) => v >= lo && v <= hi,
                    _ => true,
                });
                if matches {
                    prop_assert!(
                        keep.contains(&index),
                        "predicate {} pruned row group {index}, which contains a match",
                        predicate,
                    );
                }
            }
        }
    }
}

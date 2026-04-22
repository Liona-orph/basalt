//! Segment metadata: the footer's in-memory form and its serialisation.

use std::collections::BTreeMap;
use std::sync::Arc;

use basalt_common::{BasaltError, ByteSize, Region, Result};
use basalt_encoding::{varint, Encoding};
use basalt_types::{Schema, Value};

use crate::format::FOOTER_MAGIC;
use crate::serial;
use crate::zone_map::ZoneMap;

/// Version of the footer payload layout.
const FOOTER_VERSION: u8 = 1;

fn corrupt(offset: usize, detail: impl Into<String>) -> BasaltError {
    BasaltError::corruption("segment footer", Region::Footer, offset as u64, detail)
}

/// Where a bloom filter block lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLocation {
    /// Absolute file offset of the block *header*.
    pub offset: u64,
    /// Total block length, including the 8-byte header.
    pub length: u64,
}

impl BlockLocation {
    /// Size of the block including framing.
    pub fn size(self) -> ByteSize {
        ByteSize::new(self.length)
    }
}

/// Everything the reader knows about one column of one row group.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnChunkMetadata {
    /// Where the encoded values live.
    pub data: BlockLocation,
    /// Encoding chosen for the value stream.
    pub encoding: Encoding,
    /// In-memory size of the values, for reporting compression ratios.
    pub uncompressed_bytes: u64,
    /// Min/max/null-count/distinct summary.
    pub zone_map: ZoneMap,
    /// Bloom filter block, if one was written.
    pub bloom: Option<BlockLocation>,
}

impl ColumnChunkMetadata {
    /// Compressed size of the chunk, including block framing.
    pub fn compressed_size(&self) -> ByteSize {
        self.data.size()
    }

    /// Compression ratio against the in-memory size, or `None` if the chunk is
    /// empty.
    pub fn compression_ratio(&self) -> Option<f64> {
        ByteSize::new(self.uncompressed_bytes).ratio_to(self.data.size())
    }
}

/// One row group: a horizontal slice of the table, one chunk per column.
#[derive(Debug, Clone, PartialEq)]
pub struct RowGroupMetadata {
    /// Rows in this row group.
    pub num_rows: u64,
    /// One entry per schema field, in schema order.
    pub columns: Vec<ColumnChunkMetadata>,
}

impl RowGroupMetadata {
    /// Total on-disk size of every column chunk (excluding bloom filters).
    pub fn compressed_size(&self) -> ByteSize {
        self.columns
            .iter()
            .map(ColumnChunkMetadata::compressed_size)
            .sum()
    }
}

/// The parsed footer of a segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentMetadata {
    /// Format version read from the file header.
    pub format_version: u16,
    /// The segment's schema.
    pub schema: Arc<Schema>,
    /// Row groups, in file order.
    pub row_groups: Vec<RowGroupMetadata>,
    /// Total rows across all row groups.
    pub total_rows: u64,
    /// Free-form segment-level metadata.
    pub metadata: BTreeMap<String, String>,
}

impl SegmentMetadata {
    /// Number of row groups.
    pub fn num_row_groups(&self) -> usize {
        self.row_groups.len()
    }

    /// Total compressed size of all column data.
    pub fn compressed_size(&self) -> ByteSize {
        self.row_groups
            .iter()
            .map(RowGroupMetadata::compressed_size)
            .sum()
    }

    /// Total uncompressed size of all column data.
    pub fn uncompressed_size(&self) -> ByteSize {
        ByteSize::new(
            self.row_groups
                .iter()
                .flat_map(|group| group.columns.iter())
                .map(|chunk| chunk.uncompressed_bytes)
                .sum(),
        )
    }

    /// Serialise the footer payload (without block framing).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1024);
        out.extend_from_slice(&FOOTER_MAGIC);
        out.push(FOOTER_VERSION);
        serial::write_schema(&self.schema, &mut out);
        varint::write_uvarint(self.row_groups.len() as u64, &mut out);
        for group in &self.row_groups {
            varint::write_uvarint(group.num_rows, &mut out);
            for chunk in &group.columns {
                varint::write_uvarint(chunk.data.offset, &mut out);
                varint::write_uvarint(chunk.data.length, &mut out);
                out.push(chunk.encoding.as_u8());
                varint::write_uvarint(chunk.uncompressed_bytes, &mut out);
                write_zone_map(&chunk.zone_map, &mut out);
                match &chunk.bloom {
                    Some(location) => {
                        out.push(1);
                        varint::write_uvarint(location.offset, &mut out);
                        varint::write_uvarint(location.length, &mut out);
                    }
                    None => out.push(0),
                }
            }
        }
        varint::write_uvarint(self.total_rows, &mut out);
        serial::write_metadata(&self.metadata, &mut out);
        out
    }

    /// Parse a footer payload.
    pub fn from_bytes(bytes: &[u8], format_version: u16) -> Result<SegmentMetadata> {
        if bytes.len() < 5 || bytes[0..4] != FOOTER_MAGIC {
            return Err(corrupt(
                0,
                "footer does not start with the BFTR magic; the trailer's offset is wrong",
            ));
        }
        if bytes[4] != FOOTER_VERSION {
            return Err(corrupt(
                4,
                format!("footer version {} is not supported", bytes[4]),
            ));
        }
        let mut pos = 5usize;
        let schema = Arc::new(serial::read_schema(bytes, &mut pos)?);
        let column_count = schema.len();
        let group_count = varint::read_uvarint(bytes, &mut pos)? as usize;
        // Each row group needs at least one byte per column in the footer, so a
        // count larger than the footer itself is corruption.
        if group_count.saturating_mul(column_count.max(1)) > bytes.len() {
            return Err(corrupt(
                pos,
                format!(
                    "footer claims {group_count} row groups in {} bytes",
                    bytes.len()
                ),
            ));
        }
        let mut row_groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let num_rows = varint::read_uvarint(bytes, &mut pos)?;
            let mut columns = Vec::with_capacity(column_count);
            for _ in 0..column_count {
                let offset = varint::read_uvarint(bytes, &mut pos)?;
                let length = varint::read_uvarint(bytes, &mut pos)?;
                let encoding_code = read_u8(bytes, &mut pos)?;
                let encoding = Encoding::from_u8(encoding_code)
                    .ok_or_else(|| corrupt(pos, format!("unknown encoding id {encoding_code}")))?;
                let uncompressed_bytes = varint::read_uvarint(bytes, &mut pos)?;
                let zone_map = read_zone_map(bytes, &mut pos)?;
                let bloom = if read_u8(bytes, &mut pos)? == 1 {
                    Some(BlockLocation {
                        offset: varint::read_uvarint(bytes, &mut pos)?,
                        length: varint::read_uvarint(bytes, &mut pos)?,
                    })
                } else {
                    None
                };
                columns.push(ColumnChunkMetadata {
                    data: BlockLocation { offset, length },
                    encoding,
                    uncompressed_bytes,
                    zone_map,
                    bloom,
                });
            }
            row_groups.push(RowGroupMetadata { num_rows, columns });
        }
        let total_rows = varint::read_uvarint(bytes, &mut pos)?;
        let metadata = serial::read_metadata(bytes, &mut pos)?;

        let summed: u64 = row_groups.iter().map(|group| group.num_rows).sum();
        if summed != total_rows {
            return Err(corrupt(
                pos,
                format!("footer says {total_rows} rows but the row groups sum to {summed}"),
            ));
        }
        Ok(SegmentMetadata {
            format_version,
            schema,
            row_groups,
            total_rows,
            metadata,
        })
    }
}

fn read_u8(bytes: &[u8], pos: &mut usize) -> Result<u8> {
    let value = *bytes
        .get(*pos)
        .ok_or_else(|| corrupt(*pos, "footer ended unexpectedly"))?;
    *pos += 1;
    Ok(value)
}

fn write_zone_map(zone: &ZoneMap, out: &mut Vec<u8>) {
    write_optional_value(zone.min.as_ref(), out);
    write_optional_value(zone.max.as_ref(), out);
    varint::write_uvarint(zone.null_count, out);
    varint::write_uvarint(zone.distinct_estimate, out);
    varint::write_uvarint(zone.row_count, out);
}

fn read_zone_map(bytes: &[u8], pos: &mut usize) -> Result<ZoneMap> {
    let min = read_optional_value(bytes, pos)?;
    let max = read_optional_value(bytes, pos)?;
    let null_count = varint::read_uvarint(bytes, pos)?;
    let distinct_estimate = varint::read_uvarint(bytes, pos)?;
    let row_count = varint::read_uvarint(bytes, pos)?;
    if null_count > row_count {
        return Err(corrupt(
            *pos,
            format!("zone map reports {null_count} nulls in {row_count} rows"),
        ));
    }
    if let (Some(min), Some(max)) = (&min, &max) {
        if min > max {
            return Err(corrupt(
                *pos,
                format!("zone map has min {min} above max {max}"),
            ));
        }
    }
    Ok(ZoneMap {
        min,
        max,
        null_count,
        distinct_estimate,
        row_count,
    })
}

fn write_optional_value(value: Option<&Value>, out: &mut Vec<u8>) {
    match value {
        Some(value) => {
            out.push(1);
            serial::write_value(value, out);
        }
        None => out.push(0),
    }
}

fn read_optional_value(bytes: &[u8], pos: &mut usize) -> Result<Option<Value>> {
    match read_u8(bytes, pos)? {
        0 => Ok(None),
        1 => serial::read_value(bytes, pos).map(Some),
        other => Err(corrupt(*pos, format!("bad optional-value tag {other}"))),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::format::FORMAT_VERSION;
    use basalt_types::{DataType, Field};

    fn sample() -> SegmentMetadata {
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("city", DataType::Utf8, true),
            ])
            .unwrap(),
        );
        let chunk = |offset: u64| ColumnChunkMetadata {
            data: BlockLocation {
                offset,
                length: 128,
            },
            encoding: Encoding::DeltaBinaryPacked,
            uncompressed_bytes: 8192,
            zone_map: ZoneMap {
                min: Some(Value::Int64(1)),
                max: Some(Value::Int64(999)),
                null_count: 3,
                distinct_estimate: 900,
                row_count: 1024,
            },
            bloom: Some(BlockLocation {
                offset: offset + 128,
                length: 1200,
            }),
        };
        SegmentMetadata {
            format_version: crate::format::FORMAT_VERSION,
            schema,
            row_groups: vec![
                RowGroupMetadata {
                    num_rows: 1024,
                    columns: vec![chunk(16), chunk(1500)],
                },
                RowGroupMetadata {
                    num_rows: 512,
                    columns: vec![chunk(3000), chunk(4000)],
                },
            ],
            total_rows: 1536,
            metadata: BTreeMap::from([("writer".to_string(), "basalt".to_string())]),
        }
    }

    #[test]
    fn footer_round_trips() {
        let metadata = sample();
        let bytes = metadata.to_bytes();
        assert_eq!(
            SegmentMetadata::from_bytes(&bytes, FORMAT_VERSION).unwrap(),
            metadata
        );
        assert_eq!(metadata.num_row_groups(), 2);
        assert_eq!(metadata.compressed_size(), ByteSize::new(4 * 128));
        assert_eq!(metadata.uncompressed_size(), ByteSize::new(4 * 8192));
        assert_eq!(
            metadata.row_groups[0].columns[0].compression_ratio(),
            Some(64.0)
        );
        assert_eq!(
            metadata.row_groups[0].columns[0].compressed_size(),
            ByteSize::new(128)
        );
        assert_eq!(
            metadata.row_groups[0].columns[0].bloom.unwrap().size(),
            ByteSize::new(1200)
        );
    }

    #[test]
    fn footer_rejects_inconsistent_metadata() {
        let mut metadata = sample();
        metadata.total_rows = 99;
        let err = SegmentMetadata::from_bytes(&metadata.to_bytes(), FORMAT_VERSION).unwrap_err();
        assert!(err.to_string().contains("row groups sum to"), "{err}");

        let mut metadata = sample();
        metadata.row_groups[0].columns[0].zone_map.null_count = 99_999;
        let err = SegmentMetadata::from_bytes(&metadata.to_bytes(), FORMAT_VERSION).unwrap_err();
        assert!(err.to_string().contains("nulls in"), "{err}");

        let mut metadata = sample();
        metadata.row_groups[0].columns[0].zone_map.min = Some(Value::Int64(10_000));
        let err = SegmentMetadata::from_bytes(&metadata.to_bytes(), FORMAT_VERSION).unwrap_err();
        assert!(err.to_string().contains("above max"), "{err}");
    }

    #[test]
    fn footer_rejects_damage() {
        let bytes = sample().to_bytes();
        assert!(SegmentMetadata::from_bytes(b"", FORMAT_VERSION).is_err());
        let mut wrong_magic = bytes.clone();
        wrong_magic[1] = b'X';
        let err = SegmentMetadata::from_bytes(&wrong_magic, FORMAT_VERSION).unwrap_err();
        assert!(err.to_string().contains("BFTR magic"), "{err}");
        let mut wrong_version = bytes.clone();
        wrong_version[4] = 42;
        assert!(SegmentMetadata::from_bytes(&wrong_version, FORMAT_VERSION).is_err());
        for cut in [5, bytes.len() / 3, bytes.len() - 1] {
            assert!(
                SegmentMetadata::from_bytes(&bytes[..cut], FORMAT_VERSION).is_err(),
                "cut {cut}"
            );
        }
    }

    #[test]
    fn all_null_zone_maps_round_trip() {
        let mut metadata = sample();
        metadata.row_groups[0].columns[0].zone_map = ZoneMap {
            min: None,
            max: None,
            null_count: 1024,
            distinct_estimate: 0,
            row_count: 1024,
        };
        metadata.row_groups[0].columns[0].bloom = None;
        let bytes = metadata.to_bytes();
        assert_eq!(
            SegmentMetadata::from_bytes(&bytes, FORMAT_VERSION).unwrap(),
            metadata
        );
    }
}

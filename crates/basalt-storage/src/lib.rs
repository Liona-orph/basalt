//! The Basalt on-disk segment format.
//!
//! A **segment** is one immutable columnar file. It holds a schema, a sequence
//! of **row groups**, and per-**column chunk** metadata rich enough that a query
//! can decide not to read most of the file. The byte-level specification is in
//! [`FORMAT.md`](https://github.com/basalt-db/basalt/blob/main/crates/basalt-storage/FORMAT.md);
//! this crate is its implementation, and the two are kept in step deliberately.
//!
//! ```text
//! +---------------------+
//! | file header (16 B)  |  magic "BSLT", format version
//! +---------------------+
//! | row group 0         |  column chunk 0 : [len][crc32c][encoded values]
//! |                     |  bloom filter 0 : [len][crc32c][filter]      (optional)
//! |                     |  column chunk 1 : ...
//! +---------------------+
//! | row group 1         |  ...
//! +---------------------+
//! | footer block        |  [len][crc32c][schema + per-chunk metadata]
//! +---------------------+
//! | trailer (16 B)      |  footer offset, footer length, magic "BSLT"
//! +---------------------+
//! ```
//!
//! # The three things this crate is for
//!
//! **Corruption is detected, not returned.** Every persisted block carries a
//! [CRC32C](crc) over its payload. A flipped bit produces a
//! [`BasaltError::Corruption`](basalt_common::BasaltError::Corruption) naming
//! the structural region and the exact byte offset, never a plausible-looking
//! wrong answer.
//!
//! **Projection pushdown.** Column chunks are addressed individually in the
//! footer, so reading one column of forty reads one fortieth of the bytes. This
//! is verifiable, not just claimed: wrap the source in a [`CountingSource`] and
//! look.
//!
//! **Predicate pushdown.** Every chunk carries a [`ZoneMap`] (min, max, null
//! count, distinct estimate) and optionally a [`BloomFilter`]. A [`Predicate`]
//! is evaluated against the metadata first — zone maps are free, bloom filters
//! cost one small read and are only consulted for equality tests that survive
//! the zone-map stage — and whole row groups are skipped before any data is
//! touched.
//!
//! ```
//! use std::sync::Arc;
//! use basalt_common::Config;
//! use basalt_storage::{CountingSource, Predicate, ScanOptions, SegmentReader, SegmentWriter};
//! use basalt_types::{ColumnVector, DataType, Field, PrimitiveArray, RecordBatch, Schema, Value};
//!
//! let schema = Arc::new(Schema::new(vec![
//!     Field::new("id", DataType::Int64, false),
//!     Field::new("payload", DataType::Int64, false),
//! ])?);
//! let config = Config::default().with_batch_size(1024).with_row_group_rows(4096);
//!
//! let mut bytes = Vec::new();
//! let mut writer = SegmentWriter::new(&mut bytes, Arc::clone(&schema), config)?;
//! for group in 0..8i64 {
//!     let ids: Vec<i64> = (0..4096).map(|i| group * 4096 + i).collect();
//!     let payload: Vec<i64> = ids.iter().map(|i| i * 7).collect();
//!     writer.write_batch(&RecordBatch::try_new(
//!         Arc::clone(&schema),
//!         vec![
//!             ColumnVector::Int64(PrimitiveArray::from_vec(ids)),
//!             ColumnVector::Int64(PrimitiveArray::from_vec(payload)),
//!         ],
//!     )?)?;
//! }
//! let metadata = writer.finish()?;
//! assert_eq!(metadata.num_row_groups(), 8);
//!
//! // `id = 20000` lives in exactly one row group; the other seven are skipped
//! // using their zone maps, and only the `id` column is read.
//! let reader = SegmentReader::open(CountingSource::new(bytes))?;
//! let options = ScanOptions::new()
//!     .with_projection(vec![0])
//!     .with_predicate(Predicate::eq(0, Value::Int64(20_000)));
//! reader.source().reset();
//! let batches: Vec<RecordBatch> = reader.scan(&options)?.collect::<Result<_, _>>()?;
//! assert_eq!(batches.len(), 1);
//! assert_eq!(reader.prune_row_groups(options.predicate())?.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![forbid(unsafe_code)]

pub mod crc;
pub mod format;
mod metadata;
mod predicate;
mod reader;
mod serial;
mod writer;

pub mod bloom;
pub mod zone_map;

pub use bloom::{value_hash, BloomFilter};
pub use format::{
    BLOCK_HEADER_LEN, FILE_HEADER_LEN, FOOTER_MAGIC, FORMAT_VERSION, MAGIC, MAX_BLOCK_PAYLOAD,
    TRAILER_LEN,
};
pub use metadata::{BlockLocation, ColumnChunkMetadata, RowGroupMetadata, SegmentMetadata};
pub use predicate::{Predicate, Verdict};
pub use reader::{
    CountingSource, FileSource, PruneReport, ScanOptions, SegmentReader, SegmentScan, SegmentSource,
};
pub use writer::SegmentWriter;
pub use zone_map::{exact_distinct_count, ZoneMap};

//! Columnar compression for Basalt: codecs, and the heuristic that picks one.
//!
//! Everything here is written from first principles — bit packing, delta,
//! dictionary, run-length, frame-of-reference, varint and an LZ77 byte
//! compressor. No compression crate is used, because "we call `zstd`" is not a
//! demonstration of anything.
//!
//! # Layers
//!
//! 1. **Primitives** — [`bitpack`] (arbitrary widths 1..=64) and [`varint`]
//!    (zigzag LEB128).
//! 2. **Codecs** — [`PlainCodec`], [`VarintCodec`], [`BitPackedCodec`],
//!    [`FrameOfReferenceCodec`], [`RunLengthCodec`], [`DictionaryCodec`],
//!    [`DeltaBinaryPackedCodec`] over `[i64]`; [`PlainBytesCodec`] and
//!    [`ByteLzCodec`] over `[u8]`. All implement [`Codec`] and all write a
//!    self-describing header.
//! 3. **Selection** — [`analyze`] collects statistics in one pass, estimates the
//!    encoded size of every candidate in closed form, and returns the winner.
//!    See the [`stats`] module docs for the full decision procedure.
//! 4. **Columns** — [`encode_column`] / [`decode_column`] apply all of the above
//!    to a [`ColumnVector`](basalt_types::ColumnVector), including nulls and
//!    variable-length data.
//!
//! ```
//! use basalt_common::CompressionLevel;
//! use basalt_encoding::{analyze, decode_column, encode_column, Encoding};
//! use basalt_types::{ColumnVector, DataType, PrimitiveArray};
//!
//! // Statistics-driven selection: a clustered column becomes frame-of-reference.
//! let readings: Vec<i64> = (0..8192).map(|i| 1_699_000_000 + (i * 37) % 900).collect();
//! let plan = analyze(&readings, CompressionLevel::BALANCED);
//! assert_eq!(plan.encoding, Encoding::FrameOfReference);
//! assert!(plan.estimated_ratio() > 5.0);
//!
//! // And the whole column round-trips through the chosen encoding.
//! let column = ColumnVector::Int64(PrimitiveArray::from_vec(readings));
//! let encoded = encode_column(&column, CompressionLevel::BALANCED)?;
//! assert_eq!(decode_column(&encoded.bytes, &DataType::Int64)?, column);
//! # Ok::<(), basalt_common::BasaltError>(())
//! ```
#![forbid(unsafe_code)]

pub mod bitpack;
pub mod varint;

mod bytes;
mod codec;
mod column;
mod integer;
pub mod stats;

pub use bytes::{decode_bytes, encode_bytes_adaptive, ByteLzCodec, PlainBytesCodec};
pub use codec::{peek_encoding, Codec, Encoding, CODEC_VERSION, MAX_CHUNK_ELEMENTS};
pub use column::{
    decode_column, encode_column, time_unit_of, uncompressed_size, EncodedColumn, CHUNK_VERSION,
};
pub use integer::{
    decode_integers, encode_integers, BitPackedCodec, DeltaBinaryPackedCodec, DictionaryCodec,
    FrameOfReferenceCodec, PlainCodec, RunLengthCodec, VarintCodec, DELTA_BLOCK_SIZE,
};
pub use stats::{
    analyze, analyze_byte_column, analyze_integers, choose_integer_encoding, estimate_sizes,
    ByteLayout, BytePlan, Cardinality, EncodingPlan, IntegerStats, PlanReason, SizeEstimates,
    Sortedness, MAX_DICTIONARY_CARDINALITY,
};

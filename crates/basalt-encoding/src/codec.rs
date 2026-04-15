//! The [`Codec`] trait, the [`Encoding`] tag and the self-describing header
//! every encoded buffer starts with.

use std::fmt;

use basalt_common::{BasaltError, Region, Result};

use crate::varint;

/// Hard upper bound on the number of values a single encoded buffer may claim
/// to hold.
///
/// This is a **format limit**, not a tuning knob, and it exists for exactly one
/// reason: a damaged header can otherwise declare `2^60` elements and the
/// decoder will try to allocate for them before it ever reads a byte of body.
/// Some encodings (bit packing at width 0, run-length) legitimately describe
/// many values in very few bytes, so "the body is too short" is not a sufficient
/// defence on its own.
///
/// 2^24 = 16.7 million values per chunk, or 128 MiB as `i64`. Row groups are
/// four orders of magnitude smaller than that by default
/// ([`Config::row_group_rows`](basalt_common::Config::row_group_rows) is 65 536),
/// so no honest writer comes close.
pub const MAX_CHUNK_ELEMENTS: usize = 1 << 24;

/// Version of the codec framing. Bumped only if the header layout changes;
/// individual codecs are versioned by their [`Encoding`] id instead, so adding
/// a codec never invalidates existing files.
pub const CODEC_VERSION: u8 = 1;

/// Identifies the codec that produced a buffer.
///
/// The discriminants are part of the on-disk format and must never be reused
/// for a different meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Encoding {
    /// Fixed 8-byte little-endian integers, or raw bytes.
    Plain = 1,
    /// Zigzag LEB128 variable-length integers.
    Varint = 2,
    /// Zigzag values packed at a fixed bit width.
    BitPacked = 3,
    /// A single `min` plus bit-packed offsets from it.
    FrameOfReference = 4,
    /// (value, run length) pairs.
    RunLength = 5,
    /// A sorted dictionary plus bit-packed codes.
    Dictionary = 6,
    /// Blocks of bit-packed deltas with a per-block minimum.
    DeltaBinaryPacked = 7,
    /// Raw bytes, no transformation.
    PlainBytes = 8,
    /// The in-tree LZ77 byte compressor.
    ByteLz = 9,
}

impl Encoding {
    /// The on-disk discriminant.
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Parse a discriminant read from disk.
    pub const fn from_u8(raw: u8) -> Option<Encoding> {
        Some(match raw {
            1 => Encoding::Plain,
            2 => Encoding::Varint,
            3 => Encoding::BitPacked,
            4 => Encoding::FrameOfReference,
            5 => Encoding::RunLength,
            6 => Encoding::Dictionary,
            7 => Encoding::DeltaBinaryPacked,
            8 => Encoding::PlainBytes,
            9 => Encoding::ByteLz,
            _ => return None,
        })
    }

    /// Relative cost of *decoding* this encoding, lowest is cheapest.
    ///
    /// Used only to break ties between encodings whose estimated sizes are
    /// within a hair of each other: when two encodings produce the same number
    /// of bytes, take the one that scans faster. The ordering reflects how much
    /// per-value work the decoder does:
    ///
    /// * `Plain` — a `memcpy`, nothing per value.
    /// * `BitPacked` / `FrameOfReference` — branch-free shifts, vectorisable.
    /// * `DeltaBinaryPacked` — the above plus a serial prefix sum.
    /// * `Dictionary` — the above plus a random-access gather.
    /// * `RunLength` — data-dependent inner loop.
    /// * `Varint` — one unpredictable branch per byte, the slowest of the set.
    pub const fn decode_cost_rank(self) -> u8 {
        match self {
            Encoding::Plain | Encoding::PlainBytes => 0,
            Encoding::BitPacked | Encoding::FrameOfReference => 1,
            Encoding::DeltaBinaryPacked => 2,
            Encoding::Dictionary => 3,
            Encoding::RunLength => 4,
            Encoding::Varint => 5,
            Encoding::ByteLz => 6,
        }
    }

    /// `true` if this codec consumes integers rather than raw bytes.
    pub const fn is_integer_encoding(self) -> bool {
        !matches!(self, Encoding::PlainBytes | Encoding::ByteLz)
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Encoding::Plain => "PLAIN",
            Encoding::Varint => "VARINT",
            Encoding::BitPacked => "BITPACKED",
            Encoding::FrameOfReference => "FOR",
            Encoding::RunLength => "RLE",
            Encoding::Dictionary => "DICTIONARY",
            Encoding::DeltaBinaryPacked => "DELTA",
            Encoding::PlainBytes => "PLAIN_BYTES",
            Encoding::ByteLz => "BYTE_LZ",
        };
        f.write_str(name)
    }
}

/// A compression codec.
///
/// # One trait, two instantiations
///
/// Basalt has exactly two kinds of codec and they consume different things, so
/// `Codec` is generic over its input rather than being split into two traits:
///
/// * **Integer codecs** — `Input = [i64]`, `Output = Vec<i64>`. Every fixed-width
///   physical type is funnelled through `i64` before encoding (unsigned types
///   and floats by bit pattern, see [`crate::encode_column`]), so one set of
///   codecs covers all of them and there is one implementation to test.
/// * **Byte codecs** — `Input = [u8]`, `Output = Vec<u8>`. Used for the data
///   buffer of variable-length columns.
///
/// Every `encode` writes a **self-describing header** first
/// (`[encoding id][format version][element count]`), so a buffer can be decoded
/// with [`crate::decode_integers`] without external metadata, and a decoder that
/// is handed the wrong buffer says so instead of producing garbage.
///
/// ```
/// use basalt_encoding::{Codec, Encoding, RunLengthCodec};
///
/// let values = vec![7i64, 7, 7, 7, 9, 9];
/// let mut buf = Vec::new();
/// RunLengthCodec.encode(&values, &mut buf)?;
/// assert_eq!(buf[0], Encoding::RunLength.as_u8());
/// assert_eq!(RunLengthCodec.decode(&buf)?, values);
/// # Ok::<(), basalt_common::BasaltError>(())
/// ```
pub trait Codec {
    /// What the codec compresses.
    type Input: ?Sized;
    /// What decoding produces.
    type Output;

    /// The tag this codec writes into the header.
    fn encoding(&self) -> Encoding;

    /// Append the encoded form of `input` to `out`.
    fn encode(&self, input: &Self::Input, out: &mut Vec<u8>) -> Result<()>;

    /// Decode a buffer produced by [`Codec::encode`].
    fn decode(&self, bytes: &[u8]) -> Result<Self::Output>;

    /// Convenience: encode into a fresh `Vec`.
    fn encode_to_vec(&self, input: &Self::Input) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode(input, &mut out)?;
        Ok(out)
    }
}

/// Write `[encoding][version][count]`.
pub(crate) fn write_header(out: &mut Vec<u8>, encoding: Encoding, count: usize) {
    out.push(encoding.as_u8());
    out.push(CODEC_VERSION);
    varint::write_uvarint(count as u64, out);
}

/// Read and validate a header, returning `(element count, offset of the body)`.
///
/// `expected` is checked so that decoding the wrong buffer is a precise error
/// rather than a stream of plausible-looking nonsense.
pub(crate) fn read_header(bytes: &[u8], expected: Encoding) -> Result<(usize, usize)> {
    if bytes.len() < 2 {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            0,
            format!(
                "buffer of {} bytes is too short to hold a codec header",
                bytes.len()
            ),
        ));
    }
    let Some(found) = Encoding::from_u8(bytes[0]) else {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            0,
            format!("unknown encoding id {}", bytes[0]),
        ));
    };
    if found != expected {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            0,
            format!("expected {expected} data but the header says {found}"),
        ));
    }
    if bytes[1] != CODEC_VERSION {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            1,
            format!(
                "codec format version {} is not supported (expected {CODEC_VERSION})",
                bytes[1]
            ),
        ));
    }
    let mut pos = 2;
    let count = varint::read_uvarint(bytes, &mut pos)?;
    if count > MAX_CHUNK_ELEMENTS as u64 {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            2,
            format!("header claims {count} elements, above the {MAX_CHUNK_ELEMENTS} format limit"),
        ));
    }
    Ok((count as usize, pos))
}

/// A safe `Vec::with_capacity` hint for decoding.
///
/// Reserving `count` up front is the right thing for valid data and a denial of
/// service for damaged data, so the hint is additionally bounded by what the
/// remaining bytes could possibly describe. `min_bytes_per_value` is the
/// smallest number of body bytes one value can occupy; pass 0 when a value can
/// occupy none (bit packing at width 0), in which case only a fixed cap applies.
pub(crate) fn capacity_hint(count: usize, available: usize, min_bytes_per_value: usize) -> usize {
    const NO_BYTE_BOUND_CAP: usize = 1 << 20;
    if min_bytes_per_value == 0 {
        count.min(NO_BYTE_BOUND_CAP)
    } else {
        count.min(available / min_bytes_per_value + 1)
    }
}

/// Peek at the encoding tag of a buffer without decoding it.
pub fn peek_encoding(bytes: &[u8]) -> Result<Encoding> {
    bytes
        .first()
        .and_then(|b| Encoding::from_u8(*b))
        .ok_or_else(|| {
            BasaltError::corruption(
                "encoded chunk",
                Region::EncodedChunk,
                0,
                "buffer does not start with a known encoding id",
            )
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const ALL: &[Encoding] = &[
        Encoding::Plain,
        Encoding::Varint,
        Encoding::BitPacked,
        Encoding::FrameOfReference,
        Encoding::RunLength,
        Encoding::Dictionary,
        Encoding::DeltaBinaryPacked,
        Encoding::PlainBytes,
        Encoding::ByteLz,
    ];

    #[test]
    fn discriminants_round_trip() {
        for e in ALL {
            assert_eq!(Encoding::from_u8(e.as_u8()), Some(*e));
            assert!(!e.to_string().is_empty());
        }
        assert_eq!(Encoding::from_u8(0), None);
        assert_eq!(Encoding::from_u8(200), None);
    }

    #[test]
    fn header_round_trip() {
        let mut buf = Vec::new();
        write_header(&mut buf, Encoding::Dictionary, 1234);
        let (count, pos) = read_header(&buf, Encoding::Dictionary).unwrap();
        assert_eq!(count, 1234);
        assert_eq!(pos, buf.len());
        assert_eq!(peek_encoding(&buf).unwrap(), Encoding::Dictionary);
    }

    #[test]
    fn header_rejects_absurd_element_counts() {
        let mut buf = Vec::new();
        buf.push(Encoding::Plain.as_u8());
        buf.push(CODEC_VERSION);
        varint::write_uvarint(u64::MAX / 2, &mut buf);
        let err = read_header(&buf, Encoding::Plain).unwrap_err();
        assert!(err.to_string().contains("format limit"), "{err}");
    }

    #[test]
    fn capacity_hints_are_bounded() {
        assert_eq!(capacity_hint(10, 1000, 8), 10);
        assert_eq!(capacity_hint(1_000_000, 16, 8), 3);
        assert_eq!(capacity_hint(usize::MAX, 0, 0), 1 << 20);
    }

    #[test]
    fn header_rejects_mismatches() {
        let mut buf = Vec::new();
        write_header(&mut buf, Encoding::Plain, 3);
        let err = read_header(&buf, Encoding::RunLength).unwrap_err();
        assert!(
            err.to_string()
                .contains("expected RLE data but the header says PLAIN"),
            "{err}"
        );

        buf[1] = 99;
        let err = read_header(&buf, Encoding::Plain).unwrap_err();
        assert!(err.to_string().contains("format version 99"), "{err}");

        buf[0] = 77;
        assert!(read_header(&buf, Encoding::Plain).is_err());
        assert!(peek_encoding(&buf).is_err());
        assert!(read_header(&[], Encoding::Plain).is_err());
    }

    #[test]
    fn cost_ranks_are_ordered_sensibly() {
        assert!(Encoding::Plain.decode_cost_rank() < Encoding::BitPacked.decode_cost_rank());
        assert!(Encoding::BitPacked.decode_cost_rank() < Encoding::Dictionary.decode_cost_rank());
        assert!(Encoding::Dictionary.decode_cost_rank() < Encoding::Varint.decode_cost_rank());
        assert!(Encoding::Plain.is_integer_encoding());
        assert!(!Encoding::ByteLz.is_integer_encoding());
    }
}

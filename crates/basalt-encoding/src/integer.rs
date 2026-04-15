//! Integer codecs.
//!
//! Every fixed-width physical type in Basalt reaches these codecs as `&[i64]`
//! (see [`crate::encode_column`] for the widening and bit-casting rules), so
//! there is exactly one implementation of each technique to write, test and
//! optimise.

use basalt_common::{BasaltError, Region, Result};

use crate::bitpack::{self, packed_len, required_width};
use crate::codec::{capacity_hint, peek_encoding, read_header, write_header, Codec, Encoding};
use crate::varint;

/// Number of deltas per block in [`DeltaBinaryPackedCodec`].
///
/// 128 is the usual choice: large enough that the per-block header (a varint
/// minimum plus a width byte, so 2-11 bytes) amortises away, small enough that a
/// local run of similar deltas gets its own tight bit width instead of being
/// dragged up by an outlier at the other end of the column.
pub const DELTA_BLOCK_SIZE: usize = 128;

fn corrupt(offset: usize, detail: impl Into<String>) -> BasaltError {
    BasaltError::corruption("encoded chunk", Region::EncodedChunk, offset as u64, detail)
}

fn check_count(decoded: usize, expected: usize) -> Result<()> {
    if decoded == expected {
        Ok(())
    } else {
        Err(corrupt(
            0,
            format!("header promised {expected} values but the body held {decoded}"),
        ))
    }
}

/// Uncompressed 8-byte little-endian integers.
///
/// The baseline every other codec is measured against, and the right answer for
/// high-entropy data: it costs nothing to decode and never inflates.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlainCodec;

impl Codec for PlainCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::Plain
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::Plain, input.len());
        out.reserve(input.len() * 8);
        for value in input {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, mut pos) = read_header(bytes, Encoding::Plain)?;
        if bytes.len() - pos < count * 8 {
            return Err(corrupt(
                pos,
                format!(
                    "plain body needs {} bytes for {count} values, found {}",
                    count * 8,
                    bytes.len() - pos
                ),
            ));
        }
        let mut out = Vec::with_capacity(capacity_hint(count, bytes.len() - pos, 8));
        for _ in 0..count {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[pos..pos + 8]);
            out.push(i64::from_le_bytes(buf));
            pos += 8;
        }
        Ok(out)
    }
}

/// Zigzag LEB128 variable-length integers.
///
/// Wins on data whose magnitudes are small but whose *range* is large enough
/// that a single bit width would be wasteful — a column of mostly-small values
/// with a few large ones. Slowest to decode of the integer codecs (one
/// unpredictable branch per byte), so the selector only picks it when the size
/// win is real.
#[derive(Debug, Clone, Copy, Default)]
pub struct VarintCodec;

impl Codec for VarintCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::Varint
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::Varint, input.len());
        for &value in input {
            varint::write_varint(value, out);
        }
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, mut pos) = read_header(bytes, Encoding::Varint)?;
        let mut out = Vec::with_capacity(capacity_hint(count, bytes.len() - pos, 1));
        for _ in 0..count {
            out.push(varint::read_varint(bytes, &mut pos)?);
        }
        Ok(out)
    }
}

/// Zigzag values packed at one fixed bit width.
///
/// The width is `required_width(max zigzag value)`, so a column of small signed
/// numbers collapses to a few bits each with no per-value framing. Decoding is
/// branch-free.
#[derive(Debug, Clone, Copy, Default)]
pub struct BitPackedCodec;

impl Codec for BitPackedCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::BitPacked
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        let zigzag: Vec<u64> = input.iter().map(|&v| varint::zigzag_encode(v)).collect();
        let width = required_width(zigzag.iter().copied().max().unwrap_or(0));
        write_header(out, Encoding::BitPacked, input.len());
        out.push(width as u8);
        bitpack::pack(&zigzag, width, out)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, pos) = read_header(bytes, Encoding::BitPacked)?;
        let width = u32::from(
            *bytes
                .get(pos)
                .ok_or_else(|| corrupt(pos, "missing bit width"))?,
        );
        let packed = bitpack::unpack(&bytes[pos + 1..], width, count)?;
        Ok(packed.into_iter().map(varint::zigzag_decode).collect())
    }
}

/// A single `min` plus bit-packed unsigned offsets from it.
///
/// The classic frame-of-reference transform. It is the right encoding for
/// clustered values with a large absolute magnitude — timestamps, ids, prices —
/// where the *range* is small even though the values are not. A column of
/// microsecond timestamps within one day needs 37 bits plain but 17 after
/// subtracting the minimum.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameOfReferenceCodec;

impl Codec for FrameOfReferenceCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::FrameOfReference
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        let min = input.iter().copied().min().unwrap_or(0);
        // `wrapping_sub` on i64 then reinterpreted as u64 is the exact unsigned
        // difference whenever `value >= min`, including the full-range case
        // (i64::MIN to i64::MAX = u64::MAX) that a checked subtraction rejects.
        let offsets: Vec<u64> = input.iter().map(|&v| v.wrapping_sub(min) as u64).collect();
        let width = required_width(offsets.iter().copied().max().unwrap_or(0));
        write_header(out, Encoding::FrameOfReference, input.len());
        out.extend_from_slice(&min.to_le_bytes());
        out.push(width as u8);
        bitpack::pack(&offsets, width, out)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, pos) = read_header(bytes, Encoding::FrameOfReference)?;
        if bytes.len() < pos + 9 {
            return Err(corrupt(pos, "frame-of-reference header is truncated"));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[pos..pos + 8]);
        let min = i64::from_le_bytes(buf);
        let width = u32::from(bytes[pos + 8]);
        let offsets = bitpack::unpack(&bytes[pos + 9..], width, count)?;
        Ok(offsets
            .into_iter()
            .map(|o| min.wrapping_add(o as i64))
            .collect())
    }
}

/// `(value, run length)` pairs.
///
/// Unbeatable on low-entropy columns — flags, status codes, sorted keys with
/// repeats, and any column stored in sorted order. Degrades to slightly worse
/// than varint when there are no runs, which the selector notices.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunLengthCodec;

impl Codec for RunLengthCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::RunLength
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::RunLength, input.len());
        let mut runs: Vec<(i64, u64)> = Vec::new();
        for &value in input {
            match runs.last_mut() {
                Some((v, n)) if *v == value => *n += 1,
                _ => runs.push((value, 1)),
            }
        }
        varint::write_uvarint(runs.len() as u64, out);
        for (value, length) in runs {
            varint::write_varint(value, out);
            varint::write_uvarint(length, out);
        }
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, mut pos) = read_header(bytes, Encoding::RunLength)?;
        let run_count = varint::read_uvarint(bytes, &mut pos)? as usize;
        if run_count > count {
            return Err(corrupt(
                pos,
                format!("{run_count} runs cannot describe {count} values"),
            ));
        }
        let mut out = Vec::with_capacity(capacity_hint(count, bytes.len() - pos, 0));
        for _ in 0..run_count {
            let value = varint::read_varint(bytes, &mut pos)?;
            let length = varint::read_uvarint(bytes, &mut pos)? as usize;
            // `count - out.len()` rather than `out.len() + length` because a
            // damaged length can be close to `usize::MAX` and the addition
            // would overflow before the comparison ever ran.
            if length > count - out.len() {
                return Err(corrupt(
                    pos,
                    format!("run lengths sum to more than the {count} values in the header"),
                ));
            }
            out.resize(out.len() + length, value);
        }
        check_count(out.len(), count)?;
        Ok(out)
    }
}

/// A sorted dictionary of distinct values plus bit-packed codes.
///
/// The right answer whenever cardinality is far below the row count: a column of
/// 10 million rows drawn from 200 distinct values becomes 200 varints plus 8
/// bits per row. Sorting the dictionary is not cosmetic — it lets the dictionary
/// itself be delta-encoded, and it means a future comparison kernel can evaluate
/// `WHERE col < x` directly on the codes.
#[derive(Debug, Clone, Copy, Default)]
pub struct DictionaryCodec;

impl Codec for DictionaryCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::Dictionary
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        let mut dictionary: Vec<i64> = input.to_vec();
        dictionary.sort_unstable();
        dictionary.dedup();
        let codes: Vec<u64> = input
            .iter()
            .map(|v| dictionary.partition_point(|d| d < v) as u64)
            .collect();
        let width = required_width(dictionary.len().saturating_sub(1) as u64);

        write_header(out, Encoding::Dictionary, input.len());
        varint::write_uvarint(dictionary.len() as u64, out);
        // The dictionary is strictly increasing, so store the first value and
        // then unsigned gaps: for clustered keys this is a handful of bytes.
        if let Some(&first) = dictionary.first() {
            varint::write_varint(first, out);
            for pair in dictionary.windows(2) {
                varint::write_uvarint(pair[1].wrapping_sub(pair[0]) as u64, out);
            }
        }
        out.push(width as u8);
        bitpack::pack(&codes, width, out)
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, mut pos) = read_header(bytes, Encoding::Dictionary)?;
        let dictionary_len = varint::read_uvarint(bytes, &mut pos)? as usize;
        // Every dictionary entry costs at least one body byte, so a dictionary
        // larger than the remaining buffer is corruption by construction.
        if dictionary_len > bytes.len() - pos {
            return Err(corrupt(
                pos,
                format!(
                    "dictionary claims {dictionary_len} entries but only {} bytes remain",
                    bytes.len() - pos
                ),
            ));
        }
        let mut dictionary = Vec::with_capacity(dictionary_len);
        if dictionary_len > 0 {
            let mut current = varint::read_varint(bytes, &mut pos)?;
            dictionary.push(current);
            for _ in 1..dictionary_len {
                let gap = varint::read_uvarint(bytes, &mut pos)?;
                current = current.wrapping_add(gap as i64);
                dictionary.push(current);
            }
        }
        let width = u32::from(
            *bytes
                .get(pos)
                .ok_or_else(|| corrupt(pos, "missing code width"))?,
        );
        let codes = bitpack::unpack(&bytes[pos + 1..], width, count)?;
        let mut out = Vec::with_capacity(codes.len());
        for code in codes {
            let index = code as usize;
            let value = dictionary.get(index).ok_or_else(|| {
                corrupt(
                    pos,
                    format!(
                        "dictionary code {index} is outside a {dictionary_len}-entry dictionary"
                    ),
                )
            })?;
            out.push(*value);
        }
        Ok(out)
    }
}

/// Blocks of bit-packed deltas, each with its own minimum.
///
/// Built for monotonic data: primary keys, insertion timestamps, offsets. A
/// strictly increasing column with a stable step compresses to almost nothing,
/// because after subtracting the block minimum every delta is zero and the bit
/// width drops to 0. Per-block minima keep a single jump in the middle of a
/// column from inflating the whole thing.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaBinaryPackedCodec;

impl Codec for DeltaBinaryPackedCodec {
    type Input = [i64];
    type Output = Vec<i64>;

    fn encoding(&self) -> Encoding {
        Encoding::DeltaBinaryPacked
    }

    fn encode(&self, input: &[i64], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::DeltaBinaryPacked, input.len());
        let Some(&first) = input.first() else {
            return Ok(());
        };
        varint::write_varint(first, out);
        varint::write_uvarint(DELTA_BLOCK_SIZE as u64, out);
        let deltas: Vec<i64> = input.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
        for block in deltas.chunks(DELTA_BLOCK_SIZE) {
            let min_delta = block.iter().copied().min().unwrap_or(0);
            let offsets: Vec<u64> = block
                .iter()
                .map(|&d| d.wrapping_sub(min_delta) as u64)
                .collect();
            let width = required_width(offsets.iter().copied().max().unwrap_or(0));
            varint::write_varint(min_delta, out);
            out.push(width as u8);
            bitpack::pack(&offsets, width, out)?;
        }
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<i64>> {
        let (count, mut pos) = read_header(bytes, Encoding::DeltaBinaryPacked)?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let first = varint::read_varint(bytes, &mut pos)?;
        let block_size = varint::read_uvarint(bytes, &mut pos)? as usize;
        if block_size == 0 || block_size > 1 << 20 {
            return Err(corrupt(
                pos,
                format!("implausible delta block size {block_size}"),
            ));
        }
        let mut out = Vec::with_capacity(capacity_hint(count, bytes.len() - pos, 0));
        out.push(first);
        let mut remaining = count - 1;
        while remaining > 0 {
            let block_len = remaining.min(block_size);
            let min_delta = varint::read_varint(bytes, &mut pos)?;
            let width = u32::from(
                *bytes
                    .get(pos)
                    .ok_or_else(|| corrupt(pos, "missing delta width"))?,
            );
            pos += 1;
            let offsets = bitpack::unpack(&bytes[pos..], width, block_len)?;
            pos += packed_len(block_len, width);
            let mut current = *out.last().unwrap_or(&first);
            for offset in offsets {
                current = current.wrapping_add(min_delta.wrapping_add(offset as i64));
                out.push(current);
            }
            remaining -= block_len;
        }
        check_count(out.len(), count)?;
        Ok(out)
    }
}

/// Decode any integer buffer by looking at its header.
///
/// This is the reader-side entry point: the segment format stores which encoding
/// it used, but the buffer is self-describing anyway, so a mismatch between
/// metadata and payload is caught rather than silently mis-decoded.
///
/// ```
/// use basalt_encoding::{decode_integers, Codec, FrameOfReferenceCodec};
///
/// let values = vec![1_000_000i64, 1_000_004, 1_000_002];
/// let buf = FrameOfReferenceCodec.encode_to_vec(&values)?;
/// assert!(buf.len() < values.len() * 8);
/// assert_eq!(decode_integers(&buf)?, values);
/// # Ok::<(), basalt_common::BasaltError>(())
/// ```
pub fn decode_integers(bytes: &[u8]) -> Result<Vec<i64>> {
    match peek_encoding(bytes)? {
        Encoding::Plain => PlainCodec.decode(bytes),
        Encoding::Varint => VarintCodec.decode(bytes),
        Encoding::BitPacked => BitPackedCodec.decode(bytes),
        Encoding::FrameOfReference => FrameOfReferenceCodec.decode(bytes),
        Encoding::RunLength => RunLengthCodec.decode(bytes),
        Encoding::Dictionary => DictionaryCodec.decode(bytes),
        Encoding::DeltaBinaryPacked => DeltaBinaryPackedCodec.decode(bytes),
        other => Err(corrupt(
            0,
            format!("{other} is a byte encoding, not an integer encoding"),
        )),
    }
}

/// Encode with a named integer encoding.
///
/// # Errors
///
/// If `encoding` is not an integer encoding.
pub fn encode_integers(values: &[i64], encoding: Encoding, out: &mut Vec<u8>) -> Result<()> {
    match encoding {
        Encoding::Plain => PlainCodec.encode(values, out),
        Encoding::Varint => VarintCodec.encode(values, out),
        Encoding::BitPacked => BitPackedCodec.encode(values, out),
        Encoding::FrameOfReference => FrameOfReferenceCodec.encode(values, out),
        Encoding::RunLength => RunLengthCodec.encode(values, out),
        Encoding::Dictionary => DictionaryCodec.encode(values, out),
        Encoding::DeltaBinaryPacked => DeltaBinaryPackedCodec.encode(values, out),
        other => Err(BasaltError::unsupported(format!(
            "{other} does not encode integers"
        ))),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn all_encodings() -> Vec<Encoding> {
        vec![
            Encoding::Plain,
            Encoding::Varint,
            Encoding::BitPacked,
            Encoding::FrameOfReference,
            Encoding::RunLength,
            Encoding::Dictionary,
            Encoding::DeltaBinaryPacked,
        ]
    }

    fn adversarial_inputs() -> Vec<(&'static str, Vec<i64>)> {
        vec![
            ("empty", vec![]),
            ("single", vec![42]),
            ("single extreme", vec![i64::MIN]),
            ("all same", vec![7; 300]),
            ("all zero", vec![0; 300]),
            ("sorted", (0..300).collect()),
            ("reverse sorted", (0..300).rev().collect()),
            (
                "constant step",
                (0..300).map(|i| 1_000_000 + i * 7).collect(),
            ),
            ("two values", (0..300).map(|i| i % 2).collect()),
            ("full range", vec![i64::MIN, i64::MAX, 0, -1, 1]),
            (
                "alternating extremes",
                (0..64)
                    .map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX })
                    .collect(),
            ),
            (
                "high entropy",
                (0..300)
                    .map(|i: i64| i.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(11))
                    .collect(),
            ),
            ("one outlier", {
                let mut v = vec![5i64; 200];
                v[100] = i64::MAX;
                v
            }),
        ]
    }

    #[test]
    fn every_codec_round_trips_every_shape() {
        for encoding in all_encodings() {
            for (name, values) in adversarial_inputs() {
                let mut buf = Vec::new();
                encode_integers(&values, encoding, &mut buf).unwrap();
                let decoded = decode_integers(&buf).unwrap();
                assert_eq!(decoded, values, "{encoding} on `{name}`");
                assert_eq!(peek_encoding(&buf).unwrap(), encoding);
            }
        }
    }

    #[test]
    fn encodings_actually_compress_their_target_shape() {
        let plain_size = |n: usize| n * 8 + 4;
        let sorted: Vec<i64> = (0..1000).map(|i| 1_700_000_000 + i * 3).collect();
        let delta = DeltaBinaryPackedCodec.encode_to_vec(&sorted).unwrap();
        assert!(
            delta.len() < plain_size(1000) / 20,
            "delta on monotonic: {}",
            delta.len()
        );

        let clustered: Vec<i64> = (0..1000).map(|i| 1_000_000_000 + (i % 50)).collect();
        let for_bytes = FrameOfReferenceCodec.encode_to_vec(&clustered).unwrap();
        assert!(
            for_bytes.len() < plain_size(1000) / 8,
            "FOR on clustered: {}",
            for_bytes.len()
        );

        let runs: Vec<i64> = (0..1000).map(|i| i / 100).collect();
        let rle = RunLengthCodec.encode_to_vec(&runs).unwrap();
        assert!(rle.len() < 60, "RLE on runs: {}", rle.len());

        let low_card: Vec<i64> = (0..1000).map(|i| [10, 2000, 30_000, -5][i % 4]).collect();
        let dict = DictionaryCodec.encode_to_vec(&low_card).unwrap();
        assert!(
            dict.len() < plain_size(1000) / 8,
            "dictionary on low cardinality: {}",
            dict.len()
        );

        let small: Vec<i64> = (0..1000).map(|i| (i % 7) - 3).collect();
        let packed = BitPackedCodec.encode_to_vec(&small).unwrap();
        assert!(
            packed.len() < plain_size(1000) / 10,
            "bitpacked on small values: {}",
            packed.len()
        );
    }

    #[test]
    fn truncation_is_detected_for_every_codec() {
        let values: Vec<i64> = (0..200).map(|i| i * 3 + 1).collect();
        for encoding in all_encodings() {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            for cut in [buf.len() / 2, buf.len() - 1] {
                let err = decode_integers(&buf[..cut]);
                assert!(
                    err.is_err(),
                    "{encoding} accepted a buffer truncated to {cut}"
                );
            }
        }
    }

    #[test]
    fn dictionary_rejects_out_of_range_codes() {
        let values = vec![1i64, 2, 3];
        let mut buf = DictionaryCodec.encode_to_vec(&values).unwrap();
        // The final byte holds the packed 2-bit codes; set them all to 3, which
        // is outside a 3-entry dictionary.
        let last = buf.len() - 1;
        buf[last] = 0xff;
        let err = DictionaryCodec.decode(&buf).unwrap_err();
        assert!(
            err.to_string().contains("outside a 3-entry dictionary"),
            "{err}"
        );
    }

    #[test]
    fn run_length_rejects_oversized_runs() {
        let mut buf = Vec::new();
        write_header(&mut buf, Encoding::RunLength, 3);
        varint::write_uvarint(1, &mut buf);
        varint::write_varint(9, &mut buf);
        varint::write_uvarint(100, &mut buf);
        let err = RunLengthCodec.decode(&buf).unwrap_err();
        assert!(err.to_string().contains("sum to more than"), "{err}");
    }

    #[test]
    fn delta_rejects_implausible_block_size() {
        let mut buf = Vec::new();
        write_header(&mut buf, Encoding::DeltaBinaryPacked, 5);
        varint::write_varint(1, &mut buf);
        varint::write_uvarint(0, &mut buf);
        let err = DeltaBinaryPackedCodec.decode(&buf).unwrap_err();
        assert!(
            err.to_string().contains("implausible delta block size"),
            "{err}"
        );
    }

    #[test]
    fn wrong_codec_is_refused() {
        let buf = PlainCodec.encode_to_vec(&[1i64, 2]).unwrap();
        assert!(RunLengthCodec.decode(&buf).is_err());
        assert!(encode_integers(&[1], Encoding::ByteLz, &mut Vec::new()).is_err());
        let mut byte_buf = Vec::new();
        write_header(&mut byte_buf, Encoding::PlainBytes, 0);
        assert!(decode_integers(&byte_buf).is_err());
    }

    #[test]
    fn delta_uses_multiple_blocks() {
        let values: Vec<i64> = (0..1000).collect();
        let buf = DeltaBinaryPackedCodec.encode_to_vec(&values).unwrap();
        assert_eq!(DeltaBinaryPackedCodec.decode(&buf).unwrap(), values);
        // 999 deltas / 128 = 8 blocks, all with delta 1 and width 0.
        assert!(
            buf.len() < 60,
            "perfectly linear column should be tiny, got {}",
            buf.len()
        );
    }
}

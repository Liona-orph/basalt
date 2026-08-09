//! Byte codecs: raw storage and an in-tree LZ77 compressor.
//!
//! # Why write the compressor
//!
//! Pulling in `lz4`/`zstd` would be one line of `Cargo.toml` and would defeat
//! the point of the project. The codec below is a complete LZ77 implementation:
//! a hash-chain-free match finder over 4-byte sequences, a token stream in the
//! LZ4 block style (a nibble each for literal and match length, with 255-style
//! continuation bytes, and a 16-bit back reference), and a decoder that handles
//! overlapping copies. It is not going to beat `zstd -19`, and it is not trying
//! to; it beats "no compression" by 2-8x on the string data columnar engines
//! actually store (URLs, city names, enum-like text) while decoding at
//! memcpy-adjacent speeds.

use basalt_common::{BasaltError, Region, Result};

use crate::codec::{capacity_hint, read_header, write_header, Codec, Encoding};
use crate::varint;

/// Shortest run of bytes worth encoding as a back reference.
///
/// Below 4, the 3-byte cost of a match (token + 2-byte offset) exceeds the
/// literals it replaces.
const MIN_MATCH: usize = 4;

/// Largest back reference, bounded by the 16-bit offset field.
const MAX_OFFSET: usize = 0xFFFF;

/// Match-finder table size, as a power of two.
///
/// 2^15 entries = 128 KiB of `u32`. Big enough that a 256 KiB block (Basalt's
/// default page size) does not thrash it, small enough to stay in L2.
const HASH_LOG: u32 = 15;

#[inline]
fn hash4(sequence: u32) -> usize {
    // Knuth's multiplicative hash; the high bits of the product are the
    // best-mixed, so shift them down.
    (sequence.wrapping_mul(2_654_435_761) >> (32 - HASH_LOG)) as usize
}

#[inline]
fn read_u32(src: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([src[at], src[at + 1], src[at + 2], src[at + 3]])
}

fn corrupt(offset: usize, detail: impl Into<String>) -> BasaltError {
    BasaltError::corruption("encoded chunk", Region::EncodedChunk, offset as u64, detail)
}

fn write_extra_length(out: &mut Vec<u8>, mut value: usize) {
    while value >= 255 {
        out.push(255);
        value -= 255;
    }
    out.push(value as u8);
}

fn read_extra_length(src: &[u8], pos: &mut usize, limit: usize) -> Result<usize> {
    let mut total = 0usize;
    loop {
        let Some(&byte) = src.get(*pos) else {
            return Err(corrupt(*pos, "lz stream ended inside a length escape"));
        };
        *pos += 1;
        total += usize::from(byte);
        if byte != 255 {
            return Ok(total);
        }
        if total > limit {
            // A run of 0xff bytes can otherwise describe a length far larger
            // than the declared output, which is how a malicious or damaged
            // buffer turns into an out-of-memory.
            return Err(corrupt(
                *pos,
                "lz length escape exceeds the declared output size",
            ));
        }
    }
}

/// Raw bytes with a header. The baseline for byte columns.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlainBytesCodec;

impl Codec for PlainBytesCodec {
    type Input = [u8];
    type Output = Vec<u8>;

    fn encoding(&self) -> Encoding {
        Encoding::PlainBytes
    }

    fn encode(&self, input: &[u8], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::PlainBytes, input.len());
        out.extend_from_slice(input);
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let (count, pos) = read_header(bytes, Encoding::PlainBytes)?;
        if bytes.len() - pos < count {
            return Err(corrupt(
                pos,
                format!(
                    "plain byte body promised {count} bytes but only {} follow",
                    bytes.len() - pos
                ),
            ));
        }
        Ok(bytes[pos..pos + count].to_vec())
    }
}

/// The in-tree LZ77 compressor.
///
/// # Stream format
///
/// After the standard codec header (which carries the *uncompressed* length),
/// the body is a sequence of:
///
/// ```text
/// token : u8   high nibble = literal length (15 = "read more")
///              low nibble  = match length - 4 (15 = "read more")
/// [extra literal length bytes]     255, 255, ..., n   (sum with the nibble)
/// literals...                      literal length raw bytes
/// offset : u16 little endian       distance back into the output, 1..=65535
/// [extra match length bytes]
/// ```
///
/// The stream ends either after a match (when the last match reached the end of
/// the input) or with a literals-only sequence carrying the tail; the decoder
/// distinguishes them by checking whether the buffer is exhausted after reading
/// the literals. The encoder never emits an empty trailing sequence, so the
/// stream is canonical — every byte is load-bearing, which is what makes a
/// truncated stream detectable rather than silently valid.
///
/// Matches may overlap their own output (`offset < length`), which is how runs
/// of repeated bytes compress; the decoder therefore copies byte by byte rather
/// than with a bulk move.
///
/// ```
/// use basalt_encoding::{ByteLzCodec, Codec};
///
/// let text = "the quick brown fox jumps over the quick brown dog".repeat(20);
/// let compressed = ByteLzCodec.encode_to_vec(text.as_bytes())?;
/// assert!(compressed.len() * 4 < text.len());
/// assert_eq!(ByteLzCodec.decode(&compressed)?, text.as_bytes());
/// # Ok::<(), basalt_common::BasaltError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteLzCodec;

impl Codec for ByteLzCodec {
    type Input = [u8];
    type Output = Vec<u8>;

    fn encoding(&self) -> Encoding {
        Encoding::ByteLz
    }

    fn encode(&self, input: &[u8], out: &mut Vec<u8>) -> Result<()> {
        write_header(out, Encoding::ByteLz, input.len());
        compress(input, out);
        Ok(())
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        let (count, pos) = read_header(bytes, Encoding::ByteLz)?;
        decompress(&bytes[pos..], count)
    }
}

fn compress(src: &[u8], out: &mut Vec<u8>) {
    if src.len() <= MIN_MATCH {
        emit_final(out, src);
        return;
    }
    let mut table = vec![u32::MAX; 1 << HASH_LOG];
    let mut anchor = 0usize;
    let mut pos = 0usize;
    let last_startable = src.len() - MIN_MATCH;

    while pos <= last_startable {
        let sequence = read_u32(src, pos);
        let slot = hash4(sequence);
        let candidate = table[slot];
        table[slot] = pos as u32;

        if candidate != u32::MAX {
            let candidate = candidate as usize;
            if pos - candidate <= MAX_OFFSET && read_u32(src, candidate) == sequence {
                let mut length = MIN_MATCH;
                while pos + length < src.len() && src[candidate + length] == src[pos + length] {
                    length += 1;
                }
                emit_sequence(
                    out,
                    &src[anchor..pos],
                    (pos - candidate) as u16,
                    length - MIN_MATCH,
                );
                pos += length;
                anchor = pos;
                continue;
            }
        }
        pos += 1;
    }
    if anchor < src.len() {
        emit_final(out, &src[anchor..]);
    }
}

fn emit_sequence(out: &mut Vec<u8>, literals: &[u8], offset: u16, match_extra: usize) {
    let literal_len = literals.len();
    let token = ((literal_len.min(15) as u8) << 4) | (match_extra.min(15) as u8);
    out.push(token);
    if literal_len >= 15 {
        write_extra_length(out, literal_len - 15);
    }
    out.extend_from_slice(literals);
    out.extend_from_slice(&offset.to_le_bytes());
    if match_extra >= 15 {
        write_extra_length(out, match_extra - 15);
    }
}

fn emit_final(out: &mut Vec<u8>, literals: &[u8]) {
    let literal_len = literals.len();
    out.push((literal_len.min(15) as u8) << 4);
    if literal_len >= 15 {
        write_extra_length(out, literal_len - 15);
    }
    out.extend_from_slice(literals);
}

fn decompress(src: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(capacity_hint(expected_len, src.len(), 0));
    let mut pos = 0usize;
    while pos < src.len() {
        let token = src[pos];
        pos += 1;
        let mut literal_len = usize::from(token >> 4);
        if literal_len == 15 {
            literal_len += read_extra_length(src, &mut pos, expected_len)?;
        }
        if pos + literal_len > src.len() || out.len() + literal_len > expected_len {
            return Err(corrupt(
                pos,
                "lz literal run runs past the end of the stream",
            ));
        }
        out.extend_from_slice(&src[pos..pos + literal_len]);
        pos += literal_len;

        if pos == src.len() {
            break;
        }
        if pos + 2 > src.len() {
            return Err(corrupt(
                pos,
                "lz stream ended inside a back reference offset",
            ));
        }
        let offset = usize::from(u16::from_le_bytes([src[pos], src[pos + 1]]));
        pos += 2;
        let mut match_len = usize::from(token & 0x0f);
        if match_len == 15 {
            match_len += read_extra_length(src, &mut pos, expected_len)?;
        }
        match_len += MIN_MATCH;

        if offset == 0 || offset > out.len() {
            return Err(corrupt(
                pos,
                format!(
                    "lz back reference of {offset} bytes into {} decoded bytes",
                    out.len()
                ),
            ));
        }
        if out.len() + match_len > expected_len {
            return Err(corrupt(
                pos,
                "lz match would produce more than the declared output size",
            ));
        }
        let start = out.len() - offset;
        // Byte-at-a-time on purpose: matches may overlap their own output, which
        // is how `aaaaaa...` compresses to one short sequence.
        for i in 0..match_len {
            let byte = out[start + i];
            out.push(byte);
        }
    }
    if out.len() != expected_len {
        return Err(corrupt(
            pos,
            format!(
                "lz stream decoded {} bytes but the header declared {expected_len}",
                out.len()
            ),
        ));
    }
    Ok(out)
}

/// Decode any byte buffer by looking at its header.
pub fn decode_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    match crate::codec::peek_encoding(bytes)? {
        Encoding::PlainBytes => PlainBytesCodec.decode(bytes),
        Encoding::ByteLz => ByteLzCodec.decode(bytes),
        other => Err(corrupt(
            0,
            format!("{other} is an integer encoding, not a byte encoding"),
        )),
    }
}

/// Encode a byte buffer, choosing between [`PlainBytesCodec`] and
/// [`ByteLzCodec`] by **measuring** rather than estimating.
///
/// LZ ratios are not predictable from cheap statistics — that is the whole
/// point of a general-purpose compressor — so the honest thing is to run it and
/// keep the result only if it actually pays. Compression is one linear pass, and
/// it happens once per chunk at write time; the alternative (a guess from
/// entropy estimates) is both slower to get right and worse.
///
/// `min_gain` is the fraction of the plain size that must be saved before the
/// compressed form is accepted; see
/// [`CompressionLevel`](basalt_common::CompressionLevel).
pub fn encode_bytes_adaptive(
    input: &[u8],
    allow_compression: bool,
    min_gain: f64,
    out: &mut Vec<u8>,
) -> Result<Encoding> {
    if allow_compression && input.len() >= 64 {
        let mut compressed = Vec::new();
        ByteLzCodec.encode(input, &mut compressed)?;
        let plain_len = input.len() + 8;
        if (compressed.len() as f64) <= (plain_len as f64) * (1.0 - min_gain) {
            out.extend_from_slice(&compressed);
            return Ok(Encoding::ByteLz);
        }
    }
    PlainBytesCodec.encode(input, out)?;
    Ok(Encoding::PlainBytes)
}

/// Length prefix helper shared by the variable-length column layout.
pub(crate) fn write_length_prefixed(payload: &[u8], out: &mut Vec<u8>) {
    varint::write_uvarint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

/// Read a length-prefixed slice written by [`write_length_prefixed`].
pub(crate) fn read_length_prefixed<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8]> {
    let len = varint::read_uvarint(bytes, pos)? as usize;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| corrupt(*pos, "length prefix overflows"))?;
    if end > bytes.len() {
        return Err(corrupt(
            *pos,
            format!(
                "length prefix of {len} bytes runs past the {}-byte buffer",
                bytes.len()
            ),
        ));
    }
    let slice = &bytes[*pos..end];
    *pos = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// A 64-bit LCG: cheap, deterministic, and genuinely hard to compress.
    fn pseudo_random_bytes(n: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect()
    }

    fn shapes() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("empty", vec![]),
            ("one byte", vec![7]),
            ("under min match", vec![1, 2, 3]),
            ("exactly min match", vec![1, 2, 3, 4]),
            ("all same", vec![b'a'; 5000]),
            ("two byte cycle", (0..5000).map(|i| (i % 2) as u8).collect()),
            (
                "english-ish",
                "the quick brown fox jumps over the lazy dog. "
                    .repeat(200)
                    .into_bytes(),
            ),
            (
                "json-ish",
                r#"{"id":1,"city":"oslo","active":true}"#.repeat(300).into_bytes(),
            ),
            ("incompressible", pseudo_random_bytes(5000)),
            (
                "long literal run",
                (0..300u32).map(|i| (i * 7 % 251) as u8).collect(),
            ),
            ("long match", {
                let mut v = vec![0u8; 300];
                v.extend(std::iter::repeat_n(0u8, 3000));
                v
            }),
        ]
    }

    #[test]
    fn lz_round_trips_every_shape() {
        for (name, data) in shapes() {
            let encoded = ByteLzCodec.encode_to_vec(&data).unwrap();
            let decoded = ByteLzCodec.decode(&encoded).unwrap();
            assert_eq!(decoded, data, "shape `{name}`");
            let plain = PlainBytesCodec.encode_to_vec(&data).unwrap();
            assert_eq!(
                PlainBytesCodec.decode(&plain).unwrap(),
                data,
                "plain `{name}`"
            );
            assert_eq!(decode_bytes(&encoded).unwrap(), data);
            assert_eq!(decode_bytes(&plain).unwrap(), data);
        }
    }

    #[test]
    fn lz_actually_compresses() {
        let repetitive = "the quick brown fox jumps over the lazy dog. ".repeat(200);
        let encoded = ByteLzCodec.encode_to_vec(repetitive.as_bytes()).unwrap();
        assert!(
            encoded.len() * 8 < repetitive.len(),
            "expected >8x on highly repetitive text, got {} -> {}",
            repetitive.len(),
            encoded.len()
        );
        let runs = vec![b'z'; 100_000];
        let encoded = ByteLzCodec.encode_to_vec(&runs).unwrap();
        assert!(
            encoded.len() < 600,
            "long run should collapse, got {}",
            encoded.len()
        );
    }

    #[test]
    fn adaptive_falls_back_when_compression_does_not_pay() {
        let noise = pseudo_random_bytes(4096);
        let mut out = Vec::new();
        let chosen = encode_bytes_adaptive(&noise, true, 0.05, &mut out).unwrap();
        assert_eq!(
            chosen,
            Encoding::PlainBytes,
            "incompressible data must stay plain"
        );
        assert_eq!(decode_bytes(&out).unwrap(), noise);

        let repetitive = "abcabcabc".repeat(500).into_bytes();
        let mut out = Vec::new();
        let chosen = encode_bytes_adaptive(&repetitive, true, 0.05, &mut out).unwrap();
        assert_eq!(chosen, Encoding::ByteLz);
        assert_eq!(decode_bytes(&out).unwrap(), repetitive);

        let mut out = Vec::new();
        let chosen = encode_bytes_adaptive(&repetitive, false, 0.05, &mut out).unwrap();
        assert_eq!(
            chosen,
            Encoding::PlainBytes,
            "compression disabled must be honoured"
        );

        // Tiny inputs skip the compressor entirely.
        let mut out = Vec::new();
        assert_eq!(
            encode_bytes_adaptive(b"short", true, 0.0, &mut out).unwrap(),
            Encoding::PlainBytes
        );
    }

    #[test]
    fn corrupt_streams_are_rejected_not_expanded() {
        let data = "abcabcabcabcabcabcabc".repeat(50).into_bytes();
        let encoded = ByteLzCodec.encode_to_vec(&data).unwrap();
        for cut in [3, encoded.len() / 2, encoded.len() - 1] {
            assert!(
                ByteLzCodec.decode(&encoded[..cut]).is_err(),
                "truncation at {cut} accepted"
            );
        }
        // A back reference pointing before the start of the output.
        let mut broken = encoded.clone();
        let last = broken.len() - 1;
        broken[last] ^= 0xff;
        // Either it errors or it decodes to something of the right length; it
        // must never panic or allocate unboundedly.
        let _ = ByteLzCodec.decode(&broken);

        // A hand-built bomb: declare 16 bytes of output, then ask for a huge match.
        let mut bomb = Vec::new();
        write_header(&mut bomb, Encoding::ByteLz, 16);
        bomb.push(0x4f); // 4 literals, match escape
        bomb.extend_from_slice(b"aaaa");
        bomb.extend_from_slice(&1u16.to_le_bytes());
        bomb.extend_from_slice(&[255u8; 64]);
        bomb.push(0);
        let err = ByteLzCodec.decode(&bomb).unwrap_err();
        assert!(err.is_corruption(), "{err}");
    }

    #[test]
    fn length_prefix_helpers() {
        let mut buf = Vec::new();
        write_length_prefixed(b"hello", &mut buf);
        write_length_prefixed(b"", &mut buf);
        let mut pos = 0;
        assert_eq!(read_length_prefixed(&buf, &mut pos).unwrap(), b"hello");
        assert_eq!(read_length_prefixed(&buf, &mut pos).unwrap(), b"");
        assert_eq!(pos, buf.len());
        let mut pos = 0;
        assert!(read_length_prefixed(&[9, 1, 2], &mut pos).is_err());
    }

    #[test]
    fn wrong_family_is_refused() {
        let mut buf = Vec::new();
        write_header(&mut buf, Encoding::Plain, 0);
        assert!(decode_bytes(&buf).is_err());
    }
}

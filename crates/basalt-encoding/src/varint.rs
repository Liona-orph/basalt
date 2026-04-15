//! Zigzag + LEB128 variable-length integers.
//!
//! LEB128 stores seven payload bits per byte with the high bit as a
//! continuation flag, so small magnitudes cost one byte. Because two's
//! complement makes `-1` an all-ones pattern (ten LEB128 bytes), signed values
//! are first **zigzag** mapped so that small magnitudes of either sign stay
//! small: `0 -> 0`, `-1 -> 1`, `1 -> 2`, `-2 -> 3`, ...

use basalt_common::{BasaltError, Region, Result};

/// Map a signed integer onto an unsigned one that keeps small magnitudes small.
#[inline]
pub const fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Inverse of [`zigzag_encode`].
#[inline]
pub const fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Number of bytes [`write_uvarint`] will emit for `value`.
#[inline]
pub const fn uvarint_len(value: u64) -> usize {
    // 1 byte per 7 bits, at least one byte.
    let bits = 64 - value.leading_zeros();
    if bits == 0 {
        1
    } else {
        ((bits + 6) / 7) as usize
    }
}

/// Number of bytes a zigzag varint of `value` occupies.
#[inline]
pub const fn varint_len(value: i64) -> usize {
    uvarint_len(zigzag_encode(value))
}

/// Append an unsigned LEB128 integer.
#[inline]
pub fn write_uvarint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Append a zigzag LEB128 integer.
#[inline]
pub fn write_varint(value: i64, out: &mut Vec<u8>) {
    write_uvarint(zigzag_encode(value), out);
}

/// Read an unsigned LEB128 integer, advancing `pos`.
///
/// # Errors
///
/// [`BasaltError::Corruption`] naming the byte offset if the buffer ends
/// mid-integer or the encoding is longer than ten bytes (which cannot describe a
/// `u64` and therefore means the bytes are damaged).
pub fn read_uvarint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let start = *pos;
    let mut result: u64 = 0;
    let mut shift = 0u32;
    loop {
        let Some(&byte) = bytes.get(*pos) else {
            return Err(BasaltError::corruption(
                "encoded chunk",
                Region::EncodedChunk,
                start as u64,
                "buffer ended in the middle of a varint",
            ));
        };
        *pos += 1;
        // The bound is checked *before* shifting: `1u64 << 64` is undefined and
        // panics in debug builds, so a damaged buffer must not reach it.
        if shift >= 64 {
            return Err(BasaltError::corruption(
                "encoded chunk",
                Region::EncodedChunk,
                start as u64,
                "varint is longer than 10 bytes and cannot encode a u64",
            ));
        }
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// Read a zigzag LEB128 integer, advancing `pos`.
pub fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<i64> {
    read_uvarint(bytes, pos).map(zigzag_decode)
}

/// Append a zigzag LEB128 `i128`, used only by the decimal path.
pub fn write_varint_i128(value: i128, out: &mut Vec<u8>) {
    let mut zigzag = ((value as u128) << 1) ^ ((value >> 127) as u128);
    while zigzag >= 0x80 {
        out.push((zigzag as u8) | 0x80);
        zigzag >>= 7;
    }
    out.push(zigzag as u8);
}

/// Read a zigzag LEB128 `i128`.
pub fn read_varint_i128(bytes: &[u8], pos: &mut usize) -> Result<i128> {
    let start = *pos;
    let mut result: u128 = 0;
    let mut shift = 0u32;
    loop {
        let Some(&byte) = bytes.get(*pos) else {
            return Err(BasaltError::corruption(
                "encoded chunk",
                Region::EncodedChunk,
                start as u64,
                "buffer ended in the middle of a 128-bit varint",
            ));
        };
        *pos += 1;
        if shift >= 128 {
            return Err(BasaltError::corruption(
                "encoded chunk",
                Region::EncodedChunk,
                start as u64,
                "128-bit varint is longer than 19 bytes",
            ));
        }
        result |= u128::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(((result >> 1) as i128) ^ -((result & 1) as i128))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn zigzag_round_trip() {
        for v in [0i64, -1, 1, -2, 2, i64::MIN, i64::MAX, 12_345, -12_345] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v, "value {v}");
        }
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
    }

    #[test]
    fn varint_round_trip_and_length() {
        let mut buf = Vec::new();
        let values = [0i64, 1, -1, 63, 64, -64, 8191, i64::MIN, i64::MAX];
        for v in values {
            let before = buf.len();
            write_varint(v, &mut buf);
            assert_eq!(
                buf.len() - before,
                varint_len(v),
                "length prediction for {v}"
            );
        }
        let mut pos = 0;
        for v in values {
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v);
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn uvarint_lengths() {
        assert_eq!(uvarint_len(0), 1);
        assert_eq!(uvarint_len(127), 1);
        assert_eq!(uvarint_len(128), 2);
        assert_eq!(uvarint_len(u64::MAX), 10);
    }

    #[test]
    fn truncated_and_overlong_varints_are_corruption() {
        let mut pos = 0;
        assert!(read_uvarint(&[0x80], &mut pos).is_err());
        let overlong = [0x80u8; 12];
        let mut pos = 0;
        let err = read_uvarint(&overlong, &mut pos).unwrap_err();
        assert!(err.to_string().contains("longer than 10 bytes"), "{err}");
        assert!(err.is_corruption());
    }

    #[test]
    fn i128_varints() {
        let mut buf = Vec::new();
        let values = [0i128, 1, -1, i128::MIN, i128::MAX, 10i128.pow(30)];
        for v in values {
            write_varint_i128(v, &mut buf);
        }
        let mut pos = 0;
        for v in values {
            assert_eq!(read_varint_i128(&buf, &mut pos).unwrap(), v);
        }
        assert_eq!(pos, buf.len());
        let mut pos = 0;
        assert!(read_varint_i128(&[0x80], &mut pos).is_err());
        let mut pos = 0;
        assert!(read_varint_i128(&[0x80u8; 25], &mut pos).is_err());
    }
}

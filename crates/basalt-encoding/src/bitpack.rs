//! Fixed-width bit packing at arbitrary widths.
//!
//! Values are packed **LSB-first**: value `i` occupies bits `i*w .. (i+1)*w` of
//! a little-endian bit stream. That layout makes both directions a shift and a
//! mask against a sliding accumulator, with no special case at byte boundaries,
//! and it is the same convention Parquet uses, which matters if anyone ever
//! wants to write a converter.
//!
//! Width `0` is legal and means "every value is zero"; it occupies no bytes at
//! all, which is what makes an all-constant column after frame-of-reference
//! subtraction essentially free.

use basalt_common::{BasaltError, Region, Result};

/// Bytes needed to pack `count` values at `width` bits each.
#[inline]
pub const fn packed_len(count: usize, width: u32) -> usize {
    let bits = count * width as usize;
    bits.div_ceil(8)
}

/// Smallest width that can represent every value in `0..=max`.
#[inline]
pub const fn required_width(max: u64) -> u32 {
    64 - max.leading_zeros()
}

/// Mask with the low `width` bits set (`u64::MAX` for width 64).
#[inline]
const fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// Pack `values` at `width` bits each, appending to `out`.
///
/// Bits above `width` in any value are dropped; callers are expected to have
/// computed `width` with [`required_width`].
///
/// # Errors
///
/// If `width > 64`.
pub fn pack(values: &[u64], width: u32, out: &mut Vec<u8>) -> Result<()> {
    if width > 64 {
        return Err(BasaltError::internal(
            "bitpack::pack",
            format!("width {width} exceeds 64 bits"),
        ));
    }
    if width == 0 {
        return Ok(());
    }
    out.reserve(packed_len(values.len(), width));
    // A 128-bit accumulator is enough: at most 7 leftover bits plus one 64-bit
    // value = 71 bits are live at any time.
    let mut acc: u128 = 0;
    let mut bits: u32 = 0;
    for &value in values {
        acc |= u128::from(value & mask(width)) << bits;
        bits += width;
        while bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push(acc as u8);
    }
    Ok(())
}

/// Unpack `count` values of `width` bits from `bytes`.
///
/// # Errors
///
/// [`BasaltError::Corruption`] if `bytes` is shorter than the packing requires,
/// which is the signature of a truncated or damaged chunk.
pub fn unpack(bytes: &[u8], width: u32, count: usize) -> Result<Vec<u64>> {
    if width > 64 {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            0,
            format!("bit width {width} is impossible (maximum 64)"),
        ));
    }
    if width == 0 {
        // `count` has already been bounded by `MAX_CHUNK_ELEMENTS` in the
        // header check, so this cannot be turned into an allocation bomb.
        return Ok(vec![0; count]);
    }
    let needed = packed_len(count, width);
    if bytes.len() < needed {
        return Err(BasaltError::corruption(
            "encoded chunk",
            Region::EncodedChunk,
            bytes.len() as u64,
            format!(
                "bit-packed body needs {needed} bytes for {count} values at {width} bits, \
                     but only {} are present",
                bytes.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(count);
    let mut acc: u128 = 0;
    let mut bits: u32 = 0;
    let mut pos = 0;
    for _ in 0..count {
        while bits < width {
            acc |= u128::from(bytes[pos]) << bits;
            pos += 1;
            bits += 8;
        }
        out.push((acc as u64) & mask(width));
        acc >>= width;
        bits -= width;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn round_trip_every_width() {
        for width in 0..=64u32 {
            let count = 37;
            let values: Vec<u64> = (0..count)
                .map(|i| {
                    if width == 0 {
                        0
                    } else {
                        (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) & mask(width)
                    }
                })
                .collect();
            let mut packed = Vec::new();
            pack(&values, width, &mut packed).unwrap();
            assert_eq!(packed.len(), packed_len(count, width), "width {width}");
            assert_eq!(
                unpack(&packed, width, count).unwrap(),
                values,
                "width {width}"
            );
        }
    }

    #[test]
    fn extremes() {
        let values = vec![u64::MAX; 5];
        let mut packed = Vec::new();
        pack(&values, 64, &mut packed).unwrap();
        assert_eq!(packed.len(), 40);
        assert_eq!(unpack(&packed, 64, 5).unwrap(), values);

        let mut packed = Vec::new();
        pack(&[1, 0, 1, 1], 1, &mut packed).unwrap();
        assert_eq!(packed, vec![0b1101]);
        assert_eq!(unpack(&packed, 1, 4).unwrap(), vec![1, 0, 1, 1]);
    }

    #[test]
    fn width_zero_is_free() {
        let mut packed = Vec::new();
        pack(&[0; 100], 0, &mut packed).unwrap();
        assert!(packed.is_empty());
        assert_eq!(unpack(&packed, 0, 100).unwrap(), vec![0u64; 100]);
        assert_eq!(packed_len(100, 0), 0);
    }

    #[test]
    fn required_width_is_tight() {
        assert_eq!(required_width(0), 0);
        assert_eq!(required_width(1), 1);
        assert_eq!(required_width(2), 2);
        assert_eq!(required_width(255), 8);
        assert_eq!(required_width(256), 9);
        assert_eq!(required_width(u64::MAX), 64);
    }

    #[test]
    fn truncated_input_is_corruption() {
        let mut packed = Vec::new();
        pack(&[5, 6, 7], 5, &mut packed).unwrap();
        let err = unpack(&packed[..1], 5, 3).unwrap_err();
        assert!(err.is_corruption());
        assert!(err.to_string().contains("needs 2 bytes"), "{err}");
        assert!(unpack(&packed, 65, 1).is_err());
        assert!(pack(&[1], 65, &mut Vec::new()).is_err());
    }

    #[test]
    fn empty_input() {
        let mut packed = Vec::new();
        pack(&[], 7, &mut packed).unwrap();
        assert!(packed.is_empty());
        assert!(unpack(&packed, 7, 0).unwrap().is_empty());
    }
}

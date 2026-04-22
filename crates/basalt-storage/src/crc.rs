//! CRC32C (Castagnoli) checksums.
//!
//! Every persisted block in a segment carries one, so a flipped bit is reported
//! as [`BasaltError::Corruption`](basalt_common::BasaltError::Corruption) with
//! an offset rather than returned to the caller as data.
//!
//! # Why CRC32C and not a hash
//!
//! CRC32C detects *all* single-bit, double-bit and burst errors up to 32 bits,
//! which is exactly the failure mode of storage media and memory; a general
//! hash gives no such guarantee, only a probabilistic one. It is also the
//! polynomial with hardware support (`crc32cb` on aarch64, SSE4.2 on x86-64),
//! so a future intrinsics path is a drop-in replacement for the table below.
//! This is not a security check: it detects accident, not malice.

const POLYNOMIAL: u32 = 0x82F6_3B78;

/// Byte-at-a-time table, built at compile time so there is no lazy init and no
/// static mutable state.
const TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut index = 0usize;
    while index < 256 {
        let mut crc = index as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLYNOMIAL
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[index] = crc;
        index += 1;
    }
    table
}

/// CRC32C of `data`. O(n), one table lookup per byte.
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_continue(0, data)
}

/// Continue a CRC32C over another slice, so a large payload can be checksummed
/// without being contiguous in memory.
///
/// Pass `0` as the initial value; the standard pre/post inversion is handled
/// internally, so `crc32c_continue(crc32c_continue(0, a), b) == crc32c(a ++ b)`.
pub fn crc32c_continue(previous: u32, data: &[u8]) -> u32 {
    let mut crc = !previous;
    for &byte in data {
        crc = (crc >> 8) ^ TABLE[((crc ^ u32::from(byte)) & 0xff) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn known_vectors() {
        // The check value from the CRC catalogue for CRC-32/ISCSI.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        for split in [0, 1, 7, 64, 512, 999, 1000] {
            let (a, b) = data.split_at(split);
            assert_eq!(
                crc32c_continue(crc32c_continue(0, a), b),
                crc32c(&data),
                "split {split}"
            );
        }
    }

    #[test]
    fn detects_every_single_bit_flip() {
        let data: Vec<u8> = (0..64u8).collect();
        let baseline = crc32c(&data);
        for byte in 0..data.len() {
            for bit in 0..8 {
                let mut damaged = data.clone();
                damaged[byte] ^= 1 << bit;
                assert_ne!(crc32c(&damaged), baseline, "flip at byte {byte} bit {bit}");
            }
        }
    }
}

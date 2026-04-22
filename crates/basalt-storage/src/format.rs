//! On-disk constants and block framing.
//!
//! See `FORMAT.md` for the byte-level specification; this module is its
//! executable half and the two are kept in step deliberately.

use basalt_common::{BasaltError, Region, Result};

use crate::crc::crc32c;

/// File magic, at offset 0 and again in the last 4 bytes.
///
/// Repeating it at the end lets a reader confirm it has a whole file (not a
/// truncated copy) with a single read of the trailer.
pub const MAGIC: [u8; 4] = *b"BSLT";

/// Magic at the start of the footer payload, so a footer offset that points at
/// the wrong place is caught before anything is parsed.
pub const FOOTER_MAGIC: [u8; 4] = *b"BFTR";

/// Segment format version.
pub const FORMAT_VERSION: u16 = 1;

/// Bytes in the file header: magic(4) + version(2) + flags(2) + reserved(8).
pub const FILE_HEADER_LEN: u64 = 16;

/// Bytes in the trailer: footer offset(8) + footer length(4) + magic(4).
pub const TRAILER_LEN: u64 = 16;

/// Bytes in a block header: payload length(4) + CRC32C(4).
pub const BLOCK_HEADER_LEN: u64 = 8;

/// Largest payload a single block may hold (256 MiB).
///
/// A format limit, so a damaged length field cannot make the reader allocate an
/// arbitrary amount before the checksum has a chance to reject it.
pub const MAX_BLOCK_PAYLOAD: u32 = 256 * 1024 * 1024;

/// Build the 16-byte file header.
pub fn file_header() -> [u8; FILE_HEADER_LEN as usize] {
    let mut header = [0u8; FILE_HEADER_LEN as usize];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    // flags (2 bytes) and reserved (8 bytes) stay zero.
    header
}

/// Validate a 16-byte file header.
pub fn parse_file_header(bytes: &[u8], object: &str) -> Result<u16> {
    if bytes.len() < FILE_HEADER_LEN as usize {
        return Err(BasaltError::corruption(
            object,
            Region::FileHeader,
            0,
            format!(
                "file is only {} bytes, shorter than the 16-byte header",
                bytes.len()
            ),
        ));
    }
    if bytes[0..4] != MAGIC {
        return Err(BasaltError::corruption(
            object,
            Region::FileHeader,
            0,
            format!("bad magic {:02x?}, expected {:02x?}", &bytes[0..4], MAGIC),
        ));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != FORMAT_VERSION {
        return Err(BasaltError::corruption(
            object,
            Region::FileHeader,
            4,
            format!("segment format version {version} is not supported (this build reads {FORMAT_VERSION})"),
        ));
    }
    Ok(version)
}

/// Frame `payload` as a block: `[len u32][crc32c u32][payload]`.
pub fn frame_block(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + BLOCK_HEADER_LEN as usize);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc32c(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Parse a block header, returning the payload length and its stored checksum.
///
/// `offset` is the absolute file offset of the header, used only to make the
/// error message actionable.
pub fn parse_block_header(header: &[u8], offset: u64, object: &str) -> Result<(u32, u32)> {
    if header.len() < BLOCK_HEADER_LEN as usize {
        return Err(BasaltError::corruption(
            object,
            Region::BlockHeader,
            offset,
            "block header is truncated",
        ));
    }
    let length = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let checksum = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if length > MAX_BLOCK_PAYLOAD {
        return Err(BasaltError::corruption(
            object,
            Region::BlockHeader,
            offset,
            format!("block claims {length} bytes, above the {MAX_BLOCK_PAYLOAD}-byte limit"),
        ));
    }
    Ok((length, checksum))
}

/// Verify a block payload against its stored checksum.
///
/// `payload_offset` is the absolute file offset of the payload, so the error
/// names the exact byte range that failed.
pub fn verify_block(
    payload: &[u8],
    expected: u32,
    payload_offset: u64,
    object: &str,
) -> Result<()> {
    let actual = crc32c(payload);
    if actual != expected {
        return Err(BasaltError::corruption(
            object,
            Region::BlockPayload,
            payload_offset,
            format!(
                "crc32c mismatch over {} bytes: stored {expected:#010x}, computed {actual:#010x}",
                payload.len()
            ),
        ));
    }
    Ok(())
}

/// Build the 16-byte trailer.
pub fn trailer(footer_offset: u64, footer_length: u32) -> [u8; TRAILER_LEN as usize] {
    let mut out = [0u8; TRAILER_LEN as usize];
    out[0..8].copy_from_slice(&footer_offset.to_le_bytes());
    out[8..12].copy_from_slice(&footer_length.to_le_bytes());
    out[12..16].copy_from_slice(&MAGIC);
    out
}

/// Parse the trailer, returning `(footer block offset, footer block length)`.
pub fn parse_trailer(bytes: &[u8], file_len: u64, object: &str) -> Result<(u64, u32)> {
    let trailer_offset = file_len.saturating_sub(TRAILER_LEN);
    if bytes.len() < TRAILER_LEN as usize {
        return Err(BasaltError::corruption(
            object,
            Region::FooterTrailer,
            trailer_offset,
            "file is too short to contain a trailer",
        ));
    }
    if bytes[12..16] != MAGIC {
        return Err(BasaltError::corruption(
            object,
            Region::FooterTrailer,
            trailer_offset + 12,
            format!(
                "trailing magic {:02x?} is wrong: the file is truncated or not a Basalt segment",
                &bytes[12..16]
            ),
        ));
    }
    let mut offset_bytes = [0u8; 8];
    offset_bytes.copy_from_slice(&bytes[0..8]);
    let footer_offset = u64::from_le_bytes(offset_bytes);
    let footer_length = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let footer_end = footer_offset.saturating_add(u64::from(footer_length));
    if footer_offset < FILE_HEADER_LEN || footer_end > file_len - TRAILER_LEN {
        return Err(BasaltError::corruption(
            object,
            Region::FooterTrailer,
            trailer_offset,
            format!(
                "footer at {footer_offset}..{footer_end} does not lie inside the \
                 {file_len}-byte file"
            ),
        ));
    }
    Ok((footer_offset, footer_length))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn header_round_trip() {
        let header = file_header();
        assert_eq!(parse_file_header(&header, "test").unwrap(), FORMAT_VERSION);
        assert_eq!(header.len(), FILE_HEADER_LEN as usize);
    }

    #[test]
    fn header_errors_name_the_problem() {
        let err = parse_file_header(b"short", "seg").unwrap_err();
        assert!(
            err.to_string().contains("shorter than the 16-byte header"),
            "{err}"
        );
        let mut header = file_header();
        header[0] = b'X';
        let err = parse_file_header(&header, "seg").unwrap_err();
        assert!(err.to_string().contains("bad magic"), "{err}");
        assert_eq!(err.corruption_offset(), Some(0));
        let mut header = file_header();
        header[4] = 9;
        let err = parse_file_header(&header, "seg").unwrap_err();
        assert!(err.to_string().contains("version 9"), "{err}");
        assert_eq!(err.corruption_offset(), Some(4));
    }

    #[test]
    fn blocks_detect_damage_and_name_the_offset() {
        let payload = b"the quick brown fox".to_vec();
        let block = frame_block(&payload);
        let (length, checksum) = parse_block_header(&block, 4096, "seg").unwrap();
        assert_eq!(length as usize, payload.len());
        verify_block(&payload, checksum, 4104, "seg").unwrap();

        let mut damaged = payload.clone();
        damaged[3] ^= 0x01;
        let err = verify_block(&damaged, checksum, 4104, "seg").unwrap_err();
        assert!(err.is_corruption());
        assert_eq!(err.corruption_offset(), Some(4104));
        assert!(err.to_string().contains("crc32c mismatch"), "{err}");
    }

    #[test]
    fn block_headers_reject_absurd_lengths() {
        let mut header = [0u8; 8];
        header[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = parse_block_header(&header, 99, "seg").unwrap_err();
        assert!(err.to_string().contains("above the"), "{err}");
        assert_eq!(err.corruption_offset(), Some(99));
        assert!(parse_block_header(&[0u8; 3], 0, "seg").is_err());
    }

    #[test]
    fn trailer_round_trip_and_validation() {
        let bytes = trailer(2048, 512);
        assert_eq!(parse_trailer(&bytes, 4096, "seg").unwrap(), (2048, 512));
        // Footer that would overlap the trailer.
        let bytes = trailer(4000, 512);
        assert!(parse_trailer(&bytes, 4096, "seg").is_err());
        // Footer before the file header.
        let bytes = trailer(4, 8);
        assert!(parse_trailer(&bytes, 4096, "seg").is_err());
        // Wrong trailing magic.
        let mut bytes = trailer(2048, 512);
        bytes[15] = b'!';
        let err = parse_trailer(&bytes, 4096, "seg").unwrap_err();
        assert!(
            err.to_string()
                .contains("truncated or not a Basalt segment"),
            "{err}"
        );
        assert!(parse_trailer(&[0u8; 4], 4096, "seg").is_err());
    }
}

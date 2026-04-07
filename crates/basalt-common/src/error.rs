//! The engine-wide error type.

use std::fmt;

use thiserror::Error;

/// Convenience alias used throughout the workspace.
pub type Result<T, E = BasaltError> = std::result::Result<T, E>;

/// Where in a segment file a corruption was detected.
///
/// Corruption reports are useless without a location, so [`BasaltError::Corruption`]
/// carries both a byte offset *and* the structural region that offset belongs
/// to: "byte 4096" is much easier to act on when you also know it was the start
/// of a block header rather than the middle of a dictionary page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Region {
    /// The 8-byte file header at offset 0.
    FileHeader,
    /// The fixed-size framing header that precedes every block.
    BlockHeader,
    /// The payload of a block, i.e. the CRC-protected bytes themselves.
    BlockPayload,
    /// The serialized footer (schema + row-group metadata).
    Footer,
    /// The fixed-size trailer at the end of the file that locates the footer.
    FooterTrailer,
    /// The header or body of an encoded column chunk.
    EncodedChunk,
    /// A bloom-filter block.
    BloomFilter,
    /// A zone map inside the footer.
    ZoneMap,
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Region::FileHeader => "file header",
            Region::BlockHeader => "block header",
            Region::BlockPayload => "block payload",
            Region::Footer => "footer",
            Region::FooterTrailer => "footer trailer",
            Region::EncodedChunk => "encoded chunk",
            Region::BloomFilter => "bloom filter",
            Region::ZoneMap => "zone map",
        };
        f.write_str(s)
    }
}

/// The single error type for every Basalt crate.
///
/// Variants are *categories*, not call sites: `Schema` means "the user's schema
/// and the data disagree", `Internal` means "Basalt has a bug". Each variant
/// carries enough context that the `Display` output alone is actionable.
///
/// ```
/// use basalt_common::{BasaltError, Region};
///
/// let err = BasaltError::corruption("orders.bslt", Region::BlockPayload, 8_192, "crc32c mismatch: stored 0xdeadbeef, computed 0x12345678");
/// assert_eq!(
///     err.to_string(),
///     "corruption in orders.bslt: block payload at byte offset 8192: \
///      crc32c mismatch: stored 0xdeadbeef, computed 0x12345678",
/// );
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BasaltError {
    /// An underlying I/O operation failed.
    ///
    /// `context` describes what was being attempted ("reading footer of
    /// orders.bslt"), because `std::io::Error`'s own message never says *what*
    /// the program was doing when it got `ENOENT`.
    #[error("i/o error while {context}: {source}")]
    Io {
        /// What the engine was trying to do, phrased as a gerund.
        context: String,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// Persisted bytes failed a structural or checksum validation.
    ///
    /// Always reports *where*: the named object, the structural [`Region`] and
    /// the absolute byte offset within the file.
    #[error("corruption in {object}: {region} at byte offset {offset}: {detail}")]
    Corruption {
        /// The file (or in-memory buffer) that failed validation.
        object: String,
        /// Which structural region the offset falls in.
        region: Region,
        /// Absolute byte offset from the start of the object.
        offset: u64,
        /// What specifically was wrong.
        detail: String,
    },

    /// A schema-level problem: unknown column, duplicate name, arity mismatch,
    /// nullability violation.
    #[error("schema error: {0}")]
    Schema(String),

    /// A type-level problem: an operation was applied to a value of the wrong
    /// type, or two types could not be coerced to a common type.
    #[error("type error: {0}")]
    Type(String),

    /// The query could not be planned (unresolved reference, unsupported
    /// construct, contradictory constraints).
    #[error("planning error: {0}")]
    Plan(String),

    /// A runtime failure while executing an otherwise valid plan (overflow,
    /// division by zero, resource limit).
    #[error("execution error: {0}")]
    Execution(String),

    /// A well-formed request for something Basalt does not implement yet.
    ///
    /// Kept separate from [`BasaltError::Plan`] so that callers can distinguish
    /// "you asked for something wrong" from "we have not built that yet".
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// An invariant that Basalt itself is responsible for was violated.
    ///
    /// Seeing this in a log is always a bug report. `location` is a compile-time
    /// constant naming the module and invariant.
    #[error("internal error at {location}: {detail} (this is a bug in Basalt)")]
    Internal {
        /// Static description of the failing site, e.g. `"bitmap::and"`.
        location: &'static str,
        /// What the code expected and what it saw.
        detail: String,
    },
}

impl BasaltError {
    /// Wrap an [`std::io::Error`] with a description of the failing operation.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        BasaltError::Io {
            context: context.into(),
            source,
        }
    }

    /// Report corruption at a precise offset.
    pub fn corruption(
        object: impl Into<String>,
        region: Region,
        offset: u64,
        detail: impl Into<String>,
    ) -> Self {
        BasaltError::Corruption {
            object: object.into(),
            region,
            offset,
            detail: detail.into(),
        }
    }

    /// Construct a [`BasaltError::Schema`].
    pub fn schema(detail: impl Into<String>) -> Self {
        BasaltError::Schema(detail.into())
    }

    /// Construct a [`BasaltError::Type`].
    pub fn type_error(detail: impl Into<String>) -> Self {
        BasaltError::Type(detail.into())
    }

    /// Construct a [`BasaltError::Plan`].
    pub fn plan(detail: impl Into<String>) -> Self {
        BasaltError::Plan(detail.into())
    }

    /// Construct a [`BasaltError::Execution`].
    pub fn execution(detail: impl Into<String>) -> Self {
        BasaltError::Execution(detail.into())
    }

    /// Construct a [`BasaltError::Unsupported`].
    pub fn unsupported(detail: impl Into<String>) -> Self {
        BasaltError::Unsupported(detail.into())
    }

    /// Construct a [`BasaltError::Internal`].
    pub fn internal(location: &'static str, detail: impl Into<String>) -> Self {
        BasaltError::Internal {
            location,
            detail: detail.into(),
        }
    }

    /// `true` if the error indicates damaged persisted data rather than a
    /// caller mistake. Useful for deciding whether to quarantine a segment.
    pub fn is_corruption(&self) -> bool {
        matches!(self, BasaltError::Corruption { .. })
    }

    /// The byte offset of a corruption, if this is one.
    ///
    /// Lets recovery tooling act on the location without string-parsing the
    /// `Display` output.
    pub fn corruption_offset(&self) -> Option<u64> {
        match self {
            BasaltError::Corruption { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}

/// Adds `.io_context(..)` to `std::io::Result`, so that the ubiquitous
/// `read_exact(..)?` gains a description of what was being read.
///
/// ```
/// use basalt_common::{IoResultExt, Result};
///
/// fn read_header(r: &mut impl std::io::Read) -> Result<[u8; 8]> {
///     let mut buf = [0u8; 8];
///     r.read_exact(&mut buf).io_context("reading segment file header")?;
///     Ok(buf)
/// }
/// # let mut src: &[u8] = b"BSLT\x01\x00\x00\x00";
/// # assert!(read_header(&mut src).is_ok());
/// ```
pub trait IoResultExt<T> {
    /// Attach a description of the operation to an I/O failure.
    ///
    /// The closure-free form is intentional: the context strings are short and
    /// building one on the success path costs a single `Into<String>` call only
    /// when the value is already a `&'static str`.
    fn io_context(self, context: impl Into<String>) -> Result<T>;
}

impl<T> IoResultExt<T> for std::io::Result<T> {
    fn io_context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|source| BasaltError::io(context, source))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn corruption_display_names_the_offset() {
        let e = BasaltError::corruption("seg.bslt", Region::Footer, 1_234, "bad length prefix");
        let s = e.to_string();
        assert!(s.contains("1234"), "{s}");
        assert!(s.contains("footer"), "{s}");
        assert_eq!(e.corruption_offset(), Some(1_234));
        assert!(e.is_corruption());
    }

    #[test]
    fn io_context_is_preserved() {
        let inner = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let e: Result<()> = Err(inner).io_context("reading block payload");
        let msg = e.unwrap_err().to_string();
        assert!(msg.contains("reading block payload"), "{msg}");
        assert!(msg.contains("eof"), "{msg}");
    }

    #[test]
    fn non_corruption_has_no_offset() {
        assert_eq!(BasaltError::schema("nope").corruption_offset(), None);
        assert!(!BasaltError::internal("x", "y").is_corruption());
    }

    #[test]
    fn every_region_displays() {
        for r in [
            Region::FileHeader,
            Region::BlockHeader,
            Region::BlockPayload,
            Region::Footer,
            Region::FooterTrailer,
            Region::EncodedChunk,
            Region::BloomFilter,
            Region::ZoneMap,
        ] {
            assert!(!r.to_string().is_empty());
        }
    }
}

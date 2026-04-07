//! Engine-wide tuning knobs.

use crate::{BasaltError, ByteSize, Result};

/// How hard the encoding layer should work to make data small.
///
/// This is a *policy* dial, not a codec parameter: Basalt always picks the
/// encoding by estimating sizes (see `basalt_encoding::analyze`), and the level
/// decides how much estimated saving justifies a more expensive decode, and
/// whether the general-purpose byte codec is allowed at all.
///
/// | Level   | Candidates                              | Minimum saving to leave `Plain` |
/// |---------|-----------------------------------------|---------------------------------|
/// | 0       | `Plain` only                            | n/a                             |
/// | 1..=3   | lightweight only (no byte compressor)   | 20 %                            |
/// | 4..=6   | all                                     | 5 %                             |
/// | 7..=9   | all, exact statistics                   | 1 %                             |
///
/// The thresholds exist because a 2 % size win is not worth a 3x slower scan.
/// They are hard numbers rather than a vague "aggressiveness" so that the
/// behaviour is testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompressionLevel(u8);

impl CompressionLevel {
    /// Store everything uncompressed. Useful for debugging and for tiers where
    /// CPU is scarcer than bytes.
    pub const NONE: CompressionLevel = CompressionLevel(0);
    /// Lightweight encodings only: no general-purpose byte compression.
    pub const FAST: CompressionLevel = CompressionLevel(3);
    /// The default: all encodings, sampled statistics.
    pub const BALANCED: CompressionLevel = CompressionLevel(6);
    /// All encodings with exact statistics; slowest writes, smallest files.
    pub const MAX: CompressionLevel = CompressionLevel(9);

    /// Construct from a raw level. Returns `None` above 9.
    pub const fn new(level: u8) -> Option<CompressionLevel> {
        if level <= 9 {
            Some(CompressionLevel(level))
        } else {
            None
        }
    }

    /// The raw level, `0..=9`.
    pub const fn get(self) -> u8 {
        self.0
    }

    /// `true` if any encoding other than `Plain` may be chosen.
    pub const fn is_enabled(self) -> bool {
        self.0 > 0
    }

    /// `true` if the general-purpose byte compressor may be used for
    /// string/binary payloads.
    pub const fn allows_byte_compression(self) -> bool {
        self.0 >= 4
    }

    /// `true` if statistics should be computed exactly rather than sampled.
    pub const fn wants_exact_statistics(self) -> bool {
        self.0 >= 7
    }

    /// Fraction of the `Plain` size that must be saved before a heavier
    /// encoding is accepted, in `0.0..=1.0`.
    pub fn minimum_gain(self) -> f64 {
        match self.0 {
            0 => 1.0, // unreachable in practice: level 0 short-circuits earlier
            1..=3 => 0.20,
            4..=6 => 0.05,
            _ => 0.01,
        }
    }
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::BALANCED
    }
}

/// Engine-wide configuration.
///
/// Every field has a default that is sane for a laptop-scale analytical
/// workload, and every default is justified below. `Config` is `Clone` and
/// cheap; pass it by value.
///
/// ```
/// use basalt_common::{ByteSize, CompressionLevel, Config};
///
/// let cfg = Config::default()
///     .with_batch_size(4096)
///     .with_compression_level(CompressionLevel::MAX);
/// cfg.validate().unwrap();
/// assert_eq!(cfg.batch_size, 4096);
/// assert_eq!(cfg.page_size, ByteSize::from_kib(256));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Config {
    /// Rows per `RecordBatch` produced by the execution engine.
    ///
    /// Default **8192**. The classic vectorised-execution sweet spot: large
    /// enough that per-batch dispatch overhead disappears into the noise, small
    /// enough that one batch of a handful of 8-byte columns (8192 * 8 B = 64 KiB
    /// per column) still lives in L2 while an operator works on it. Powers of
    /// two make the bitmap arithmetic exact (8192 = 128 * 64-bit words).
    pub batch_size: usize,

    /// Target size of an on-disk block before compression.
    ///
    /// Default **256 KiB**. Blocks are the unit of checksumming and of I/O; too
    /// small and the per-block CRC and metadata dominate, too large and
    /// predicate pushdown loses resolution and the reader's transient buffer
    /// grows. 256 KiB is ~16 sequential 16 KiB reads on a spinning disk and one
    /// comfortable NVMe request.
    pub page_size: ByteSize,

    /// How aggressively to compress. Default [`CompressionLevel::BALANCED`].
    pub compression_level: CompressionLevel,

    /// Rows per row group in a segment.
    ///
    /// Default **65 536**. A row group is the unit of predicate-based skipping,
    /// so this is the knob that trades metadata size against skip resolution:
    /// at 64 Ki rows a billion-row table has ~15 000 row groups, whose zone maps
    /// still fit comfortably in memory, while a selective predicate can still
    /// discard 99.99 % of the file.
    pub row_group_rows: usize,

    /// Write a bloom filter per column chunk. Default **true**.
    ///
    /// Bloom filters only help equality predicates and cost roughly
    /// `-1.44 * log2(fpp)` bits per distinct value, so they are disabled for
    /// chunks whose cardinality is close to the row count (the filter would be
    /// as large as the data and never skip anything).
    pub write_bloom_filters: bool,

    /// Target false-positive probability for bloom filters. Default **0.01**.
    ///
    /// 1 % is the usual knee: going to 0.1 % costs 50 % more bits for a benefit
    /// that only shows up when the predicate is already highly selective.
    pub bloom_filter_fpp: f64,

    /// Upper bound on the memory a single [`SegmentWriter`]-style component may
    /// buffer before it must flush. Default **64 MiB**.
    ///
    /// [`SegmentWriter`]: https://docs.rs/basalt-storage
    pub write_buffer_limit: ByteSize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            batch_size: 8192,
            page_size: ByteSize::from_kib(256),
            compression_level: CompressionLevel::default(),
            row_group_rows: 65_536,
            write_bloom_filters: true,
            bloom_filter_fpp: 0.01,
            write_buffer_limit: ByteSize::from_mib(64),
        }
    }
}

impl Config {
    /// Alias for [`Config::default`], for readability at call sites.
    pub fn new() -> Self {
        Config::default()
    }

    /// Override [`Config::batch_size`].
    #[must_use]
    pub fn with_batch_size(mut self, rows: usize) -> Self {
        self.batch_size = rows;
        self
    }

    /// Override [`Config::page_size`].
    #[must_use]
    pub fn with_page_size(mut self, size: ByteSize) -> Self {
        self.page_size = size;
        self
    }

    /// Override [`Config::compression_level`].
    #[must_use]
    pub fn with_compression_level(mut self, level: CompressionLevel) -> Self {
        self.compression_level = level;
        self
    }

    /// Override [`Config::row_group_rows`].
    #[must_use]
    pub fn with_row_group_rows(mut self, rows: usize) -> Self {
        self.row_group_rows = rows;
        self
    }

    /// Override [`Config::write_bloom_filters`].
    #[must_use]
    pub fn with_bloom_filters(mut self, enabled: bool) -> Self {
        self.write_bloom_filters = enabled;
        self
    }

    /// Override [`Config::bloom_filter_fpp`].
    #[must_use]
    pub fn with_bloom_filter_fpp(mut self, fpp: f64) -> Self {
        self.bloom_filter_fpp = fpp;
        self
    }

    /// Check the knobs against each other and against hard engine limits.
    ///
    /// Called once at engine start-up rather than on every use, so that hot
    /// paths can assume a valid `Config`.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(BasaltError::plan(
                "config.batch_size must be greater than 0",
            ));
        }
        if self.batch_size > 1 << 20 {
            return Err(BasaltError::plan(format!(
                "config.batch_size {} exceeds the 1048576-row limit; larger batches defeat \
                 cache-resident execution",
                self.batch_size
            )));
        }
        if self.row_group_rows == 0 {
            return Err(BasaltError::plan(
                "config.row_group_rows must be greater than 0",
            ));
        }
        if self.row_group_rows < self.batch_size {
            return Err(BasaltError::plan(format!(
                "config.row_group_rows ({}) must be at least config.batch_size ({}): a row group \
                 that cannot fill one batch wastes a skip opportunity per batch",
                self.row_group_rows, self.batch_size
            )));
        }
        if self.page_size.bytes() < 4096 {
            return Err(BasaltError::plan(format!(
                "config.page_size {} is below the 4 KiB minimum (one OS page)",
                self.page_size
            )));
        }
        if !(0.0..1.0).contains(&self.bloom_filter_fpp) || self.bloom_filter_fpp == 0.0 {
            return Err(BasaltError::plan(format!(
                "config.bloom_filter_fpp must be in (0.0, 1.0), got {}",
                self.bloom_filter_fpp
            )));
        }
        if self.write_buffer_limit.bytes() < self.page_size.bytes() {
            return Err(BasaltError::plan(format!(
                "config.write_buffer_limit ({}) must be at least config.page_size ({})",
                self.write_buffer_limit, self.page_size
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn defaults_validate() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn rejects_bad_combinations() {
        assert!(Config::default().with_batch_size(0).validate().is_err());
        assert!(Config::default()
            .with_batch_size(1 << 21)
            .validate()
            .is_err());
        assert!(Config::default()
            .with_row_group_rows(16)
            .validate()
            .is_err());
        assert!(Config::default()
            .with_page_size(ByteSize::new(10))
            .validate()
            .is_err());
        assert!(Config::default()
            .with_bloom_filter_fpp(0.0)
            .validate()
            .is_err());
        assert!(Config::default()
            .with_bloom_filter_fpp(1.5)
            .validate()
            .is_err());
    }

    #[test]
    fn error_message_is_actionable() {
        let err = Config::default()
            .with_row_group_rows(16)
            .validate()
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("row_group_rows"), "{msg}");
        assert!(msg.contains("16"), "{msg}");
        assert!(msg.contains("8192"), "{msg}");
    }

    #[test]
    fn compression_levels_are_ordered() {
        assert!(!CompressionLevel::NONE.is_enabled());
        assert!(!CompressionLevel::FAST.allows_byte_compression());
        assert!(CompressionLevel::BALANCED.allows_byte_compression());
        assert!(CompressionLevel::MAX.wants_exact_statistics());
        assert!(!CompressionLevel::BALANCED.wants_exact_statistics());
        assert!(CompressionLevel::FAST.minimum_gain() > CompressionLevel::MAX.minimum_gain());
        assert_eq!(CompressionLevel::new(10), None);
        assert_eq!(CompressionLevel::new(9).map(CompressionLevel::get), Some(9));
    }
}

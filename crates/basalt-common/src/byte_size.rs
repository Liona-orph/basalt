//! A byte count that formats the way a human would write it.

use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::str::FromStr;

/// Number of bytes, with human-readable [`Display`](fmt::Display) and
/// [`FromStr`] using **binary** (power-of-two) units.
///
/// Basalt reports a lot of sizes — block lengths, buffer capacities, memory
/// budgets — and every one of them was, before this type existed, a bare `u64`
/// that some call site printed raw. Wrapping the count also stops the classic
/// "is this a row count or a byte count?" mix-up at compile time.
///
/// Only binary units are supported (`KiB` = 1024 B) and the aliases `KB`/`K`
/// are *also* parsed as 1024: a storage engine that silently switched between
/// 1000 and 1024 depending on spelling would be a bug factory. The `Display`
/// side always writes the unambiguous `KiB`/`MiB` spelling.
///
/// ```
/// use basalt_common::ByteSize;
///
/// assert_eq!(ByteSize::from_kib(1).to_string(), "1 KiB");
/// assert_eq!(ByteSize::new(1_536).to_string(), "1.5 KiB");
/// assert_eq!(ByteSize::new(999).to_string(), "999 B");
/// assert_eq!("64MiB".parse::<ByteSize>().unwrap(), ByteSize::from_mib(64));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ByteSize(u64);

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;
const TIB: u64 = 1024 * GIB;
const PIB: u64 = 1024 * TIB;

impl ByteSize {
    /// Zero bytes.
    pub const ZERO: ByteSize = ByteSize(0);

    /// Construct from a raw byte count.
    pub const fn new(bytes: u64) -> Self {
        ByteSize(bytes)
    }

    /// `n` kibibytes (`n * 1024` bytes).
    pub const fn from_kib(n: u64) -> Self {
        ByteSize(n.saturating_mul(KIB))
    }

    /// `n` mebibytes.
    pub const fn from_mib(n: u64) -> Self {
        ByteSize(n.saturating_mul(MIB))
    }

    /// `n` gibibytes.
    pub const fn from_gib(n: u64) -> Self {
        ByteSize(n.saturating_mul(GIB))
    }

    /// The raw byte count.
    pub const fn bytes(self) -> u64 {
        self.0
    }

    /// The byte count as `usize`, saturating on 32-bit targets.
    ///
    /// Used when a size has to become a capacity; saturation is the right
    /// behaviour because the subsequent allocation will fail loudly anyway.
    pub const fn as_usize(self) -> usize {
        if self.0 > usize::MAX as u64 {
            usize::MAX
        } else {
            self.0 as usize
        }
    }

    /// Ratio `self / other` as a compression ratio, or `None` if `other` is 0.
    ///
    /// Used by the encoding layer to report "3.4x" numbers without every call
    /// site re-implementing the zero check.
    pub fn ratio_to(self, other: ByteSize) -> Option<f64> {
        if other.0 == 0 {
            None
        } else {
            Some(self.0 as f64 / other.0 as f64)
        }
    }
}

impl fmt::Display for ByteSize {
    /// Formats with at most two decimal places, trailing zeros trimmed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (unit, suffix) = match self.0 {
            0..=1_023 => (1, "B"),
            v if v < MIB => (KIB, "KiB"),
            v if v < GIB => (MIB, "MiB"),
            v if v < TIB => (GIB, "GiB"),
            v if v < PIB => (TIB, "TiB"),
            _ => (PIB, "PiB"),
        };
        if unit == 1 {
            return write!(f, "{} B", self.0);
        }
        let scaled = self.0 as f64 / unit as f64;
        let mut text = format!("{scaled:.2}");
        // Trim "1.50" -> "1.5" and "1.00" -> "1"; a size report should not
        // pretend to a precision it does not have.
        while text.contains('.') && (text.ends_with('0') || text.ends_with('.')) {
            text.pop();
        }
        write!(f, "{text} {suffix}")
    }
}

/// Error returned by [`ByteSize`]'s [`FromStr`] implementation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseByteSizeError {
    /// The numeric part was missing or not a number.
    #[error("`{0}` is not a valid byte size: expected a number, optionally followed by a unit")]
    NotANumber(String),
    /// The unit suffix was not recognised.
    #[error("`{0}` is not a known unit (expected one of B, KiB, MiB, GiB, TiB, PiB)")]
    UnknownUnit(String),
    /// The value overflowed `u64`.
    #[error("byte size `{0}` overflows u64")]
    Overflow(String),
}

impl FromStr for ByteSize {
    type Err = ParseByteSizeError;

    /// Accepts `"1024"`, `"1 KiB"`, `"1.5MiB"`, `"64 gb"` (case-insensitive,
    /// always binary units).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        let split = trimmed
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(trimmed.len());
        let (num, unit) = trimmed.split_at(split);
        let num = num.trim();
        if num.is_empty() {
            return Err(ParseByteSizeError::NotANumber(s.to_string()));
        }
        let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
            "" | "b" => 1,
            "k" | "kb" | "kib" => KIB,
            "m" | "mb" | "mib" => MIB,
            "g" | "gb" | "gib" => GIB,
            "t" | "tb" | "tib" => TIB,
            "p" | "pb" | "pib" => PIB,
            other => return Err(ParseByteSizeError::UnknownUnit(other.to_string())),
        };
        if let Ok(int) = num.parse::<u64>() {
            return int
                .checked_mul(multiplier)
                .map(ByteSize)
                .ok_or_else(|| ParseByteSizeError::Overflow(s.to_string()));
        }
        let float: f64 = num
            .parse()
            .map_err(|_| ParseByteSizeError::NotANumber(s.to_string()))?;
        if !float.is_finite() || float < 0.0 {
            return Err(ParseByteSizeError::NotANumber(s.to_string()));
        }
        let scaled = float * multiplier as f64;
        if scaled > u64::MAX as f64 {
            return Err(ParseByteSizeError::Overflow(s.to_string()));
        }
        Ok(ByteSize(scaled as u64))
    }
}

impl From<u64> for ByteSize {
    fn from(v: u64) -> Self {
        ByteSize(v)
    }
}

impl From<usize> for ByteSize {
    fn from(v: usize) -> Self {
        ByteSize(v as u64)
    }
}

impl From<ByteSize> for u64 {
    fn from(v: ByteSize) -> Self {
        v.0
    }
}

impl Add for ByteSize {
    type Output = ByteSize;
    /// Saturating: a size report should never wrap around to a small number.
    fn add(self, rhs: ByteSize) -> ByteSize {
        ByteSize(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for ByteSize {
    fn add_assign(&mut self, rhs: ByteSize) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for ByteSize {
    type Output = ByteSize;
    /// Saturating at zero.
    fn sub(self, rhs: ByteSize) -> ByteSize {
        ByteSize(self.0.saturating_sub(rhs.0))
    }
}

impl Mul<u64> for ByteSize {
    type Output = ByteSize;
    fn mul(self, rhs: u64) -> ByteSize {
        ByteSize(self.0.saturating_mul(rhs))
    }
}

impl std::iter::Sum for ByteSize {
    fn sum<I: Iterator<Item = ByteSize>>(iter: I) -> ByteSize {
        iter.fold(ByteSize::ZERO, Add::add)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn formats_each_unit() {
        assert_eq!(ByteSize::new(0).to_string(), "0 B");
        assert_eq!(ByteSize::new(1).to_string(), "1 B");
        assert_eq!(ByteSize::new(1023).to_string(), "1023 B");
        assert_eq!(ByteSize::new(1024).to_string(), "1 KiB");
        assert_eq!(ByteSize::new(1536).to_string(), "1.5 KiB");
        assert_eq!(ByteSize::new(1_234_567).to_string(), "1.18 MiB");
        assert_eq!(ByteSize::from_gib(3).to_string(), "3 GiB");
        // We stop at PiB rather than inventing EiB/ZiB units nobody reads.
        assert_eq!(ByteSize::new(u64::MAX).to_string(), "16384 PiB");
    }

    #[test]
    fn parses_round_trip() {
        for s in ["0", "1", "1023", "1024", "64KiB", "1.5 MiB", "3gb", "7 TiB"] {
            let parsed: ByteSize = s.parse().unwrap();
            let reparsed: ByteSize = parsed.to_string().parse().unwrap();
            assert_eq!(parsed, reparsed, "input {s}");
        }
    }

    #[test]
    fn rejects_nonsense() {
        assert!("".parse::<ByteSize>().is_err());
        assert!("KiB".parse::<ByteSize>().is_err());
        assert!("12 quatloos".parse::<ByteSize>().is_err());
        assert!("99999999999999999999 PiB".parse::<ByteSize>().is_err());
    }

    #[test]
    fn arithmetic_saturates() {
        assert_eq!(ByteSize::new(5) - ByteSize::new(9), ByteSize::ZERO);
        assert_eq!(
            ByteSize::new(u64::MAX) + ByteSize::new(1),
            ByteSize::new(u64::MAX)
        );
        let total: ByteSize = [ByteSize::new(1), ByteSize::new(2)].into_iter().sum();
        assert_eq!(total, ByteSize::new(3));
    }

    #[test]
    fn ratio() {
        assert_eq!(ByteSize::new(100).ratio_to(ByteSize::new(25)), Some(4.0));
        assert_eq!(ByteSize::new(100).ratio_to(ByteSize::ZERO), None);
    }
}

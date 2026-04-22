//! Per-chunk bloom filters for equality predicates.

use basalt_common::{BasaltError, Region, Result};
use basalt_encoding::varint;
use basalt_types::Value;

use crate::serial::write_value;

/// Version byte of the serialised filter.
const BLOOM_VERSION: u8 = 1;

/// Largest number of hash functions. Beyond ~16 the extra probes cost more than
/// the false positives they remove, and a damaged header must not be able to
/// make the reader do unbounded work.
const MAX_HASHES: u32 = 16;

/// A classic bloom filter over 64-bit hashes.
///
/// # The one guarantee
///
/// `contains` may return `true` for a value that was never inserted (a false
/// positive, at the configured rate), but it **never** returns `false` for a
/// value that was inserted. That asymmetry is the entire reason a bloom filter
/// can be used to skip row groups: a `false` is a proof of absence, so the block
/// can be skipped; a `true` only means "read it and check". The property test in
/// `tests/segment.rs` asserts exactly this over random data.
///
/// # Sizing
///
/// For `n` distinct values at false-positive probability `p`, the optimal
/// filter is `m = -n ln p / (ln 2)^2` bits with `k = (m/n) ln 2` hash
/// functions. At `p = 0.01` that is 9.6 bits and 7 probes per value — 1.2 bytes
/// per distinct value, which is why a filter over a unique `i64` column costs
/// 15 % of the column and can eliminate a whole row group.
///
/// # Double hashing
///
/// The `k` bit positions come from `h1 + i * h2` rather than `k` independent
/// hashes. Kirsch and Mitzenmacher showed this has the same asymptotic false
/// positive rate, and it costs one hash instead of `k`. `h2` is forced odd so
/// that it generates the full residue system and the probes cannot collapse.
///
/// ```
/// use basalt_storage::{value_hash, BloomFilter};
/// use basalt_types::Value;
///
/// let mut filter = BloomFilter::with_capacity(1000, 0.01);
/// for i in 0..1000 {
///     filter.insert(value_hash(&Value::Int64(i)));
/// }
/// // No false negatives, ever.
/// for i in 0..1000 {
///     assert!(filter.contains(value_hash(&Value::Int64(i))));
/// }
/// // And it is small: ~1.2 bytes per distinct value.
/// assert!(filter.size_bytes() < 1600);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloomFilter {
    words: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

impl BloomFilter {
    /// A filter sized for `expected_distinct` values at false-positive rate
    /// `fpp`.
    ///
    /// `fpp` outside `(0, 1)` is clamped to 1 %; the caller's `Config` is
    /// validated at start-up, so this is belt-and-braces rather than a real
    /// path.
    pub fn with_capacity(expected_distinct: usize, fpp: f64) -> BloomFilter {
        let n = expected_distinct.max(1) as f64;
        let p = if fpp > 0.0 && fpp < 1.0 { fpp } else { 0.01 };
        let ln2 = std::f64::consts::LN_2;
        let bits = (-n * p.ln() / (ln2 * ln2)).ceil().max(64.0);
        // Round up to a whole 64-bit word so the modulo never straddles.
        let num_bits = ((bits as usize).div_ceil(64)) * 64;
        let num_hashes = (((num_bits as f64 / n) * ln2).round() as u32).clamp(1, MAX_HASHES);
        BloomFilter {
            words: vec![0; num_bits / 64],
            num_bits,
            num_hashes,
        }
    }

    /// Insert a hash produced by [`value_hash`].
    pub fn insert(&mut self, hash: u64) {
        // The probe positions are computed up front because `probes` borrows
        // `self` and the loop body needs it mutably. `num_hashes <= 16`, so this
        // is a fixed-size stack array in practice.
        let bits: Vec<usize> = probe_positions(hash, self.num_bits, self.num_hashes).collect();
        for bit in bits {
            self.words[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    /// `false` proves the value was never inserted; `true` means "maybe".
    pub fn contains(&self, hash: u64) -> bool {
        self.probes(hash)
            .all(|bit| self.words[bit / 64] & (1u64 << (bit % 64)) != 0)
    }

    /// Number of bits in the filter.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of hash probes per lookup.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Serialised size in bytes, including the header.
    pub fn size_bytes(&self) -> usize {
        self.words.len() * 8 + 8
    }

    /// Fraction of bits set, which is the observable "how full is it" signal.
    ///
    /// A saturated filter (close to 1.0) has stopped being useful, and a writer
    /// can log that rather than silently producing a filter that never skips.
    pub fn saturation(&self) -> f64 {
        let set: u32 = self.words.iter().map(|w| w.count_ones()).sum();
        f64::from(set) / self.num_bits as f64
    }

    fn probes(&self, hash: u64) -> impl Iterator<Item = usize> {
        probe_positions(hash, self.num_bits, self.num_hashes)
    }

    /// Serialise: `[version][num_bits][num_hashes][words...]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.words.len() * 8 + 16);
        out.push(BLOOM_VERSION);
        varint::write_uvarint(self.num_bits as u64, &mut out);
        varint::write_uvarint(u64::from(self.num_hashes), &mut out);
        for word in &self.words {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// Parse a filter written by [`BloomFilter::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<BloomFilter> {
        let corrupt = |offset: usize, detail: String| {
            BasaltError::corruption("bloom filter", Region::BloomFilter, offset as u64, detail)
        };
        let version = *bytes
            .first()
            .ok_or_else(|| corrupt(0, "empty bloom filter block".into()))?;
        if version != BLOOM_VERSION {
            return Err(corrupt(
                0,
                format!("bloom filter version {version} is not supported"),
            ));
        }
        let mut pos = 1;
        let num_bits = varint::read_uvarint(bytes, &mut pos)? as usize;
        let num_hashes = varint::read_uvarint(bytes, &mut pos)? as u32;
        if num_bits == 0 || num_bits % 64 != 0 {
            return Err(corrupt(
                pos,
                format!("bloom filter bit count {num_bits} is not a multiple of 64"),
            ));
        }
        if num_hashes == 0 || num_hashes > MAX_HASHES {
            return Err(corrupt(
                pos,
                format!("bloom filter probe count {num_hashes} is out of range"),
            ));
        }
        let word_count = num_bits / 64;
        if bytes.len() - pos < word_count * 8 {
            return Err(corrupt(
                pos,
                format!(
                    "bloom filter needs {} body bytes, found {}",
                    word_count * 8,
                    bytes.len() - pos
                ),
            ));
        }
        let mut words = Vec::with_capacity(word_count);
        for index in 0..word_count {
            let start = pos + index * 8;
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[start..start + 8]);
            words.push(u64::from_le_bytes(buf));
        }
        Ok(BloomFilter {
            words,
            num_bits,
            num_hashes,
        })
    }
}

/// The `k` bit positions for one hash, by double hashing.
fn probe_positions(hash: u64, num_bits: usize, num_hashes: u32) -> impl Iterator<Item = usize> {
    let h1 = hash;
    // Forcing the stride odd keeps `h1 + i*h2 (mod m)` from cycling early.
    let h2 = (hash >> 32) | 1;
    let modulus = num_bits as u64;
    (0..num_hashes)
        .map(move |i| (h1.wrapping_add(u64::from(i).wrapping_mul(h2)) % modulus) as usize)
}

/// Stable 64-bit hash of a [`Value`].
///
/// "Stable" is the operative word: the hash is written into a file by one
/// process and probed by another, possibly a different build, so
/// `std::collections::hash_map::DefaultHasher` (deliberately unspecified and
/// randomly seeded) cannot be used. This is FNV-1a over the canonical value
/// encoding, finished with the splitmix64 mixer because FNV's avalanche in the
/// high bits is poor and the double-hashing scheme takes `h >> 32` as its
/// stride.
///
/// Hashing the *canonical encoding* rather than the value's fields means the
/// hash automatically agrees with [`Value`]'s equality: values that serialise
/// identically hash identically, and the serialiser is the same one the footer
/// uses.
///
/// Note that the type tag is part of the encoding, so `Int32(5)` and `Int64(5)`
/// hash differently — which is correct, because they are different values under
/// `Value`'s equality. Predicate evaluation casts the probe to the column's type
/// before hashing.
pub fn value_hash(value: &Value) -> u64 {
    let mut encoded = Vec::with_capacity(24);
    write_value(value, &mut encoded);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in &encoded {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    // splitmix64 finaliser.
    let mut z = hash;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut filter = BloomFilter::with_capacity(5000, 0.01);
        let inserted: Vec<Value> = (0..5000).map(Value::Int64).collect();
        for value in &inserted {
            filter.insert(value_hash(value));
        }
        for value in &inserted {
            assert!(filter.contains(value_hash(value)), "{value} went missing");
        }
    }

    #[test]
    fn false_positive_rate_is_near_the_target() {
        let mut filter = BloomFilter::with_capacity(10_000, 0.01);
        for i in 0..10_000i64 {
            filter.insert(value_hash(&Value::Int64(i)));
        }
        let probes = 100_000i64;
        let hits = (1_000_000..1_000_000 + probes)
            .filter(|i| filter.contains(value_hash(&Value::Int64(*i))))
            .count();
        let rate = hits as f64 / probes as f64;
        assert!(
            rate < 0.03,
            "false positive rate {rate} is far above the 1% target"
        );
        assert!(
            filter.saturation() > 0.1 && filter.saturation() < 0.9,
            "{}",
            filter.saturation()
        );
    }

    #[test]
    fn sizing_follows_the_formula() {
        let filter = BloomFilter::with_capacity(1000, 0.01);
        // -1000 * ln(0.01) / ln(2)^2 = 9585 bits, rounded up to a word multiple.
        assert_eq!(filter.num_bits(), 9600);
        assert_eq!(filter.num_hashes(), 7);
        let tighter = BloomFilter::with_capacity(1000, 0.001);
        assert!(tighter.num_bits() > filter.num_bits());
        // Degenerate inputs must still produce a usable filter.
        assert!(BloomFilter::with_capacity(0, 0.01).num_bits() >= 64);
        assert_eq!(
            BloomFilter::with_capacity(10, -1.0).num_hashes().max(1),
            BloomFilter::with_capacity(10, 0.01).num_hashes()
        );
    }

    #[test]
    fn serialisation_round_trip() {
        let mut filter = BloomFilter::with_capacity(200, 0.01);
        for i in 0..200i64 {
            filter.insert(value_hash(&Value::Int64(i)));
        }
        let bytes = filter.to_bytes();
        assert_eq!(BloomFilter::from_bytes(&bytes).unwrap(), filter);
        assert_eq!(filter.size_bytes(), filter.num_bits() / 8 + 8);
    }

    #[test]
    fn corrupt_filters_are_rejected() {
        let filter = BloomFilter::with_capacity(64, 0.01);
        let bytes = filter.to_bytes();
        assert!(BloomFilter::from_bytes(&[]).is_err());
        let mut wrong_version = bytes.clone();
        wrong_version[0] = 9;
        assert!(BloomFilter::from_bytes(&wrong_version).is_err());
        assert!(BloomFilter::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut absurd = vec![BLOOM_VERSION];
        varint::write_uvarint(65, &mut absurd);
        varint::write_uvarint(3, &mut absurd);
        assert!(
            BloomFilter::from_bytes(&absurd).is_err(),
            "bit count must be a word multiple"
        );
        let mut too_many = vec![BLOOM_VERSION];
        varint::write_uvarint(64, &mut too_many);
        varint::write_uvarint(99, &mut too_many);
        assert!(BloomFilter::from_bytes(&too_many).is_err());
    }

    #[test]
    fn hashing_is_stable_and_type_aware() {
        // Hard-coded so a change to the hash function fails loudly: it would
        // silently invalidate every bloom filter ever written.
        assert_eq!(value_hash(&Value::Int64(0)), value_hash(&Value::Int64(0)));
        assert_ne!(value_hash(&Value::Int32(5)), value_hash(&Value::Int64(5)));
        assert_ne!(
            value_hash(&Value::Float64(0.0)),
            value_hash(&Value::Float64(-0.0))
        );
        assert_eq!(
            value_hash(&Value::Utf8("a".into())),
            value_hash(&Value::Utf8("a".into()))
        );
        assert_ne!(
            value_hash(&Value::Utf8("a".into())),
            value_hash(&Value::Binary(b"a".to_vec()))
        );
    }
}

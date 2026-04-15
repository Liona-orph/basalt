//! Column statistics and the encoding-selection heuristic.
//!
//! # How Basalt picks an encoding
//!
//! The selector is a *size estimator plus an argmin*, not a pile of heuristics.
//! One linear pass over the chunk collects statistics; each candidate encoding
//! then reports what it would cost, in bytes, using a closed-form expression;
//! the smallest wins, with ties broken toward the encoding that decodes fastest;
//! and finally the winner has to beat `Plain` by a configured margin or `Plain`
//! is used anyway.
//!
//! Two properties make this worth doing:
//!
//! * **It is honest.** Most of the estimates are *exact*, not approximations:
//!   the statistics pass accumulates the true varint and run-length body sizes
//!   as it goes, and bit-packing sizes are closed-form. Only the delta estimate
//!   is approximate, and it is deliberately conservative (it uses the global
//!   delta range where the codec uses per-block ranges), so delta is picked only
//!   when it wins by more than the slack.
//! * **It is testable.** [`estimate_sizes`] is a pure function of
//!   [`IntegerStats`], so the decision procedure can be unit-tested without
//!   encoding anything, and the encoders can be checked against their own
//!   estimates.
//!
//! ## The estimates
//!
//! With `n` values, `h` = header bytes and `pack(n, w) = ceil(n*w/8)`:
//!
//! | Encoding | Estimated body | Exact? |
//! |---|---|---|
//! | `Plain` | `8n` | yes |
//! | `Varint` | sum of zigzag varint lengths | yes (accumulated) |
//! | `BitPacked` | `1 + pack(n, width(max zigzag))` | yes |
//! | `FrameOfReference` | `9 + pack(n, width(max - min))` | yes |
//! | `RunLength` | sum over runs of `varint(value) + varint(length)` | yes (accumulated) |
//! | `Dictionary` | dictionary gaps + `1 + pack(n, width(k-1))` | yes when `k <= 16384` |
//! | `DeltaBinaryPacked` | per-block minima + `pack(n-1, width(delta range))` | conservative |
//!
//! ## Why the `Plain` margin exists
//!
//! Compression is not free at read time. A 2 % size win that makes a scan 3x
//! slower is a bad trade for an analytical engine whose queries are usually
//! scan-bound, so [`CompressionLevel::minimum_gain`] sets how much saving is
//! required before leaving `Plain`: 20 % at level 1-3, 5 % at 4-6, 1 % at 7-9.

use std::collections::HashSet;

use basalt_common::CompressionLevel;

use crate::bitpack::{packed_len, required_width};
use crate::codec::Encoding;
use crate::varint::{self, uvarint_len, varint_len, zigzag_encode};

/// Largest dictionary the selector will consider.
///
/// 2^14 entries means 14-bit codes, i.e. 1.75 bytes per value — still a 4.5x
/// win over `Plain`. Above that the dictionary itself starts to dominate, the
/// gather loses cache locality, and the analysis pass would have to hold an
/// unbounded set, so Basalt stops looking.
pub const MAX_DICTIONARY_CARDINALITY: usize = 1 << 14;

/// Number of distinct values, exactly or as a lower bound.
///
/// The analysis pass stops counting once a dictionary is out of the question,
/// so callers must handle "at least this many".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// The exact number of distinct values.
    Exact(usize),
    /// At least this many; counting was abandoned.
    AtLeast(usize),
}

impl Cardinality {
    /// The number, whether exact or a lower bound.
    pub const fn value(self) -> usize {
        match self {
            Cardinality::Exact(n) | Cardinality::AtLeast(n) => n,
        }
    }

    /// `true` if the count is exact.
    pub const fn is_exact(self) -> bool {
        matches!(self, Cardinality::Exact(_))
    }
}

/// Whether a column is already ordered, which is what makes delta encoding pay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sortedness {
    /// Fewer than two values, or all values equal.
    Constant,
    /// Non-decreasing.
    Ascending,
    /// Non-increasing.
    Descending,
    /// Neither.
    Unsorted,
}

/// Everything the selector needs, collected in one pass.
#[derive(Debug, Clone)]
pub struct IntegerStats {
    /// Number of values analysed.
    pub count: usize,
    /// The first value, which delta encoding stores verbatim.
    pub first: i64,
    /// Smallest value (0 for an empty chunk).
    pub min: i64,
    /// Largest value (0 for an empty chunk).
    pub max: i64,
    /// Number of maximal runs of equal values.
    pub run_count: usize,
    /// Ordering of the values.
    pub sortedness: Sortedness,
    /// Distinct value count, exact or bounded.
    pub cardinality: Cardinality,
    /// `max - min` as an unsigned span (exact even across the full `i64` range).
    pub value_span: u64,
    /// Largest zigzag-transformed value, i.e. what fixes the bit-packing width.
    pub max_zigzag: u64,
    /// Exact size of a varint body, accumulated during the pass.
    pub varint_body_bytes: usize,
    /// Exact size of a run-length body (excluding the run count prefix).
    pub run_length_body_bytes: usize,
    /// Smallest consecutive delta (0 for fewer than two values).
    pub delta_min: i64,
    /// Largest consecutive delta.
    pub delta_max: i64,
    /// `delta_max - delta_min` as an unsigned span.
    pub delta_span: u64,
    /// The sorted distinct values, when there were few enough to keep.
    distinct_sorted: Option<Vec<i64>>,
}

impl IntegerStats {
    /// The sorted distinct values, if the cardinality was low enough to record.
    pub fn distinct_values(&self) -> Option<&[i64]> {
        self.distinct_sorted.as_deref()
    }

    /// Fraction of values that start a new run, in `0.0..=1.0`. Low means runs
    /// are long, which is what run-length encoding needs.
    pub fn run_density(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.run_count as f64 / self.count as f64
        }
    }
}

const fn header_len(count: usize) -> usize {
    2 + uvarint_len(count as u64)
}

/// Collect [`IntegerStats`] in a single pass. O(n) time, O(min(k, 16384)) space.
///
/// The pass does five things at once — extremes, runs, sortedness, distinct
/// values and the exact varint/run-length body sizes — because the data is only
/// worth touching once; a separate pass per statistic would cost more than the
/// encoding itself on a wide table.
// One pass, five statistics, and two 20-field struct literals. Splitting the
// loop would mean several passes over the data, which is the one thing this
// function exists to avoid.
#[allow(clippy::too_many_lines)]
pub fn analyze_integers(values: &[i64]) -> IntegerStats {
    if values.is_empty() {
        return IntegerStats {
            count: 0,
            first: 0,
            min: 0,
            max: 0,
            run_count: 0,
            sortedness: Sortedness::Constant,
            cardinality: Cardinality::Exact(0),
            value_span: 0,
            max_zigzag: 0,
            varint_body_bytes: 0,
            run_length_body_bytes: 0,
            delta_min: 0,
            delta_max: 0,
            delta_span: 0,
            distinct_sorted: Some(Vec::new()),
        };
    }

    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut max_zigzag = 0u64;
    let mut varint_body_bytes = 0usize;
    let mut run_length_body_bytes = 0usize;
    let mut run_count = 0usize;
    let mut ascending = true;
    let mut descending = true;
    let mut delta_min = i64::MAX;
    let mut delta_max = i64::MIN;
    let mut distinct: Option<HashSet<i64>> = Some(HashSet::new());

    let mut run_value = values[0];
    let mut run_len = 0usize;

    for (i, &value) in values.iter().enumerate() {
        min = min.min(value);
        max = max.max(value);
        let zigzag = zigzag_encode(value);
        max_zigzag = max_zigzag.max(zigzag);
        varint_body_bytes += uvarint_len(zigzag);

        if i > 0 {
            let previous = values[i - 1];
            if value < previous {
                ascending = false;
            }
            if value > previous {
                descending = false;
            }
            let delta = value.wrapping_sub(previous);
            delta_min = delta_min.min(delta);
            delta_max = delta_max.max(delta);
        }

        if value == run_value {
            run_len += 1;
        } else {
            run_count += 1;
            run_length_body_bytes += varint_len(run_value) + uvarint_len(run_len as u64);
            run_value = value;
            run_len = 1;
        }

        if let Some(set) = &mut distinct {
            set.insert(value);
            if set.len() > MAX_DICTIONARY_CARDINALITY {
                distinct = None;
            }
        }
    }
    run_count += 1;
    run_length_body_bytes += varint_len(run_value) + uvarint_len(run_len as u64);

    let sortedness = match (ascending, descending) {
        (true, true) => Sortedness::Constant,
        (true, false) => Sortedness::Ascending,
        (false, true) => Sortedness::Descending,
        (false, false) => Sortedness::Unsorted,
    };
    let (cardinality, distinct_sorted) = match distinct {
        Some(set) => {
            let mut sorted: Vec<i64> = set.into_iter().collect();
            sorted.sort_unstable();
            (Cardinality::Exact(sorted.len()), Some(sorted))
        }
        None => (Cardinality::AtLeast(MAX_DICTIONARY_CARDINALITY + 1), None),
    };
    let delta_span = if values.len() < 2 {
        0
    } else {
        delta_max.wrapping_sub(delta_min) as u64
    };

    IntegerStats {
        count: values.len(),
        first: values[0],
        min,
        max,
        run_count,
        sortedness,
        cardinality,
        value_span: max.wrapping_sub(min) as u64,
        max_zigzag,
        varint_body_bytes,
        run_length_body_bytes,
        delta_min: if values.len() < 2 { 0 } else { delta_min },
        delta_max: if values.len() < 2 { 0 } else { delta_max },
        delta_span,
        distinct_sorted,
    }
}

/// Predicted encoded size, in bytes, of every candidate encoding.
///
/// `None` means the encoding is not applicable to this chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeEstimates {
    /// Size with [`Encoding::Plain`].
    pub plain: usize,
    /// Size with [`Encoding::Varint`].
    pub varint: usize,
    /// Size with [`Encoding::BitPacked`].
    pub bit_packed: usize,
    /// Size with [`Encoding::FrameOfReference`].
    pub frame_of_reference: usize,
    /// Size with [`Encoding::RunLength`].
    pub run_length: usize,
    /// Size with [`Encoding::Dictionary`], if the cardinality is low enough.
    pub dictionary: Option<usize>,
    /// Size with [`Encoding::DeltaBinaryPacked`], if there are at least 2 values.
    pub delta: Option<usize>,
}

impl SizeEstimates {
    /// `(encoding, size)` for every applicable candidate.
    pub fn candidates(&self) -> Vec<(Encoding, usize)> {
        let mut out = vec![
            (Encoding::Plain, self.plain),
            (Encoding::Varint, self.varint),
            (Encoding::BitPacked, self.bit_packed),
            (Encoding::FrameOfReference, self.frame_of_reference),
            (Encoding::RunLength, self.run_length),
        ];
        if let Some(size) = self.dictionary {
            out.push((Encoding::Dictionary, size));
        }
        if let Some(size) = self.delta {
            out.push((Encoding::DeltaBinaryPacked, size));
        }
        out
    }
}

/// Closed-form size predictions for every encoding. Pure function of the stats.
pub fn estimate_sizes(stats: &IntegerStats) -> SizeEstimates {
    let n = stats.count;
    let header = header_len(n);

    let plain = header + n * 8;
    let varint = header + stats.varint_body_bytes;
    let bit_packed = header + 1 + packed_len(n, required_width(stats.max_zigzag));
    let frame_of_reference = header + 8 + 1 + packed_len(n, required_width(stats.value_span));
    let run_length = header + uvarint_len(stats.run_count as u64) + stats.run_length_body_bytes;

    let dictionary = stats.distinct_sorted.as_ref().map(|dictionary| {
        // First value as a zigzag varint, then unsigned gaps.
        let mut body = uvarint_len(dictionary.len() as u64);
        if let Some(&first) = dictionary.first() {
            body += varint_len(first);
            for pair in dictionary.windows(2) {
                body += uvarint_len(pair[1].wrapping_sub(pair[0]) as u64);
            }
        }
        let code_width = required_width(dictionary.len().saturating_sub(1) as u64);
        header + body + 1 + packed_len(n, code_width)
    });

    let delta = if n >= 2 {
        let blocks = (n - 1).div_ceil(crate::integer::DELTA_BLOCK_SIZE);
        // Conservative on two counts: it charges the widest possible per-block
        // minimum varint, and it uses the *global* delta range where the codec
        // uses a per-block range. Delta therefore has to win comfortably.
        let per_block_min = varint_len(stats.delta_min).max(varint_len(stats.delta_max));
        Some(
            header
                + varint_len(stats.first)
                + uvarint_len(crate::integer::DELTA_BLOCK_SIZE as u64)
                + blocks * (per_block_min + 1)
                + packed_len(n - 1, required_width(stats.delta_span)),
        )
    } else {
        None
    };

    SizeEstimates {
        plain,
        varint,
        bit_packed,
        frame_of_reference,
        run_length,
        dictionary,
        delta,
    }
}

/// Why the selector chose what it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanReason {
    /// The chunk was empty, so `Plain` costs nothing.
    Empty,
    /// [`CompressionLevel::NONE`] was requested.
    CompressionDisabled,
    /// This encoding had the smallest estimate.
    SmallestEstimate,
    /// Something was smaller, but not by enough to justify a slower decode.
    GainBelowThreshold,
}

/// The chosen encoding and the numbers behind the choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodingPlan {
    /// The encoding to use.
    pub encoding: Encoding,
    /// Predicted size with that encoding.
    pub estimated_bytes: usize,
    /// Predicted size with [`Encoding::Plain`], for comparison.
    pub plain_bytes: usize,
    /// Why.
    pub reason: PlanReason,
}

impl EncodingPlan {
    /// Predicted compression ratio against `Plain` (`1.0` means no gain).
    pub fn estimated_ratio(&self) -> f64 {
        if self.estimated_bytes == 0 {
            1.0
        } else {
            self.plain_bytes as f64 / self.estimated_bytes as f64
        }
    }
}

/// Choose an encoding from statistics alone.
///
/// Separated from [`analyze`] so the decision procedure can be tested against
/// hand-written statistics, including shapes that are expensive to generate.
pub fn choose_integer_encoding(stats: &IntegerStats, level: CompressionLevel) -> EncodingPlan {
    let estimates = estimate_sizes(stats);
    let plain_bytes = estimates.plain;

    if stats.count == 0 {
        return EncodingPlan {
            encoding: Encoding::Plain,
            estimated_bytes: plain_bytes,
            plain_bytes,
            reason: PlanReason::Empty,
        };
    }
    if !level.is_enabled() {
        return EncodingPlan {
            encoding: Encoding::Plain,
            estimated_bytes: plain_bytes,
            plain_bytes,
            reason: PlanReason::CompressionDisabled,
        };
    }

    // argmin over (size, decode cost): when two encodings tie on bytes, the one
    // that scans faster wins.
    let best = estimates
        .candidates()
        .into_iter()
        .min_by_key(|(encoding, size)| (*size, encoding.decode_cost_rank()));
    let Some((encoding, estimated_bytes)) = best else {
        return EncodingPlan {
            encoding: Encoding::Plain,
            estimated_bytes: plain_bytes,
            plain_bytes,
            reason: PlanReason::Empty,
        };
    };

    let threshold = (plain_bytes as f64) * (1.0 - level.minimum_gain());
    if encoding != Encoding::Plain && (estimated_bytes as f64) > threshold {
        return EncodingPlan {
            encoding: Encoding::Plain,
            estimated_bytes: plain_bytes,
            plain_bytes,
            reason: PlanReason::GainBelowThreshold,
        };
    }
    EncodingPlan {
        encoding,
        estimated_bytes,
        plain_bytes,
        reason: PlanReason::SmallestEstimate,
    }
}

/// Analyse a chunk and choose an encoding for it.
///
/// ```
/// use basalt_common::CompressionLevel;
/// use basalt_encoding::{analyze, Encoding};
///
/// // A monotonic id column: delta encoding wins by a mile.
/// let ids: Vec<i64> = (0..10_000).map(|i| 1_000_000 + i * 2).collect();
/// let plan = analyze(&ids, CompressionLevel::BALANCED);
/// assert_eq!(plan.encoding, Encoding::DeltaBinaryPacked);
/// assert!(plan.estimated_ratio() > 50.0);
///
/// // High-entropy values: nothing beats plain, so plain it is.
/// let mut state = 0x2545_F491_4F6C_DD1Du64;
/// let noise: Vec<i64> = (0..10_000)
///     .map(|_| {
///         state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
///         state as i64
///     })
///     .collect();
/// assert_eq!(analyze(&noise, CompressionLevel::BALANCED).encoding, Encoding::Plain);
/// ```
pub fn analyze(values: &[i64], level: CompressionLevel) -> EncodingPlan {
    choose_integer_encoding(&analyze_integers(values), level)
}

// ---------------------------------------------------------------------------
// Variable-length columns
// ---------------------------------------------------------------------------

/// How a variable-length column should be laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteLayout {
    /// Per-row lengths plus one concatenated data buffer.
    Direct,
    /// A dictionary of distinct values plus integer codes.
    Dictionary,
}

/// The chosen layout for a variable-length column, and the numbers behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytePlan {
    /// The layout to use.
    pub layout: ByteLayout,
    /// Predicted size with that layout.
    pub estimated_bytes: usize,
    /// Predicted size with [`ByteLayout::Direct`], for comparison.
    pub direct_bytes: usize,
    /// Distinct value count (bounded).
    pub cardinality: Cardinality,
}

/// Decide between a direct and a dictionary layout for a string/binary column.
///
/// The comparison is made on the *uncompressed* sizes of the two layouts,
/// deliberately: the general-purpose byte codec runs on whichever data buffer
/// comes out of this decision, and it compresses both about equally well
/// (a dictionary's data buffer is the direct buffer with the duplicates
/// removed), so including it would add noise, not information. What the
/// dictionary genuinely saves is the *repetition*, and that is exactly what this
/// comparison measures.
pub fn analyze_byte_column(values: &[&[u8]]) -> BytePlan {
    let count = values.len();
    let total_bytes: usize = values.iter().map(|v| v.len()).sum();
    // Lengths are small non-negative integers; the direct layout stores them
    // through the integer pipeline, so estimate them the same way.
    let lengths: Vec<i64> = values.iter().map(|v| v.len() as i64).collect();
    let length_bytes = estimate_sizes(&analyze_integers(&lengths))
        .candidates()
        .into_iter()
        .map(|(_, size)| size)
        .min()
        .unwrap_or(0);
    let direct_bytes = length_bytes + total_bytes;

    let mut distinct: HashSet<&[u8]> = HashSet::new();
    let mut abandoned = false;
    for value in values {
        distinct.insert(value);
        if distinct.len() > MAX_DICTIONARY_CARDINALITY {
            abandoned = true;
            break;
        }
    }
    if abandoned {
        return BytePlan {
            layout: ByteLayout::Direct,
            estimated_bytes: direct_bytes,
            direct_bytes,
            cardinality: Cardinality::AtLeast(MAX_DICTIONARY_CARDINALITY + 1),
        };
    }

    let cardinality = Cardinality::Exact(distinct.len());
    let dictionary_data: usize = distinct.iter().map(|v| v.len()).sum();
    let dictionary_length_bytes = {
        let lengths: Vec<i64> = distinct.iter().map(|v| v.len() as i64).collect();
        estimate_sizes(&analyze_integers(&lengths))
            .candidates()
            .into_iter()
            .map(|(_, size)| size)
            .min()
            .unwrap_or(0)
    };
    // The codes are bounded by the cardinality, so the bit-packed size is exact
    // whatever permutation they happen to take.
    let code_bytes = header_len(count)
        + 1
        + packed_len(
            count,
            required_width(distinct.len().saturating_sub(1) as u64),
        );
    let dictionary_bytes = varint::uvarint_len(distinct.len() as u64)
        + dictionary_length_bytes
        + dictionary_data
        + code_bytes;

    if dictionary_bytes < direct_bytes {
        BytePlan {
            layout: ByteLayout::Dictionary,
            estimated_bytes: dictionary_bytes,
            direct_bytes,
            cardinality,
        }
    } else {
        BytePlan {
            layout: ByteLayout::Direct,
            estimated_bytes: direct_bytes,
            direct_bytes,
            cardinality,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::codec::Codec;
    use crate::integer::{encode_integers, PlainCodec};

    /// A 64-bit LCG. `i * constant` is *not* random enough for these tests:
    /// its consecutive deltas are constant, so delta encoding wins on it.
    fn pseudo_random_i64(n: usize) -> Vec<i64> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                state as i64
            })
            .collect()
    }

    #[test]
    fn statistics_are_correct() {
        let values = vec![5i64, 5, 5, 7, 7, 1];
        let stats = analyze_integers(&values);
        assert_eq!(stats.count, 6);
        assert_eq!(stats.min, 1);
        assert_eq!(stats.max, 7);
        assert_eq!(stats.run_count, 3);
        assert_eq!(stats.cardinality, Cardinality::Exact(3));
        assert_eq!(stats.distinct_values(), Some(&[1i64, 5, 7][..]));
        assert_eq!(stats.sortedness, Sortedness::Unsorted);
        assert_eq!(stats.value_span, 6);
        assert!((stats.run_density() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn sortedness_detection() {
        assert_eq!(analyze_integers(&[]).sortedness, Sortedness::Constant);
        assert_eq!(analyze_integers(&[3]).sortedness, Sortedness::Constant);
        assert_eq!(
            analyze_integers(&[3, 3, 3]).sortedness,
            Sortedness::Constant
        );
        assert_eq!(
            analyze_integers(&[1, 2, 2, 5]).sortedness,
            Sortedness::Ascending
        );
        assert_eq!(
            analyze_integers(&[5, 2, 2, 1]).sortedness,
            Sortedness::Descending
        );
        assert_eq!(
            analyze_integers(&[1, 5, 2]).sortedness,
            Sortedness::Unsorted
        );
    }

    #[test]
    fn cardinality_stops_counting_when_a_dictionary_is_hopeless() {
        let values: Vec<i64> = (0..(MAX_DICTIONARY_CARDINALITY as i64 + 100)).collect();
        let stats = analyze_integers(&values);
        assert!(!stats.cardinality.is_exact());
        assert_eq!(stats.cardinality.value(), MAX_DICTIONARY_CARDINALITY + 1);
        assert!(stats.distinct_values().is_none());
        assert!(estimate_sizes(&stats).dictionary.is_none());
    }

    /// The estimates are the whole basis of the decision, so they had better
    /// match what the encoders actually produce.
    #[test]
    fn estimates_match_reality() {
        let shapes: Vec<(&str, Vec<i64>)> = vec![
            ("runs", (0..500).map(|i| i / 50).collect()),
            ("small", (0..500).map(|i| (i % 9) - 4).collect()),
            (
                "clustered",
                (0..500).map(|i| 1_000_000_000 + (i % 37)).collect(),
            ),
            ("monotonic", (0..500).map(|i| 42 + i * 5).collect()),
            (
                "low cardinality",
                (0..500).map(|i| [3i64, -100, 70_000][i % 3]).collect(),
            ),
            ("noise", pseudo_random_i64(500)),
        ];
        for (name, values) in shapes {
            let stats = analyze_integers(&values);
            let estimates = estimate_sizes(&stats);
            for (encoding, estimated) in estimates.candidates() {
                let mut buf = Vec::new();
                encode_integers(&values, encoding, &mut buf).unwrap();
                let actual = buf.len();
                match encoding {
                    // Delta's estimate is deliberately an upper bound.
                    Encoding::DeltaBinaryPacked => assert!(
                        actual <= estimated,
                        "{name}/{encoding}: actual {actual} exceeded the conservative estimate {estimated}"
                    ),
                    _ => assert_eq!(
                        actual, estimated,
                        "{name}/{encoding}: estimate {estimated} != actual {actual}"
                    ),
                }
            }
        }
    }

    #[test]
    fn selection_picks_the_right_encoding_for_each_shape() {
        let cases: Vec<(&str, Vec<i64>, Encoding)> = vec![
            (
                "long runs",
                (0..2000).map(|i| i / 200).collect(),
                Encoding::RunLength,
            ),
            (
                "monotonic ids",
                (0..2000).map(|i| 5_000_000 + i * 3).collect(),
                Encoding::DeltaBinaryPacked,
            ),
            (
                "clustered timestamps",
                (0..2000)
                    .map(|i| 1_700_000_000_000_000 + (i * 7919 % 100_000))
                    .collect(),
                Encoding::FrameOfReference,
            ),
            (
                "tiny signed values",
                (0..2000).map(|i| (i % 11) - 5).collect(),
                Encoding::BitPacked,
            ),
            ("random 64-bit", pseudo_random_i64(2000), Encoding::Plain),
        ];
        for (name, values, expected) in cases {
            let plan = analyze(&values, CompressionLevel::BALANCED);
            assert_eq!(plan.encoding, expected, "{name} -> {plan:?}");
            // And the choice really is (near-)optimal among the candidates.
            let estimates = estimate_sizes(&analyze_integers(&values));
            let best = estimates
                .candidates()
                .into_iter()
                .map(|(_, s)| s)
                .min()
                .unwrap_or(0);
            if plan.reason == PlanReason::SmallestEstimate {
                assert_eq!(plan.estimated_bytes, best, "{name}");
            }
        }
    }

    #[test]
    fn dictionary_wins_on_wide_low_cardinality_data() {
        // Values too far apart for FOR or bit packing, but only four of them.
        let values: Vec<i64> = (0..4000)
            .map(|i| [i64::MIN / 3, 0, i64::MAX / 7, -99_999_999_999][i % 4])
            .collect();
        let plan = analyze(&values, CompressionLevel::BALANCED);
        assert_eq!(plan.encoding, Encoding::Dictionary, "{plan:?}");
        assert!(plan.estimated_ratio() > 3.0, "{plan:?}");
    }

    #[test]
    fn varint_wins_when_most_values_are_small_but_the_range_is_wide() {
        // Cardinality above the dictionary cap, magnitudes small, one huge
        // outlier that ruins any fixed bit width.
        let mut values: Vec<i64> = (0..20_000).map(|i| (i * 7919) % 20_000).collect();
        values[0] = i64::MAX;
        let plan = analyze(&values, CompressionLevel::BALANCED);
        assert_eq!(plan.encoding, Encoding::Varint, "{plan:?}");
    }

    #[test]
    fn compression_level_gates_the_choice() {
        // A modest 30% win: taken at BALANCED, refused at FAST (20% threshold
        // is met) -- pick a shape with a small win to show the threshold bite.
        let values: Vec<i64> = (0..1000).map(|i| (i % 1_000_000) * 4_000_000_000).collect();
        let none = analyze(&values, CompressionLevel::NONE);
        assert_eq!(none.encoding, Encoding::Plain);
        assert_eq!(none.reason, PlanReason::CompressionDisabled);

        let stats = analyze_integers(&values);
        let estimates = estimate_sizes(&stats);
        let best = estimates
            .candidates()
            .into_iter()
            .map(|(_, s)| s)
            .min()
            .unwrap_or(0);
        let gain = 1.0 - best as f64 / estimates.plain as f64;
        // Whatever the shape turns out to be, the threshold rule must hold.
        for level in [
            CompressionLevel::FAST,
            CompressionLevel::BALANCED,
            CompressionLevel::MAX,
        ] {
            let plan = choose_integer_encoding(&stats, level);
            if gain >= level.minimum_gain() {
                assert_ne!(
                    plan.reason,
                    PlanReason::GainBelowThreshold,
                    "level {level:?}"
                );
            } else {
                assert_eq!(plan.encoding, Encoding::Plain, "level {level:?}");
                assert_eq!(plan.reason, PlanReason::GainBelowThreshold);
            }
        }
    }

    #[test]
    fn empty_and_single_value_chunks() {
        let plan = analyze(&[], CompressionLevel::MAX);
        assert_eq!(plan.encoding, Encoding::Plain);
        assert_eq!(plan.reason, PlanReason::Empty);
        assert!((plan.estimated_ratio() - 1.0).abs() < 1.0);

        let stats = analyze_integers(&[7]);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.delta_span, 0);
        assert!(estimate_sizes(&stats).delta.is_none());
        let buf = PlainCodec.encode_to_vec(&[7i64]).unwrap();
        assert_eq!(buf.len(), estimate_sizes(&stats).plain);
    }

    #[test]
    fn ties_are_broken_towards_cheaper_decoding() {
        // All-zero data: several encodings collapse to nearly nothing, and the
        // cheapest to decode among the smallest must win.
        let values = vec![0i64; 1024];
        let plan = analyze(&values, CompressionLevel::MAX);
        let estimates = estimate_sizes(&analyze_integers(&values));
        let smallest = estimates
            .candidates()
            .into_iter()
            .map(|(_, s)| s)
            .min()
            .unwrap_or(0);
        assert_eq!(plan.estimated_bytes, smallest);
        let tied_ranks: Vec<u8> = estimates
            .candidates()
            .into_iter()
            .filter(|(_, s)| *s == smallest)
            .map(|(e, _)| e.decode_cost_rank())
            .collect();
        assert_eq!(
            plan.encoding.decode_cost_rank(),
            tied_ranks.into_iter().min().unwrap_or(0)
        );
    }

    #[test]
    fn byte_column_layout_selection() {
        let repeated: Vec<&[u8]> = (0..1000)
            .map(|i| [&b"amsterdam"[..], &b"reykjavik"[..], &b"ushuaia"[..]][i % 3])
            .collect();
        let plan = analyze_byte_column(&repeated);
        assert_eq!(plan.layout, ByteLayout::Dictionary, "{plan:?}");
        assert!(plan.estimated_bytes < plan.direct_bytes);
        assert_eq!(plan.cardinality, Cardinality::Exact(3));

        let owned: Vec<Vec<u8>> = (0..1000)
            .map(|i| format!("row-{i:09}").into_bytes())
            .collect();
        let unique: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let plan = analyze_byte_column(&unique);
        assert_eq!(plan.layout, ByteLayout::Direct, "{plan:?}");

        let empty: Vec<&[u8]> = Vec::new();
        assert_eq!(analyze_byte_column(&empty).layout, ByteLayout::Direct);
    }

    #[test]
    fn byte_column_gives_up_on_huge_cardinality() {
        let owned: Vec<Vec<u8>> = (0..MAX_DICTIONARY_CARDINALITY + 10)
            .map(|i| format!("{i}").into_bytes())
            .collect();
        let values: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let plan = analyze_byte_column(&values);
        assert_eq!(plan.layout, ByteLayout::Direct);
        assert!(!plan.cardinality.is_exact());
    }
}

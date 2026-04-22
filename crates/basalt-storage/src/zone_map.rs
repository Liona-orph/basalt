//! Per-chunk summary statistics used to skip row groups.

use std::collections::HashSet;
use std::fmt;

use basalt_types::{ColumnVector, DataType, Value};

use crate::bloom::value_hash;

/// Number of bits in the linear-counting sketch used for distinct estimation.
///
/// 8192 bits (1 KiB, transient) keeps the relative error below ~2 % up to a few
/// thousand distinct values, which is the range where the estimate changes a
/// decision (whether to build a bloom filter, and how big).
const SKETCH_BITS: usize = 8192;

/// Summary of one column chunk.
///
/// This is what makes predicate pushdown possible: it lives in the footer, so
/// the reader can decide whether a row group can possibly satisfy a predicate
/// **without reading a single data byte**.
///
/// # Nulls
///
/// `min` and `max` ignore nulls entirely — they are the extremes of the *present*
/// values — and `null_count` carries the rest of the story. A chunk that is
/// entirely null has `min == max == None` and `null_count == row_count`, which
/// is exactly what `IS NULL` / `IS NOT NULL` pruning needs. Folding nulls into
/// the bounds (the mistake of treating `NULL` as negative infinity) would make
/// `col > 5` fail to skip an all-null chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneMap {
    /// Smallest non-null value, or `None` if every value is null.
    pub min: Option<Value>,
    /// Largest non-null value, or `None` if every value is null.
    pub max: Option<Value>,
    /// Number of nulls in the chunk.
    pub null_count: u64,
    /// Estimated number of distinct non-null values (see
    /// [`ZoneMap::from_column`] for the estimator and its error).
    pub distinct_estimate: u64,
    /// Number of rows the chunk covers.
    pub row_count: u64,
}

impl ZoneMap {
    /// Compute a zone map in one pass over the column.
    ///
    /// O(n) with a single materialised [`Value`] per comparison; for
    /// variable-length columns the bounds are tracked by *index* and
    /// materialised once at the end, so a string column does not allocate per
    /// row.
    ///
    /// The distinct estimate uses **linear counting**: hash each value into an
    /// 8192-bit sketch, then `n̂ = -m ln(z/m)` where `z` is the number of bits
    /// still clear. Linear counting is chosen over HyperLogLog because the
    /// interesting range here is small (does this column have 3 distinct values
    /// or 30 000?), and linear counting is both simpler and more accurate below
    /// a few thousand; it degrades gracefully by saturating, and the result is
    /// clamped to the row count.
    pub fn from_column(column: &ColumnVector) -> ZoneMap {
        let row_count = column.len() as u64;
        let null_count = column.null_count() as u64;
        if row_count == null_count {
            return ZoneMap {
                min: None,
                max: None,
                null_count,
                distinct_estimate: 0,
                row_count,
            };
        }

        let mut sketch = vec![0u64; SKETCH_BITS / 64];
        let mut min_index = usize::MAX;
        let mut max_index = usize::MAX;

        // For fixed-width types the `Value` is Copy-ish and cheap; for
        // variable-length types we compare the borrowed bytes and only
        // materialise at the end.
        match column {
            ColumnVector::Utf8(array) => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        continue;
                    }
                    let value = array.value(i);
                    if min_index == usize::MAX || value < array.value(min_index) {
                        min_index = i;
                    }
                    if max_index == usize::MAX || value > array.value(max_index) {
                        max_index = i;
                    }
                    set_sketch_bit(&mut sketch, hash_bytes(value.as_bytes(), true));
                }
            }
            ColumnVector::Binary(array) => {
                for i in 0..array.len() {
                    if array.is_null(i) {
                        continue;
                    }
                    let value = array.value(i);
                    if min_index == usize::MAX || value < array.value(min_index) {
                        min_index = i;
                    }
                    if max_index == usize::MAX || value > array.value(max_index) {
                        max_index = i;
                    }
                    set_sketch_bit(&mut sketch, hash_bytes(value, false));
                }
            }
            other => {
                let mut min_value: Option<Value> = None;
                let mut max_value: Option<Value> = None;
                for i in 0..other.len() {
                    if other.is_null(i) {
                        continue;
                    }
                    let value = other.value(i);
                    set_sketch_bit(&mut sketch, value_hash(&value));
                    match &min_value {
                        Some(current) if *current <= value => {}
                        _ => min_value = Some(value.clone()),
                    }
                    match &max_value {
                        Some(current) if *current >= value => {}
                        _ => max_value = Some(value),
                    }
                }
                return ZoneMap {
                    min: min_value,
                    max: max_value,
                    null_count,
                    distinct_estimate: linear_count(&sketch, row_count - null_count),
                    row_count,
                };
            }
        }

        ZoneMap {
            min: (min_index != usize::MAX).then(|| column.value(min_index)),
            max: (max_index != usize::MAX).then(|| column.value(max_index)),
            null_count,
            distinct_estimate: linear_count(&sketch, row_count - null_count),
            row_count,
        }
    }

    /// `true` if every value in the chunk is null.
    pub fn is_all_null(&self) -> bool {
        self.null_count == self.row_count
    }

    /// `true` if the chunk contains no nulls.
    pub fn has_no_nulls(&self) -> bool {
        self.null_count == 0
    }

    /// `true` if `value` falls inside `[min, max]`.
    ///
    /// A chunk whose bounds exclude the value cannot contain it, which is the
    /// whole basis of zone-map pruning. Returns `false` for an all-null chunk.
    pub fn may_contain(&self, value: &Value) -> bool {
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => value >= min && value <= max,
            _ => false,
        }
    }

    /// `true` if a bloom filter is worth building for this chunk.
    ///
    /// Two values or fewer and the zone map already answers every equality
    /// question exactly; float and null columns are excluded because grouping
    /// or point-looking-up a float is almost always a mistake and the filter
    /// would just be dead weight.
    pub fn wants_bloom_filter(&self, data_type: &DataType) -> bool {
        data_type.is_hash_groupable() && self.distinct_estimate >= 4
    }
}

impl fmt::Display for ZoneMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.min, &self.max) {
            (Some(min), Some(max)) => write!(f, "[{min}, {max}]")?,
            _ => f.write_str("[all null]")?,
        }
        write!(
            f,
            " nulls={}/{} ndv~{}",
            self.null_count, self.row_count, self.distinct_estimate
        )
    }
}

fn hash_bytes(bytes: &[u8], is_utf8: bool) -> u64 {
    // Must agree with `value_hash` so the sketch and the bloom filter count the
    // same things; both go through the canonical value encoding.
    if is_utf8 {
        match std::str::from_utf8(bytes) {
            Ok(text) => value_hash(&Value::Utf8(text.to_string())),
            Err(_) => value_hash(&Value::Binary(bytes.to_vec())),
        }
    } else {
        value_hash(&Value::Binary(bytes.to_vec()))
    }
}

fn set_sketch_bit(sketch: &mut [u64], hash: u64) {
    let bit = (hash % SKETCH_BITS as u64) as usize;
    sketch[bit / 64] |= 1u64 << (bit % 64);
}

/// Linear counting: `n̂ = -m ln(z / m)`, clamped to `[1, upper_bound]`.
fn linear_count(sketch: &[u64], upper_bound: u64) -> u64 {
    let set: u32 = sketch.iter().map(|word| word.count_ones()).sum();
    let m = SKETCH_BITS as f64;
    let zeros = m - f64::from(set);
    if zeros <= 0.0 {
        // Saturated: every bit is set, so all we can say is "a lot".
        return upper_bound;
    }
    let estimate = -m * (zeros / m).ln();
    (estimate.round().max(1.0) as u64).min(upper_bound)
}

/// Exact distinct count, for tests and for callers that can afford it.
///
/// Not used by the writer: it needs a `HashSet` of materialised values, which is
/// exactly the allocation the sketch exists to avoid.
pub fn exact_distinct_count(column: &ColumnVector) -> u64 {
    let mut seen = HashSet::new();
    for i in 0..column.len() {
        if !column.is_null(i) {
            seen.insert(column.value(i));
        }
    }
    seen.len() as u64
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use basalt_types::{NullArray, PrimitiveArray, StringArray};

    #[test]
    fn bounds_ignore_nulls() {
        let column = ColumnVector::Int32(PrimitiveArray::from_opt_vec(vec![
            Some(5),
            None,
            Some(-3),
            None,
            Some(9),
        ]));
        let zone = ZoneMap::from_column(&column);
        assert_eq!(zone.min, Some(Value::Int32(-3)));
        assert_eq!(zone.max, Some(Value::Int32(9)));
        assert_eq!(zone.null_count, 2);
        assert_eq!(zone.row_count, 5);
        assert_eq!(zone.distinct_estimate, 3);
        assert!(!zone.is_all_null());
        assert!(!zone.has_no_nulls());
        assert!(zone.may_contain(&Value::Int32(0)));
        assert!(!zone.may_contain(&Value::Int32(10)));
        assert!(zone.to_string().contains("nulls=2/5"));
    }

    #[test]
    fn all_null_chunks() {
        let column = ColumnVector::Int64(PrimitiveArray::from_opt_vec(vec![None, None, None]));
        let zone = ZoneMap::from_column(&column);
        assert_eq!(zone.min, None);
        assert_eq!(zone.max, None);
        assert!(zone.is_all_null());
        assert_eq!(zone.distinct_estimate, 0);
        assert!(!zone.may_contain(&Value::Int64(0)));
        assert!(!zone.wants_bloom_filter(&DataType::Int64));
        assert_eq!(zone.to_string(), "[all null] nulls=3/3 ndv~0");

        let null_column = ColumnVector::Null(NullArray::new(4));
        let zone = ZoneMap::from_column(&null_column);
        assert!(zone.is_all_null());
        assert_eq!(zone.row_count, 4);
    }

    #[test]
    fn string_bounds_are_lexicographic() {
        let column = ColumnVector::Utf8(StringArray::from_strs(&["pear", "apple", "quince"]));
        let zone = ZoneMap::from_column(&column);
        assert_eq!(zone.min, Some(Value::Utf8("apple".into())));
        assert_eq!(zone.max, Some(Value::Utf8("quince".into())));
        assert_eq!(zone.distinct_estimate, 3);
    }

    #[test]
    fn distinct_estimate_is_accurate_at_useful_scales() {
        for distinct in [1usize, 2, 10, 100, 1000, 4000] {
            let values: Vec<i64> = (0..8000).map(|i| (i % distinct) as i64).collect();
            let column = ColumnVector::Int64(PrimitiveArray::from_vec(values));
            let zone = ZoneMap::from_column(&column);
            let exact = exact_distinct_count(&column);
            assert_eq!(exact, distinct as u64);
            let error = (zone.distinct_estimate as f64 - exact as f64).abs() / exact as f64;
            assert!(
                error < 0.10,
                "distinct={distinct} estimate={} error={error:.3}",
                zone.distinct_estimate
            );
        }
    }

    #[test]
    fn distinct_estimate_saturates_gracefully() {
        let values: Vec<i64> = (0..200_000).collect();
        let column = ColumnVector::Int64(PrimitiveArray::from_vec(values));
        let zone = ZoneMap::from_column(&column);
        // The sketch is full; the estimate must not exceed the row count.
        assert!(zone.distinct_estimate <= zone.row_count);
        assert!(zone.distinct_estimate > 8_000, "{}", zone.distinct_estimate);
    }

    #[test]
    fn bloom_filter_policy() {
        let low = ZoneMap {
            min: Some(Value::Boolean(false)),
            max: Some(Value::Boolean(true)),
            null_count: 0,
            distinct_estimate: 2,
            row_count: 1000,
        };
        assert!(
            !low.wants_bloom_filter(&DataType::Boolean),
            "zone map already decides"
        );
        let high = ZoneMap {
            distinct_estimate: 900,
            ..low.clone()
        };
        assert!(high.wants_bloom_filter(&DataType::Int64));
        assert!(
            !high.wants_bloom_filter(&DataType::Float64),
            "floats are excluded"
        );
        assert!(!high.wants_bloom_filter(&DataType::Null));
    }

    #[test]
    fn float_bounds_use_total_order() {
        let column = ColumnVector::Float64(PrimitiveArray::from_vec(vec![
            1.0,
            f64::NAN,
            -1.0,
            f64::INFINITY,
        ]));
        let zone = ZoneMap::from_column(&column);
        assert_eq!(zone.min, Some(Value::Float64(-1.0)));
        // NaN is the largest value under IEEE-754 total order.
        assert!(matches!(zone.max, Some(Value::Float64(v)) if v.is_nan()));
    }
}

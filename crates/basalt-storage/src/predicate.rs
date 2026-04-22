//! Predicates and row-group pruning.

use std::fmt;

use basalt_common::{BasaltError, Result};
use basalt_types::{DataType, Schema, Value};

use crate::metadata::RowGroupMetadata;
use crate::zone_map::ZoneMap;

/// Result of testing a predicate against a row group's metadata.
///
/// Deliberately two-valued rather than three: metadata can prove that a row
/// group contains **no** matching row, but it can never prove that every row
/// matches (the bounds are a summary, not the data). So the answer is either
/// "skip it" or "you will have to look".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// No row in this group can satisfy the predicate. Skip it entirely.
    Skip,
    /// Some row might match. The group must be read.
    Maybe,
}

impl Verdict {
    /// `true` if the row group can be skipped.
    pub fn is_skip(self) -> bool {
        self == Verdict::Skip
    }

    fn and(self, other: Verdict) -> Verdict {
        // A conjunction is false if either side is false.
        if self == Verdict::Skip || other == Verdict::Skip {
            Verdict::Skip
        } else {
            Verdict::Maybe
        }
    }

    fn or(self, other: Verdict) -> Verdict {
        // A disjunction is false only if both sides are.
        if self == Verdict::Skip && other == Verdict::Skip {
            Verdict::Skip
        } else {
            Verdict::Maybe
        }
    }
}

/// A comparison a scan can push down to the storage layer.
///
/// Column references are *positions* in the segment's schema, not names,
/// because by the time a predicate reaches storage the planner has already
/// resolved them; carrying names would mean a hash lookup per row group.
///
/// # What pushdown does and does not do here
///
/// A predicate is used to **skip row groups**, not to filter rows. Whatever
/// survives pruning is returned in full, and the executor applies the residual
/// filter with a [`Selection`](basalt_types::Selection). That split is the right
/// one: storage owns the coarse decision (is this megabyte worth reading?) and
/// the executor owns the fine one (which of these 65 536 rows match?).
///
/// ```
/// use basalt_storage::Predicate;
/// use basalt_types::Value;
///
/// // WHERE id = 42 AND city > 'lisbon'
/// let predicate = Predicate::eq(0, Value::Int64(42))
///     .and(Predicate::gt(1, Value::Utf8("lisbon".into())));
/// assert_eq!(predicate.column_indices(), vec![0, 1]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    /// Always true; prunes nothing.
    AlwaysTrue,
    /// Always false; prunes everything.
    AlwaysFalse,
    /// `column = value`.
    Eq(usize, Value),
    /// `column <> value`.
    NotEq(usize, Value),
    /// `column < value`.
    Lt(usize, Value),
    /// `column <= value`.
    LtEq(usize, Value),
    /// `column > value`.
    Gt(usize, Value),
    /// `column >= value`.
    GtEq(usize, Value),
    /// `column BETWEEN low AND high`, inclusive.
    Between(usize, Value, Value),
    /// `column IS NULL`.
    IsNull(usize),
    /// `column IS NOT NULL`.
    IsNotNull(usize),
    /// Conjunction.
    And(Box<Predicate>, Box<Predicate>),
    /// Disjunction.
    Or(Box<Predicate>, Box<Predicate>),
    /// Negation.
    Not(Box<Predicate>),
}

impl Predicate {
    /// `column = value`.
    pub fn eq(column: usize, value: Value) -> Predicate {
        Predicate::Eq(column, value)
    }

    /// `column < value`.
    pub fn lt(column: usize, value: Value) -> Predicate {
        Predicate::Lt(column, value)
    }

    /// `column > value`.
    pub fn gt(column: usize, value: Value) -> Predicate {
        Predicate::Gt(column, value)
    }

    /// `column BETWEEN low AND high`.
    pub fn between(column: usize, low: Value, high: Value) -> Predicate {
        Predicate::Between(column, low, high)
    }

    /// Conjunction (builder style).
    #[must_use]
    pub fn and(self, other: Predicate) -> Predicate {
        Predicate::And(Box::new(self), Box::new(other))
    }

    /// Disjunction (builder style).
    #[must_use]
    pub fn or(self, other: Predicate) -> Predicate {
        Predicate::Or(Box::new(self), Box::new(other))
    }

    /// Negation (builder style).
    #[must_use]
    pub fn negate(self) -> Predicate {
        Predicate::Not(Box::new(self))
    }

    /// Every column position the predicate mentions, sorted and deduplicated.
    pub fn column_indices(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.collect_columns(&mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_columns(&self, out: &mut Vec<usize>) {
        match self {
            Predicate::AlwaysTrue | Predicate::AlwaysFalse => {}
            Predicate::Eq(c, _)
            | Predicate::NotEq(c, _)
            | Predicate::Lt(c, _)
            | Predicate::LtEq(c, _)
            | Predicate::Gt(c, _)
            | Predicate::GtEq(c, _)
            | Predicate::Between(c, _, _)
            | Predicate::IsNull(c)
            | Predicate::IsNotNull(c) => out.push(*c),
            Predicate::And(a, b) | Predicate::Or(a, b) => {
                a.collect_columns(out);
                b.collect_columns(out);
            }
            Predicate::Not(inner) => inner.collect_columns(out),
        }
    }

    /// Check that every column position exists in `schema`.
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        for index in self.column_indices() {
            if index >= schema.len() {
                return Err(BasaltError::plan(format!(
                    "predicate references column {index} but the segment has {} columns",
                    schema.len()
                )));
            }
        }
        Ok(())
    }

    /// Equality tests that sit in a purely conjunctive position.
    ///
    /// Only these may be answered by a bloom filter. Under an `OR` or a `NOT`,
    /// "the filter says this value is absent" does **not** make the whole
    /// predicate false, so consulting the filter there would skip row groups
    /// that contain matching rows. Walking only the `AND` spine is what makes
    /// bloom pruning sound.
    pub fn conjunctive_equalities(&self) -> Vec<(usize, &Value)> {
        let mut out = Vec::new();
        self.collect_equalities(&mut out);
        out
    }

    fn collect_equalities<'a>(&'a self, out: &mut Vec<(usize, &'a Value)>) {
        match self {
            Predicate::Eq(column, value) => out.push((*column, value)),
            Predicate::And(a, b) => {
                a.collect_equalities(out);
                b.collect_equalities(out);
            }
            _ => {}
        }
    }

    /// Test the predicate against a row group's zone maps.
    ///
    /// Reads nothing: zone maps live in the footer, which is already in memory.
    ///
    /// # Errors
    ///
    /// If a column position is out of range for the schema.
    pub fn evaluate_zone_maps(&self, group: &RowGroupMetadata, schema: &Schema) -> Result<Verdict> {
        let zone = |index: usize| -> Result<(&ZoneMap, &DataType)> {
            let chunk = group.columns.get(index).ok_or_else(|| {
                BasaltError::plan(format!(
                    "predicate references column {index} but the row group has {} chunks",
                    group.columns.len()
                ))
            })?;
            Ok((&chunk.zone_map, schema.field(index).data_type()))
        };
        Ok(match self {
            Predicate::AlwaysTrue => Verdict::Maybe,
            Predicate::AlwaysFalse => Verdict::Skip,
            Predicate::IsNull(column) => {
                let (zone, _) = zone(*column)?;
                if zone.has_no_nulls() {
                    Verdict::Skip
                } else {
                    Verdict::Maybe
                }
            }
            Predicate::IsNotNull(column) => {
                let (zone, _) = zone(*column)?;
                if zone.is_all_null() {
                    Verdict::Skip
                } else {
                    Verdict::Maybe
                }
            }
            Predicate::And(a, b) => a
                .evaluate_zone_maps(group, schema)?
                .and(b.evaluate_zone_maps(group, schema)?),
            Predicate::Or(a, b) => a
                .evaluate_zone_maps(group, schema)?
                .or(b.evaluate_zone_maps(group, schema)?),
            // Negation cannot be pruned from bounds: knowing that every value
            // lies in [10, 20] tells you nothing about `NOT (col = 15)`.
            // Returning `Maybe` is always sound, just not helpful.
            Predicate::Not(_) => Verdict::Maybe,
            Predicate::Eq(column, value) => {
                let (zone, data_type) = zone(*column)?;
                match exact_cast(value, data_type) {
                    // No value of the column's type equals the probe, so the
                    // comparison is false for every row.
                    None => Verdict::Skip,
                    Some(probe) => {
                        if zone.may_contain(&probe) {
                            Verdict::Maybe
                        } else {
                            Verdict::Skip
                        }
                    }
                }
            }
            Predicate::NotEq(column, value) => {
                let (zone, data_type) = zone(*column)?;
                match (exact_cast(value, data_type), &zone.min, &zone.max) {
                    (Some(probe), Some(min), Some(max)) if *min == probe && *max == probe => {
                        // Constant chunk equal to the probe: every row fails,
                        // and nulls are not `true` either.
                        Verdict::Skip
                    }
                    _ if zone.is_all_null() => Verdict::Skip,
                    _ => Verdict::Maybe,
                }
            }
            Predicate::Lt(column, value) => {
                let (zone, data_type) = zone(*column)?;
                ordered(zone, exact_cast(value, data_type), |min, _, probe| {
                    min >= probe
                })
            }
            Predicate::LtEq(column, value) => {
                let (zone, data_type) = zone(*column)?;
                ordered(zone, exact_cast(value, data_type), |min, _, probe| {
                    min > probe
                })
            }
            Predicate::Gt(column, value) => {
                let (zone, data_type) = zone(*column)?;
                ordered(zone, exact_cast(value, data_type), |_, max, probe| {
                    max <= probe
                })
            }
            Predicate::GtEq(column, value) => {
                let (zone, data_type) = zone(*column)?;
                ordered(zone, exact_cast(value, data_type), |_, max, probe| {
                    max < probe
                })
            }
            Predicate::Between(column, low, high) => {
                let (zone, data_type) = zone(*column)?;
                match (&zone.min, &zone.max) {
                    (Some(min), Some(max)) => {
                        let low = exact_cast(low, data_type);
                        let high = exact_cast(high, data_type);
                        let above = high.is_some_and(|high| *min > high);
                        let below = low.is_some_and(|low| *max < low);
                        if above || below {
                            Verdict::Skip
                        } else {
                            Verdict::Maybe
                        }
                    }
                    _ => Verdict::Skip,
                }
            }
        })
    }
}

/// Apply an ordering test to a zone map. All-null chunks always skip: every
/// comparison against `NULL` is `NULL`, which is not `true`.
fn ordered(
    zone: &ZoneMap,
    probe: Option<Value>,
    can_skip: impl Fn(&Value, &Value, &Value) -> bool,
) -> Verdict {
    match (&zone.min, &zone.max, probe) {
        (Some(min), Some(max), Some(probe)) => {
            if can_skip(min, max, &probe) {
                Verdict::Skip
            } else {
                Verdict::Maybe
            }
        }
        // All-null chunk.
        (None, _, _) | (_, None, _) => Verdict::Skip,
        // The probe does not convert exactly, so be conservative.
        _ => Verdict::Maybe,
    }
}

/// Cast `value` to `data_type` **only if the conversion is exact**.
///
/// `WHERE small_int_col = 100000` must prune everything, and
/// `WHERE int_col = 1.5` must too: no `i32` equals `1.5`. A plain cast would
/// truncate to `1` and wrongly keep row groups (or, worse, miss them). The
/// round-trip check makes "does any value of this column's type equal the
/// probe?" precise.
fn exact_cast(value: &Value, data_type: &DataType) -> Option<Value> {
    if value.data_type() == *data_type {
        return Some(value.clone());
    }
    let cast = value.cast_to(data_type).ok()?;
    let back = cast.cast_to(&value.data_type()).ok()?;
    (back == *value).then_some(cast)
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::AlwaysTrue => f.write_str("TRUE"),
            Predicate::AlwaysFalse => f.write_str("FALSE"),
            Predicate::Eq(c, v) => write!(f, "#{c} = {v}"),
            Predicate::NotEq(c, v) => write!(f, "#{c} <> {v}"),
            Predicate::Lt(c, v) => write!(f, "#{c} < {v}"),
            Predicate::LtEq(c, v) => write!(f, "#{c} <= {v}"),
            Predicate::Gt(c, v) => write!(f, "#{c} > {v}"),
            Predicate::GtEq(c, v) => write!(f, "#{c} >= {v}"),
            Predicate::Between(c, lo, hi) => write!(f, "#{c} BETWEEN {lo} AND {hi}"),
            Predicate::IsNull(c) => write!(f, "#{c} IS NULL"),
            Predicate::IsNotNull(c) => write!(f, "#{c} IS NOT NULL"),
            Predicate::And(a, b) => write!(f, "({a} AND {b})"),
            Predicate::Or(a, b) => write!(f, "({a} OR {b})"),
            Predicate::Not(p) => write!(f, "NOT {p}"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use basalt_encoding::Encoding;
    use basalt_types::Field;

    use crate::metadata::{BlockLocation, ColumnChunkMetadata};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("small", DataType::Int8, true),
        ])
        .unwrap()
    }

    fn group(zones: Vec<ZoneMap>) -> RowGroupMetadata {
        RowGroupMetadata {
            num_rows: zones.first().map_or(0, |z| z.row_count),
            columns: zones
                .into_iter()
                .map(|zone_map| ColumnChunkMetadata {
                    data: BlockLocation {
                        offset: 0,
                        length: 0,
                    },
                    encoding: Encoding::Plain,
                    uncompressed_bytes: 0,
                    zone_map,
                    bloom: None,
                })
                .collect(),
        }
    }

    fn zone(min: i64, max: i64, nulls: u64, rows: u64) -> ZoneMap {
        ZoneMap {
            min: Some(Value::Int64(min)),
            max: Some(Value::Int64(max)),
            null_count: nulls,
            distinct_estimate: 10,
            row_count: rows,
        }
    }

    fn small_zone(min: i8, max: i8) -> ZoneMap {
        ZoneMap {
            min: Some(Value::Int8(min)),
            max: Some(Value::Int8(max)),
            null_count: 0,
            distinct_estimate: 5,
            row_count: 100,
        }
    }

    fn verdict(p: &Predicate, zones: Vec<ZoneMap>) -> Verdict {
        p.evaluate_zone_maps(&group(zones), &schema()).unwrap()
    }

    #[test]
    fn range_pruning() {
        let zones = || vec![zone(10, 20, 0, 100), small_zone(0, 5)];
        assert_eq!(
            verdict(&Predicate::eq(0, Value::Int64(15)), zones()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&Predicate::eq(0, Value::Int64(21)), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::lt(0, Value::Int64(10)), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::lt(0, Value::Int64(11)), zones()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&Predicate::gt(0, Value::Int64(20)), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::gt(0, Value::Int64(19)), zones()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&Predicate::LtEq(0, Value::Int64(9)), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::GtEq(0, Value::Int64(21)), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(
                &Predicate::between(0, Value::Int64(0), Value::Int64(9)),
                zones()
            ),
            Verdict::Skip
        );
        assert_eq!(
            verdict(
                &Predicate::between(0, Value::Int64(15), Value::Int64(99)),
                zones()
            ),
            Verdict::Maybe
        );
    }

    #[test]
    fn null_pruning() {
        let no_nulls = vec![zone(1, 2, 0, 100), small_zone(0, 1)];
        let all_null = vec![
            ZoneMap {
                min: None,
                max: None,
                null_count: 100,
                distinct_estimate: 0,
                row_count: 100,
            },
            small_zone(0, 1),
        ];
        assert_eq!(
            verdict(&Predicate::IsNull(0), no_nulls.clone()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::IsNotNull(0), no_nulls.clone()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&Predicate::IsNull(0), all_null.clone()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&Predicate::IsNotNull(0), all_null.clone()),
            Verdict::Skip
        );
        // Any comparison against an all-null chunk is NULL, never true.
        assert_eq!(
            verdict(&Predicate::eq(0, Value::Int64(1)), all_null.clone()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::lt(0, Value::Int64(9999)), all_null.clone()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(
                &Predicate::between(0, Value::Int64(0), Value::Int64(9)),
                all_null
            ),
            Verdict::Skip
        );
    }

    #[test]
    fn out_of_range_probes_prune_or_stay_safe() {
        let zones = || vec![zone(10, 20, 0, 100), small_zone(0, 5)];
        // 1000 does not fit an Int8, so nothing in that column can equal it.
        assert_eq!(
            verdict(&Predicate::eq(1, Value::Int64(1000)), zones()),
            Verdict::Skip
        );
        // But `small < 1000` is true for every row: must not skip.
        assert_eq!(
            verdict(&Predicate::lt(1, Value::Int64(1000)), zones()),
            Verdict::Maybe
        );
        // A non-integral probe cannot equal any integer.
        assert_eq!(
            verdict(&Predicate::eq(0, Value::Float64(1.5)), zones()),
            Verdict::Skip
        );
    }

    #[test]
    fn boolean_combinations() {
        let zones = || vec![zone(10, 20, 0, 100), small_zone(0, 5)];
        let hit = Predicate::eq(0, Value::Int64(15));
        let miss = Predicate::eq(0, Value::Int64(99));
        assert_eq!(
            verdict(&hit.clone().and(miss.clone()), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&miss.clone().and(miss.clone()), zones()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&hit.clone().or(miss.clone()), zones()),
            Verdict::Maybe
        );
        assert_eq!(
            verdict(&miss.clone().or(miss.clone()), zones()),
            Verdict::Skip
        );
        assert_eq!(verdict(&miss.clone().negate(), zones()), Verdict::Maybe);
        assert_eq!(verdict(&Predicate::AlwaysFalse, zones()), Verdict::Skip);
        assert_eq!(verdict(&Predicate::AlwaysTrue, zones()), Verdict::Maybe);
        assert!(Verdict::Skip.is_skip());
    }

    #[test]
    fn not_equal_pruning() {
        let constant = vec![zone(7, 7, 0, 100), small_zone(0, 1)];
        assert_eq!(
            verdict(&Predicate::NotEq(0, Value::Int64(7)), constant.clone()),
            Verdict::Skip
        );
        assert_eq!(
            verdict(&Predicate::NotEq(0, Value::Int64(8)), constant),
            Verdict::Maybe
        );
    }

    #[test]
    fn conjunctive_equalities_stop_at_or_and_not() {
        let a = Predicate::eq(0, Value::Int64(1));
        let b = Predicate::eq(1, Value::Int8(2));
        assert_eq!(a.clone().and(b.clone()).conjunctive_equalities().len(), 2);
        assert_eq!(a.clone().or(b.clone()).conjunctive_equalities().len(), 0);
        assert_eq!(a.clone().negate().conjunctive_equalities().len(), 0);
        assert_eq!(
            a.clone()
                .and(b.clone().or(a.clone()))
                .conjunctive_equalities()
                .len(),
            1
        );
    }

    #[test]
    fn validation_and_display() {
        let p = Predicate::eq(0, Value::Int64(1)).and(Predicate::gt(1, Value::Int8(2)));
        p.validate(&schema()).unwrap();
        assert_eq!(p.column_indices(), vec![0, 1]);
        let bad = Predicate::eq(9, Value::Int64(1));
        assert!(bad.validate(&schema()).is_err());
        assert!(bad
            .evaluate_zone_maps(&group(vec![zone(1, 2, 0, 1)]), &schema())
            .is_err());
        assert_eq!(p.to_string(), "(#0 = 1 AND #1 > 2)");
        assert_eq!(Predicate::IsNull(0).to_string(), "#0 IS NULL");
        assert_eq!(Predicate::AlwaysTrue.to_string(), "TRUE");
        assert_eq!(
            Predicate::between(0, Value::Int64(1), Value::Int64(2))
                .negate()
                .to_string(),
            "NOT #0 BETWEEN 1 AND 2"
        );
    }
}

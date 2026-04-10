//! The logical type system and its coercion lattice.

use std::fmt;

/// Resolution of a [`DataType::Timestamp`].
///
/// Ordered from coarse to fine, so `max(a, b)` is the finer unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TimeUnit {
    /// Seconds since the Unix epoch.
    Second,
    /// Milliseconds since the Unix epoch.
    Millisecond,
    /// Microseconds since the Unix epoch.
    Microsecond,
    /// Nanoseconds since the Unix epoch.
    Nanosecond,
}

impl TimeUnit {
    /// Ticks of this unit in one second.
    pub const fn per_second(self) -> i64 {
        match self {
            TimeUnit::Second => 1,
            TimeUnit::Millisecond => 1_000,
            TimeUnit::Microsecond => 1_000_000,
            TimeUnit::Nanosecond => 1_000_000_000,
        }
    }

    /// Short suffix used in [`Display`](fmt::Display) output: `s`, `ms`, `us`, `ns`.
    pub const fn suffix(self) -> &'static str {
        match self {
            TimeUnit::Second => "s",
            TimeUnit::Millisecond => "ms",
            TimeUnit::Microsecond => "us",
            TimeUnit::Nanosecond => "ns",
        }
    }
}

/// Maximum precision of [`DataType::Decimal128`]: `i128` holds
/// `1.7 x 10^38`, so 38 decimal digits always fit and 39 sometimes do not.
pub const DECIMAL128_MAX_PRECISION: u8 = 38;

/// A logical column type.
///
/// "Logical" is the operative word: [`DataType::Date32`] and [`DataType::Int32`]
/// have the same physical representation (`i32`) but different meaning, and the
/// coercion rules treat them very differently. The physical layout lives in
/// [`ColumnVector`](crate::ColumnVector).
///
/// ```
/// use basalt_types::{DataType, TimeUnit};
///
/// assert_eq!(DataType::Int32.byte_width(), Some(4));
/// assert!(DataType::Utf8.byte_width().is_none());
/// assert_eq!(DataType::Timestamp(TimeUnit::Microsecond).to_string(), "TIMESTAMP(us)");
/// assert_eq!(DataType::decimal128(10, 2), Some(DataType::Decimal128(10, 2)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DataType {
    /// The type of an expression that is always `NULL`.
    ///
    /// It is the bottom of the coercion lattice: `coerce(Null, T) == T` for
    /// every `T`, which is what makes `SELECT NULL AS x UNION SELECT 1` work.
    Null,
    /// `true` / `false`.
    Boolean,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 8-bit unsigned integer.
    UInt8,
    /// 16-bit unsigned integer.
    UInt16,
    /// 32-bit unsigned integer.
    UInt32,
    /// 64-bit unsigned integer.
    UInt64,
    /// IEEE-754 binary32.
    Float32,
    /// IEEE-754 binary64.
    Float64,
    /// UTF-8 encoded text, variable length.
    Utf8,
    /// Arbitrary bytes, variable length.
    Binary,
    /// Days since the Unix epoch, signed.
    Date32,
    /// Ticks of the given [`TimeUnit`] since the Unix epoch, signed.
    Timestamp(TimeUnit),
    /// Fixed-point decimal stored in an `i128`.
    ///
    /// The tuple is `(precision, scale)`: `precision` is the total number of
    /// significant digits (`1..=38`) and `scale` the number of digits after the
    /// point (`0..=precision`). Use [`DataType::decimal128`] to build one with
    /// those invariants checked.
    Decimal128(u8, i8),
}

impl DataType {
    /// Build a validated `Decimal128`.
    ///
    /// Returns `None` when `precision` is 0, exceeds
    /// [`DECIMAL128_MAX_PRECISION`], or `scale` is negative or larger than
    /// `precision`. Negative scale (a multiplier) is deliberately rejected: it
    /// is rare, doubles the number of cases in every decimal code path, and
    /// nothing in Basalt needs it.
    pub const fn decimal128(precision: u8, scale: i8) -> Option<DataType> {
        if precision == 0 || precision > DECIMAL128_MAX_PRECISION {
            return None;
        }
        if scale < 0 || scale as u8 > precision {
            return None;
        }
        Some(DataType::Decimal128(precision, scale))
    }

    /// Size in bytes of one value, for fixed-width types.
    ///
    /// `None` for [`DataType::Utf8`] and [`DataType::Binary`] (variable length)
    /// and for [`DataType::Boolean`], which is stored one *bit* per value —
    /// returning `1` there would make every size calculation eight times too
    /// large. Use [`DataType::bit_width`] if you want a uniform answer.
    pub const fn byte_width(&self) -> Option<usize> {
        match self {
            DataType::Int8 | DataType::UInt8 => Some(1),
            DataType::Int16 | DataType::UInt16 => Some(2),
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => Some(4),
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Timestamp(_) => {
                Some(8)
            }
            DataType::Decimal128(_, _) => Some(16),
            DataType::Boolean | DataType::Utf8 | DataType::Binary | DataType::Null => None,
        }
    }

    /// Size in *bits* of one value, for fixed-width types (1 for `Boolean`).
    pub const fn bit_width(&self) -> Option<usize> {
        match self {
            DataType::Boolean => Some(1),
            DataType::Null | DataType::Utf8 | DataType::Binary => None,
            other => match other.byte_width() {
                Some(w) => Some(w * 8),
                None => None,
            },
        }
    }

    /// `true` if values occupy a constant number of bits.
    pub const fn is_fixed_width(&self) -> bool {
        self.bit_width().is_some()
    }

    /// `true` for variable-length types ([`DataType::Utf8`], [`DataType::Binary`]).
    pub const fn is_variable_width(&self) -> bool {
        matches!(self, DataType::Utf8 | DataType::Binary)
    }

    /// `true` for signed integer types.
    pub const fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
        )
    }

    /// `true` for unsigned integer types.
    pub const fn is_unsigned_integer(&self) -> bool {
        matches!(
            self,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
        )
    }

    /// `true` for any integer type.
    pub const fn is_integer(&self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    /// `true` for `Float32` / `Float64`.
    pub const fn is_floating(&self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// `true` for integers, floats and decimals.
    pub const fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_floating() || matches!(self, DataType::Decimal128(_, _))
    }

    /// `true` for `Date32` and `Timestamp`.
    pub const fn is_temporal(&self) -> bool {
        matches!(self, DataType::Date32 | DataType::Timestamp(_))
    }

    /// `true` if `<`, `<=`, `>`, `>=` are meaningful for this type.
    ///
    /// Everything is ordered except [`DataType::Null`], where there is only one
    /// value and comparison carries no information. Note that ordering floats
    /// uses IEEE-754 *total* order (see [`Value`](crate::Value)), so `NaN` has a
    /// defined position rather than making comparison partial.
    pub const fn is_comparable(&self) -> bool {
        !matches!(self, DataType::Null)
    }

    /// `true` if `=` is meaningful. True for every type, including `Null`
    /// (where the answer is always `NULL` under SQL three-valued logic).
    pub const fn supports_equality(&self) -> bool {
        true
    }

    /// `true` if the type can be the key of a hash aggregation or join.
    ///
    /// Excludes floats: hashing `NaN` and `-0.0` consistently is possible (see
    /// [`Value`](crate::Value)) but grouping by a float is almost always a bug,
    /// so Basalt makes the planner ask for it explicitly.
    pub const fn is_hash_groupable(&self) -> bool {
        !matches!(self, DataType::Float32 | DataType::Float64 | DataType::Null)
    }

    /// Number of decimal digits needed to write any value of an integer type.
    const fn integer_digits(&self) -> Option<u8> {
        Some(match self {
            DataType::Int8 | DataType::UInt8 => 3,
            DataType::Int16 | DataType::UInt16 => 5,
            DataType::Int32 | DataType::UInt32 => 10,
            DataType::Int64 => 19,
            DataType::UInt64 => 20,
            _ => return None,
        })
    }

    /// `(width in bits, signed)` for integer types.
    const fn integer_shape(&self) -> Option<(u32, bool)> {
        Some(match self {
            DataType::Int8 => (8, true),
            DataType::Int16 => (16, true),
            DataType::Int32 => (32, true),
            DataType::Int64 => (64, true),
            DataType::UInt8 => (8, false),
            DataType::UInt16 => (16, false),
            DataType::UInt32 => (32, false),
            DataType::UInt64 => (64, false),
            _ => return None,
        })
    }

    const fn signed_of_width(bits: u32) -> Option<DataType> {
        Some(match bits {
            8 => DataType::Int8,
            16 => DataType::Int16,
            32 => DataType::Int32,
            64 => DataType::Int64,
            _ => return None,
        })
    }

    const fn unsigned_of_width(bits: u32) -> Option<DataType> {
        Some(match bits {
            8 => DataType::UInt8,
            16 => DataType::UInt16,
            32 => DataType::UInt32,
            64 => DataType::UInt64,
            _ => return None,
        })
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Null => f.write_str("NULL"),
            DataType::Boolean => f.write_str("BOOLEAN"),
            DataType::Int8 => f.write_str("INT8"),
            DataType::Int16 => f.write_str("INT16"),
            DataType::Int32 => f.write_str("INT32"),
            DataType::Int64 => f.write_str("INT64"),
            DataType::UInt8 => f.write_str("UINT8"),
            DataType::UInt16 => f.write_str("UINT16"),
            DataType::UInt32 => f.write_str("UINT32"),
            DataType::UInt64 => f.write_str("UINT64"),
            DataType::Float32 => f.write_str("FLOAT32"),
            DataType::Float64 => f.write_str("FLOAT64"),
            DataType::Utf8 => f.write_str("UTF8"),
            DataType::Binary => f.write_str("BINARY"),
            DataType::Date32 => f.write_str("DATE32"),
            DataType::Timestamp(u) => write!(f, "TIMESTAMP({})", u.suffix()),
            DataType::Decimal128(p, s) => write!(f, "DECIMAL({p}, {s})"),
        }
    }
}

/// Find the least common supertype of `a` and `b`, or `None` if there is none.
///
/// # The lattice
///
/// Coercion is a partial join on the type lattice. The rules, in the order they
/// are applied, are:
///
/// 1. **Reflexivity.** `coerce(T, T) = T`.
/// 2. **`Null` is bottom.** `coerce(Null, T) = T`. An all-`NULL` literal has to
///    take the other side's type or nothing would ever type-check.
/// 3. **Integer widening, same signedness.** `coerce(Int8, Int32) = Int32`.
///    Always exact.
/// 4. **Mixed signedness.** The result is the narrowest *signed* type that
///    contains both domains: `UInt8 + Int8 -> Int16`, `UInt16 + Int32 -> Int32`,
///    `UInt32 + Int8 -> Int64`. `UInt64` mixed with any signed type has **no**
///    common type and returns `None` — `Int64` cannot hold `u64::MAX` and
///    `Float64` would silently round it. Callers that want the SQL-engine
///    behaviour of "just make it a double" must ask for it explicitly.
/// 5. **Integer to float.** `Float32` absorbs integers up to 16 bits (a
///    `binary32` represents every integer below `2^24` exactly). Anything wider
///    goes to `Float64`. `Int64`/`UInt64` to `Float64` is the *only* lossy
///    promotion the lattice performs: it loses precision above `2^53`. It is
///    permitted because refusing it would break `avg(bigint_column)` and every
///    other piece of ordinary SQL, but [`is_lossless_coercion`] reports it as
///    lossy so a planner can warn.
/// 6. **Float widening.** `coerce(Float32, Float64) = Float64`, exact.
/// 7. **Decimal reconciliation.** `coerce(Decimal(p1,s1), Decimal(p2,s2))`
///    keeps every integral digit and the finer scale:
///    `scale = max(s1, s2)`, `integral = max(p1-s1, p2-s2)`,
///    `precision = integral + scale`. If that exceeds 38 the answer is `None`
///    rather than a silently truncated scale — dropping fractional digits is
///    exactly the failure mode decimals exist to prevent.
/// 8. **Decimal with integer.** The integer becomes `Decimal(digits, 0)` and
///    rule 7 applies.
/// 9. **Decimal with float.** `None`, deliberately. Binary floating point and
///    decimal have no common exact type; the usual "promote to double" rule
///    destroys the exactness that made someone choose `DECIMAL` in the first
///    place. Make the cast explicit.
/// 10. **Boolean with numeric.** `None`. Implicit `true -> 1` hides real bugs
///     (`WHERE flag = 1` against a boolean column) and no SQL dialect Basalt
///     targets requires it.
/// 11. **`Utf8` with `Binary`.** `Binary`. UTF-8 is a subset of bytes and the
///     byte-wise order agrees with the code-point order, so the embedding is
///     exact and order-preserving.
/// 12. **Temporal.** `Date32 + Timestamp(u) = Timestamp(u)` (a date is midnight
///     UTC). `Timestamp(u1) + Timestamp(u2)` takes the *finer* unit, so no
///     sub-second information is dropped. Dates and integers do **not** coerce:
///     `date_col > 5` is a bug, not a query.
///
/// The relation is commutative and idempotent by construction, and both
/// properties are covered by property tests.
///
/// ```
/// use basalt_types::{coerce, DataType, TimeUnit};
///
/// assert_eq!(coerce(&DataType::Int8, &DataType::Int64), Some(DataType::Int64));
/// assert_eq!(coerce(&DataType::UInt16, &DataType::Int8), Some(DataType::Int32));
/// assert_eq!(coerce(&DataType::UInt64, &DataType::Int8), None);
/// assert_eq!(coerce(&DataType::Int16, &DataType::Float32), Some(DataType::Float32));
/// assert_eq!(coerce(&DataType::Int32, &DataType::Float32), Some(DataType::Float64));
/// assert_eq!(
///     coerce(&DataType::Decimal128(10, 2), &DataType::Decimal128(5, 4)),
///     Some(DataType::Decimal128(12, 4)),
/// );
/// assert_eq!(coerce(&DataType::Decimal128(10, 2), &DataType::Float64), None);
/// assert_eq!(
///     coerce(&DataType::Date32, &DataType::Timestamp(TimeUnit::Second)),
///     Some(DataType::Timestamp(TimeUnit::Second)),
/// );
/// ```
pub fn coerce(a: &DataType, b: &DataType) -> Option<DataType> {
    // 1. Reflexivity.
    if a == b {
        return Some(a.clone());
    }
    // 2. Null is the bottom of the lattice.
    match (a, b) {
        (DataType::Null, other) | (other, DataType::Null) => return Some(other.clone()),
        _ => {}
    }

    // 3 & 4. Integer / integer.
    if let (Some((wa, sa)), Some((wb, sb))) = (a.integer_shape(), b.integer_shape()) {
        if sa == sb {
            let width = wa.max(wb);
            return if sa {
                DataType::signed_of_width(width)
            } else {
                DataType::unsigned_of_width(width)
            };
        }
        let (signed_w, unsigned_w) = if sa { (wa, wb) } else { (wb, wa) };
        // A signed type of W bits holds 0..=2^(W-1)-1, so it contains an
        // unsigned type of U bits only when W >= U + 1.
        let needed = (unsigned_w + 1).max(signed_w);
        let rounded = needed.next_power_of_two().max(8);
        return DataType::signed_of_width(rounded);
    }

    // 5 & 6. Anything numeric with a float.
    if a.is_floating() || b.is_floating() {
        let (float, other) = if a.is_floating() { (a, b) } else { (b, a) };
        if other.is_floating() {
            return Some(DataType::Float64);
        }
        if let Some((width, _)) = other.integer_shape() {
            let exact_in_f32 = width <= 16;
            return Some(if *float == DataType::Float32 && exact_in_f32 {
                DataType::Float32
            } else {
                DataType::Float64
            });
        }
        // 9. Decimal with float: no exact common type. Refuse.
        return None;
    }

    // 7 & 8. Decimals.
    if let (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) = (a, b) {
        return reconcile_decimal(*p1, *s1, *p2, *s2);
    }
    if let (DataType::Decimal128(p, s), other) | (other, DataType::Decimal128(p, s)) = (a, b) {
        if let Some(digits) = other.integer_digits() {
            return reconcile_decimal(*p, *s, digits, 0);
        }
    }

    // 11. Text and bytes.
    match (a, b) {
        (DataType::Utf8, DataType::Binary) | (DataType::Binary, DataType::Utf8) => {
            return Some(DataType::Binary)
        }
        _ => {}
    }

    // 12. Temporal.
    match (a, b) {
        (DataType::Date32, DataType::Timestamp(u)) | (DataType::Timestamp(u), DataType::Date32) => {
            return Some(DataType::Timestamp(*u))
        }
        (DataType::Timestamp(u1), DataType::Timestamp(u2)) => {
            return Some(DataType::Timestamp(*u1.max(u2)))
        }
        _ => {}
    }

    // 10 and everything else: refuse.
    None
}

fn reconcile_decimal(p1: u8, s1: i8, p2: u8, s2: i8) -> Option<DataType> {
    let integral = (i32::from(p1) - i32::from(s1)).max(i32::from(p2) - i32::from(s2));
    let scale = s1.max(s2);
    let precision = integral + i32::from(scale);
    if precision <= 0 || precision > i32::from(DECIMAL128_MAX_PRECISION) {
        return None;
    }
    DataType::decimal128(precision as u8, scale)
}

/// `true` if every value of `from` is representable in `to` without loss.
///
/// This is *domain* containment, not "the values I happen to have fit". It is
/// deliberately strict:
///
/// * `Int64 -> Float64` is **lossy** (integers above `2^53` round), even though
///   [`coerce`] performs it.
/// * `Timestamp(Second) -> Timestamp(Nanosecond)` is **lossy**, because the
///   multiplication by `10^9` overflows `i64` for timestamps far from the
///   epoch, even though it is exact for any plausible real timestamp.
/// * `Date32 -> Timestamp(Microsecond)` is **lossy** for the same reason:
///   `i32` days spans about +/-5.8 million years, microseconds only +/-292 000.
///
/// A planner can use this to decide between silently coercing and requiring an
/// explicit `CAST`.
pub fn is_lossless_coercion(from: &DataType, to: &DataType) -> bool {
    if from == to || matches!(from, DataType::Null) {
        return true;
    }
    if let (Some((wf, sf)), Some((wt, st))) = (from.integer_shape(), to.integer_shape()) {
        return match (sf, st) {
            (true, true) | (false, false) => wt >= wf,
            // unsigned -> signed needs one extra bit
            (false, true) => wt > wf,
            // signed -> unsigned always loses the negative half
            (true, false) => false,
        };
    }
    match (from, to) {
        (DataType::Float32, DataType::Float64) => true,
        (DataType::Utf8, DataType::Binary) => true,
        // binary32 is exact for |v| < 2^24, binary64 for |v| < 2^53.
        (f, DataType::Float32) if f.is_integer() => f.integer_shape().is_some_and(|(w, _)| w <= 16),
        (f, DataType::Float64) if f.is_integer() => f.integer_shape().is_some_and(|(w, _)| w <= 32),
        (DataType::Decimal128(p1, s1), DataType::Decimal128(p2, s2)) => {
            s2 >= s1 && (i32::from(*p2) - i32::from(*s2)) >= (i32::from(*p1) - i32::from(*s1))
        }
        (i, DataType::Decimal128(p, s)) if i.is_integer() => i
            .integer_digits()
            .is_some_and(|d| (i32::from(*p) - i32::from(*s)) >= i32::from(d)),
        // i32 days reaches ~5.8e6 years; only second and millisecond
        // timestamps span that in an i64.
        (DataType::Date32, DataType::Timestamp(u)) => {
            matches!(u, TimeUnit::Second | TimeUnit::Millisecond)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const ALL: &[DataType] = &[
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float32,
        DataType::Float64,
        DataType::Utf8,
        DataType::Binary,
        DataType::Date32,
        DataType::Timestamp(TimeUnit::Second),
        DataType::Timestamp(TimeUnit::Nanosecond),
        DataType::Decimal128(10, 2),
        DataType::Decimal128(38, 10),
    ];

    #[test]
    fn coercion_is_commutative_and_idempotent() {
        for a in ALL {
            assert_eq!(coerce(a, a).as_ref(), Some(a));
            for b in ALL {
                assert_eq!(coerce(a, b), coerce(b, a), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn coercion_result_absorbs_both_inputs() {
        // If a common type exists, coercing either input with it is a no-op:
        // this is the "join" property of the lattice.
        for a in ALL {
            for b in ALL {
                if let Some(c) = coerce(a, b) {
                    assert_eq!(coerce(a, &c).as_ref(), Some(&c), "{a} join {b} = {c}");
                    assert_eq!(coerce(b, &c).as_ref(), Some(&c), "{a} join {b} = {c}");
                }
            }
        }
    }

    #[test]
    fn integer_widening() {
        assert_eq!(
            coerce(&DataType::Int8, &DataType::Int64),
            Some(DataType::Int64)
        );
        assert_eq!(
            coerce(&DataType::UInt8, &DataType::UInt32),
            Some(DataType::UInt32)
        );
        assert_eq!(
            coerce(&DataType::UInt8, &DataType::Int8),
            Some(DataType::Int16)
        );
        assert_eq!(
            coerce(&DataType::UInt16, &DataType::Int8),
            Some(DataType::Int32)
        );
        assert_eq!(
            coerce(&DataType::UInt32, &DataType::Int64),
            Some(DataType::Int64)
        );
        assert_eq!(
            coerce(&DataType::UInt32, &DataType::Int8),
            Some(DataType::Int64)
        );
    }

    #[test]
    fn uint64_refuses_to_mix_with_signed() {
        for signed in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        ] {
            assert_eq!(
                coerce(&DataType::UInt64, &signed),
                None,
                "uint64 vs {signed}"
            );
        }
        assert_eq!(
            coerce(&DataType::UInt64, &DataType::UInt8),
            Some(DataType::UInt64)
        );
    }

    #[test]
    fn float_promotion_respects_exactness() {
        assert_eq!(
            coerce(&DataType::Int8, &DataType::Float32),
            Some(DataType::Float32)
        );
        assert_eq!(
            coerce(&DataType::UInt16, &DataType::Float32),
            Some(DataType::Float32)
        );
        assert_eq!(
            coerce(&DataType::Int32, &DataType::Float32),
            Some(DataType::Float64)
        );
        assert_eq!(
            coerce(&DataType::UInt64, &DataType::Float32),
            Some(DataType::Float64)
        );
        assert_eq!(
            coerce(&DataType::Float32, &DataType::Float64),
            Some(DataType::Float64)
        );
        assert!(is_lossless_coercion(&DataType::Int16, &DataType::Float32));
        assert!(!is_lossless_coercion(&DataType::Int32, &DataType::Float32));
        assert!(is_lossless_coercion(&DataType::Int32, &DataType::Float64));
        assert!(!is_lossless_coercion(&DataType::Int64, &DataType::Float64));
    }

    #[test]
    fn decimal_reconciliation() {
        assert_eq!(
            coerce(&DataType::Decimal128(10, 2), &DataType::Decimal128(10, 4)),
            Some(DataType::Decimal128(12, 4))
        );
        assert_eq!(
            coerce(&DataType::Decimal128(38, 0), &DataType::Decimal128(10, 4)),
            None,
            "would need 42 digits; refuse rather than drop scale"
        );
        assert_eq!(
            coerce(&DataType::Decimal128(10, 2), &DataType::Int32),
            Some(DataType::Decimal128(12, 2))
        );
        assert_eq!(
            coerce(&DataType::Decimal128(10, 2), &DataType::Float32),
            None
        );
        assert_eq!(
            coerce(&DataType::Decimal128(10, 2), &DataType::Float64),
            None
        );
    }

    #[test]
    fn refusals() {
        assert_eq!(coerce(&DataType::Boolean, &DataType::Int32), None);
        assert_eq!(coerce(&DataType::Date32, &DataType::Int32), None);
        assert_eq!(coerce(&DataType::Utf8, &DataType::Int64), None);
        assert_eq!(coerce(&DataType::Boolean, &DataType::Utf8), None);
    }

    #[test]
    fn text_and_temporal() {
        assert_eq!(
            coerce(&DataType::Utf8, &DataType::Binary),
            Some(DataType::Binary)
        );
        assert_eq!(
            coerce(
                &DataType::Timestamp(TimeUnit::Second),
                &DataType::Timestamp(TimeUnit::Nanosecond)
            ),
            Some(DataType::Timestamp(TimeUnit::Nanosecond))
        );
        assert_eq!(
            coerce(
                &DataType::Date32,
                &DataType::Timestamp(TimeUnit::Millisecond)
            ),
            Some(DataType::Timestamp(TimeUnit::Millisecond))
        );
        assert!(is_lossless_coercion(
            &DataType::Date32,
            &DataType::Timestamp(TimeUnit::Second)
        ));
        assert!(!is_lossless_coercion(
            &DataType::Date32,
            &DataType::Timestamp(TimeUnit::Nanosecond)
        ));
    }

    #[test]
    fn null_is_bottom() {
        for t in ALL {
            assert_eq!(coerce(&DataType::Null, t).as_ref(), Some(t));
            assert!(is_lossless_coercion(&DataType::Null, t));
        }
    }

    #[test]
    fn predicates_and_widths() {
        assert_eq!(DataType::Boolean.byte_width(), None);
        assert_eq!(DataType::Boolean.bit_width(), Some(1));
        assert_eq!(DataType::Decimal128(10, 2).byte_width(), Some(16));
        assert_eq!(
            DataType::Timestamp(TimeUnit::Nanosecond).byte_width(),
            Some(8)
        );
        assert!(DataType::Utf8.is_variable_width());
        assert!(!DataType::Utf8.is_fixed_width());
        assert!(DataType::Date32.is_temporal());
        assert!(DataType::Decimal128(1, 0).is_numeric());
        assert!(!DataType::Null.is_comparable());
        assert!(DataType::Null.supports_equality());
        assert!(!DataType::Float64.is_hash_groupable());
        assert!(DataType::Utf8.is_hash_groupable());
        assert_eq!(DataType::decimal128(0, 0), None);
        assert_eq!(DataType::decimal128(39, 0), None);
        assert_eq!(DataType::decimal128(5, 6), None);
        assert_eq!(DataType::decimal128(5, -1), None);
        assert_eq!(DataType::decimal128(5, 5), Some(DataType::Decimal128(5, 5)));
    }

    #[test]
    fn display_is_unambiguous() {
        assert_eq!(
            DataType::Timestamp(TimeUnit::Microsecond).to_string(),
            "TIMESTAMP(us)"
        );
        assert_eq!(DataType::Decimal128(10, 2).to_string(), "DECIMAL(10, 2)");
        assert_eq!(DataType::UInt8.to_string(), "UINT8");
        assert_eq!(TimeUnit::Nanosecond.per_second(), 1_000_000_000);
        for t in ALL {
            assert!(!t.to_string().is_empty());
        }
    }
}

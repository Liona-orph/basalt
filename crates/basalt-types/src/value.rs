//! Owned scalar values with a total order.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use basalt_common::{BasaltError, Result};

use crate::{DataType, TimeUnit};

/// An owned scalar.
///
/// `Value` is what a literal, a zone-map bound or a single cell read out of a
/// column looks like. It is deliberately *not* used inside arrays — a
/// `Vec<Value>` would be the row-store representation this whole engine exists
/// to avoid.
///
/// # Ordering, equality and hashing
///
/// `Value` implements [`Ord`], [`Eq`] and [`Hash`], and the three agree with
/// each other. Getting that right requires three explicit decisions:
///
/// **1. `NULL` sorts first.** `Value::Null` is less than every non-null value.
/// This is a *type-level* total order used for sorting keys, building indexes
/// and comparing zone-map bounds — it is not SQL's `ORDER BY` semantics.
/// SQL's `NULLS FIRST` / `NULLS LAST` is a property of the sort operator, which
/// applies it on top of this order; and SQL's three-valued `=` (where
/// `NULL = NULL` is `NULL`, not `true`) is a property of the comparison
/// *expression*, not of this type. Making `Null` the bottom element is the only
/// choice that keeps `Ord` total while matching the lattice, where
/// [`DataType::Null`] is also the bottom.
///
/// **2. Comparison is physical, not cross-type.** `Value::Int32(5)` and
/// `Value::Int64(5)` are *not* equal, and different variants are ordered by a
/// fixed variant rank. This is intentional. A cross-type numeric `Ord` cannot be
/// consistent with `Hash` and `Eq` without a canonical numeric form for every
/// combination (what is the hash of `Int64(5)` such that it matches
/// `Float64(5.0)` and `Decimal(500, 2)`?), and every attempt produces subtle
/// bugs. Instead, the planner coerces both sides with [`coerce`](crate::coerce)
/// and [`Value::cast_to`] before comparing — and in the place where ordering
/// actually matters at runtime (a single column, a zone map, a sort key) every
/// value already has the same type, so physical order *is* semantic order.
///
/// **3. Floats use IEEE-754 total order.** [`f64::total_cmp`] gives
/// `-NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN`. Two floats compare
/// equal exactly when their bit patterns match, so hashing the bits agrees with
/// the ordering. The consequences are worth stating: `NaN == NaN` is `true` and
/// `0.0 == -0.0` is `false` for `Value`. Both are wrong for IEEE arithmetic and
/// right for a storage key, which is what `Value`'s `Eq` is for; expression
/// evaluation implements IEEE semantics separately.
///
/// **Decimals are the one exception to "physical".** `Decimal128(100, _, 2)`
/// (`1.00`) and `Decimal128(10, _, 1)` (`1.0`) compare *equal*, and hash the
/// same, because they are the same number and a column's zone map must not
/// depend on how many trailing zeros the writer happened to store. Comparison
/// rescales exactly using `i128` and falls back to sign analysis if a rescale
/// would overflow, so it is always correct.
///
/// ```
/// use basalt_types::Value;
///
/// let mut vs = vec![
///     Value::Int32(3),
///     Value::Null,
///     Value::Int32(-1),
/// ];
/// vs.sort();
/// assert_eq!(vs, vec![Value::Null, Value::Int32(-1), Value::Int32(3)]);
///
/// // Decimals compare numerically regardless of stored scale.
/// assert_eq!(Value::Decimal128(100, 5, 2), Value::Decimal128(10, 5, 1));
/// ```
#[derive(Debug, Clone)]
pub enum Value {
    /// SQL `NULL`. Bottom of the order.
    Null,
    /// Boolean.
    Boolean(bool),
    /// 8-bit signed integer.
    Int8(i8),
    /// 16-bit signed integer.
    Int16(i16),
    /// 32-bit signed integer.
    Int32(i32),
    /// 64-bit signed integer.
    Int64(i64),
    /// 8-bit unsigned integer.
    UInt8(u8),
    /// 16-bit unsigned integer.
    UInt16(u16),
    /// 32-bit unsigned integer.
    UInt32(u32),
    /// 64-bit unsigned integer.
    UInt64(u64),
    /// IEEE-754 binary32.
    Float32(f32),
    /// IEEE-754 binary64.
    Float64(f64),
    /// UTF-8 text.
    Utf8(String),
    /// Arbitrary bytes.
    Binary(Vec<u8>),
    /// Days since the Unix epoch.
    Date32(i32),
    /// Ticks since the Unix epoch, in the given unit.
    Timestamp(i64, TimeUnit),
    /// Unscaled value, precision, scale. The number is `value * 10^-scale`.
    Decimal128(i128, u8, i8),
}

impl Value {
    /// The logical type of this value; [`DataType::Null`] for `Null`.
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null => DataType::Null,
            Value::Boolean(_) => DataType::Boolean,
            Value::Int8(_) => DataType::Int8,
            Value::Int16(_) => DataType::Int16,
            Value::Int32(_) => DataType::Int32,
            Value::Int64(_) => DataType::Int64,
            Value::UInt8(_) => DataType::UInt8,
            Value::UInt16(_) => DataType::UInt16,
            Value::UInt32(_) => DataType::UInt32,
            Value::UInt64(_) => DataType::UInt64,
            Value::Float32(_) => DataType::Float32,
            Value::Float64(_) => DataType::Float64,
            Value::Utf8(_) => DataType::Utf8,
            Value::Binary(_) => DataType::Binary,
            Value::Date32(_) => DataType::Date32,
            Value::Timestamp(_, u) => DataType::Timestamp(*u),
            Value::Decimal128(_, p, s) => DataType::Decimal128(*p, *s),
        }
    }

    /// `true` for [`Value::Null`].
    pub const fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// The value as `i128` if it is an integer (or a date, in days).
    ///
    /// Provided so that integer code paths can be written once instead of nine
    /// times; `i128` is the only type that holds every `i64` and every `u64`.
    pub fn as_i128(&self) -> Option<i128> {
        Some(match self {
            Value::Int8(v) => i128::from(*v),
            Value::Int16(v) => i128::from(*v),
            Value::Int32(v) | Value::Date32(v) => i128::from(*v),
            Value::Int64(v) | Value::Timestamp(v, _) => i128::from(*v),
            Value::UInt8(v) => i128::from(*v),
            Value::UInt16(v) => i128::from(*v),
            Value::UInt32(v) => i128::from(*v),
            Value::UInt64(v) => i128::from(*v),
            _ => return None,
        })
    }

    /// The value as `f64` if it is numeric. Lossy for large integers and for
    /// decimals; use only where the caller has already accepted that.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float32(v) => Some(f64::from(*v)),
            Value::Float64(v) => Some(*v),
            Value::Decimal128(v, _, s) => Some(*v as f64 / 10f64.powi(i32::from(*s))),
            other => other.as_i128().map(|v| v as f64),
        }
    }

    /// The text if this is a [`Value::Utf8`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Utf8(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// The bytes if this is a [`Value::Binary`] or [`Value::Utf8`].
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Binary(b) => Some(b.as_slice()),
            Value::Utf8(s) => Some(s.as_bytes()),
            _ => None,
        }
    }

    /// The boolean if this is a [`Value::Boolean`].
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Convert to `target`, failing on anything that would silently corrupt the
    /// value.
    ///
    /// This is SQL's explicit `CAST`, so it is more permissive than
    /// [`coerce`](crate::coerce): narrowing integer casts are allowed but
    /// **range-checked**, float-to-integer truncates toward zero and is
    /// range-checked, and decimal-to-integer truncates. Conversions that cannot
    /// be defined without inventing a policy — float to decimal (which rounding?
    /// which scale?) and text to number (which locale?) — return
    /// [`BasaltError::Unsupported`] rather than guessing.
    ///
    /// `Null` casts to `Null` of any type.
    pub fn cast_to(&self, target: &DataType) -> Result<Value> {
        if self.is_null() {
            return Ok(Value::Null);
        }
        if self.data_type() == *target {
            return Ok(self.clone());
        }
        match (self, target) {
            // ---- integers and dates ------------------------------------
            (v, t) if v.plain_int().is_some() && t.is_integer() => {
                let raw = v.plain_int().unwrap_or_default();
                int_from_i128(raw, t).ok_or_else(|| range_error(self, target))
            }
            (v, DataType::Float32) if v.plain_int().is_some() => {
                Ok(Value::Float32(v.plain_int().unwrap_or_default() as f32))
            }
            (v, DataType::Float64) if v.plain_int().is_some() => {
                Ok(Value::Float64(v.plain_int().unwrap_or_default() as f64))
            }
            (v, DataType::Decimal128(p, s)) if v.plain_int().is_some() => {
                let raw = v.plain_int().unwrap_or_default();
                let scaled = pow10(*s)
                    .and_then(|m| raw.checked_mul(m))
                    .ok_or_else(|| range_error(self, target))?;
                check_precision(scaled, *p).ok_or_else(|| range_error(self, target))?;
                Ok(Value::Decimal128(scaled, *p, *s))
            }
            // ---- floats -------------------------------------------------
            (Value::Float32(v), DataType::Float64) => Ok(Value::Float64(f64::from(*v))),
            (Value::Float64(v), DataType::Float32) => Ok(Value::Float32(*v as f32)),
            (Value::Float32(_) | Value::Float64(_), t) if t.is_integer() => {
                let f = self.as_f64().unwrap_or_default();
                if !f.is_finite() {
                    return Err(range_error(self, target));
                }
                let truncated = f.trunc();
                if truncated < -(2f64.powi(127)) || truncated > 2f64.powi(127) {
                    return Err(range_error(self, target));
                }
                int_from_i128(truncated as i128, t).ok_or_else(|| range_error(self, target))
            }
            // ---- decimals ----------------------------------------------
            (Value::Decimal128(v, _, s), DataType::Decimal128(p2, s2)) => {
                let rescaled = rescale(*v, *s, *s2).ok_or_else(|| range_error(self, target))?;
                check_precision(rescaled, *p2).ok_or_else(|| range_error(self, target))?;
                Ok(Value::Decimal128(rescaled, *p2, *s2))
            }
            (Value::Decimal128(v, _, s), t) if t.is_integer() => {
                let divisor = pow10(*s).ok_or_else(|| range_error(self, target))?;
                int_from_i128(v / divisor, t).ok_or_else(|| range_error(self, target))
            }
            (Value::Decimal128(_, _, _), DataType::Float64) => {
                Ok(Value::Float64(self.as_f64().unwrap_or_default()))
            }
            (Value::Decimal128(_, _, _), DataType::Float32) => {
                Ok(Value::Float32(self.as_f64().unwrap_or_default() as f32))
            }
            // ---- text and bytes ----------------------------------------
            (Value::Utf8(s), DataType::Binary) => Ok(Value::Binary(s.clone().into_bytes())),
            (Value::Binary(b), DataType::Utf8) => match String::from_utf8(b.clone()) {
                Ok(s) => Ok(Value::Utf8(s)),
                Err(e) => Err(BasaltError::type_error(format!(
                    "cannot cast BINARY to UTF8: invalid UTF-8 at byte {}",
                    e.utf8_error().valid_up_to()
                ))),
            },
            (v, DataType::Utf8) => Ok(Value::Utf8(v.to_string())),
            // ---- temporal ------------------------------------------------
            (Value::Date32(days), DataType::Timestamp(u)) => i64::from(*days)
                .checked_mul(86_400)
                .and_then(|secs| secs.checked_mul(u.per_second()))
                .map(|ticks| Value::Timestamp(ticks, *u))
                .ok_or_else(|| range_error(self, target)),
            (Value::Timestamp(ticks, u), DataType::Date32) => {
                let per_day = u.per_second().checked_mul(86_400).unwrap_or(i64::MAX);
                let days = ticks.div_euclid(per_day);
                i32::try_from(days)
                    .map(Value::Date32)
                    .map_err(|_| range_error(self, target))
            }
            (Value::Timestamp(ticks, from), DataType::Timestamp(to)) => {
                let (f, t) = (from.per_second(), to.per_second());
                let converted = if t >= f {
                    ticks.checked_mul(t / f)
                } else {
                    Some(ticks.div_euclid(f / t))
                };
                converted
                    .map(|v| Value::Timestamp(v, *to))
                    .ok_or_else(|| range_error(self, target))
            }
            _ => Err(unsupported_cast(self, target)),
        }
    }

    /// The value as `i128` only when it is a *plain* integer.
    ///
    /// Unlike [`Value::as_i128`] this rejects `Date32` and `Timestamp`, whose
    /// numeric payload is meaningless without its unit. Casting those to a bare
    /// integer therefore has to be spelled out via an explicit `EXTRACT`-style
    /// operation rather than falling out of a numeric cast.
    const fn plain_int(&self) -> Option<i128> {
        match self {
            Value::Int8(v) => Some(*v as i128),
            Value::Int16(v) => Some(*v as i128),
            Value::Int32(v) => Some(*v as i128),
            Value::Int64(v) => Some(*v as i128),
            Value::UInt8(v) => Some(*v as i128),
            Value::UInt16(v) => Some(*v as i128),
            Value::UInt32(v) => Some(*v as i128),
            Value::UInt64(v) => Some(*v as i128),
            _ => None,
        }
    }

    /// Variant rank used by the total order. Stable across releases because it
    /// is persisted indirectly (sorted files, zone-map comparisons).
    const fn rank(&self) -> u8 {
        match self {
            Value::Null => 0,
            Value::Boolean(_) => 1,
            Value::Int8(_) => 2,
            Value::Int16(_) => 3,
            Value::Int32(_) => 4,
            Value::Int64(_) => 5,
            Value::UInt8(_) => 6,
            Value::UInt16(_) => 7,
            Value::UInt32(_) => 8,
            Value::UInt64(_) => 9,
            Value::Float32(_) => 10,
            Value::Float64(_) => 11,
            Value::Utf8(_) => 12,
            Value::Binary(_) => 13,
            Value::Date32(_) => 14,
            Value::Timestamp(_, _) => 15,
            Value::Decimal128(_, _, _) => 16,
        }
    }
}

fn unsupported_cast(v: &Value, target: &DataType) -> BasaltError {
    BasaltError::unsupported(format!("cannot cast {} to {target}", v.data_type()))
}

fn range_error(v: &Value, target: &DataType) -> BasaltError {
    BasaltError::type_error(format!(
        "value {v} of type {} does not fit in {target}",
        v.data_type()
    ))
}

/// `10^exp` as `i128`, or `None` if `exp` is out of range.
fn pow10(exp: i8) -> Option<i128> {
    if !(0..=38).contains(&exp) {
        return None;
    }
    let mut acc: i128 = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(10)?;
    }
    Some(acc)
}

/// Move a decimal from `from` scale to `to` scale, exactly.
///
/// Scaling down (fewer fractional digits) truncates toward zero, which is what
/// SQL `CAST` does; scaling up is checked for overflow.
fn rescale(value: i128, from: i8, to: i8) -> Option<i128> {
    match to.cmp(&from) {
        Ordering::Equal => Some(value),
        Ordering::Greater => pow10(to - from).and_then(|m| value.checked_mul(m)),
        Ordering::Less => pow10(from - to).map(|d| value / d),
    }
}

/// `Some(())` if `value` fits in `precision` decimal digits.
fn check_precision(value: i128, precision: u8) -> Option<()> {
    let limit = pow10(precision as i8)?;
    if value.unsigned_abs() < limit.unsigned_abs() {
        Some(())
    } else {
        None
    }
}

fn int_from_i128(raw: i128, target: &DataType) -> Option<Value> {
    Some(match target {
        DataType::Int8 => Value::Int8(i8::try_from(raw).ok()?),
        DataType::Int16 => Value::Int16(i16::try_from(raw).ok()?),
        DataType::Int32 => Value::Int32(i32::try_from(raw).ok()?),
        DataType::Int64 => Value::Int64(i64::try_from(raw).ok()?),
        DataType::UInt8 => Value::UInt8(u8::try_from(raw).ok()?),
        DataType::UInt16 => Value::UInt16(u16::try_from(raw).ok()?),
        DataType::UInt32 => Value::UInt32(u32::try_from(raw).ok()?),
        DataType::UInt64 => Value::UInt64(u64::try_from(raw).ok()?),
        _ => return None,
    })
}

/// Canonical form of a decimal for hashing and equality: strip trailing zeros
/// so that `1.0` and `1.00` agree.
fn canonical_decimal(value: i128, scale: i8) -> (i128, i8) {
    let mut v = value;
    let mut s = scale;
    while s > 0 && v % 10 == 0 && v != 0 {
        v /= 10;
        s -= 1;
    }
    if v == 0 {
        (0, 0)
    } else {
        (v, s)
    }
}

/// Exact numeric comparison of two decimals with possibly different scales.
///
/// Rescales the coarser one; if that would overflow `i128` the rescaled value is
/// necessarily larger in magnitude than anything representable, so the sign of
/// the operand being scaled up decides the answer.
fn cmp_decimal(a: i128, sa: i8, b: i128, sb: i8) -> Ordering {
    match sa.cmp(&sb) {
        Ordering::Equal => a.cmp(&b),
        Ordering::Less => match rescale(a, sa, sb) {
            Some(scaled) => scaled.cmp(&b),
            None => {
                if a > 0 {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
        },
        Ordering::Greater => match rescale(b, sb, sa) {
            Some(scaled) => a.cmp(&scaled),
            None => {
                if b > 0 {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        },
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            (Value::Int8(a), Value::Int8(b)) => a.cmp(b),
            (Value::Int16(a), Value::Int16(b)) => a.cmp(b),
            (Value::Int32(a), Value::Int32(b)) => a.cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
            (Value::UInt8(a), Value::UInt8(b)) => a.cmp(b),
            (Value::UInt16(a), Value::UInt16(b)) => a.cmp(b),
            (Value::UInt32(a), Value::UInt32(b)) => a.cmp(b),
            (Value::UInt64(a), Value::UInt64(b)) => a.cmp(b),
            (Value::Float32(a), Value::Float32(b)) => a.total_cmp(b),
            (Value::Float64(a), Value::Float64(b)) => a.total_cmp(b),
            (Value::Utf8(a), Value::Utf8(b)) => a.cmp(b),
            (Value::Binary(a), Value::Binary(b)) => a.cmp(b),
            (Value::Date32(a), Value::Date32(b)) => a.cmp(b),
            // Different units are different physical types; order by unit first
            // so the relation stays total. Within a column the unit is fixed, so
            // this degenerates to comparing ticks.
            (Value::Timestamp(a, ua), Value::Timestamp(b, ub)) => ua.cmp(ub).then(a.cmp(b)),
            (Value::Decimal128(a, _, sa), Value::Decimal128(b, _, sb)) => {
                cmp_decimal(*a, *sa, *b, *sb)
            }
            (a, b) => a.rank().cmp(&b.rank()),
        }
    }
}

impl Hash for Value {
    /// Agrees with [`Ord`]: values that compare equal hash equal.
    ///
    /// Floats hash their raw bits, which is exactly the equivalence
    /// `total_cmp` induces; decimals hash their canonical (trailing-zero
    /// stripped) form, matching the numeric equality above.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rank().hash(state);
        match self {
            Value::Null => {}
            Value::Boolean(v) => v.hash(state),
            Value::Int8(v) => v.hash(state),
            Value::Int16(v) => v.hash(state),
            Value::Int32(v) | Value::Date32(v) => v.hash(state),
            Value::Int64(v) => v.hash(state),
            Value::UInt8(v) => v.hash(state),
            Value::UInt16(v) => v.hash(state),
            Value::UInt32(v) => v.hash(state),
            Value::UInt64(v) => v.hash(state),
            Value::Float32(v) => v.to_bits().hash(state),
            Value::Float64(v) => v.to_bits().hash(state),
            Value::Utf8(v) => v.hash(state),
            Value::Binary(v) => v.hash(state),
            Value::Timestamp(v, u) => {
                u.hash(state);
                v.hash(state);
            }
            Value::Decimal128(v, _, s) => canonical_decimal(*v, *s).hash(state),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("NULL"),
            Value::Boolean(v) => write!(f, "{v}"),
            Value::Int8(v) => write!(f, "{v}"),
            Value::Int16(v) => write!(f, "{v}"),
            Value::Int32(v) => write!(f, "{v}"),
            Value::Int64(v) => write!(f, "{v}"),
            Value::UInt8(v) => write!(f, "{v}"),
            Value::UInt16(v) => write!(f, "{v}"),
            Value::UInt32(v) => write!(f, "{v}"),
            Value::UInt64(v) => write!(f, "{v}"),
            Value::Float32(v) => write!(f, "{v}"),
            Value::Float64(v) => write!(f, "{v}"),
            Value::Utf8(v) => f.write_str(v),
            Value::Binary(v) => {
                f.write_str("0x")?;
                for b in v {
                    write!(f, "{b:02x}")?;
                }
                Ok(())
            }
            Value::Date32(v) => write!(f, "{v} days"),
            Value::Timestamp(v, u) => write!(f, "{v}{}", u.suffix()),
            Value::Decimal128(v, _, s) => write_decimal(f, *v, *s),
        }
    }
}

fn write_decimal(f: &mut fmt::Formatter<'_>, value: i128, scale: i8) -> fmt::Result {
    if scale <= 0 {
        return write!(f, "{value}");
    }
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let scale = scale as usize;
    let padded = if digits.len() <= scale {
        format!("{}{digits}", "0".repeat(scale - digits.len() + 1))
    } else {
        digits
    };
    let split = padded.len() - scale;
    if negative {
        f.write_str("-")?;
    }
    write!(f, "{}.{}", &padded[..split], &padded[split..])
}

macro_rules! impl_from {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl From<$t> for Value {
            fn from(v: $t) -> Value { Value::$variant(v) }
        })*
    };
}

impl_from! {
    bool => Boolean,
    i8 => Int8,
    i16 => Int16,
    i32 => Int32,
    i64 => Int64,
    u8 => UInt8,
    u16 => UInt16,
    u32 => UInt32,
    u64 => UInt64,
    f32 => Float32,
    f64 => Float64,
    String => Utf8,
    Vec<u8> => Binary,
}

impl From<&str> for Value {
    fn from(v: &str) -> Value {
        Value::Utf8(v.to_string())
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Value {
        match v {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::hash_map::DefaultHasher;

    use super::*;

    fn hash_of(v: &Value) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn null_sorts_first() {
        let mut vs = [Value::Int64(5), Value::Null, Value::Int64(-9), Value::Null];
        vs.sort();
        assert_eq!(vs[0], Value::Null);
        assert_eq!(vs[1], Value::Null);
        assert_eq!(vs[2], Value::Int64(-9));
    }

    #[test]
    fn hash_agrees_with_eq() {
        let pairs = [
            (Value::Int32(7), Value::Int32(7)),
            (Value::Float64(f64::NAN), Value::Float64(f64::NAN)),
            (Value::Decimal128(100, 10, 2), Value::Decimal128(10, 5, 1)),
            (Value::Decimal128(0, 10, 5), Value::Decimal128(0, 3, 0)),
            (Value::Utf8("a".into()), Value::Utf8("a".into())),
            (Value::Null, Value::Null),
        ];
        for (a, b) in pairs {
            assert_eq!(a, b, "{a:?} vs {b:?}");
            assert_eq!(hash_of(&a), hash_of(&b), "{a:?} vs {b:?}");
        }
        // And unequal values must (almost surely) hash differently.
        assert_ne!(Value::Float64(0.0), Value::Float64(-0.0));
        assert_ne!(
            hash_of(&Value::Float64(0.0)),
            hash_of(&Value::Float64(-0.0))
        );
        assert_ne!(Value::Int32(1), Value::Int64(1));
    }

    #[test]
    fn float_total_order() {
        let mut vs = [
            Value::Float64(f64::NAN),
            Value::Float64(1.0),
            Value::Float64(f64::NEG_INFINITY),
            Value::Float64(-0.0),
            Value::Float64(0.0),
            Value::Float64(f64::INFINITY),
        ];
        vs.sort();
        assert_eq!(vs[0], Value::Float64(f64::NEG_INFINITY));
        assert_eq!(vs[1], Value::Float64(-0.0));
        assert_eq!(vs[2], Value::Float64(0.0));
        assert_eq!(vs[3], Value::Float64(1.0));
        assert_eq!(vs[4], Value::Float64(f64::INFINITY));
        assert!(vs[5].as_f64().unwrap().is_nan());
    }

    #[test]
    fn decimal_ordering_across_scales() {
        assert!(Value::Decimal128(50, 3, 2) < Value::Decimal128(10, 2, 1)); // 0.50 < 1.0
        assert!(Value::Decimal128(-50, 3, 2) < Value::Decimal128(-10, 3, 2));
        // Overflow path: scaling 10^37 up by 10^30 cannot fit in i128.
        let huge = Value::Decimal128(i128::MAX / 2, 38, 0);
        let small = Value::Decimal128(1, 38, 30);
        assert!(huge > small);
        assert!(small < huge);
    }

    #[test]
    fn casts_that_should_work() {
        assert_eq!(
            Value::Int8(5).cast_to(&DataType::Int64).unwrap(),
            Value::Int64(5)
        );
        assert_eq!(
            Value::Int64(5).cast_to(&DataType::Int8).unwrap(),
            Value::Int8(5)
        );
        assert_eq!(
            Value::Int32(3).cast_to(&DataType::Float64).unwrap(),
            Value::Float64(3.0)
        );
        assert_eq!(
            Value::Float64(3.9).cast_to(&DataType::Int32).unwrap(),
            Value::Int32(3)
        );
        assert_eq!(
            Value::Float64(-3.9).cast_to(&DataType::Int32).unwrap(),
            Value::Int32(-3)
        );
        assert_eq!(
            Value::Int32(12)
                .cast_to(&DataType::Decimal128(10, 2))
                .unwrap(),
            Value::Decimal128(1200, 10, 2)
        );
        assert_eq!(
            Value::Decimal128(1234, 10, 2)
                .cast_to(&DataType::Int32)
                .unwrap(),
            Value::Int32(12)
        );
        assert_eq!(
            Value::Decimal128(1234, 10, 2)
                .cast_to(&DataType::Decimal128(12, 4))
                .unwrap(),
            Value::Decimal128(123_400, 12, 4)
        );
        assert_eq!(
            Value::Utf8("hi".into()).cast_to(&DataType::Binary).unwrap(),
            Value::Binary(b"hi".to_vec())
        );
        assert_eq!(
            Value::Binary(b"hi".to_vec())
                .cast_to(&DataType::Utf8)
                .unwrap(),
            Value::Utf8("hi".into())
        );
        assert_eq!(
            Value::Date32(2)
                .cast_to(&DataType::Timestamp(TimeUnit::Second))
                .unwrap(),
            Value::Timestamp(172_800, TimeUnit::Second)
        );
        assert_eq!(
            Value::Timestamp(172_800, TimeUnit::Second)
                .cast_to(&DataType::Date32)
                .unwrap(),
            Value::Date32(2)
        );
        assert_eq!(
            Value::Timestamp(1_500, TimeUnit::Millisecond)
                .cast_to(&DataType::Timestamp(TimeUnit::Second))
                .unwrap(),
            Value::Timestamp(1, TimeUnit::Second)
        );
        assert_eq!(Value::Null.cast_to(&DataType::Int8).unwrap(), Value::Null);
        assert_eq!(
            Value::Int32(7).cast_to(&DataType::Utf8).unwrap(),
            Value::Utf8("7".into())
        );
    }

    #[test]
    fn casts_that_must_fail() {
        assert!(Value::Int64(300).cast_to(&DataType::Int8).is_err());
        assert!(Value::Int64(-1).cast_to(&DataType::UInt8).is_err());
        assert!(Value::Float64(f64::NAN).cast_to(&DataType::Int32).is_err());
        assert!(Value::Float64(1e30).cast_to(&DataType::Int32).is_err());
        assert!(Value::Float64(1.5)
            .cast_to(&DataType::Decimal128(10, 2))
            .is_err());
        assert!(Value::Utf8("12".into()).cast_to(&DataType::Int32).is_err());
        assert!(Value::Date32(1).cast_to(&DataType::Int32).is_err());
        assert!(Value::Binary(vec![0xff]).cast_to(&DataType::Utf8).is_err());
        assert!(Value::Int32(1234)
            .cast_to(&DataType::Decimal128(3, 0))
            .is_err());
        let err = Value::Int64(300).cast_to(&DataType::Int8).unwrap_err();
        assert!(err.to_string().contains("300"), "{err}");
    }

    #[test]
    fn display() {
        assert_eq!(Value::Decimal128(1234, 10, 2).to_string(), "12.34");
        assert_eq!(Value::Decimal128(-1234, 10, 2).to_string(), "-12.34");
        assert_eq!(Value::Decimal128(5, 10, 3).to_string(), "0.005");
        assert_eq!(Value::Decimal128(1234, 10, 0).to_string(), "1234");
        assert_eq!(Value::Binary(vec![0xde, 0xad]).to_string(), "0xdead");
        assert_eq!(Value::Null.to_string(), "NULL");
        assert_eq!(Value::Timestamp(5, TimeUnit::Nanosecond).to_string(), "5ns");
        assert_eq!(Value::Date32(-1).to_string(), "-1 days");
    }

    #[test]
    fn accessors_and_conversions() {
        assert_eq!(Value::from(3i32).data_type(), DataType::Int32);
        assert_eq!(Value::from("x").as_str(), Some("x"));
        assert_eq!(Value::from(vec![1u8]).as_bytes(), Some(&[1u8][..]));
        assert_eq!(Value::from("x").as_bytes(), Some(&b"x"[..]));
        assert_eq!(Value::from(true).as_bool(), Some(true));
        assert_eq!(Value::from(Some(1u8)), Value::UInt8(1));
        assert_eq!(Value::from(Option::<u8>::None), Value::Null);
        assert_eq!(
            Value::UInt64(u64::MAX).as_i128(),
            Some(i128::from(u64::MAX))
        );
        assert_eq!(Value::Utf8("a".into()).as_i128(), None);
        assert!((Value::Decimal128(1234, 10, 2).as_f64().unwrap() - 12.34).abs() < 1e-9);
        assert!(Value::Null.is_null());
    }
}

//! Append-only builders for every array type.
//!
//! All builders share the same shape:
//!
//! * `new()` / `with_capacity(n)` — one allocation up front if you know `n`.
//! * `append_value` / `append_null` / `append_option`.
//! * `finish()` — consumes the builder and hands its buffers to the array
//!   without copying them.
//!
//! Growth is amortised O(1): the value buffers are `Vec`s, which double, so
//! building `n` values costs `O(n)` with `O(log n)` reallocations. The validity
//! mask is created **lazily** — a builder that never sees a null never allocates
//! a bitmap and produces an array with `validity == None`, which is what lets
//! scan kernels take the null-free fast path.

use basalt_common::{BasaltError, Result};

use crate::{
    BinaryArray, Bitmap, BitmapBuilder, BooleanArray, Buffer, ColumnVector, DataType, NativeType,
    NullArray, PrimitiveArray, StringArray, TimeUnit, Value,
};

/// Tracks nulls without allocating until the first one arrives.
#[derive(Debug, Default, Clone)]
struct ValidityTracker {
    bits: Option<BitmapBuilder>,
    len: usize,
}

impl ValidityTracker {
    fn with_capacity(_cap: usize) -> Self {
        ValidityTracker::default()
    }

    fn push(&mut self, valid: bool) {
        match (&mut self.bits, valid) {
            (Some(b), _) => b.push(valid),
            (None, true) => {}
            (None, false) => {
                // First null: back-fill the all-valid prefix.
                let mut b = BitmapBuilder::with_capacity(self.len + 1);
                b.push_n(true, self.len);
                b.push(false);
                self.bits = Some(b);
            }
        }
        self.len += 1;
    }

    fn finish(self) -> Option<Bitmap> {
        self.bits.map(BitmapBuilder::finish)
    }
}

/// Builder for [`PrimitiveArray`].
///
/// ```
/// use basalt_types::{PrimitiveBuilder};
///
/// let mut b = PrimitiveBuilder::<i32>::with_capacity(3);
/// b.append_value(1);
/// b.append_null();
/// b.append_option(Some(3));
/// let arr = b.finish();
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.get(1), None);
/// assert_eq!(arr.null_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PrimitiveBuilder<T: NativeType> {
    values: Vec<T>,
    validity: ValidityTracker,
}

impl<T: NativeType> PrimitiveBuilder<T> {
    /// An empty builder.
    pub fn new() -> Self {
        PrimitiveBuilder {
            values: Vec::new(),
            validity: ValidityTracker::default(),
        }
    }

    /// A builder pre-sized for `capacity` values.
    pub fn with_capacity(capacity: usize) -> Self {
        PrimitiveBuilder {
            values: Vec::with_capacity(capacity),
            validity: ValidityTracker::with_capacity(capacity),
        }
    }

    /// Number of appended slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if nothing has been appended.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Reserve room for `additional` more values.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    /// Append a value.
    #[inline]
    pub fn append_value(&mut self, value: T) {
        self.values.push(value);
        self.validity.push(true);
    }

    /// Append a null. The slot holds `T::default()`, which nothing may read.
    #[inline]
    pub fn append_null(&mut self) {
        self.values.push(T::default());
        self.validity.push(false);
    }

    /// Append `Some(v)` or a null.
    #[inline]
    pub fn append_option(&mut self, value: Option<T>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    /// Append every item of an iterator.
    pub fn extend<I: IntoIterator<Item = Option<T>>>(&mut self, iter: I) {
        for v in iter {
            self.append_option(v);
        }
    }

    /// Finish. Moves the value buffer into the array without copying.
    pub fn finish(self) -> PrimitiveArray<T> {
        PrimitiveArray::try_new(Buffer::from_vec(self.values), self.validity.finish())
            .unwrap_or_else(|_| PrimitiveArray::from_vec(Vec::new()))
    }
}

impl<T: NativeType> Default for PrimitiveBuilder<T> {
    fn default() -> Self {
        PrimitiveBuilder::new()
    }
}

/// Builder for [`BooleanArray`]; values are bit-packed as they arrive.
#[derive(Debug, Clone, Default)]
pub struct BooleanBuilder {
    values: BitmapBuilder,
    validity: ValidityTracker,
}

impl BooleanBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        BooleanBuilder::default()
    }

    /// A builder pre-sized for `capacity` values.
    pub fn with_capacity(capacity: usize) -> Self {
        BooleanBuilder {
            values: BitmapBuilder::with_capacity(capacity),
            validity: ValidityTracker::with_capacity(capacity),
        }
    }

    /// Number of appended slots.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if nothing has been appended.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Append a value.
    pub fn append_value(&mut self, value: bool) {
        self.values.push(value);
        self.validity.push(true);
    }

    /// Append a null.
    pub fn append_null(&mut self) {
        self.values.push(false);
        self.validity.push(false);
    }

    /// Append `Some(v)` or a null.
    pub fn append_option(&mut self, value: Option<bool>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    /// Finish.
    pub fn finish(self) -> BooleanArray {
        BooleanArray::try_new(self.values.finish(), self.validity.finish())
            .unwrap_or_else(|_| BooleanArray::from_bools(&[]))
    }
}

/// Builder for [`BinaryArray`], writing into one shared data buffer.
#[derive(Debug, Clone)]
pub struct BinaryBuilder {
    offsets: Vec<i32>,
    data: Vec<u8>,
    validity: ValidityTracker,
}

impl BinaryBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        BinaryBuilder {
            offsets: vec![0],
            data: Vec::new(),
            validity: ValidityTracker::default(),
        }
    }

    /// A builder pre-sized for `values` values totalling `bytes` bytes.
    ///
    /// Two capacities because the two buffers grow independently; guessing the
    /// byte total badly is the usual cause of repeated reallocation in string
    /// building.
    pub fn with_capacity(values: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(values + 1);
        offsets.push(0);
        BinaryBuilder {
            offsets,
            data: Vec::with_capacity(bytes),
            validity: ValidityTracker::with_capacity(values),
        }
    }

    /// Number of appended slots.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `true` if nothing has been appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a value.
    ///
    /// # Errors
    ///
    /// If the data buffer would exceed `i32::MAX` bytes. 32-bit offsets are a
    /// deliberate trade (half the metadata of 64-bit ones for the same data);
    /// a column chunk above 2 GiB must be split, which the writer does anyway.
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) -> Result<()> {
        let bytes = value.as_ref();
        let end = self.data.len() + bytes.len();
        if end > i32::MAX as usize {
            return Err(BasaltError::execution(format!(
                "binary column chunk would exceed the 2 GiB offset limit ({end} bytes); \
                 split the chunk"
            )));
        }
        self.data.extend_from_slice(bytes);
        self.offsets.push(end as i32);
        self.validity.push(true);
        Ok(())
    }

    /// Append a null (an empty value with the validity bit cleared).
    pub fn append_null(&mut self) {
        self.offsets.push(self.data.len() as i32);
        self.validity.push(false);
    }

    /// Append `Some(v)` or a null.
    pub fn append_option(&mut self, value: Option<impl AsRef<[u8]>>) -> Result<()> {
        if let Some(v) = value {
            self.append_value(v)
        } else {
            self.append_null();
            Ok(())
        }
    }

    /// Finish.
    pub fn finish(self) -> Result<BinaryArray> {
        BinaryArray::try_new(
            Buffer::from_vec(self.offsets),
            Buffer::from_vec(self.data),
            self.validity.finish(),
        )
    }
}

impl Default for BinaryBuilder {
    fn default() -> Self {
        BinaryBuilder::new()
    }
}

/// Builder for [`StringArray`].
///
/// Takes `&str`, so the UTF-8 invariant holds by construction and `finish` does
/// not need to re-validate.
#[derive(Debug, Clone, Default)]
pub struct StringBuilder {
    inner: BinaryBuilder,
}

impl StringBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        StringBuilder {
            inner: BinaryBuilder::new(),
        }
    }

    /// A builder pre-sized for `values` strings totalling `bytes` bytes.
    pub fn with_capacity(values: usize, bytes: usize) -> Self {
        StringBuilder {
            inner: BinaryBuilder::with_capacity(values, bytes),
        }
    }

    /// Number of appended slots.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if nothing has been appended.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Append a string.
    pub fn append_value(&mut self, value: impl AsRef<str>) -> Result<()> {
        self.inner.append_value(value.as_ref().as_bytes())
    }

    /// Append a null.
    pub fn append_null(&mut self) {
        self.inner.append_null();
    }

    /// Append `Some(v)` or a null.
    pub fn append_option(&mut self, value: Option<impl AsRef<str>>) -> Result<()> {
        if let Some(v) = value {
            self.append_value(v)
        } else {
            self.append_null();
            Ok(())
        }
    }

    /// Finish. Skips UTF-8 validation, which the `&str` inputs already provide.
    pub fn finish(self) -> Result<StringArray> {
        let binary = self.inner.finish()?;
        // Every value came from a `&str`, so re-validating would be O(bytes) of
        // provably redundant work. Reconstructing through `try_new` here would
        // be the "safe" choice; instead we assert the invariant in debug builds
        // and keep release builds linear-free.
        Ok(StringArray::from_binary_unchecked(binary))
    }
}

/// A type-erased builder that appends [`Value`]s.
///
/// Used where the type is only known at runtime: decoding a segment, building a
/// batch from parsed literals, tests. Per-value dispatch makes it slower than
/// the typed builders, so hot paths should use those.
#[derive(Debug)]
pub enum ColumnBuilder {
    /// Builds a `NULL`-typed column; only counts rows.
    Null(usize),
    /// Boolean column.
    Boolean(BooleanBuilder),
    /// `Int8` column.
    Int8(PrimitiveBuilder<i8>),
    /// `Int16` column.
    Int16(PrimitiveBuilder<i16>),
    /// `Int32` column.
    Int32(PrimitiveBuilder<i32>),
    /// `Int64` column.
    Int64(PrimitiveBuilder<i64>),
    /// `UInt8` column.
    UInt8(PrimitiveBuilder<u8>),
    /// `UInt16` column.
    UInt16(PrimitiveBuilder<u16>),
    /// `UInt32` column.
    UInt32(PrimitiveBuilder<u32>),
    /// `UInt64` column.
    UInt64(PrimitiveBuilder<u64>),
    /// `Float32` column.
    Float32(PrimitiveBuilder<f32>),
    /// `Float64` column.
    Float64(PrimitiveBuilder<f64>),
    /// `Utf8` column.
    Utf8(StringBuilder),
    /// `Binary` column.
    Binary(BinaryBuilder),
    /// `Date32` column.
    Date32(PrimitiveBuilder<i32>),
    /// `Timestamp` column with its unit.
    Timestamp(PrimitiveBuilder<i64>, TimeUnit),
    /// `Decimal128` column with its precision and scale.
    Decimal128(PrimitiveBuilder<i128>, u8, i8),
}

impl ColumnBuilder {
    /// A builder for `data_type`.
    pub fn new(data_type: &DataType) -> Result<ColumnBuilder> {
        ColumnBuilder::with_capacity(data_type, 0)
    }

    /// A builder for `data_type`, pre-sized for `capacity` rows.
    pub fn with_capacity(data_type: &DataType, capacity: usize) -> Result<ColumnBuilder> {
        Ok(match data_type {
            DataType::Null => ColumnBuilder::Null(0),
            DataType::Boolean => ColumnBuilder::Boolean(BooleanBuilder::with_capacity(capacity)),
            DataType::Int8 => ColumnBuilder::Int8(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Int16 => ColumnBuilder::Int16(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Int32 => ColumnBuilder::Int32(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Int64 => ColumnBuilder::Int64(PrimitiveBuilder::with_capacity(capacity)),
            DataType::UInt8 => ColumnBuilder::UInt8(PrimitiveBuilder::with_capacity(capacity)),
            DataType::UInt16 => ColumnBuilder::UInt16(PrimitiveBuilder::with_capacity(capacity)),
            DataType::UInt32 => ColumnBuilder::UInt32(PrimitiveBuilder::with_capacity(capacity)),
            DataType::UInt64 => ColumnBuilder::UInt64(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Float32 => ColumnBuilder::Float32(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Float64 => ColumnBuilder::Float64(PrimitiveBuilder::with_capacity(capacity)),
            // 16 bytes per string is a guess that avoids the first few growths
            // without over-committing for short keys.
            DataType::Utf8 => {
                ColumnBuilder::Utf8(StringBuilder::with_capacity(capacity, capacity * 16))
            }
            DataType::Binary => {
                ColumnBuilder::Binary(BinaryBuilder::with_capacity(capacity, capacity * 16))
            }
            DataType::Date32 => ColumnBuilder::Date32(PrimitiveBuilder::with_capacity(capacity)),
            DataType::Timestamp(u) => {
                ColumnBuilder::Timestamp(PrimitiveBuilder::with_capacity(capacity), *u)
            }
            DataType::Decimal128(p, s) => {
                ColumnBuilder::Decimal128(PrimitiveBuilder::with_capacity(capacity), *p, *s)
            }
        })
    }

    /// The type this builder produces.
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnBuilder::Null(_) => DataType::Null,
            ColumnBuilder::Boolean(_) => DataType::Boolean,
            ColumnBuilder::Int8(_) => DataType::Int8,
            ColumnBuilder::Int16(_) => DataType::Int16,
            ColumnBuilder::Int32(_) => DataType::Int32,
            ColumnBuilder::Int64(_) => DataType::Int64,
            ColumnBuilder::UInt8(_) => DataType::UInt8,
            ColumnBuilder::UInt16(_) => DataType::UInt16,
            ColumnBuilder::UInt32(_) => DataType::UInt32,
            ColumnBuilder::UInt64(_) => DataType::UInt64,
            ColumnBuilder::Float32(_) => DataType::Float32,
            ColumnBuilder::Float64(_) => DataType::Float64,
            ColumnBuilder::Utf8(_) => DataType::Utf8,
            ColumnBuilder::Binary(_) => DataType::Binary,
            ColumnBuilder::Date32(_) => DataType::Date32,
            ColumnBuilder::Timestamp(_, u) => DataType::Timestamp(*u),
            ColumnBuilder::Decimal128(_, p, s) => DataType::Decimal128(*p, *s),
        }
    }

    /// Number of appended rows.
    pub fn len(&self) -> usize {
        match self {
            ColumnBuilder::Null(n) => *n,
            ColumnBuilder::Boolean(b) => b.len(),
            ColumnBuilder::Int8(b) => b.len(),
            ColumnBuilder::Int16(b) => b.len(),
            ColumnBuilder::Int32(b) | ColumnBuilder::Date32(b) => b.len(),
            ColumnBuilder::Int64(b) | ColumnBuilder::Timestamp(b, _) => b.len(),
            ColumnBuilder::UInt8(b) => b.len(),
            ColumnBuilder::UInt16(b) => b.len(),
            ColumnBuilder::UInt32(b) => b.len(),
            ColumnBuilder::UInt64(b) => b.len(),
            ColumnBuilder::Float32(b) => b.len(),
            ColumnBuilder::Float64(b) => b.len(),
            ColumnBuilder::Utf8(b) => b.len(),
            ColumnBuilder::Binary(b) => b.len(),
            ColumnBuilder::Decimal128(b, _, _) => b.len(),
        }
    }

    /// `true` if nothing has been appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a null.
    pub fn append_null(&mut self) {
        match self {
            ColumnBuilder::Null(n) => *n += 1,
            ColumnBuilder::Boolean(b) => b.append_null(),
            ColumnBuilder::Int8(b) => b.append_null(),
            ColumnBuilder::Int16(b) => b.append_null(),
            ColumnBuilder::Int32(b) | ColumnBuilder::Date32(b) => b.append_null(),
            ColumnBuilder::Int64(b) | ColumnBuilder::Timestamp(b, _) => b.append_null(),
            ColumnBuilder::UInt8(b) => b.append_null(),
            ColumnBuilder::UInt16(b) => b.append_null(),
            ColumnBuilder::UInt32(b) => b.append_null(),
            ColumnBuilder::UInt64(b) => b.append_null(),
            ColumnBuilder::Float32(b) => b.append_null(),
            ColumnBuilder::Float64(b) => b.append_null(),
            ColumnBuilder::Utf8(b) => b.append_null(),
            ColumnBuilder::Binary(b) => b.append_null(),
            ColumnBuilder::Decimal128(b, _, _) => b.append_null(),
        }
    }

    /// Append a [`Value`], which must have this builder's type (or be `Null`).
    ///
    /// # Errors
    ///
    /// [`BasaltError::Type`] naming both types on a mismatch. Decimals are also
    /// checked for a matching scale: appending `1.0` to a `DECIMAL(10, 3)`
    /// column would otherwise store `1` and silently mean `0.001`.
    pub fn append_value(&mut self, value: &Value) -> Result<()> {
        if value.is_null() {
            self.append_null();
            return Ok(());
        }
        let (source, target) = (value.data_type(), self.data_type());
        let mismatch = move || {
            BasaltError::type_error(format!(
                "cannot append a {source} value to a {target} column"
            ))
        };
        match (self, value) {
            // `ColumnBuilder::Null` accepts nothing but nulls, which returned
            // above, so it falls through to the wildcard arm.
            (ColumnBuilder::Boolean(b), Value::Boolean(v)) => b.append_value(*v),
            (ColumnBuilder::Int8(b), Value::Int8(v)) => b.append_value(*v),
            (ColumnBuilder::Int16(b), Value::Int16(v)) => b.append_value(*v),
            (ColumnBuilder::Int32(b), Value::Int32(v)) => b.append_value(*v),
            (ColumnBuilder::Int64(b), Value::Int64(v)) => b.append_value(*v),
            (ColumnBuilder::UInt8(b), Value::UInt8(v)) => b.append_value(*v),
            (ColumnBuilder::UInt16(b), Value::UInt16(v)) => b.append_value(*v),
            (ColumnBuilder::UInt32(b), Value::UInt32(v)) => b.append_value(*v),
            (ColumnBuilder::UInt64(b), Value::UInt64(v)) => b.append_value(*v),
            (ColumnBuilder::Float32(b), Value::Float32(v)) => b.append_value(*v),
            (ColumnBuilder::Float64(b), Value::Float64(v)) => b.append_value(*v),
            (ColumnBuilder::Utf8(b), Value::Utf8(v)) => b.append_value(v)?,
            (ColumnBuilder::Binary(b), Value::Binary(v)) => b.append_value(v)?,
            (ColumnBuilder::Date32(b), Value::Date32(v)) => b.append_value(*v),
            (ColumnBuilder::Timestamp(b, u), Value::Timestamp(v, vu)) if u == vu => {
                b.append_value(*v);
            }
            (ColumnBuilder::Decimal128(b, _, s), Value::Decimal128(v, _, vs)) if s == vs => {
                b.append_value(*v);
            }
            _ => return Err(mismatch()),
        }
        Ok(())
    }

    /// Finish, producing a [`ColumnVector`].
    pub fn finish(self) -> ColumnVector {
        match self {
            ColumnBuilder::Null(n) => ColumnVector::Null(NullArray::new(n)),
            ColumnBuilder::Boolean(b) => ColumnVector::Boolean(b.finish()),
            ColumnBuilder::Int8(b) => ColumnVector::Int8(b.finish()),
            ColumnBuilder::Int16(b) => ColumnVector::Int16(b.finish()),
            ColumnBuilder::Int32(b) => ColumnVector::Int32(b.finish()),
            ColumnBuilder::Int64(b) => ColumnVector::Int64(b.finish()),
            ColumnBuilder::UInt8(b) => ColumnVector::UInt8(b.finish()),
            ColumnBuilder::UInt16(b) => ColumnVector::UInt16(b.finish()),
            ColumnBuilder::UInt32(b) => ColumnVector::UInt32(b.finish()),
            ColumnBuilder::UInt64(b) => ColumnVector::UInt64(b.finish()),
            ColumnBuilder::Float32(b) => ColumnVector::Float32(b.finish()),
            ColumnBuilder::Float64(b) => ColumnVector::Float64(b.finish()),
            // `finish` on the var-length builders can only fail on invariants
            // this builder maintains itself, so an empty array is an
            // unreachable fallback that keeps `finish` infallible for callers.
            ColumnBuilder::Utf8(b) => ColumnVector::Utf8(
                b.finish()
                    .unwrap_or_else(|_| StringArray::from_strs::<&str>(&[])),
            ),
            ColumnBuilder::Binary(b) => ColumnVector::Binary(
                b.finish()
                    .unwrap_or_else(|_| BinaryArray::from_slices::<&[u8]>(&[])),
            ),
            ColumnBuilder::Date32(b) => ColumnVector::Date32(b.finish()),
            ColumnBuilder::Timestamp(b, u) => ColumnVector::Timestamp(b.finish(), u),
            ColumnBuilder::Decimal128(b, p, s) => ColumnVector::Decimal128(b.finish(), p, s),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn validity_is_lazy() {
        let mut b = PrimitiveBuilder::<i32>::new();
        b.append_value(1);
        b.append_value(2);
        let arr = b.finish();
        assert!(arr.validity().is_none(), "no nulls means no bitmap");

        let mut b = PrimitiveBuilder::<i32>::new();
        b.append_value(1);
        b.append_null();
        b.append_value(3);
        let arr = b.finish();
        assert_eq!(arr.null_count(), 1);
        assert_eq!(arr.iter().collect::<Vec<_>>(), vec![Some(1), None, Some(3)]);
    }

    #[test]
    fn back_fill_after_many_valid_values() {
        let mut b = PrimitiveBuilder::<u16>::with_capacity(200);
        for i in 0..200u16 {
            b.append_value(i);
        }
        b.append_null();
        let arr = b.finish();
        assert_eq!(arr.len(), 201);
        assert_eq!(arr.null_count(), 1);
        for i in 0..200 {
            assert_eq!(arr.get(i), Some(i as u16));
        }
        assert_eq!(arr.get(200), None);
    }

    #[test]
    fn boolean_and_string_builders() {
        let mut b = BooleanBuilder::with_capacity(4);
        b.append_value(true);
        b.append_null();
        b.append_option(Some(false));
        b.append_option(None);
        assert_eq!(b.len(), 4);
        assert!(!b.is_empty());
        let arr = b.finish();
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(false), None]
        );

        let mut s = StringBuilder::with_capacity(2, 8);
        s.append_value("hello").unwrap();
        s.append_null();
        s.append_option(Some("x")).unwrap();
        s.append_option(Option::<&str>::None).unwrap();
        assert_eq!(s.len(), 4);
        let arr = s.finish().unwrap();
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            vec![Some("hello"), None, Some("x"), None]
        );
    }

    #[test]
    fn binary_builder() {
        let mut b = BinaryBuilder::new();
        assert!(b.is_empty());
        b.append_value([1u8, 2]).unwrap();
        b.append_option(Some([3u8])).unwrap();
        b.append_option(Option::<[u8; 1]>::None).unwrap();
        let arr = b.finish().unwrap();
        assert_eq!(
            arr.iter().collect::<Vec<_>>(),
            vec![Some(&[1u8, 2][..]), Some(&[3u8][..]), None]
        );
    }

    #[test]
    fn column_builder_round_trip_for_every_type() {
        let cases: Vec<(DataType, Value)> = vec![
            (DataType::Boolean, Value::Boolean(true)),
            (DataType::Int8, Value::Int8(-1)),
            (DataType::Int16, Value::Int16(-2)),
            (DataType::Int32, Value::Int32(-3)),
            (DataType::Int64, Value::Int64(-4)),
            (DataType::UInt8, Value::UInt8(1)),
            (DataType::UInt16, Value::UInt16(2)),
            (DataType::UInt32, Value::UInt32(3)),
            (DataType::UInt64, Value::UInt64(4)),
            (DataType::Float32, Value::Float32(1.5)),
            (DataType::Float64, Value::Float64(2.5)),
            (DataType::Utf8, Value::Utf8("hi".into())),
            (DataType::Binary, Value::Binary(vec![7, 8])),
            (DataType::Date32, Value::Date32(19_000)),
            (
                DataType::Timestamp(TimeUnit::Microsecond),
                Value::Timestamp(1_700_000_000_000_000, TimeUnit::Microsecond),
            ),
            (DataType::Decimal128(10, 2), Value::Decimal128(1234, 10, 2)),
        ];
        for (dt, v) in cases {
            let mut b = ColumnBuilder::with_capacity(&dt, 4).unwrap();
            assert_eq!(b.data_type(), dt);
            assert!(b.is_empty());
            b.append_value(&v).unwrap();
            b.append_null();
            b.append_value(&Value::Null).unwrap();
            assert_eq!(b.len(), 3);
            let col = b.finish();
            assert_eq!(col.data_type(), dt);
            assert_eq!(col.len(), 3);
            assert_eq!(col.value(0), v, "{dt}");
            assert_eq!(col.null_count(), 2, "{dt}");
        }
    }

    #[test]
    fn column_builder_rejects_mismatches() {
        let mut b = ColumnBuilder::new(&DataType::Int32).unwrap();
        let err = b.append_value(&Value::Int64(1)).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot append a INT64 value to a INT32 column"),
            "{err}"
        );

        let mut b = ColumnBuilder::new(&DataType::Decimal128(10, 2)).unwrap();
        assert!(
            b.append_value(&Value::Decimal128(1, 10, 3)).is_err(),
            "scale must match"
        );
        assert!(b.append_value(&Value::Decimal128(1, 10, 2)).is_ok());

        let mut b = ColumnBuilder::new(&DataType::Timestamp(TimeUnit::Second)).unwrap();
        assert!(b
            .append_value(&Value::Timestamp(1, TimeUnit::Nanosecond))
            .is_err());

        let mut b = ColumnBuilder::new(&DataType::Null).unwrap();
        assert!(b.append_value(&Value::Int32(1)).is_err());
        b.append_null();
        assert_eq!(b.finish().len(), 1);
    }

    #[test]
    fn extend_and_reserve() {
        let mut b = PrimitiveBuilder::<i64>::new();
        b.reserve(10);
        b.extend([Some(1i64), None, Some(3)]);
        let arr = b.finish();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 1);
        assert!(PrimitiveBuilder::<i64>::default().is_empty());
    }
}

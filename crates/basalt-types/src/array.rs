//! Columnar arrays: contiguous values plus a bitmap validity mask.

use std::fmt;

use basalt_common::{BasaltError, Result};

use crate::{Bitmap, Buffer, DataType, TimeUnit, Value};

/// A type that can live in a [`PrimitiveArray`]'s value buffer.
///
/// Sealed in practice by the set of impls below: these are exactly the physical
/// layouts Basalt stores. There is deliberately no `DATA_TYPE` associated
/// constant, because the mapping is not injective — `Int32` and `Date32` share
/// `i32` — so the *logical* type lives on the [`ColumnVector`] variant instead.
pub trait NativeType: Copy + fmt::Debug + Default + PartialEq + Send + Sync + 'static {
    /// Width of one value in bytes.
    const WIDTH: usize;

    /// Equality used when comparing arrays.
    ///
    /// Defaults to `PartialEq`, but the float types override it to compare bit
    /// patterns so that `NaN == NaN`. Without that, two arrays holding the same
    /// bytes would compare unequal, and a round trip through the storage layer
    /// could not be asserted. It also keeps array equality consistent with
    /// [`Value`]'s total order, which treats floats the same way.
    #[inline]
    fn total_eq(self, other: Self) -> bool {
        self == other
    }
}

macro_rules! impl_native {
    ($($t:ty),*) => { $(impl NativeType for $t { const WIDTH: usize = std::mem::size_of::<$t>(); })* };
}
impl_native!(i8, i16, i32, i64, u8, u16, u32, u64, i128);

impl NativeType for f32 {
    const WIDTH: usize = 4;
    #[inline]
    fn total_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl NativeType for f64 {
    const WIDTH: usize = 8;
    #[inline]
    fn total_eq(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

/// A fixed-width column of `T` with an optional validity mask.
///
/// The values buffer is *dense*: it stores a `T` for every slot, including the
/// null ones (whose contents are unspecified). Densely storing nulls is what
/// makes the array a plain `&[T]` that the compiler can auto-vectorise; the
/// alternative — compacting valid values and indirecting through a rank
/// operation — costs a popcount per access on every read to save memory on a
/// case (mostly-null columns) that the encoding layer handles far better on
/// disk anyway.
///
/// `validity` is `None` when the column has no nulls at all. That is not just an
/// optimisation: it lets scan loops branch once per array instead of once per
/// value.
#[derive(Clone)]
pub struct PrimitiveArray<T: NativeType> {
    values: Buffer<T>,
    validity: Option<Bitmap>,
}

impl<T: NativeType> PrimitiveArray<T> {
    /// Build from a values buffer and an optional validity mask (1 = valid).
    pub fn try_new(values: Buffer<T>, validity: Option<Bitmap>) -> Result<Self> {
        if let Some(v) = &validity {
            if v.len() != values.len() {
                return Err(BasaltError::schema(format!(
                    "validity mask has {} bits but the values buffer has {} entries",
                    v.len(),
                    values.len()
                )));
            }
        }
        Ok(PrimitiveArray { values, validity })
    }

    /// Build a null-free array from a `Vec`.
    pub fn from_vec(values: Vec<T>) -> Self {
        PrimitiveArray {
            values: Buffer::from_vec(values),
            validity: None,
        }
    }

    /// Build from optional values, materialising a validity mask only if some
    /// value is `None`.
    pub fn from_opt_vec(values: Vec<Option<T>>) -> Self {
        let has_null = values.iter().any(Option::is_none);
        let validity = if has_null {
            Some(values.iter().map(Option::is_some).collect::<Bitmap>())
        } else {
            None
        };
        let raw = values
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect::<Vec<_>>();
        PrimitiveArray {
            values: Buffer::from_vec(raw),
            validity,
        }
    }

    /// Number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if there are no slots.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The dense values, including undefined entries under nulls.
    #[inline]
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    /// The values buffer, for zero-copy hand-off to the encoding layer.
    #[inline]
    pub fn values_buffer(&self) -> &Buffer<T> {
        &self.values
    }

    /// The validity mask, or `None` when the array has no nulls.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// `true` if slot `i` is null.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| !v.get(i))
    }

    /// `true` if slot `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// Number of nulls. O(len / 64) — not cached, because the callers that need
    /// it repeatedly (statistics, writers) already keep their own counter.
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::count_zeros)
    }

    /// The raw value at `i`, regardless of validity.
    ///
    /// # Panics
    ///
    /// If `i >= len`.
    #[inline]
    pub fn value(&self, i: usize) -> T {
        self.values.as_slice()[i]
    }

    /// The value at `i`, or `None` if it is null.
    #[inline]
    pub fn get(&self, i: usize) -> Option<T> {
        if self.is_valid(i) {
            Some(self.value(i))
        } else {
            None
        }
    }

    /// Iterate over optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<T>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// A zero-copy view of `len` slots starting at `offset`.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        PrimitiveArray {
            values: self.values.slice(offset, len),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
        }
    }
}

impl<T: NativeType> PartialEq for PrimitiveArray<T> {
    /// Semantic comparison: same length, same null positions, same values at
    /// the valid positions. Bytes under a null are explicitly unspecified, so
    /// they must not participate.
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        (0..self.len()).all(|i| match (self.get(i), other.get(i)) {
            (Some(a), Some(b)) => a.total_eq(b),
            (None, None) => true,
            _ => false,
        })
    }
}

impl<T: NativeType> fmt::Debug for PrimitiveArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries((0..self.len()).map(|i| self.get(i)))
            .finish()
    }
}

/// A column of booleans, stored one bit per value.
///
/// Both the values and the validity mask are [`Bitmap`]s, so a million-row
/// boolean column is 125 KiB, not 1 MiB.
#[derive(Clone)]
pub struct BooleanArray {
    values: Bitmap,
    validity: Option<Bitmap>,
}

impl BooleanArray {
    /// Build from a values bitmap and an optional validity mask.
    pub fn try_new(values: Bitmap, validity: Option<Bitmap>) -> Result<Self> {
        if let Some(v) = &validity {
            if v.len() != values.len() {
                return Err(BasaltError::schema(format!(
                    "validity mask has {} bits but the values bitmap has {}",
                    v.len(),
                    values.len()
                )));
            }
        }
        Ok(BooleanArray { values, validity })
    }

    /// Build a null-free array.
    pub fn from_bools(values: &[bool]) -> Self {
        BooleanArray {
            values: Bitmap::from_bools(values),
            validity: None,
        }
    }

    /// Build from optional booleans.
    pub fn from_opt_vec(values: Vec<Option<bool>>) -> Self {
        let has_null = values.iter().any(Option::is_none);
        let validity = if has_null {
            Some(values.iter().map(Option::is_some).collect::<Bitmap>())
        } else {
            None
        };
        let bits = values
            .into_iter()
            .map(|v| v.unwrap_or(false))
            .collect::<Bitmap>();
        BooleanArray {
            values: bits,
            validity,
        }
    }

    /// Number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The packed values.
    #[inline]
    pub fn values(&self) -> &Bitmap {
        &self.values
    }

    /// The validity mask, if any.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// `true` if slot `i` is null.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| !v.get(i))
    }

    /// `true` if slot `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// Number of nulls. O(len / 64).
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::count_zeros)
    }

    /// The raw bit at `i`.
    #[inline]
    pub fn value(&self, i: usize) -> bool {
        self.values.get(i)
    }

    /// The value at `i`, or `None` if null.
    #[inline]
    pub fn get(&self, i: usize) -> Option<bool> {
        if self.is_valid(i) {
            Some(self.value(i))
        } else {
            None
        }
    }

    /// Iterate over optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<bool>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// A zero-copy view.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        BooleanArray {
            values: self.values.slice(offset, len),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
        }
    }
}

impl PartialEq for BooleanArray {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.get(i) == other.get(i))
    }
}

impl fmt::Debug for BooleanArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A column of variable-length byte strings.
///
/// Layout is `offsets` (`len + 1` monotonically non-decreasing `i32`s) plus one
/// contiguous `data` buffer. **Not** `Vec<Vec<u8>>`, and this matters enough to
/// spell out:
///
/// * `Vec<Vec<u8>>` needs one heap allocation *per value*. A ten-million-row
///   string column becomes ten million `malloc`s at build time and ten million
///   `free`s at drop time, which alone can dominate a scan.
/// * Every access is a pointer chase into an unpredictable cache line, so a scan
///   over `Vec<Vec<u8>>` is bounded by memory latency instead of bandwidth. With
///   offsets + one buffer, the data is read sequentially and the hardware
///   prefetcher does its job.
/// * The three-word `Vec` header per value costs 24 bytes of overhead, which for
///   short strings exceeds the payload.
/// * Only a contiguous buffer can be handed to a codec, written to disk, or
///   memory-mapped back without a serialisation pass.
///
/// Slicing keeps the *whole* data buffer and only narrows the offsets window, so
/// it stays O(1); offsets are absolute indices into `data`. Call
/// [`BinaryArray::compact`] to shed the untouched bytes.
#[derive(Clone)]
pub struct BinaryArray {
    offsets: Buffer<i32>,
    data: Buffer<u8>,
    validity: Option<Bitmap>,
}

impl BinaryArray {
    /// Build from raw parts, validating the offset invariants.
    ///
    /// Checks that `offsets` is non-empty, non-decreasing, non-negative and
    /// within `data`. O(len).
    pub fn try_new(
        offsets: Buffer<i32>,
        data: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Result<Self> {
        if offsets.is_empty() {
            return Err(BasaltError::schema(
                "binary array offsets must have at least one entry (len + 1)",
            ));
        }
        let len = offsets.len() - 1;
        if let Some(v) = &validity {
            if v.len() != len {
                return Err(BasaltError::schema(format!(
                    "validity mask has {} bits but the array has {len} values",
                    v.len()
                )));
            }
        }
        let slice = offsets.as_slice();
        if slice[0] < 0 {
            return Err(BasaltError::schema(format!(
                "negative first offset {}",
                slice[0]
            )));
        }
        for w in slice.windows(2) {
            if w[1] < w[0] {
                return Err(BasaltError::schema(format!(
                    "offsets must be non-decreasing, found {} after {}",
                    w[1], w[0]
                )));
            }
        }
        let last = slice[len] as usize;
        if last > data.len() {
            return Err(BasaltError::schema(format!(
                "last offset {last} exceeds the {}-byte data buffer",
                data.len()
            )));
        }
        Ok(BinaryArray {
            offsets,
            data,
            validity,
        })
    }

    /// Build a null-free array from byte slices.
    pub fn from_slices<S: AsRef<[u8]>>(values: &[S]) -> Self {
        let total: usize = values.iter().map(|v| v.as_ref().len()).sum();
        let mut offsets = Vec::with_capacity(values.len() + 1);
        let mut data = Vec::with_capacity(total);
        offsets.push(0i32);
        for v in values {
            data.extend_from_slice(v.as_ref());
            offsets.push(data.len() as i32);
        }
        BinaryArray {
            offsets: Buffer::from_vec(offsets),
            data: Buffer::from_vec(data),
            validity: None,
        }
    }

    /// Number of values.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `len + 1` offsets.
    #[inline]
    pub fn offsets(&self) -> &[i32] {
        self.offsets.as_slice()
    }

    /// The shared data buffer. Offsets index into this directly.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// The validity mask, if any.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.validity.as_ref()
    }

    /// `true` if slot `i` is null.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.validity.as_ref().is_some_and(|v| !v.get(i))
    }

    /// `true` if slot `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// Number of nulls. O(len / 64).
    pub fn null_count(&self) -> usize {
        self.validity.as_ref().map_or(0, Bitmap::count_zeros)
    }

    /// The bytes at `i` (empty for a null slot).
    ///
    /// # Panics
    ///
    /// If `i >= len`.
    #[inline]
    pub fn value(&self, i: usize) -> &[u8] {
        let offsets = self.offsets.as_slice();
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        &self.data.as_slice()[start..end]
    }

    /// The bytes at `i`, or `None` if null.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        if self.is_valid(i) {
            Some(self.value(i))
        } else {
            None
        }
    }

    /// Iterate over optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<&[u8]>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Total bytes referenced by this view (not the size of the data buffer).
    pub fn value_bytes(&self) -> usize {
        let offsets = self.offsets.as_slice();
        (offsets[self.len()] - offsets[0]).max(0) as usize
    }

    /// A zero-copy view: narrows the offsets window, keeps the data buffer.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        BinaryArray {
            offsets: self.offsets.slice(offset, len + 1),
            data: self.data.clone(),
            validity: self.validity.as_ref().map(|v| v.slice(offset, len)),
        }
    }

    /// Copy the referenced bytes into a fresh buffer, releasing the rest.
    ///
    /// O(value_bytes). Worth doing when a small slice of a huge column outlives
    /// the column.
    pub fn compact(&self) -> Self {
        let values: Vec<&[u8]> = (0..self.len()).map(|i| self.value(i)).collect();
        let mut compacted = BinaryArray::from_slices(&values);
        compacted.validity.clone_from(&self.validity);
        compacted
    }
}

impl PartialEq for BinaryArray {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.get(i) == other.get(i))
    }
}

impl fmt::Debug for BinaryArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A column of UTF-8 strings.
///
/// Same layout as [`BinaryArray`] with one extra invariant, checked once at
/// construction: every value is valid UTF-8. Because slice boundaries always
/// fall on value boundaries, slicing preserves it.
#[derive(Clone, PartialEq)]
pub struct StringArray {
    inner: BinaryArray,
}

impl StringArray {
    /// Build from raw parts, validating both the offsets and UTF-8. O(bytes).
    pub fn try_new(
        offsets: Buffer<i32>,
        data: Buffer<u8>,
        validity: Option<Bitmap>,
    ) -> Result<Self> {
        let inner = BinaryArray::try_new(offsets, data, validity)?;
        for i in 0..inner.len() {
            let bytes = inner.value(i);
            if let Err(e) = std::str::from_utf8(bytes) {
                return Err(BasaltError::schema(format!(
                    "value {i} of a UTF8 column is not valid UTF-8 (invalid byte at {})",
                    e.valid_up_to()
                )));
            }
        }
        Ok(StringArray { inner })
    }

    /// Build a null-free array from strings.
    pub fn from_strs<S: AsRef<str>>(values: &[S]) -> Self {
        let refs: Vec<&[u8]> = values.iter().map(|s| s.as_ref().as_bytes()).collect();
        StringArray {
            inner: BinaryArray::from_slices(&refs),
        }
    }

    /// Wrap a [`BinaryArray`] that is already known to hold UTF-8.
    ///
    /// Crate-internal and reachable only from [`StringBuilder`], whose input is
    /// `&str`, so the invariant holds by typing rather than by inspection. This
    /// exists purely to avoid an O(bytes) revalidation pass on every `finish`.
    ///
    /// [`StringBuilder`]: crate::StringBuilder
    pub(crate) fn from_binary_unchecked(inner: BinaryArray) -> StringArray {
        debug_assert!(
            (0..inner.len()).all(|i| std::str::from_utf8(inner.value(i)).is_ok()),
            "from_binary_unchecked called with non-UTF-8 data"
        );
        StringArray { inner }
    }

    /// The underlying byte array.
    #[inline]
    pub fn as_binary(&self) -> &BinaryArray {
        &self.inner
    }

    /// Number of values.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The `len + 1` offsets.
    #[inline]
    pub fn offsets(&self) -> &[i32] {
        self.inner.offsets()
    }

    /// The shared data buffer.
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.inner.data()
    }

    /// The validity mask, if any.
    #[inline]
    pub fn validity(&self) -> Option<&Bitmap> {
        self.inner.validity()
    }

    /// `true` if slot `i` is null.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        self.inner.is_null(i)
    }

    /// `true` if slot `i` holds a value.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        self.inner.is_valid(i)
    }

    /// Number of nulls. O(len / 64).
    pub fn null_count(&self) -> usize {
        self.inner.null_count()
    }

    /// The string at `i`.
    ///
    /// # Panics
    ///
    /// If `i >= len`. Never panics on the UTF-8 decode: `try_new` and the
    /// builders are the only ways to make a `StringArray`, and both validate
    /// every value, so the `expect` below is unreachable by construction.
    #[inline]
    pub fn value(&self, i: usize) -> &str {
        std::str::from_utf8(self.inner.value(i))
            .expect("StringArray invariant: every value was UTF-8 validated at construction")
    }

    /// The string at `i`, or `None` if null.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&str> {
        if self.is_valid(i) {
            Some(self.value(i))
        } else {
            None
        }
    }

    /// Iterate over optional values.
    pub fn iter(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// Total bytes referenced by this view.
    pub fn value_bytes(&self) -> usize {
        self.inner.value_bytes()
    }

    /// A zero-copy view.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        StringArray {
            inner: self.inner.slice(offset, len),
        }
    }

    /// Copy the referenced bytes into a fresh buffer.
    pub fn compact(&self) -> Self {
        StringArray {
            inner: self.inner.compact(),
        }
    }
}

impl fmt::Debug for StringArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// A column whose every value is `NULL`.
///
/// Stores only a length. This is the array behind `SELECT NULL`, and keeping it
/// as its own type means the rest of the engine never has to allocate a buffer
/// of values nobody will read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NullArray {
    len: usize,
}

impl NullArray {
    /// A column of `len` nulls.
    pub const fn new(len: usize) -> Self {
        NullArray { len }
    }

    /// Number of slots.
    #[inline]
    pub const fn len(self) -> usize {
        self.len
    }

    /// `true` if empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Every slot is null, so this equals [`NullArray::len`]. O(1).
    #[inline]
    pub const fn null_count(self) -> usize {
        self.len
    }

    /// Always `true`.
    #[inline]
    pub const fn is_null(self, _index: usize) -> bool {
        true
    }

    /// Always `false`.
    #[inline]
    pub const fn is_valid(self, _index: usize) -> bool {
        false
    }

    /// A view, i.e. a shorter `NullArray`.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds.
    pub fn slice(self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "NullArray::slice out of bounds");
        NullArray { len }
    }
}

/// A type-tagged column of values: the unit the whole engine passes around.
///
/// # Enum, not `dyn Array`
///
/// Basalt uses a closed enum rather than a `Box<dyn Array>` trait object, and
/// the choice is load-bearing:
///
/// * **Static dispatch.** Every kernel (filter, compare, aggregate) matches once
///   per *array*, then runs a monomorphised loop over `&[T]`. With trait objects
///   the compiler cannot see through the vtable, so the inner loop keeps an
///   indirect call per value and never vectorises.
/// * **No downcasting.** The Arrow-style `as_any().downcast_ref::<Int32Array>()`
///   pattern turns a type error into a runtime `Option`. A `match` is exhaustive:
///   adding `Decimal256` later is a compile error at every site that must handle
///   it, which is exactly what you want in an engine.
/// * **The set really is closed.** Physical layouts are a property of the
///   storage format, not an extension point. A user-defined type would be a
///   *logical* type over an existing physical layout.
///
/// The cost is that the enum is as large as its largest variant and that adding
/// a type touches many matches. Both are acceptable; the matches are mechanical
/// and the compiler finds them all.
#[derive(Clone, PartialEq)]
pub enum ColumnVector {
    /// All nulls.
    Null(NullArray),
    /// Booleans, bit-packed.
    Boolean(BooleanArray),
    /// 8-bit signed integers.
    Int8(PrimitiveArray<i8>),
    /// 16-bit signed integers.
    Int16(PrimitiveArray<i16>),
    /// 32-bit signed integers.
    Int32(PrimitiveArray<i32>),
    /// 64-bit signed integers.
    Int64(PrimitiveArray<i64>),
    /// 8-bit unsigned integers.
    UInt8(PrimitiveArray<u8>),
    /// 16-bit unsigned integers.
    UInt16(PrimitiveArray<u16>),
    /// 32-bit unsigned integers.
    UInt32(PrimitiveArray<u32>),
    /// 64-bit unsigned integers.
    UInt64(PrimitiveArray<u64>),
    /// binary32 floats.
    Float32(PrimitiveArray<f32>),
    /// binary64 floats.
    Float64(PrimitiveArray<f64>),
    /// UTF-8 strings.
    Utf8(StringArray),
    /// Byte strings.
    Binary(BinaryArray),
    /// Days since the Unix epoch.
    Date32(PrimitiveArray<i32>),
    /// Ticks since the Unix epoch, with their unit.
    Timestamp(PrimitiveArray<i64>, TimeUnit),
    /// Fixed-point decimals, with precision and scale.
    Decimal128(PrimitiveArray<i128>, u8, i8),
}

/// Apply the same expression to every [`ColumnVector`] variant.
macro_rules! for_each_variant {
    ($self:expr, $arr:ident => $body:expr) => {
        match $self {
            ColumnVector::Null($arr) => $body,
            ColumnVector::Boolean($arr) => $body,
            ColumnVector::Int8($arr) => $body,
            ColumnVector::Int16($arr) => $body,
            ColumnVector::Int32($arr) => $body,
            ColumnVector::Int64($arr) => $body,
            ColumnVector::UInt8($arr) => $body,
            ColumnVector::UInt16($arr) => $body,
            ColumnVector::UInt32($arr) => $body,
            ColumnVector::UInt64($arr) => $body,
            ColumnVector::Float32($arr) => $body,
            ColumnVector::Float64($arr) => $body,
            ColumnVector::Utf8($arr) => $body,
            ColumnVector::Binary($arr) => $body,
            ColumnVector::Date32($arr) => $body,
            ColumnVector::Timestamp($arr, _) => $body,
            ColumnVector::Decimal128($arr, _, _) => $body,
        }
    };
}

impl ColumnVector {
    /// The logical type of the column.
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnVector::Null(_) => DataType::Null,
            ColumnVector::Boolean(_) => DataType::Boolean,
            ColumnVector::Int8(_) => DataType::Int8,
            ColumnVector::Int16(_) => DataType::Int16,
            ColumnVector::Int32(_) => DataType::Int32,
            ColumnVector::Int64(_) => DataType::Int64,
            ColumnVector::UInt8(_) => DataType::UInt8,
            ColumnVector::UInt16(_) => DataType::UInt16,
            ColumnVector::UInt32(_) => DataType::UInt32,
            ColumnVector::UInt64(_) => DataType::UInt64,
            ColumnVector::Float32(_) => DataType::Float32,
            ColumnVector::Float64(_) => DataType::Float64,
            ColumnVector::Utf8(_) => DataType::Utf8,
            ColumnVector::Binary(_) => DataType::Binary,
            ColumnVector::Date32(_) => DataType::Date32,
            ColumnVector::Timestamp(_, u) => DataType::Timestamp(*u),
            ColumnVector::Decimal128(_, p, s) => DataType::Decimal128(*p, *s),
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        for_each_variant!(self, a => a.len())
    }

    /// `true` if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of nulls. O(len / 64).
    pub fn null_count(&self) -> usize {
        for_each_variant!(self, a => a.null_count())
    }

    /// `true` if row `i` is null.
    pub fn is_null(&self, i: usize) -> bool {
        for_each_variant!(self, a => a.is_null(i))
    }

    /// `true` if row `i` holds a value.
    pub fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// The validity mask, or `None` when the column has no nulls.
    pub fn validity(&self) -> Option<&Bitmap> {
        match self {
            ColumnVector::Null(_) => None,
            ColumnVector::Boolean(a) => a.validity(),
            ColumnVector::Utf8(a) => a.validity(),
            ColumnVector::Binary(a) => a.validity(),
            ColumnVector::Int8(a) => a.validity(),
            ColumnVector::Int16(a) => a.validity(),
            ColumnVector::Int32(a) | ColumnVector::Date32(a) => a.validity(),
            ColumnVector::Int64(a) | ColumnVector::Timestamp(a, _) => a.validity(),
            ColumnVector::UInt8(a) => a.validity(),
            ColumnVector::UInt16(a) => a.validity(),
            ColumnVector::UInt32(a) => a.validity(),
            ColumnVector::UInt64(a) => a.validity(),
            ColumnVector::Float32(a) => a.validity(),
            ColumnVector::Float64(a) => a.validity(),
            ColumnVector::Decimal128(a, _, _) => a.validity(),
        }
    }

    /// Materialise row `i` as an owned [`Value`].
    ///
    /// Allocates for [`DataType::Utf8`] and [`DataType::Binary`]. Meant for
    /// scalar contexts (zone maps, `LIMIT 1`, error messages), never for scans.
    ///
    /// # Panics
    ///
    /// If `i >= len`.
    pub fn value(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        match self {
            ColumnVector::Null(_) => Value::Null,
            ColumnVector::Boolean(a) => Value::Boolean(a.value(i)),
            ColumnVector::Int8(a) => Value::Int8(a.value(i)),
            ColumnVector::Int16(a) => Value::Int16(a.value(i)),
            ColumnVector::Int32(a) => Value::Int32(a.value(i)),
            ColumnVector::Int64(a) => Value::Int64(a.value(i)),
            ColumnVector::UInt8(a) => Value::UInt8(a.value(i)),
            ColumnVector::UInt16(a) => Value::UInt16(a.value(i)),
            ColumnVector::UInt32(a) => Value::UInt32(a.value(i)),
            ColumnVector::UInt64(a) => Value::UInt64(a.value(i)),
            ColumnVector::Float32(a) => Value::Float32(a.value(i)),
            ColumnVector::Float64(a) => Value::Float64(a.value(i)),
            ColumnVector::Utf8(a) => Value::Utf8(a.value(i).to_string()),
            ColumnVector::Binary(a) => Value::Binary(a.value(i).to_vec()),
            ColumnVector::Date32(a) => Value::Date32(a.value(i)),
            ColumnVector::Timestamp(a, u) => Value::Timestamp(a.value(i), *u),
            ColumnVector::Decimal128(a, p, s) => Value::Decimal128(a.value(i), *p, *s),
        }
    }

    /// A zero-copy view of `len` rows starting at `offset`.
    ///
    /// O(1) for every variant: buffers and bitmaps are `Arc`-shared and only the
    /// offset/length pair changes.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds.
    pub fn slice(&self, offset: usize, len: usize) -> ColumnVector {
        match self {
            ColumnVector::Null(a) => ColumnVector::Null(a.slice(offset, len)),
            ColumnVector::Boolean(a) => ColumnVector::Boolean(a.slice(offset, len)),
            ColumnVector::Int8(a) => ColumnVector::Int8(a.slice(offset, len)),
            ColumnVector::Int16(a) => ColumnVector::Int16(a.slice(offset, len)),
            ColumnVector::Int32(a) => ColumnVector::Int32(a.slice(offset, len)),
            ColumnVector::Int64(a) => ColumnVector::Int64(a.slice(offset, len)),
            ColumnVector::UInt8(a) => ColumnVector::UInt8(a.slice(offset, len)),
            ColumnVector::UInt16(a) => ColumnVector::UInt16(a.slice(offset, len)),
            ColumnVector::UInt32(a) => ColumnVector::UInt32(a.slice(offset, len)),
            ColumnVector::UInt64(a) => ColumnVector::UInt64(a.slice(offset, len)),
            ColumnVector::Float32(a) => ColumnVector::Float32(a.slice(offset, len)),
            ColumnVector::Float64(a) => ColumnVector::Float64(a.slice(offset, len)),
            ColumnVector::Utf8(a) => ColumnVector::Utf8(a.slice(offset, len)),
            ColumnVector::Binary(a) => ColumnVector::Binary(a.slice(offset, len)),
            ColumnVector::Date32(a) => ColumnVector::Date32(a.slice(offset, len)),
            ColumnVector::Timestamp(a, u) => ColumnVector::Timestamp(a.slice(offset, len), *u),
            ColumnVector::Decimal128(a, p, s) => {
                ColumnVector::Decimal128(a.slice(offset, len), *p, *s)
            }
        }
    }

    /// Gather rows by index, materialising a new column. O(indices.len()).
    ///
    /// This is the "give up and copy" path used after a sort or a hash join.
    /// Filters should prefer [`Selection`](crate::Selection), which defers the
    /// copy until something actually needs contiguous data.
    pub fn take(&self, indices: &[u32]) -> Result<ColumnVector> {
        let len = self.len();
        if let Some(bad) = indices.iter().find(|i| **i as usize >= len) {
            return Err(BasaltError::execution(format!(
                "take index {bad} out of bounds for a column of {len} rows"
            )));
        }
        Ok(match self {
            ColumnVector::Null(_) => ColumnVector::Null(NullArray::new(indices.len())),
            ColumnVector::Boolean(a) => ColumnVector::Boolean(BooleanArray::from_opt_vec(
                indices.iter().map(|&i| a.get(i as usize)).collect(),
            )),
            ColumnVector::Int8(a) => ColumnVector::Int8(take_primitive(a, indices)),
            ColumnVector::Int16(a) => ColumnVector::Int16(take_primitive(a, indices)),
            ColumnVector::Int32(a) => ColumnVector::Int32(take_primitive(a, indices)),
            ColumnVector::Int64(a) => ColumnVector::Int64(take_primitive(a, indices)),
            ColumnVector::UInt8(a) => ColumnVector::UInt8(take_primitive(a, indices)),
            ColumnVector::UInt16(a) => ColumnVector::UInt16(take_primitive(a, indices)),
            ColumnVector::UInt32(a) => ColumnVector::UInt32(take_primitive(a, indices)),
            ColumnVector::UInt64(a) => ColumnVector::UInt64(take_primitive(a, indices)),
            ColumnVector::Float32(a) => ColumnVector::Float32(take_primitive(a, indices)),
            ColumnVector::Float64(a) => ColumnVector::Float64(take_primitive(a, indices)),
            ColumnVector::Date32(a) => ColumnVector::Date32(take_primitive(a, indices)),
            ColumnVector::Timestamp(a, u) => {
                ColumnVector::Timestamp(take_primitive(a, indices), *u)
            }
            ColumnVector::Decimal128(a, p, s) => {
                ColumnVector::Decimal128(take_primitive(a, indices), *p, *s)
            }
            ColumnVector::Utf8(a) => {
                let inner = take_binary(a.as_binary(), indices)?;
                ColumnVector::Utf8(StringArray { inner })
            }
            ColumnVector::Binary(a) => ColumnVector::Binary(take_binary(a, indices)?),
        })
    }

    /// Keep the rows whose bit is set. O(len).
    ///
    /// # Errors
    ///
    /// If `mask` is not the same length as the column.
    pub fn filter(&self, mask: &Bitmap) -> Result<ColumnVector> {
        if mask.len() != self.len() {
            return Err(BasaltError::execution(format!(
                "filter mask has {} bits but the column has {} rows",
                mask.len(),
                self.len()
            )));
        }
        let indices: Vec<u32> = mask.iter_set_indices().map(|i| i as u32).collect();
        self.take(&indices)
    }
}

/// Concatenate columns of the same type into one contiguous column.
///
/// This is the kernel the segment writer uses to turn a row group's worth of
/// buffered batches into one chunk. It copies (there is no way not to), but it
/// copies *buffers*, not values: primitive columns become one `extend_from_slice`
/// per input and variable-length columns become one `extend_from_slice` of the
/// data plus a shifted offset walk. Going through `Value` instead would allocate
/// a `String` per row of every string column.
///
/// O(total rows). Allocates exactly one output buffer per component.
///
/// # Errors
///
/// If `columns` is empty or the types disagree.
///
/// ```
/// use basalt_types::{concat, ColumnVector, PrimitiveArray, Value};
///
/// let a = ColumnVector::Int32(PrimitiveArray::from_vec(vec![1, 2]));
/// let b = ColumnVector::Int32(PrimitiveArray::from_opt_vec(vec![None, Some(4)]));
/// let joined = concat(&[a, b])?;
/// assert_eq!(joined.len(), 4);
/// assert_eq!(joined.value(2), Value::Null);
/// assert_eq!(joined.value(3), Value::Int32(4));
/// # Ok::<(), basalt_common::BasaltError>(())
/// ```
// One arm per physical layout, plus a macro expansion each. There is no
// meaningful way to split an exhaustive dispatch without hiding the fact that
// every variant is handled.
#[allow(clippy::too_many_lines)]
pub fn concat(columns: &[ColumnVector]) -> Result<ColumnVector> {
    let Some(first) = columns.first() else {
        return Err(BasaltError::execution("cannot concatenate zero columns"));
    };
    let data_type = first.data_type();
    for column in columns {
        if column.data_type() != data_type {
            return Err(BasaltError::schema(format!(
                "cannot concatenate a {} column with a {data_type} column",
                column.data_type()
            )));
        }
    }
    if columns.len() == 1 {
        return Ok(first.clone());
    }
    let total: usize = columns.iter().map(ColumnVector::len).sum();
    let any_nulls = columns.iter().any(|c| c.null_count() > 0);

    macro_rules! concat_primitive {
        ($variant:ident) => {{
            let parts: Vec<_> = columns
                .iter()
                .map(|c| match c {
                    ColumnVector::$variant(a) => Ok(a),
                    other => Err(BasaltError::internal(
                        "array::concat",
                        format!("variant mismatch after type check: {}", other.data_type()),
                    )),
                })
                .collect::<Result<Vec<_>>>()?;
            concat_primitive_arrays(&parts, total, any_nulls)?
        }};
    }

    Ok(match first {
        ColumnVector::Null(_) => ColumnVector::Null(NullArray::new(total)),
        ColumnVector::Boolean(_) => {
            let mut values = crate::BitmapBuilder::with_capacity(total);
            let mut validity = any_nulls.then(|| crate::BitmapBuilder::with_capacity(total));
            for column in columns {
                let ColumnVector::Boolean(array) = column else {
                    return Err(BasaltError::internal("array::concat", "variant mismatch"));
                };
                for i in 0..array.len() {
                    values.push(array.value(i));
                    if let Some(bits) = &mut validity {
                        bits.push(array.is_valid(i));
                    }
                }
            }
            ColumnVector::Boolean(BooleanArray::try_new(
                values.finish(),
                validity.map(crate::BitmapBuilder::finish),
            )?)
        }
        ColumnVector::Int8(_) => ColumnVector::Int8(concat_primitive!(Int8)),
        ColumnVector::Int16(_) => ColumnVector::Int16(concat_primitive!(Int16)),
        ColumnVector::Int32(_) => ColumnVector::Int32(concat_primitive!(Int32)),
        ColumnVector::Int64(_) => ColumnVector::Int64(concat_primitive!(Int64)),
        ColumnVector::UInt8(_) => ColumnVector::UInt8(concat_primitive!(UInt8)),
        ColumnVector::UInt16(_) => ColumnVector::UInt16(concat_primitive!(UInt16)),
        ColumnVector::UInt32(_) => ColumnVector::UInt32(concat_primitive!(UInt32)),
        ColumnVector::UInt64(_) => ColumnVector::UInt64(concat_primitive!(UInt64)),
        ColumnVector::Float32(_) => ColumnVector::Float32(concat_primitive!(Float32)),
        ColumnVector::Float64(_) => ColumnVector::Float64(concat_primitive!(Float64)),
        ColumnVector::Date32(_) => ColumnVector::Date32(concat_primitive!(Date32)),
        ColumnVector::Timestamp(_, unit) => {
            let parts: Vec<&PrimitiveArray<i64>> = columns
                .iter()
                .filter_map(|c| match c {
                    ColumnVector::Timestamp(a, _) => Some(a),
                    _ => None,
                })
                .collect();
            ColumnVector::Timestamp(concat_primitive_arrays(&parts, total, any_nulls)?, *unit)
        }
        ColumnVector::Decimal128(_, precision, scale) => {
            let parts: Vec<&PrimitiveArray<i128>> = columns
                .iter()
                .filter_map(|c| match c {
                    ColumnVector::Decimal128(a, _, _) => Some(a),
                    _ => None,
                })
                .collect();
            ColumnVector::Decimal128(
                concat_primitive_arrays(&parts, total, any_nulls)?,
                *precision,
                *scale,
            )
        }
        ColumnVector::Utf8(_) => {
            let parts: Vec<&BinaryArray> = columns
                .iter()
                .filter_map(|c| match c {
                    ColumnVector::Utf8(a) => Some(a.as_binary()),
                    _ => None,
                })
                .collect();
            let inner = concat_binary_arrays(&parts, total, any_nulls)?;
            // Every input satisfied the UTF-8 invariant and concatenation only
            // moves whole values, so it still holds.
            ColumnVector::Utf8(StringArray::from_binary_unchecked(inner))
        }
        ColumnVector::Binary(_) => {
            let parts: Vec<&BinaryArray> = columns
                .iter()
                .filter_map(|c| match c {
                    ColumnVector::Binary(a) => Some(a),
                    _ => None,
                })
                .collect();
            ColumnVector::Binary(concat_binary_arrays(&parts, total, any_nulls)?)
        }
    })
}

fn concat_primitive_arrays<T: NativeType>(
    parts: &[&PrimitiveArray<T>],
    total: usize,
    any_nulls: bool,
) -> Result<PrimitiveArray<T>> {
    let mut values = Vec::with_capacity(total);
    let mut validity = any_nulls.then(|| crate::BitmapBuilder::with_capacity(total));
    for array in parts {
        values.extend_from_slice(array.values());
        if let Some(bits) = &mut validity {
            match array.validity() {
                Some(mask) => {
                    for i in 0..array.len() {
                        bits.push(mask.get(i));
                    }
                }
                None => bits.push_n(true, array.len()),
            }
        }
    }
    PrimitiveArray::try_new(
        Buffer::from_vec(values),
        validity.map(crate::BitmapBuilder::finish),
    )
}

fn concat_binary_arrays(
    parts: &[&BinaryArray],
    total: usize,
    any_nulls: bool,
) -> Result<BinaryArray> {
    let total_bytes: usize = parts.iter().map(|a| a.value_bytes()).sum();
    let mut offsets = Vec::with_capacity(total + 1);
    let mut data = Vec::with_capacity(total_bytes);
    let mut validity = any_nulls.then(|| crate::BitmapBuilder::with_capacity(total));
    offsets.push(0i32);
    for array in parts {
        for i in 0..array.len() {
            data.extend_from_slice(array.value(i));
            offsets.push(data.len() as i32);
            if let Some(bits) = &mut validity {
                bits.push(array.is_valid(i));
            }
        }
    }
    BinaryArray::try_new(
        Buffer::from_vec(offsets),
        Buffer::from_vec(data),
        validity.map(crate::BitmapBuilder::finish),
    )
}

fn take_primitive<T: NativeType>(a: &PrimitiveArray<T>, indices: &[u32]) -> PrimitiveArray<T> {
    if a.validity().is_none() {
        let values = indices
            .iter()
            .map(|&i| a.value(i as usize))
            .collect::<Vec<_>>();
        return PrimitiveArray::from_vec(values);
    }
    PrimitiveArray::from_opt_vec(indices.iter().map(|&i| a.get(i as usize)).collect())
}

fn take_binary(a: &BinaryArray, indices: &[u32]) -> Result<BinaryArray> {
    let values: Vec<&[u8]> = indices.iter().map(|&i| a.value(i as usize)).collect();
    let mut out = BinaryArray::from_slices(&values);
    if a.validity().is_some() {
        let validity: Bitmap = indices.iter().map(|&i| a.is_valid(i as usize)).collect();
        out = BinaryArray::try_new(out.offsets.clone(), out.data.clone(), Some(validity))?;
    }
    Ok(out)
}

impl fmt::Debug for ColumnVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data_type())?;
        f.write_str(" ")?;
        f.debug_list()
            .entries((0..self.len()).map(|i| self.value(i)))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn primitive_basics() {
        let a = PrimitiveArray::from_opt_vec(vec![Some(1i32), None, Some(3)]);
        assert_eq!(a.len(), 3);
        assert!(!a.is_empty());
        assert_eq!(a.null_count(), 1);
        assert!(a.is_null(1));
        assert!(a.is_valid(0));
        assert_eq!(a.get(1), None);
        assert_eq!(a.value(0), 1);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![Some(1), None, Some(3)]);
        assert_eq!(a.values().len(), 3);
        assert_eq!(a.values_buffer().len(), 3);
        let s = a.slice(1, 2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![None, Some(3)]);
        assert_eq!(format!("{s:?}"), "[None, Some(3)]");
    }

    #[test]
    fn equality_ignores_bytes_under_nulls() {
        let a = PrimitiveArray::try_new(
            Buffer::from_vec(vec![1i32, 999, 3]),
            Some(Bitmap::from_bools(&[true, false, true])),
        )
        .unwrap();
        let b = PrimitiveArray::try_new(
            Buffer::from_vec(vec![1i32, -7, 3]),
            Some(Bitmap::from_bools(&[true, false, true])),
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn validity_length_is_checked() {
        let err =
            PrimitiveArray::try_new(Buffer::from_vec(vec![1i32, 2]), Some(Bitmap::new_set(3)))
                .unwrap_err();
        assert!(err.to_string().contains("validity mask"), "{err}");
        assert!(BooleanArray::try_new(Bitmap::new_set(2), Some(Bitmap::new_set(3))).is_err());
    }

    #[test]
    fn boolean_array() {
        let a = BooleanArray::from_opt_vec(vec![Some(true), None, Some(false)]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.null_count(), 1);
        assert!(a.value(0));
        assert_eq!(a.get(2), Some(false));
        assert_eq!(a.values().len(), 3);
        assert!(a.validity().is_some());
        let s = a.slice(1, 2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![None, Some(false)]);
        assert!(!BooleanArray::from_bools(&[true]).is_empty());
    }

    #[test]
    fn binary_and_string() {
        let s = StringArray::from_strs(&["alpha", "", "gamma"]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.value(0), "alpha");
        assert_eq!(s.value(1), "");
        assert_eq!(s.get(2), Some("gamma"));
        assert_eq!(s.offsets(), &[0, 5, 5, 10]);
        assert_eq!(s.data().len(), 10);
        assert_eq!(s.value_bytes(), 10);
        assert_eq!(s.null_count(), 0);
        assert!(s.validity().is_none());
        assert!(s.is_valid(0));

        let sliced = s.slice(1, 2);
        assert_eq!(sliced.value_bytes(), 5);
        assert_eq!(
            sliced.iter().collect::<Vec<_>>(),
            vec![Some(""), Some("gamma")]
        );
        // Slicing keeps the whole data buffer; compact releases it.
        assert_eq!(sliced.data().len(), 10);
        assert_eq!(sliced.compact().data().len(), 5);
        assert_eq!(sliced.compact(), sliced);
    }

    #[test]
    fn string_validates_utf8_once() {
        let offsets = Buffer::from_vec(vec![0i32, 2]);
        let data = Buffer::from_vec(vec![0xff, 0xfe]);
        let err = StringArray::try_new(offsets, data, None).unwrap_err();
        assert!(err.to_string().contains("not valid UTF-8"), "{err}");
    }

    #[test]
    fn binary_offset_validation() {
        assert!(BinaryArray::try_new(Buffer::empty(), Buffer::empty(), None).is_err());
        assert!(BinaryArray::try_new(
            Buffer::from_vec(vec![0i32, 5, 3]),
            Buffer::from_vec(vec![0u8; 5]),
            None
        )
        .is_err());
        assert!(BinaryArray::try_new(
            Buffer::from_vec(vec![-1i32, 5]),
            Buffer::from_vec(vec![0u8; 5]),
            None
        )
        .is_err());
        assert!(BinaryArray::try_new(
            Buffer::from_vec(vec![0i32, 9]),
            Buffer::from_vec(vec![0u8; 5]),
            None
        )
        .is_err());
        assert!(BinaryArray::try_new(
            Buffer::from_vec(vec![0i32, 1]),
            Buffer::from_vec(vec![0u8; 5]),
            Some(Bitmap::new_set(4))
        )
        .is_err());
    }

    #[test]
    fn column_vector_dispatch() {
        let c = ColumnVector::Int32(PrimitiveArray::from_opt_vec(vec![Some(1), None, Some(3)]));
        assert_eq!(c.data_type(), DataType::Int32);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert_eq!(c.null_count(), 1);
        assert!(c.is_null(1));
        assert!(c.is_valid(0));
        assert_eq!(c.value(0), Value::Int32(1));
        assert_eq!(c.value(1), Value::Null);
        assert!(c.validity().is_some());
        assert_eq!(c.slice(2, 1).value(0), Value::Int32(3));
        assert!(format!("{c:?}").starts_with("INT32 ["));
    }

    #[test]
    fn null_column() {
        let c = ColumnVector::Null(NullArray::new(4));
        assert_eq!(c.data_type(), DataType::Null);
        assert_eq!(c.null_count(), 4);
        assert!(c.is_null(0));
        assert!(c.validity().is_none());
        assert_eq!(c.slice(1, 2).len(), 2);
        assert_eq!(c.value(0), Value::Null);
        assert!(NullArray::new(0).is_empty());
        assert_eq!(c.take(&[0, 0, 1]).unwrap().len(), 3);
    }

    #[test]
    fn take_and_filter() {
        let c = ColumnVector::Utf8(StringArray::from_strs(&["a", "bb", "ccc"]));
        let taken = c.take(&[2, 0]).unwrap();
        assert_eq!(taken.value(0), Value::Utf8("ccc".into()));
        assert_eq!(taken.value(1), Value::Utf8("a".into()));
        assert!(c.take(&[5]).is_err());

        let mask = Bitmap::from_bools(&[true, false, true]);
        let filtered = c.filter(&mask).unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered.value(1), Value::Utf8("ccc".into()));
        assert!(c.filter(&Bitmap::new_set(2)).is_err());
    }

    #[test]
    fn take_preserves_nulls() {
        let c = ColumnVector::Int64(PrimitiveArray::from_opt_vec(vec![Some(1), None, Some(3)]));
        let t = c.take(&[1, 2, 1]).unwrap();
        assert_eq!(t.null_count(), 2);
        assert_eq!(t.value(1), Value::Int64(3));

        let b = BinaryArray::try_new(
            Buffer::from_vec(vec![0i32, 1, 2, 3]),
            Buffer::from_vec(b"xyz".to_vec()),
            Some(Bitmap::from_bools(&[true, false, true])),
        )
        .unwrap();
        let c = ColumnVector::Binary(b);
        let t = c.take(&[0, 1]).unwrap();
        assert_eq!(t.null_count(), 1);
        assert_eq!(t.value(0), Value::Binary(b"x".to_vec()));
    }

    #[test]
    fn concat_preserves_values_and_nulls() {
        let a = ColumnVector::Utf8(StringArray::from_strs(&["one", "two"]));
        let b = ColumnVector::Utf8(StringArray::from_strs(&["three"]));
        let joined = concat(&[a.clone(), b]).unwrap();
        assert_eq!(joined.len(), 3);
        assert_eq!(joined.value(2), Value::Utf8("three".into()));
        assert_eq!(concat(std::slice::from_ref(&a)).unwrap(), a);

        let with_nulls = ColumnVector::Int64(PrimitiveArray::from_opt_vec(vec![Some(1), None]));
        let dense = ColumnVector::Int64(PrimitiveArray::from_vec(vec![3]));
        let joined = concat(&[with_nulls, dense]).unwrap();
        assert_eq!(joined.null_count(), 1);
        assert_eq!(joined.value(0), Value::Int64(1));
        assert_eq!(joined.value(2), Value::Int64(3));

        assert!(concat(&[]).is_err());
        assert!(concat(&[
            ColumnVector::Int32(PrimitiveArray::from_vec(vec![1])),
            ColumnVector::Int64(PrimitiveArray::from_vec(vec![1])),
        ])
        .is_err());
    }

    #[test]
    fn concat_every_variant() {
        let cases: Vec<ColumnVector> = vec![
            ColumnVector::Null(NullArray::new(2)),
            ColumnVector::Boolean(BooleanArray::from_opt_vec(vec![Some(true), None])),
            ColumnVector::Int8(PrimitiveArray::from_vec(vec![1i8, 2])),
            ColumnVector::Int16(PrimitiveArray::from_vec(vec![1i16, 2])),
            ColumnVector::Int32(PrimitiveArray::from_vec(vec![1i32, 2])),
            ColumnVector::Int64(PrimitiveArray::from_vec(vec![1i64, 2])),
            ColumnVector::UInt8(PrimitiveArray::from_vec(vec![1u8, 2])),
            ColumnVector::UInt16(PrimitiveArray::from_vec(vec![1u16, 2])),
            ColumnVector::UInt32(PrimitiveArray::from_vec(vec![1u32, 2])),
            ColumnVector::UInt64(PrimitiveArray::from_vec(vec![1u64, 2])),
            ColumnVector::Float32(PrimitiveArray::from_vec(vec![1f32, 2.0])),
            ColumnVector::Float64(PrimitiveArray::from_vec(vec![1f64, 2.0])),
            ColumnVector::Utf8(StringArray::from_strs(&["a", "b"])),
            ColumnVector::Binary(BinaryArray::from_slices(&[b"a", b"b"])),
            ColumnVector::Date32(PrimitiveArray::from_vec(vec![1i32, 2])),
            ColumnVector::Timestamp(PrimitiveArray::from_vec(vec![1i64, 2]), TimeUnit::Second),
            ColumnVector::Decimal128(PrimitiveArray::from_vec(vec![1i128, 2]), 10, 2),
        ];
        for column in cases {
            let joined = concat(&[column.clone(), column.clone()]).unwrap();
            assert_eq!(joined.len(), 4, "{}", column.data_type());
            assert_eq!(joined.data_type(), column.data_type());
            assert_eq!(joined.slice(0, 2), column);
            assert_eq!(joined.slice(2, 2), column);
        }
    }

    #[test]
    fn all_variants_report_their_type() {
        let cases: Vec<(ColumnVector, DataType)> = vec![
            (
                ColumnVector::Boolean(BooleanArray::from_bools(&[true])),
                DataType::Boolean,
            ),
            (
                ColumnVector::Int8(PrimitiveArray::from_vec(vec![1i8])),
                DataType::Int8,
            ),
            (
                ColumnVector::Int16(PrimitiveArray::from_vec(vec![1i16])),
                DataType::Int16,
            ),
            (
                ColumnVector::Int64(PrimitiveArray::from_vec(vec![1i64])),
                DataType::Int64,
            ),
            (
                ColumnVector::UInt8(PrimitiveArray::from_vec(vec![1u8])),
                DataType::UInt8,
            ),
            (
                ColumnVector::UInt16(PrimitiveArray::from_vec(vec![1u16])),
                DataType::UInt16,
            ),
            (
                ColumnVector::UInt32(PrimitiveArray::from_vec(vec![1u32])),
                DataType::UInt32,
            ),
            (
                ColumnVector::UInt64(PrimitiveArray::from_vec(vec![1u64])),
                DataType::UInt64,
            ),
            (
                ColumnVector::Float32(PrimitiveArray::from_vec(vec![1f32])),
                DataType::Float32,
            ),
            (
                ColumnVector::Float64(PrimitiveArray::from_vec(vec![1f64])),
                DataType::Float64,
            ),
            (
                ColumnVector::Binary(BinaryArray::from_slices(&[b"x"])),
                DataType::Binary,
            ),
            (
                ColumnVector::Date32(PrimitiveArray::from_vec(vec![1i32])),
                DataType::Date32,
            ),
            (
                ColumnVector::Timestamp(PrimitiveArray::from_vec(vec![1i64]), TimeUnit::Second),
                DataType::Timestamp(TimeUnit::Second),
            ),
            (
                ColumnVector::Decimal128(PrimitiveArray::from_vec(vec![1i128]), 10, 2),
                DataType::Decimal128(10, 2),
            ),
        ];
        for (c, t) in cases {
            assert_eq!(c.data_type(), t);
            assert_eq!(c.len(), 1);
            assert_eq!(c.null_count(), 0);
            assert_eq!(c.slice(0, 1).len(), 1);
            assert_eq!(c.take(&[0]).unwrap().value(0), c.value(0));
            assert!(c.validity().is_none());
        }
    }
}

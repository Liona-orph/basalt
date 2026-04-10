//! Immutable, cheaply sliceable value buffers.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// An immutable, reference-counted, cheaply sliceable run of `T`.
///
/// Every columnar array in Basalt stores its values in a `Buffer`. The point is
/// [`Buffer::slice`]: slicing a batch of a million rows down to a thousand must
/// not copy a million values, because the executor slices constantly (batching,
/// limit, window frames). A `Buffer` is therefore a *view*: an `Arc`'d backing
/// `Vec<T>` plus an offset and a length.
///
/// * `clone` — O(1), one atomic increment.
/// * `slice` — O(1), no allocation.
/// * `as_slice` — O(1), returns a plain `&[T]` so that hot loops see ordinary
///   slice code that LLVM can vectorise.
///
/// The cost of this design is that a small slice keeps its whole backing buffer
/// alive. That is the right default for a query engine (slices are short-lived
/// and the alternative is copying on every operator boundary); callers that want
/// to release the backing memory can call [`Buffer::compact`].
///
/// ```
/// use basalt_types::Buffer;
///
/// let buf = Buffer::from_vec(vec![1i32, 2, 3, 4, 5]);
/// let mid = buf.slice(1, 3);
/// assert_eq!(mid.as_slice(), &[2, 3, 4]);
/// assert_eq!(mid.slice(1, 1).as_slice(), &[3]);
/// ```
pub struct Buffer<T> {
    data: Arc<Vec<T>>,
    offset: usize,
    len: usize,
}

impl<T> Buffer<T> {
    /// Take ownership of a `Vec` without copying it.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Buffer {
            data: Arc::new(data),
            offset: 0,
            len,
        }
    }

    /// An empty buffer. Allocates an empty `Arc<Vec<_>>` (no heap data).
    pub fn empty() -> Self {
        Buffer::from_vec(Vec::new())
    }

    /// Number of elements visible through this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The visible elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// A sub-view. O(1).
    ///
    /// # Panics
    ///
    /// If `offset + len` exceeds this view's length. Slicing out of bounds is a
    /// programming error, not a runtime condition, so it panics like `&v[a..b]`.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Buffer<T> {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "Buffer::slice({offset}, {len}) out of bounds for length {}",
            self.len
        );
        Buffer {
            data: Arc::clone(&self.data),
            offset: self.offset + offset,
            len,
        }
    }

    /// Byte offset of this view inside its backing allocation, in elements.
    ///
    /// Exposed because the encoding layer wants to know whether a buffer is
    /// aligned to the start of its allocation (offset 0) and can therefore be
    /// handed to a codec without a copy.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl<T: Clone> Buffer<T> {
    /// Copy the view into a fresh allocation, dropping the reference to the
    /// backing buffer. O(len).
    pub fn compact(&self) -> Buffer<T> {
        Buffer::from_vec(self.as_slice().to_vec())
    }

    /// Copy the visible elements out.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<T> Clone for Buffer<T> {
    /// O(1): bumps the refcount, never copies data.
    fn clone(&self) -> Self {
        Buffer {
            data: Arc::clone(&self.data),
            offset: self.offset,
            len: self.len,
        }
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    /// Compares *contents*, not identity: two buffers that show the same values
    /// are equal even if they have different backing allocations.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for Buffer<T> {}

impl<T> From<Vec<T>> for Buffer<T> {
    fn from(v: Vec<T>) -> Self {
        Buffer::from_vec(v)
    }
}

impl<T: Clone> From<&[T]> for Buffer<T> {
    fn from(v: &[T]) -> Self {
        Buffer::from_vec(v.to_vec())
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Buffer::from_vec(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn slice_is_a_view() {
        let buf = Buffer::from_vec((0i64..100).collect::<Vec<_>>());
        let s = buf.slice(10, 20);
        assert_eq!(s.len(), 20);
        assert_eq!(s.as_slice()[0], 10);
        assert_eq!(s.offset(), 10);
        let s2 = s.slice(5, 5);
        assert_eq!(s2.as_slice(), &[15, 16, 17, 18, 19]);
        assert_eq!(s2.offset(), 15);
    }

    #[test]
    fn empty_and_edge_slices() {
        let buf: Buffer<u8> = Buffer::empty();
        assert!(buf.is_empty());
        assert_eq!(buf.slice(0, 0).len(), 0);
        let buf = Buffer::from_vec(vec![1u8, 2, 3]);
        assert_eq!(buf.slice(3, 0).len(), 0);
        assert_eq!(buf.slice(0, 3).as_slice(), &[1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_out_of_bounds_panics() {
        let buf = Buffer::from_vec(vec![1u8, 2, 3]);
        let _ = buf.slice(2, 2);
    }

    #[test]
    fn compact_detaches() {
        let buf = Buffer::from_vec((0u32..1000).collect::<Vec<_>>());
        let s = buf.slice(500, 2);
        let c = s.compact();
        assert_eq!(c.as_slice(), s.as_slice());
        assert_eq!(c.offset(), 0);
    }

    #[test]
    fn equality_is_by_content() {
        let a = Buffer::from_vec(vec![1, 2, 3]);
        let b = Buffer::from_vec(vec![0, 1, 2, 3]).slice(1, 3);
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), "[1, 2, 3]");
    }
}

//! A packed bit vector, used for validity (null) masks and filter results.

use std::fmt;
use std::sync::Arc;

use basalt_common::{BasaltError, Result};

/// Bits per backing word. Chosen as 64 so that `count_ones` is one `popcnt`
/// per word and `iter_set_indices` is one `tzcnt` per set bit.
const BITS: usize = 64;

#[inline]
const fn words_for(bits: usize) -> usize {
    bits.div_ceil(BITS)
}

/// An immutable, cheaply sliceable packed bit vector.
///
/// Basalt uses `Bitmap` for two things:
///
/// 1. **Validity masks.** A set bit means *valid* (non-null), the same polarity
///    Arrow uses. The polarity matters: with 1 = valid, "is this row usable?"
///    is a plain `AND` of the masks of all inputs, and a column with no nulls
///    can omit the bitmap entirely (`None`) instead of storing all-ones.
/// 2. **Filter results.** A predicate evaluates to one bit per row, which the
///    executor then either applies directly or converts into a
///    [`Selection`](crate::Selection).
///
/// Storage is `Arc<Vec<u64>>` plus a bit offset and a bit length, so
/// [`Bitmap::slice`] is O(1) and does not copy. Because a slice may start
/// mid-word, all reads go through an internal word-extraction helper that
/// shifts two adjacent words together; that costs two loads and two shifts per
/// 64 bits, which is still far cheaper than materialising a copy.
///
/// # Complexity
///
/// | Operation           | Cost                    |
/// |---------------------|-------------------------|
/// | `get` / `set`       | O(1) (`set` may clone, see below) |
/// | `count_ones`        | O(len / 64)             |
/// | `and` / `or` / `not`| O(len / 64), allocates  |
/// | `slice`             | O(1), no allocation     |
/// | `iter_set_indices`  | O(len / 64 + popcount)  |
///
/// [`Bitmap::set`] is copy-on-write: if the backing buffer is shared it is
/// cloned first, so a `set` loop on a freshly built bitmap is O(1) amortised but
/// the first `set` on a shared bitmap is O(len). Build with [`BitmapBuilder`]
/// when constructing from scratch.
///
/// ```
/// use basalt_types::Bitmap;
///
/// let a = Bitmap::from_bools(&[true, true, false, true]);
/// let b = Bitmap::from_bools(&[true, false, false, true]);
/// assert_eq!(a.count_ones(), 3);
/// assert_eq!(a.and(&b).unwrap().to_bools(), vec![true, false, false, true]);
/// assert_eq!(a.not().to_bools(), vec![false, false, true, false]);
/// assert_eq!(a.iter_set_indices().collect::<Vec<_>>(), vec![0, 1, 3]);
/// ```
#[derive(Clone)]
pub struct Bitmap {
    words: Arc<Vec<u64>>,
    /// Bit offset of this view into `words`.
    offset: usize,
    /// Number of bits visible through this view.
    len: usize,
}

impl Bitmap {
    /// A bitmap of `len` bits, all set (all valid).
    pub fn new_set(len: usize) -> Bitmap {
        let mut words = vec![u64::MAX; words_for(len)];
        mask_tail(&mut words, len);
        Bitmap {
            words: Arc::new(words),
            offset: 0,
            len,
        }
    }

    /// A bitmap of `len` bits, all clear (all null).
    pub fn new_unset(len: usize) -> Bitmap {
        Bitmap {
            words: Arc::new(vec![0u64; words_for(len)]),
            offset: 0,
            len,
        }
    }

    /// Build from booleans. O(n).
    pub fn from_bools(bits: &[bool]) -> Bitmap {
        let mut b = BitmapBuilder::with_capacity(bits.len());
        for &v in bits {
            b.push(v);
        }
        b.finish()
    }

    /// Build from raw little-endian words, taking `len` bits.
    ///
    /// Bits beyond `len` in the final word are ignored by every operation, so
    /// callers do not have to zero them.
    pub fn from_words(words: Vec<u64>, len: usize) -> Result<Bitmap> {
        if words.len() < words_for(len) {
            return Err(BasaltError::internal(
                "bitmap::from_words",
                format!(
                    "need {} words for {len} bits, got {}",
                    words_for(len),
                    words.len()
                ),
            ));
        }
        Ok(Bitmap {
            words: Arc::new(words),
            offset: 0,
            len,
        })
    }

    /// Build from a little-endian byte buffer (the on-disk representation).
    pub fn from_bytes(bytes: &[u8], len: usize) -> Result<Bitmap> {
        let needed = len.div_ceil(8);
        if bytes.len() < needed {
            return Err(BasaltError::internal(
                "bitmap::from_bytes",
                format!("need {needed} bytes for {len} bits, got {}", bytes.len()),
            ));
        }
        let mut words = vec![0u64; words_for(len)];
        for (i, chunk) in bytes[..needed].chunks(8).enumerate() {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            words[i] = u64::from_le_bytes(w);
        }
        Ok(Bitmap {
            words: Arc::new(words),
            offset: 0,
            len,
        })
    }

    /// Serialise to little-endian bytes, `ceil(len / 8)` of them.
    ///
    /// The view is normalised to offset 0 first, so the output of a sliced
    /// bitmap round-trips through [`Bitmap::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.len.div_ceil(8));
        let mut bit = 0;
        while bit < self.len {
            let word = self.word_at(bit);
            let remaining_bits = self.len - bit;
            let n = remaining_bits.min(BITS).div_ceil(8);
            out.extend_from_slice(&word.to_le_bytes()[..n]);
            bit += BITS;
        }
        out
    }

    /// Number of bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no bits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read bit `i`.
    ///
    /// # Panics
    ///
    /// If `i >= len`. Use [`Bitmap::try_get`] when the index is data-dependent.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(
            i < self.len,
            "Bitmap index {i} out of bounds for length {}",
            self.len
        );
        let idx = self.offset + i;
        (self.words[idx / BITS] >> (idx % BITS)) & 1 == 1
    }

    /// Read bit `i`, or `None` if out of bounds.
    #[inline]
    pub fn try_get(&self, i: usize) -> Option<bool> {
        if i < self.len {
            let idx = self.offset + i;
            Some((self.words[idx / BITS] >> (idx % BITS)) & 1 == 1)
        } else {
            None
        }
    }

    /// Write bit `i`.
    ///
    /// Copy-on-write: if the backing buffer is shared with another `Bitmap`,
    /// it is cloned first (O(len)); otherwise this is O(1).
    ///
    /// # Panics
    ///
    /// If `i >= len`.
    pub fn set(&mut self, i: usize, value: bool) {
        assert!(
            i < self.len,
            "Bitmap index {i} out of bounds for length {}",
            self.len
        );
        let idx = self.offset + i;
        let words = Arc::make_mut(&mut self.words);
        let mask = 1u64 << (idx % BITS);
        if value {
            words[idx / BITS] |= mask;
        } else {
            words[idx / BITS] &= !mask;
        }
    }

    /// A view of `len` bits starting at `offset`. O(1).
    ///
    /// # Panics
    ///
    /// If `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Bitmap {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= self.len),
            "Bitmap::slice({offset}, {len}) out of bounds for length {}",
            self.len
        );
        Bitmap {
            words: Arc::clone(&self.words),
            offset: self.offset + offset,
            len,
        }
    }

    /// Number of set bits. O(len / 64), one `popcnt` per word.
    pub fn count_ones(&self) -> usize {
        let mut total = 0;
        let mut bit = 0;
        while bit < self.len {
            total += self.word_at(bit).count_ones() as usize;
            bit += BITS;
        }
        total
    }

    /// Number of clear bits. O(len / 64).
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// `true` if every bit is set. O(len / 64).
    pub fn is_all_set(&self) -> bool {
        self.count_ones() == self.len
    }

    /// `true` if no bit is set. Short-circuits on the first non-zero word.
    pub fn is_all_unset(&self) -> bool {
        let mut bit = 0;
        while bit < self.len {
            if self.word_at(bit) != 0 {
                return false;
            }
            bit += BITS;
        }
        true
    }

    /// Iterate over every bit as a `bool`.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }

    /// Iterate over the indices of set bits, cheapest-first.
    ///
    /// Uses `trailing_zeros` so the cost is proportional to the number of set
    /// bits plus one word scan, not to `len` times a branch. This is the
    /// primitive the executor uses to turn a filter result into row indices.
    pub fn iter_set_indices(&self) -> SetIndices<'_> {
        SetIndices {
            bitmap: self,
            word_index: 0,
            current: if self.len == 0 { 0 } else { self.word_at(0) },
        }
    }

    /// Collect into a `Vec<bool>`. Mainly for tests and debugging.
    pub fn to_bools(&self) -> Vec<bool> {
        self.iter().collect()
    }

    /// Bitwise AND. Allocates a fresh, offset-0 bitmap.
    pub fn and(&self, other: &Bitmap) -> Result<Bitmap> {
        self.binary_op(other, "bitmap::and", |a, b| a & b)
    }

    /// Bitwise OR. Allocates a fresh, offset-0 bitmap.
    pub fn or(&self, other: &Bitmap) -> Result<Bitmap> {
        self.binary_op(other, "bitmap::or", |a, b| a | b)
    }

    /// Bitwise XOR. Allocates a fresh, offset-0 bitmap.
    pub fn xor(&self, other: &Bitmap) -> Result<Bitmap> {
        self.binary_op(other, "bitmap::xor", |a, b| a ^ b)
    }

    /// Bitwise NOT. Allocates a fresh, offset-0 bitmap.
    ///
    /// Bits past `len` in the final word are left clear so that the result is
    /// canonical and `to_bytes` round-trips.
    pub fn not(&self) -> Bitmap {
        let mut words = Vec::with_capacity(words_for(self.len));
        let mut bit = 0;
        while bit < self.len {
            words.push(!self.word_at(bit));
            bit += BITS;
        }
        mask_tail(&mut words, self.len);
        Bitmap {
            words: Arc::new(words),
            offset: 0,
            len: self.len,
        }
    }

    fn binary_op(
        &self,
        other: &Bitmap,
        location: &'static str,
        op: impl Fn(u64, u64) -> u64,
    ) -> Result<Bitmap> {
        if self.len != other.len {
            return Err(BasaltError::internal(
                location,
                format!("length mismatch: {} vs {}", self.len, other.len),
            ));
        }
        let mut words = Vec::with_capacity(words_for(self.len));
        let mut bit = 0;
        while bit < self.len {
            words.push(op(self.word_at(bit), other.word_at(bit)));
            bit += BITS;
        }
        mask_tail(&mut words, self.len);
        Ok(Bitmap {
            words: Arc::new(words),
            offset: 0,
            len: self.len,
        })
    }

    /// The 64 bits starting at logical position `bit_index`, zero-padded past
    /// the end of the view.
    ///
    /// This is the workhorse that makes offset slicing free: it stitches the
    /// tail of one backing word to the head of the next.
    #[inline]
    fn word_at(&self, bit_index: usize) -> u64 {
        debug_assert!(bit_index < self.len);
        let start = self.offset + bit_index;
        let w = start / BITS;
        let s = start % BITS;
        let lo = self.words.get(w).copied().unwrap_or(0) >> s;
        let hi = if s == 0 {
            0
        } else {
            self.words.get(w + 1).copied().unwrap_or(0) << (BITS - s)
        };
        let mut value = lo | hi;
        let remaining = self.len - bit_index;
        if remaining < BITS {
            value &= (1u64 << remaining) - 1;
        }
        value
    }
}

/// Zero every bit at or after `len` in the final word, keeping the canonical
/// form (needed so that `not` and `to_bytes` do not leak junk).
fn mask_tail(words: &mut [u64], len: usize) {
    let tail = len % BITS;
    if tail != 0 {
        if let Some(last) = words.get_mut(len / BITS) {
            *last &= (1u64 << tail) - 1;
        }
    }
}

impl PartialEq for Bitmap {
    /// Compares visible bits only; backing storage and offset are irrelevant.
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        let mut bit = 0;
        while bit < self.len {
            if self.word_at(bit) != other.word_at(bit) {
                return false;
            }
            bit += BITS;
        }
        true
    }
}

impl Eq for Bitmap {}

impl fmt::Debug for Bitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const PREVIEW: usize = 64;
        write!(f, "Bitmap[{}; ", self.len)?;
        for i in 0..self.len.min(PREVIEW) {
            f.write_str(if self.get(i) { "1" } else { "0" })?;
        }
        if self.len > PREVIEW {
            write!(f, "... +{}", self.len - PREVIEW)?;
        }
        f.write_str("]")
    }
}

impl FromIterator<bool> for Bitmap {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut b = BitmapBuilder::new();
        for v in iter {
            b.push(v);
        }
        b.finish()
    }
}

/// Iterator over the indices of set bits in a [`Bitmap`].
#[derive(Debug)]
pub struct SetIndices<'a> {
    bitmap: &'a Bitmap,
    word_index: usize,
    current: u64,
}

impl Iterator for SetIndices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current != 0 {
                let tz = self.current.trailing_zeros() as usize;
                // Clear the lowest set bit.
                self.current &= self.current - 1;
                return Some(self.word_index * BITS + tz);
            }
            self.word_index += 1;
            let start = self.word_index * BITS;
            if start >= self.bitmap.len {
                return None;
            }
            self.current = self.bitmap.word_at(start);
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.bitmap.len))
    }
}

/// Append-only builder for [`Bitmap`].
///
/// Growth is amortised O(1): the backing `Vec<u64>` doubles, so building `n`
/// bits costs `O(n)` total with at most `log2(n / 64)` reallocations.
///
/// ```
/// use basalt_types::BitmapBuilder;
///
/// let mut b = BitmapBuilder::with_capacity(128);
/// b.push(true);
/// b.push_n(false, 62);
/// b.push(true);
/// let bm = b.finish();
/// assert_eq!(bm.len(), 64);
/// assert_eq!(bm.iter_set_indices().collect::<Vec<_>>(), vec![0, 63]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BitmapBuilder {
    words: Vec<u64>,
    len: usize,
}

impl BitmapBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        BitmapBuilder::default()
    }

    /// A builder pre-sized for `bits` bits (one allocation).
    pub fn with_capacity(bits: usize) -> Self {
        BitmapBuilder {
            words: Vec::with_capacity(words_for(bits)),
            len: 0,
        }
    }

    /// Number of bits pushed so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if nothing has been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one bit.
    #[inline]
    pub fn push(&mut self, value: bool) {
        if self.len % BITS == 0 {
            self.words.push(0);
        }
        if value {
            let idx = self.len;
            // The word was just pushed or already exists; indexing is in range.
            self.words[idx / BITS] |= 1u64 << (idx % BITS);
        }
        self.len += 1;
    }

    /// Append `count` copies of `value`. O(count / 64) for the common runs.
    pub fn push_n(&mut self, value: bool, count: usize) {
        // Fill the partial word one bit at a time, then whole words at once.
        let mut remaining = count;
        while remaining > 0 && self.len % BITS != 0 {
            self.push(value);
            remaining -= 1;
        }
        while remaining >= BITS {
            self.words.push(if value { u64::MAX } else { 0 });
            self.len += BITS;
            remaining -= BITS;
        }
        for _ in 0..remaining {
            self.push(value);
        }
    }

    /// Overwrite bit `i`.
    ///
    /// # Panics
    ///
    /// If `i` has not been pushed yet.
    pub fn set(&mut self, i: usize, value: bool) {
        assert!(
            i < self.len,
            "BitmapBuilder index {i} out of bounds for length {}",
            self.len
        );
        let mask = 1u64 << (i % BITS);
        if value {
            self.words[i / BITS] |= mask;
        } else {
            self.words[i / BITS] &= !mask;
        }
    }

    /// Finish, producing an immutable [`Bitmap`]. Does not copy.
    pub fn finish(mut self) -> Bitmap {
        mask_tail(&mut self.words, self.len);
        Bitmap {
            words: Arc::new(self.words),
            offset: 0,
            len: self.len,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn reference(bits: &[bool]) -> Bitmap {
        Bitmap::from_bools(bits)
    }

    #[test]
    fn get_set_round_trip() {
        let mut bm = Bitmap::new_unset(200);
        for i in (0..200).step_by(7) {
            bm.set(i, true);
        }
        for i in 0..200 {
            assert_eq!(bm.get(i), i % 7 == 0, "bit {i}");
        }
        assert_eq!(bm.count_ones(), (0..200).step_by(7).count());
        assert_eq!(bm.count_zeros(), 200 - (0..200).step_by(7).count());
    }

    #[test]
    fn all_set_all_unset() {
        assert!(Bitmap::new_set(0).is_all_set());
        assert!(Bitmap::new_set(0).is_all_unset());
        assert!(Bitmap::new_set(130).is_all_set());
        assert!(!Bitmap::new_set(130).is_all_unset());
        assert!(Bitmap::new_unset(130).is_all_unset());
        assert_eq!(Bitmap::new_set(130).count_ones(), 130);
    }

    #[test]
    fn slicing_across_word_boundaries() {
        let bits: Vec<bool> = (0..300).map(|i| i % 3 == 0).collect();
        let bm = reference(&bits);
        for start in [0usize, 1, 7, 63, 64, 65, 129] {
            for len in [0usize, 1, 5, 63, 64, 65, 100] {
                if start + len > 300 {
                    continue;
                }
                let s = bm.slice(start, len);
                assert_eq!(s.len(), len);
                assert_eq!(
                    s.to_bools(),
                    &bits[start..start + len],
                    "slice({start},{len})"
                );
                assert_eq!(
                    s.count_ones(),
                    bits[start..start + len].iter().filter(|b| **b).count()
                );
            }
        }
    }

    #[test]
    fn logical_ops_match_reference() {
        let a: Vec<bool> = (0..137).map(|i| i % 2 == 0).collect();
        let b: Vec<bool> = (0..137).map(|i| i % 3 == 0).collect();
        let ba = reference(&a);
        let bb = reference(&b);
        let and_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x && *y).collect();
        let or_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x || *y).collect();
        let xor_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x ^ *y).collect();
        let not_ref: Vec<bool> = a.iter().map(|x| !*x).collect();
        assert_eq!(ba.and(&bb).unwrap().to_bools(), and_ref);
        assert_eq!(ba.or(&bb).unwrap().to_bools(), or_ref);
        assert_eq!(ba.xor(&bb).unwrap().to_bools(), xor_ref);
        assert_eq!(ba.not().to_bools(), not_ref);
    }

    #[test]
    fn ops_reject_length_mismatch() {
        let a = Bitmap::new_set(10);
        let b = Bitmap::new_set(11);
        let err = a.and(&b).unwrap_err();
        assert!(err.to_string().contains("length mismatch"), "{err}");
    }

    #[test]
    fn bytes_round_trip_including_slices() {
        let bits: Vec<bool> = (0..201).map(|i| (i * 7) % 5 < 2).collect();
        let bm = reference(&bits);
        let restored = Bitmap::from_bytes(&bm.to_bytes(), bm.len()).unwrap();
        assert_eq!(restored, bm);
        let s = bm.slice(3, 100);
        let restored = Bitmap::from_bytes(&s.to_bytes(), s.len()).unwrap();
        assert_eq!(restored, s);
    }

    #[test]
    fn set_indices_matches_scan() {
        let bits: Vec<bool> = (0..500).map(|i| i % 11 == 0 || i == 499).collect();
        let bm = reference(&bits);
        let expect: Vec<usize> = bits
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i) } else { None })
            .collect();
        assert_eq!(bm.iter_set_indices().collect::<Vec<_>>(), expect);
        // Same, through an unaligned slice.
        let s = bm.slice(5, 400);
        let expect: Vec<usize> = bits[5..405]
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i) } else { None })
            .collect();
        assert_eq!(s.iter_set_indices().collect::<Vec<_>>(), expect);
    }

    #[test]
    fn builder_push_n() {
        let mut b = BitmapBuilder::new();
        b.push(true);
        b.push_n(false, 130);
        b.push_n(true, 3);
        let bm = b.finish();
        assert_eq!(bm.len(), 134);
        assert_eq!(
            bm.iter_set_indices().collect::<Vec<_>>(),
            vec![0, 131, 132, 133]
        );
        let mut b2 = BitmapBuilder::with_capacity(8);
        b2.push_n(true, 5);
        b2.set(2, false);
        assert_eq!(b2.len(), 5);
        assert!(!b2.is_empty());
        assert_eq!(b2.finish().to_bools(), vec![true, true, false, true, true]);
    }

    #[test]
    fn copy_on_write_does_not_disturb_the_original() {
        let original = Bitmap::from_bools(&[true, false, true]);
        let mut copy = original.clone();
        copy.set(1, true);
        assert_eq!(original.to_bools(), vec![true, false, true]);
        assert_eq!(copy.to_bools(), vec![true, true, true]);
    }

    #[test]
    fn equality_ignores_offset() {
        let bits: Vec<bool> = (0..100).map(|i| i % 4 == 1).collect();
        let a = Bitmap::from_bools(&bits[10..50]);
        let b = Bitmap::from_bools(&bits).slice(10, 40);
        assert_eq!(a, b);
        assert_ne!(a, Bitmap::from_bools(&bits[10..51]));
    }

    #[test]
    fn debug_is_readable() {
        let bm = Bitmap::from_bools(&[true, false, true]);
        assert_eq!(format!("{bm:?}"), "Bitmap[3; 101]");
        let big = Bitmap::new_set(70);
        assert!(format!("{big:?}").contains("+6"));
    }

    #[test]
    fn from_words_and_iter_collect() {
        let bm = Bitmap::from_words(vec![0b1011], 4).unwrap();
        assert_eq!(bm.to_bools(), vec![true, true, false, true]);
        assert!(Bitmap::from_words(vec![], 1).is_err());
        assert!(Bitmap::from_bytes(&[], 1).is_err());
        let collected: Bitmap = [true, false, true].into_iter().collect();
        assert_eq!(collected.to_bools(), vec![true, false, true]);
        assert_eq!(bm.try_get(9), None);
        assert_eq!(bm.try_get(0), Some(true));
    }
}

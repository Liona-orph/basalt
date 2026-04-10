//! Selection vectors: filter results that are not yet materialised.

use std::fmt;

use basalt_common::{BasaltError, Result};

use crate::{Bitmap, ColumnVector};

/// The set of rows still alive in a pipeline, as row indices rather than data.
///
/// A scan that reads 64 Ki rows and applies three predicates should not copy the
/// surviving rows three times. Instead, each predicate narrows a `Selection` —
/// a list of indices into the *original* batch — and the copy happens once, at
/// the point where an operator genuinely needs contiguous data (a hash join
/// build, an output batch). This is the standard "late materialisation" design
/// and it is why filter pushdown pays off.
///
/// The `All` variant is not an optimisation detail: an unfiltered scan is the
/// common case, and representing it as `0..n` in a `Vec` would cost a
/// megabyte-scale allocation and destroy the sequential access pattern for no
/// reason. Kernels are expected to match on it and take the dense path.
///
/// Indices are always **strictly increasing**, which keeps `intersect`/`union`
/// linear and preserves the source order that so many operators rely on.
///
/// ```
/// use basalt_types::{Bitmap, Selection};
///
/// let sel = Selection::all(6);
/// assert!(sel.is_all());
///
/// // First predicate keeps the even rows...
/// let sel = sel.refine(&Bitmap::from_bools(&[true, false, true, false, true, false])).unwrap();
/// assert_eq!(sel.indices().collect::<Vec<_>>(), vec![0, 2, 4]);
///
/// // ...the second runs only over those three rows.
/// let sel = sel.refine(&Bitmap::from_bools(&[false, true, true])).unwrap();
/// assert_eq!(sel.indices().collect::<Vec<_>>(), vec![2, 4]);
/// assert_eq!(sel.selectivity(), 2.0 / 6.0);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub enum Selection {
    /// Every row of a batch of this length is selected.
    All(usize),
    /// The listed rows are selected, in strictly increasing order.
    Some {
        /// Selected row indices, strictly increasing and `< source_len`.
        indices: Vec<u32>,
        /// Length of the batch the indices refer to.
        source_len: usize,
    },
}

impl Selection {
    /// Everything in a batch of `len` rows.
    pub fn all(len: usize) -> Selection {
        Selection::All(len)
    }

    /// Nothing, out of a batch of `len` rows.
    pub fn none(len: usize) -> Selection {
        Selection::Some {
            indices: Vec::new(),
            source_len: len,
        }
    }

    /// Build from explicit indices, checking the ordering and range invariants.
    pub fn from_indices(indices: Vec<u32>, source_len: usize) -> Result<Selection> {
        for pair in indices.windows(2) {
            if pair[0] >= pair[1] {
                return Err(BasaltError::execution(format!(
                    "selection indices must be strictly increasing, found {} then {}",
                    pair[0], pair[1]
                )));
            }
        }
        if let Some(&last) = indices.last() {
            if last as usize >= source_len {
                return Err(BasaltError::execution(format!(
                    "selection index {last} out of range for {source_len} rows"
                )));
            }
        }
        Ok(Selection::Some {
            indices,
            source_len,
        })
    }

    /// Build from a predicate result. O(source_len / 64 + selected).
    pub fn from_bitmap(mask: &Bitmap) -> Selection {
        if mask.count_ones() == mask.len() {
            return Selection::All(mask.len());
        }
        Selection::Some {
            indices: mask.iter_set_indices().map(|i| i as u32).collect(),
            source_len: mask.len(),
        }
    }

    /// Number of selected rows.
    pub fn len(&self) -> usize {
        match self {
            Selection::All(n) => *n,
            Selection::Some { indices, .. } => indices.len(),
        }
    }

    /// `true` if nothing is selected.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Length of the batch these indices refer to.
    pub fn source_len(&self) -> usize {
        match self {
            Selection::All(n) => *n,
            Selection::Some { source_len, .. } => *source_len,
        }
    }

    /// `true` if this is the dense [`Selection::All`] case.
    pub fn is_all(&self) -> bool {
        matches!(self, Selection::All(_))
    }

    /// Fraction of the source rows that survive, in `0.0..=1.0`.
    ///
    /// An empty source is defined as fully selective (`1.0`) so that cost
    /// estimates never divide by zero.
    pub fn selectivity(&self) -> f64 {
        if self.source_len() == 0 {
            1.0
        } else {
            self.len() as f64 / self.source_len() as f64
        }
    }

    /// Iterate the selected row indices in increasing order.
    pub fn indices(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        match self {
            Selection::All(n) => Box::new(0..*n as u32),
            Selection::Some { indices, .. } => Box::new(indices.iter().copied()),
        }
    }

    /// Materialise the indices as a slice, allocating for [`Selection::All`].
    pub fn to_indices(&self) -> Vec<u32> {
        self.indices().collect()
    }

    /// Render as a bitmap over the source rows. O(source_len).
    pub fn to_bitmap(&self) -> Bitmap {
        match self {
            Selection::All(n) => Bitmap::new_set(*n),
            Selection::Some {
                indices,
                source_len,
            } => {
                let mut b = Bitmap::new_unset(*source_len);
                for &i in indices {
                    b.set(i as usize, true);
                }
                b
            }
        }
    }

    /// Narrow this selection with a predicate evaluated over the *selected*
    /// rows.
    ///
    /// This is the operation that makes chained predicates cheap: `mask` has one
    /// bit per currently-selected row (so `mask.len() == self.len()`), not one
    /// per source row, so the second predicate only pays for the rows the first
    /// one let through.
    ///
    /// # Errors
    ///
    /// If `mask.len() != self.len()`.
    pub fn refine(&self, mask: &Bitmap) -> Result<Selection> {
        if mask.len() != self.len() {
            return Err(BasaltError::execution(format!(
                "refinement mask has {} bits but {} rows are selected",
                mask.len(),
                self.len()
            )));
        }
        let source_len = self.source_len();
        match self {
            Selection::All(_) => Ok(Selection::from_bitmap(mask)),
            Selection::Some { indices, .. } => {
                let kept: Vec<u32> = mask.iter_set_indices().map(|i| indices[i]).collect();
                Ok(Selection::Some {
                    indices: kept,
                    source_len,
                })
            }
        }
    }

    /// Rows selected by both. O(a + b).
    ///
    /// # Errors
    ///
    /// If the two selections describe batches of different lengths.
    pub fn intersect(&self, other: &Selection) -> Result<Selection> {
        self.check_compatible(other, "intersect")?;
        match (self, other) {
            (Selection::All(_), s) | (s, Selection::All(_)) => Ok(s.clone()),
            (
                Selection::Some {
                    indices: left,
                    source_len,
                },
                Selection::Some { indices: right, .. },
            ) => {
                let mut out = Vec::with_capacity(left.len().min(right.len()));
                let (mut li, mut ri) = (0, 0);
                while li < left.len() && ri < right.len() {
                    match left[li].cmp(&right[ri]) {
                        std::cmp::Ordering::Equal => {
                            out.push(left[li]);
                            li += 1;
                            ri += 1;
                        }
                        std::cmp::Ordering::Less => li += 1,
                        std::cmp::Ordering::Greater => ri += 1,
                    }
                }
                Ok(Selection::Some {
                    indices: out,
                    source_len: *source_len,
                })
            }
        }
    }

    /// Rows selected by either. O(a + b).
    ///
    /// # Errors
    ///
    /// If the two selections describe batches of different lengths.
    pub fn union(&self, other: &Selection) -> Result<Selection> {
        self.check_compatible(other, "union")?;
        match (self, other) {
            (Selection::All(n), _) | (_, Selection::All(n)) => Ok(Selection::All(*n)),
            (
                Selection::Some {
                    indices: left,
                    source_len,
                },
                Selection::Some { indices: right, .. },
            ) => {
                let mut out = Vec::with_capacity(left.len() + right.len());
                let (mut li, mut ri) = (0, 0);
                while li < left.len() || ri < right.len() {
                    let take_left = match (left.get(li), right.get(ri)) {
                        (Some(x), Some(y)) => x <= y,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    let value = if take_left { left[li] } else { right[ri] };
                    if out.last() != Some(&value) {
                        out.push(value);
                    }
                    if take_left {
                        li += 1;
                    } else {
                        ri += 1;
                    }
                }
                Ok(Selection::Some {
                    indices: out,
                    source_len: *source_len,
                })
            }
        }
    }

    /// Materialise the selected rows of a column.
    ///
    /// O(1) when the selection is [`Selection::All`] (the column is returned as
    /// a cheap clone); O(selected) otherwise.
    pub fn apply(&self, column: &ColumnVector) -> Result<ColumnVector> {
        if column.len() != self.source_len() {
            return Err(BasaltError::execution(format!(
                "selection covers {} rows but the column has {}",
                self.source_len(),
                column.len()
            )));
        }
        match self {
            Selection::All(_) => Ok(column.clone()),
            Selection::Some { indices, .. } => column.take(indices),
        }
    }

    fn check_compatible(&self, other: &Selection, op: &str) -> Result<()> {
        if self.source_len() != other.source_len() {
            return Err(BasaltError::execution(format!(
                "cannot {op} selections over {} and {} rows",
                self.source_len(),
                other.source_len()
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Selection::All(n) => write!(f, "Selection::All({n})"),
            Selection::Some {
                indices,
                source_len,
            } => {
                const PREVIEW: usize = 16;
                write!(f, "Selection::Some({}/{source_len}: ", indices.len())?;
                for (i, v) in indices.iter().take(PREVIEW).enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{v}")?;
                }
                if indices.len() > PREVIEW {
                    write!(f, ", ...")?;
                }
                f.write_str(")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::{PrimitiveArray, Value};

    #[test]
    fn from_bitmap_collapses_to_all() {
        assert_eq!(
            Selection::from_bitmap(&Bitmap::new_set(5)),
            Selection::All(5)
        );
        let s = Selection::from_bitmap(&Bitmap::from_bools(&[true, false, true]));
        assert_eq!(s.to_indices(), vec![0, 2]);
        assert_eq!(s.source_len(), 3);
        assert!(!s.is_all());
    }

    #[test]
    fn refine_composes() {
        let s = Selection::all(8);
        let s = s
            .refine(&Bitmap::from_bools(&[
                true, true, false, true, false, true, false, true,
            ]))
            .unwrap();
        assert_eq!(s.to_indices(), vec![0, 1, 3, 5, 7]);
        let s = s
            .refine(&Bitmap::from_bools(&[false, true, true, false, true]))
            .unwrap();
        assert_eq!(s.to_indices(), vec![1, 3, 7]);
        assert_eq!(s.source_len(), 8);
        assert!(s.refine(&Bitmap::new_set(9)).is_err());
    }

    #[test]
    fn set_operations() {
        let a = Selection::from_indices(vec![1, 3, 5, 7], 10).unwrap();
        let b = Selection::from_indices(vec![3, 4, 5, 9], 10).unwrap();
        assert_eq!(a.intersect(&b).unwrap().to_indices(), vec![3, 5]);
        assert_eq!(a.union(&b).unwrap().to_indices(), vec![1, 3, 4, 5, 7, 9]);
        assert_eq!(a.intersect(&Selection::all(10)).unwrap(), a);
        assert_eq!(a.union(&Selection::all(10)).unwrap(), Selection::All(10));
        assert!(a.intersect(&Selection::all(9)).is_err());
        assert!(a.union(&Selection::none(9)).is_err());
        assert_eq!(a.intersect(&Selection::none(10)).unwrap().len(), 0);
    }

    #[test]
    fn invariants_are_enforced() {
        assert!(Selection::from_indices(vec![1, 1], 5).is_err());
        assert!(Selection::from_indices(vec![2, 1], 5).is_err());
        assert!(Selection::from_indices(vec![0, 9], 5).is_err());
        assert!(Selection::from_indices(vec![], 5).is_ok());
    }

    #[test]
    fn bitmap_round_trip() {
        let mask = Bitmap::from_bools(&[false, true, true, false, true]);
        let s = Selection::from_bitmap(&mask);
        assert_eq!(s.to_bitmap(), mask);
        assert_eq!(Selection::all(4).to_bitmap(), Bitmap::new_set(4));
        assert_eq!(Selection::none(4).to_bitmap(), Bitmap::new_unset(4));
    }

    #[test]
    fn apply_materialises() {
        let col = ColumnVector::Int32(PrimitiveArray::from_vec(vec![10, 20, 30, 40]));
        let s = Selection::from_indices(vec![1, 3], 4).unwrap();
        let out = s.apply(&col).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out.value(0), Value::Int32(20));
        assert_eq!(Selection::all(4).apply(&col).unwrap(), col);
        assert!(s.apply(&col.slice(0, 3)).is_err());
    }

    #[test]
    fn selectivity_and_debug() {
        assert!((Selection::none(0).selectivity() - 1.0).abs() < f64::EPSILON);
        let quarter = Selection::from_indices(vec![0], 4).unwrap().selectivity();
        assert!((quarter - 0.25).abs() < f64::EPSILON);
        assert_eq!(format!("{:?}", Selection::all(3)), "Selection::All(3)");
        let s = Selection::from_indices((0..20).collect(), 40).unwrap();
        assert!(format!("{s:?}").contains("..."));
        assert!(Selection::none(3).is_empty());
    }
}

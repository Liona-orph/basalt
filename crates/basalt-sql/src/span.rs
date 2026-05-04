//! Byte ranges into the original SQL text.

use std::fmt;

/// A half-open byte range `[start, end)` into the SQL source string.
///
/// Every token, every error and every AST node carries one. Byte offsets are
/// used rather than line/column pairs for three reasons:
///
/// * they are what the lexer already has, so producing them is free;
/// * they compose — the span of `a + b` is `a.span.to(b.span)` — whereas
///   line/column pairs need the source text to be merged correctly;
/// * an editor integration wants byte offsets anyway, and a human wants
///   line/column, which [`Span::line_col`] computes on demand from the source.
///
/// Spans are *provenance*, not identity: AST equality deliberately ignores them
/// (see [`Expr`](crate::ast::Expr)), so that a printed-and-reparsed tree
/// compares equal to the original.
///
/// ```
/// use basalt_sql::Span;
///
/// let sql = "SELECT a\nFROM t";
/// let span = Span::new(9, 13);
/// assert_eq!(span.slice(sql), "FROM");
/// assert_eq!(span.line_col(sql), (2, 1));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span {
    /// Byte offset of the first byte of the range.
    pub start: usize,
    /// Byte offset one past the last byte of the range.
    pub end: usize,
}

impl Span {
    /// A span covering `[start, end)`.
    ///
    /// `end` is clamped up to `start` so that a `Span` is never inverted, which
    /// means callers rendering a caret never have to defend against it.
    pub const fn new(start: usize, end: usize) -> Span {
        Span {
            start,
            end: if end < start { start } else { end },
        }
    }

    /// The empty span at `offset`, used for "end of input" diagnostics.
    pub const fn at(offset: usize) -> Span {
        Span {
            start: offset,
            end: offset,
        }
    }

    /// The smallest span covering both `self` and `other`.
    #[must_use]
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// Length of the range in bytes.
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// `true` if the range covers no bytes.
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The source text this span covers.
    ///
    /// Returns `""` if the span does not lie on character boundaries of
    /// `source` or runs past its end, rather than panicking: a diagnostic
    /// printer must never be the thing that crashes the program.
    pub fn slice<'a>(&self, source: &'a str) -> &'a str {
        source.get(self.start..self.end).unwrap_or("")
    }

    /// 1-based `(line, column)` of [`Span::start`], counting columns in
    /// characters rather than bytes so that a caret lands under the right glyph
    /// for ASCII and for anything else that is one column wide.
    pub fn line_col(&self, source: &str) -> (usize, usize) {
        let upto = source.get(..self.start).unwrap_or(source);
        let line = upto.bytes().filter(|b| *b == b'\n').count() + 1;
        let line_start = upto.rfind('\n').map_or(0, |i| i + 1);
        let column = source
            .get(line_start..self.start)
            .unwrap_or("")
            .chars()
            .count()
            + 1;
        (line, column)
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "byte {}", self.start)
        } else {
            write!(f, "bytes {}..{}", self.start, self.end)
        }
    }
}

/// A [`Span`] on a syntax node that has no other content.
///
/// `*` in a projection list and `*` in `count(*)` are nodes whose entire
/// payload is their location. A bare `Span` field there would make AST equality
/// span-sensitive for exactly those two nodes and nowhere else, quietly breaking
/// the round-trip property. `NodeSpan` keeps the location for diagnostics while
/// taking no part in equality or hashing — the same rule the rest of the tree
/// follows, made explicit in the type.
///
/// ```
/// use basalt_sql::{NodeSpan, Span};
///
/// assert_eq!(NodeSpan(Span::new(0, 1)), NodeSpan(Span::new(90, 91)));
/// assert_eq!(NodeSpan(Span::new(3, 4)).0.start, 3);
/// ```
#[derive(Debug, Clone, Copy, Eq)]
pub struct NodeSpan(pub Span);

impl PartialEq for NodeSpan {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::hash::Hash for NodeSpan {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn merge_and_slice() {
        let sql = "SELECT a + b";
        let a = Span::new(7, 8);
        let b = Span::new(11, 12);
        assert_eq!(a.to(b).slice(sql), "a + b");
        assert_eq!(b.to(a), a.to(b), "merging is commutative");
        assert_eq!(a.len(), 1);
        assert!(!a.is_empty());
        assert!(Span::at(3).is_empty());
    }

    #[test]
    fn inverted_ranges_are_clamped() {
        let s = Span::new(10, 4);
        assert_eq!(s.start, 10);
        assert_eq!(s.end, 10);
        assert!(s.is_empty());
    }

    #[test]
    fn line_col_counts_from_one() {
        let sql = "SELECT a\nFROM t\nWHERE b";
        assert_eq!(Span::at(0).line_col(sql), (1, 1));
        assert_eq!(Span::at(7).line_col(sql), (1, 8));
        assert_eq!(Span::at(9).line_col(sql), (2, 1));
        assert_eq!(Span::at(16).line_col(sql), (3, 1));
    }

    #[test]
    fn out_of_bounds_slices_are_empty_not_panics() {
        assert_eq!(Span::new(0, 100).slice("abc"), "");
        // Byte 1 is inside the two-byte 'é'.
        assert_eq!(Span::new(0, 1).slice("é"), "");
    }

    #[test]
    fn display_distinguishes_point_from_range() {
        assert_eq!(Span::at(4).to_string(), "byte 4");
        assert_eq!(Span::new(4, 9).to_string(), "bytes 4..9");
    }
}

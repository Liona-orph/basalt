//! Diagnostics for the SQL front end.
//!
//! A parser is judged by its errors, so this module is deliberately richer than
//! "syntax error at position 12". Every [`SqlError`] carries:
//!
//! * the [`Span`] of the offending text, so a caller can underline it;
//! * a message that names **what was expected** and **what was found**;
//! * optionally a *hint*: a concrete next action, not a restatement of the
//!   problem. Hints are only attached where the fix is unambiguous — a wrong
//!   guess is worse than no guess.

use std::fmt;

use basalt_common::BasaltError;

use crate::span::Span;

/// A syntax or lexical error, located in the source text.
///
/// ```
/// use basalt_sql::parse;
///
/// let err = parse("SELECT * FORM t").unwrap_err();
/// assert!(err.to_string().contains("did you mean `FROM`"), "{err}");
/// assert_eq!(err.span().slice("SELECT * FORM t"), "FORM");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqlError {
    span: Span,
    message: String,
    hint: Option<String>,
}

impl SqlError {
    /// A plain error at `span`.
    pub fn new(span: Span, message: impl Into<String>) -> SqlError {
        SqlError {
            span,
            message: message.into(),
            hint: None,
        }
    }

    /// Attach a suggested fix (builder style).
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> SqlError {
        self.hint = Some(hint.into());
        self
    }

    /// Attach a suggested fix only when there is one.
    #[must_use]
    pub fn with_optional_hint(mut self, hint: Option<String>) -> SqlError {
        self.hint = hint;
        self
    }

    /// The byte range this error refers to.
    pub fn span(&self) -> Span {
        self.span
    }

    /// The message, without the hint.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The suggested fix, if the error carries one.
    pub fn hint(&self) -> Option<&str> {
        self.hint.as_deref()
    }

    /// Render a multi-line diagnostic with the offending line and a caret.
    ///
    /// `source` must be the same string the error was produced from; if it is
    /// not, the output degrades to the single-line [`Display`](fmt::Display)
    /// form rather than pointing somewhere misleading.
    ///
    /// ```
    /// use basalt_sql::parse;
    ///
    /// let sql = "SELECT a FROM";
    /// let rendered = parse(sql).unwrap_err().render(sql);
    /// assert!(rendered.contains("1:14"), "{rendered}");
    /// assert!(rendered.contains('^'), "{rendered}");
    /// ```
    pub fn render(&self, source: &str) -> String {
        if self.span.end > source.len() {
            return self.to_string();
        }
        let (line, column) = self.span.line_col(source);
        let line_start = source
            .get(..self.span.start)
            .and_then(|s| s.rfind('\n').map(|i| i + 1))
            .unwrap_or(0);
        let line_end = source[line_start..]
            .find('\n')
            .map_or(source.len(), |i| line_start + i);
        let text = source.get(line_start..line_end).unwrap_or("");
        // A tab is one byte but many columns; rendering it as a space keeps the
        // caret aligned without needing to know the reader's tab stop.
        let text: String = text
            .chars()
            .map(|c| if c == '\t' { ' ' } else { c })
            .collect();

        let width = self.span.len().max(1);
        let gutter = format!("{line}");
        let pad = " ".repeat(gutter.len());
        let mut out = format!("error: {}\n", self.message);
        out.push_str(&format!("{pad}--> {line}:{column}\n"));
        out.push_str(&format!("{pad} |\n"));
        out.push_str(&format!("{gutter} | {text}\n"));
        out.push_str(&format!(
            "{pad} | {}{}\n",
            " ".repeat(column - 1),
            "^".repeat(width)
        ));
        if let Some(hint) = &self.hint {
            out.push_str(&format!("{pad} = hint: {hint}\n"));
        }
        out
    }
}

impl fmt::Display for SqlError {
    /// Single-line, greppable form. [`SqlError::render`] is the pretty one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)?;
        if let Some(hint) = &self.hint {
            write!(f, " (hint: {hint})")?;
        }
        Ok(())
    }
}

impl std::error::Error for SqlError {}

impl From<SqlError> for BasaltError {
    /// SQL errors surface to the rest of the engine as
    /// [`BasaltError::Plan`], which is the category for "this query cannot be
    /// planned". The span is folded into the message because `BasaltError` is
    /// deliberately source-agnostic — it is also raised by callers who never
    /// went through SQL at all.
    fn from(err: SqlError) -> BasaltError {
        BasaltError::plan(err.to_string())
    }
}

/// Format a list of alternatives the way a person would say them.
///
/// `["a"] -> "a"`, `["a", "b"] -> "a or b"`, `["a", "b", "c"] -> "a, b or c"`.
pub(crate) fn join_alternatives(items: &[&str]) -> String {
    match items {
        [] => String::from("nothing"),
        [only] => (*only).to_string(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

/// Optimal string alignment distance (Levenshtein plus adjacent
/// transposition), capped: returns `None` as soon as the distance is known to
/// exceed `max`.
///
/// Transpositions cost 1 rather than 2 because the single most common SQL typo
/// is `FORM` for `FROM`, and plain Levenshtein rates that as two edits — the
/// same as `FROM` versus `GROUP`-shaped noise. Counting it as one is what makes
/// a tight threshold usable.
///
/// Used only for "did you mean" suggestions, where a wrong suggestion is
/// actively harmful, so the cap is tight (see [`did_you_mean`]).
fn edit_distance_within(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    // Three rows: the transposition rule reaches two rows back.
    let mut before = vec![0usize; b.len() + 1];
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 0..a.len() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for j in 0..b.len() {
            let cost = usize::from(a[i] != b[j]);
            let mut best = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
            if i > 0 && j > 0 && a[i] == b[j - 1] && a[i - 1] == b[j] {
                best = best.min(before[j - 1] + 1);
            }
            cur[j + 1] = best;
            row_min = row_min.min(best);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut before, &mut prev);
        std::mem::swap(&mut prev, &mut cur);
    }
    let distance = prev[b.len()];
    if distance <= max {
        Some(distance)
    } else {
        None
    }
}

/// The closest candidate to `word`, if one is close enough to be worth saying.
///
/// The threshold scales with the length of the word (one edit for short words,
/// two for anything from five characters) because "did you mean X" is only
/// helpful when it is usually right: suggesting `SET` for `AS` would be noise.
/// Comparison is case-insensitive so that `slect` suggests `SELECT`.
pub(crate) fn did_you_mean<'a>(
    word: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    if word.len() < 3 {
        return None;
    }
    let max = if word.chars().count() >= 5 { 2 } else { 1 };
    let lowered = word.to_ascii_lowercase();
    let mut best: Option<(usize, &str)> = None;
    for candidate in candidates {
        let Some(d) = edit_distance_within(&lowered, &candidate.to_ascii_lowercase(), max) else {
            continue;
        };
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, candidate));
        }
        if d == 0 {
            break;
        }
    }
    best.map(|(_, c)| c)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn display_includes_span_and_hint() {
        let e = SqlError::new(Span::new(3, 7), "expected `FROM`").with_hint("add a FROM clause");
        assert_eq!(
            e.to_string(),
            "expected `FROM` at bytes 3..7 (hint: add a FROM clause)"
        );
        assert_eq!(e.message(), "expected `FROM`");
        assert_eq!(e.hint(), Some("add a FROM clause"));
        assert_eq!(e.span(), Span::new(3, 7));
    }

    #[test]
    fn render_points_at_the_right_line() {
        let sql = "SELECT a,\n       b c d\nFROM t";
        let e = SqlError::new(Span::new(19, 20), "unexpected token").with_hint("remove it");
        let out = e.render(sql);
        assert!(out.contains("2:10"), "{out}");
        assert!(out.contains("       b c d"), "{out}");
        assert!(out.contains("hint: remove it"), "{out}");
        // The caret is under column 10 of the rendered line, after the
        // `1 | ` gutter.
        let caret_line = out.lines().find(|l| l.contains('^')).unwrap();
        assert_eq!(caret_line.find('^'), Some(4 + 9));
    }

    #[test]
    fn render_degrades_gracefully_for_a_foreign_source() {
        let e = SqlError::new(Span::new(100, 110), "boom");
        assert_eq!(e.render("short"), e.to_string());
    }

    #[test]
    fn alternatives_read_like_english() {
        assert_eq!(join_alternatives(&[]), "nothing");
        assert_eq!(join_alternatives(&["a"]), "a");
        assert_eq!(join_alternatives(&["a", "b"]), "a or b");
        assert_eq!(join_alternatives(&["a", "b", "c"]), "a, b or c");
    }

    #[test]
    fn suggestions_are_conservative() {
        let words = ["SELECT", "FROM", "WHERE", "GROUP"];
        assert_eq!(did_you_mean("FORM", words), Some("FROM"));
        assert_eq!(did_you_mean("slect", words), Some("SELECT"));
        assert_eq!(did_you_mean("wehre", words), Some("WHERE"));
        assert_eq!(did_you_mean("gropu", words), Some("GROUP"));
        // Too short to guess about, and too far away to guess about.
        assert_eq!(did_you_mean("xy", words), None);
        assert_eq!(did_you_mean("banana", words), None);
    }
}

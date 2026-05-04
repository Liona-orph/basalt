//! "Never panic" tests.
//!
//! The front end is the part of the engine that a caller feeds arbitrary text
//! to, so its total contract is: **any input produces `Ok` or `Err`, never a
//! panic and never a hang**. The two ways a recursive-descent parser breaks that
//! contract are a slice index on a multi-byte character boundary and a stack
//! overflow on deeply nested input; both are covered here.

#![allow(clippy::unwrap_used)]

use basalt_sql::{parse, parse_expression, parse_many, tokenize};
use proptest::prelude::*;

/// Run every entry point. Returning `()` is the assertion: reaching this line
/// means nothing panicked.
fn exercise(input: &str) {
    let _ = tokenize(input);
    let _ = parse(input);
    let _ = parse_many(input);
    let _ = parse_expression(input);
    // Errors must be renderable against the source they came from.
    if let Err(e) = parse(input) {
        let rendered = e.render(input);
        assert!(!rendered.is_empty());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Arbitrary bytes, interpreted as lossy UTF-8.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..96)) {
        exercise(&String::from_utf8_lossy(&bytes));
    }

    /// Arbitrary text over the SQL alphabet, which reaches far more of the
    /// grammar than random bytes do.
    #[test]
    fn arbitrary_sql_shaped_text_never_panics(
        input in r#"[a-zA-Z0-9_ \t\n'"()*,.;=<>+\-/%|$?!&:@#\\]{0,120}"#
    ) {
        exercise(&input);
    }

    /// Fragments of real SQL glued together in random order: the most effective
    /// generator, because every piece is individually meaningful.
    #[test]
    fn shuffled_sql_fragments_never_panic(
        pieces in prop::collection::vec(
            prop::sample::select(vec![
                "SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
                "JOIN", "ON", "USING", "NATURAL", "LEFT", "UNION", "ALL", "WITH", "AS",
                "CASE", "WHEN", "THEN", "END", "CAST", "EXTRACT", "OVER", "PARTITION",
                "(", ")", ",", "*", "+", "-", "1", "'x'", "a", "t.b", "NULL", "NOT",
                "AND", "OR", "IS", "IN", "BETWEEN", "LIKE", "EXISTS", "VALUES", ";",
                "/*", "*/", "--", "\"", "'", "$1", "?",
            ]),
            0..24,
        ),
    ) {
        exercise(&pieces.join(" "));
    }
}

#[test]
fn pathological_shapes_terminate() {
    // Each of these has broken a hand-written parser somewhere.
    for input in [
        "(".repeat(10_000),
        ")".repeat(10_000),
        "/*".repeat(10_000),
        "*/".repeat(10_000),
        "-".repeat(10_000),
        "SELECT ".repeat(5_000),
        format!("SELECT {}", "NOT ".repeat(10_000)),
        format!("SELECT {}", "-".repeat(10_000)),
        format!("SELECT a{}", " + a".repeat(10_000)),
        format!("SELECT {}1{}", "(".repeat(5_000), ")".repeat(5_000)),
        format!("SELECT * FROM {}", "t JOIN ".repeat(2_000)),
        format!("{}SELECT 1", "WITH c AS (".repeat(2_000)),
        "'".repeat(10_000),
        "\"".repeat(10_000),
        "\u{10FFFF}".repeat(1_000),
    ] {
        exercise(&input);
    }
}

#[test]
fn multi_byte_characters_do_not_split() {
    // Every diagnostic slices the source with a span; a span that lands inside a
    // character would panic in `Index<Range<usize>>` if `Span::slice` used it.
    for input in [
        "SELECT é",
        "SELECT 'é",
        "SELECT \"é",
        "SELECT é!é",
        "SELECT ¿",
        "SELECT 🦀 FROM t",
        "SELECT a FROM 🦀",
        "-- é\nSELECT ✓",
        "/* é",
    ] {
        exercise(input);
        if let Err(e) = parse(input) {
            // The span must land on character boundaries of the input.
            assert!(input.is_char_boundary(e.span().start), "{input}: {e}");
            assert!(input.is_char_boundary(e.span().end), "{input}: {e}");
        }
    }
}

#[test]
fn deeply_nested_input_is_rejected_rather_than_overflowing() {
    for input in [
        format!("SELECT {}1", "(".repeat(10_000)),
        format!("SELECT * FROM {}t", "(".repeat(10_000)),
        format!("SELECT {}1{}", "(SELECT ".repeat(2_000), ")".repeat(2_000)),
    ] {
        let err = parse(&input).unwrap_err();
        assert!(
            err.message().contains("nests more than") || err.message().contains("expected"),
            "{err}"
        );
    }
}

//! The rejected-statement corpus.
//!
//! Each case asserts three things: that the statement is rejected, that the
//! message names the actual problem, and that the span underlines the *right
//! text*. The last one is the part that rots silently — a parser can drift into
//! reporting every error at the end of the input and still pass a test that only
//! checks `is_err()`.

#![allow(clippy::unwrap_used)]

use basalt_sql::{parse, parse_expression, tokenize};

/// Assert that `sql` is rejected with a message containing `needle`, underlining
/// exactly `underlined`.
#[track_caller]
fn reject(sql: &str, needle: &str, underlined: &str) {
    let err = match parse(sql) {
        Err(e) => e,
        Ok(statement) => panic!("`{sql}` was accepted as `{statement}`"),
    };
    assert!(
        err.message().contains(needle),
        "`{sql}`\nexpected the message to contain `{needle}`, got:\n{}",
        err.render(sql)
    );
    assert_eq!(
        err.span().slice(sql),
        underlined,
        "`{sql}` underlined the wrong text:\n{}",
        err.render(sql)
    );
}

/// As [`reject`], for a bare expression.
#[track_caller]
fn reject_expr(sql: &str, needle: &str, underlined: &str) {
    let err = match parse_expression(sql) {
        Err(e) => e,
        Ok(expr) => panic!("`{sql}` was accepted as `{expr}`"),
    };
    assert!(
        err.message().contains(needle),
        "`{sql}`\nexpected the message to contain `{needle}`, got:\n{}",
        err.render(sql)
    );
    assert_eq!(err.span().slice(sql), underlined, "`{sql}`");
}

/// Assert the hint, which is the part a user actually acts on.
#[track_caller]
fn hint(sql: &str, needle: &str) {
    let err = parse(sql).unwrap_err();
    let hint = err
        .hint()
        .unwrap_or_else(|| panic!("`{sql}` produced no hint:\n{}", err.render(sql)));
    assert!(
        hint.contains(needle),
        "`{sql}`\nexpected the hint to contain `{needle}`, got `{hint}`"
    );
}

// ---------------------------------------------------------------------------
// Lexical
// ---------------------------------------------------------------------------

#[test]
fn unterminated_literals() {
    reject("SELECT 'abc", "unterminated string literal", "'abc");
    reject("SELECT \"abc", "unterminated quoted identifier", "\"abc");
    reject("SELECT /* a", "unterminated block comment", "/* a");
    reject(
        "SELECT /* a /* b */ FROM t",
        "unterminated block comment",
        "/* a /* b */ FROM t",
    );
    reject("SELECT \"\" FROM t", "must not be empty", "\"\"");
}

#[test]
fn unknown_characters() {
    reject("SELECT a ! b", "unexpected character `!`", "!");
    reject("SELECT a & b", "unexpected character `&`", "&");
    // The lexer quotes the offending character with backticks, which reads
    // oddly when the character *is* a backtick; the hint carries the advice.
    reject("SELECT `a`", "unexpected character", "`");
    hint("SELECT `a`", "double quotes");
    reject("SELECT a::INT32", "`::` is not supported", "::");
    reject("SELECT 1abc", "is not a valid number", "1abc");
    reject("SELECT $0", "parameter numbers start at 1", "$0");
    reject("SELECT $x", "expected a digit after `$`", "$x");
    reject("SELECT ?, $2", "but not both", "$2");
}

#[test]
fn lexical_hints() {
    hint("SELECT a ! b", "!=");
    hint("SELECT a & b", "AND");
    hint("SELECT a::INT32", "CAST(expr AS type)");
    hint("SELECT 'abc", "closing `'`");
    hint("SELECT \\", "backslash");
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[test]
fn statements_that_start_wrong() {
    reject("", "expected SELECT", "");
    reject("orders", "expected SELECT", "orders");
    reject("SELCT 1", "expected SELECT", "SELCT");
    reject("UPDATE t SET a = 1", "expected SELECT", "UPDATE");
}

#[test]
fn misspelled_clauses_are_diagnosed_where_they_are_written() {
    reject("SELECT * FORM t", "expected FROM", "FORM");
    reject("SELECT * FROM t WERE a", "is not a valid alias", "WERE");
    hint("SELECT * FROM t WERE a", "did you mean `WHERE`?");
    reject("SELECT * FROM t GROUP a", "expected GROUP BY", "a");
    reject("SELECT * FROM t ORDER a", "expected ORDER BY", "a");
    hint("SELECT * FORM t", "did you mean `FROM`?");
    hint("SELECT * FROM t GROUP a", "GROUP BY");
}

#[test]
fn incomplete_statements_point_at_the_end() {
    reject("SELECT", "expected an expression", "");
    reject("SELECT a FROM", "expected an identifier", "");
    reject("SELECT a FROM t WHERE", "expected an expression", "");
    reject("SELECT a FROM t GROUP BY", "expected an expression", "");
    reject("SELECT a FROM t ORDER BY", "expected an expression", "");
    reject("SELECT a FROM t LIMIT", "expected an expression", "");
    reject_expr("a +", "expected an expression", "");
    reject_expr("(a", "expected `)`", "");
}

#[test]
fn select_list_problems() {
    reject("SELECT a, FROM t", "trailing comma", ",");
    reject("SELECT a b c FROM t", "expected FROM", "c");
    reject("SELECT select FROM t", "expected an expression", "select");
    reject("SELECT a AS FROM t", "expected an identifier", "FROM");
    // The select list needs a comma after `*`, so the error belongs on the
    // token that should have been one, not on the item after it.
    reject("SELECT * t.* extra FROM t", "expected FROM", "t");
}

#[test]
fn join_problems() {
    reject("SELECT * FROM a JOIN b", "expected ON", "");
    reject("SELECT * FROM a LEFT b ON a.x = b.x", "expected JOIN", "b");
    reject(
        "SELECT * FROM a NATURAL JOIN b ON a.x = b.x",
        "cannot also have an ON or USING clause",
        "ON",
    );
    reject("SELECT * FROM a JOIN b USING x", "expected `(`", "x");
    hint("SELECT * FROM a JOIN b", "CROSS JOIN");
}

#[test]
fn expression_problems() {
    reject_expr("a IS b", "expected NULL", "b");
    reject_expr("a IS TRUE", "expected NULL", "TRUE");
    reject_expr("a BETWEEN 1", "expected AND", "");
    reject_expr("a IN 1", "expected `(`", "1");
    reject_expr("CASE END", "expected WHEN", "END");
    reject_expr("CASE WHEN a THEN 1", "expected END", "");
    reject_expr("CASE WHEN a 1 END", "expected THEN", "1");
    reject_expr("CAST(a AS)", "expected a type name", ")");
    reject_expr("CAST(a AS nosuchtype)", "unknown type name", "nosuchtype");
    reject_expr("CAST(a, INT32)", "expected AS", ",");
    reject_expr("EXTRACT(YEAR ts)", "expected FROM", "ts");
    reject_expr("EXTRACT(NOPE FROM ts)", "not a date or time field", "NOPE");
    reject_expr("EXISTS 1", "expected `(`", "1");
    reject_expr("f(a,)", "expected an expression", ")");
    reject_expr("*", "`*` is not an expression", "*");
    // A near miss gets a suggestion; `nosuchtype` above deliberately does not.
    hint("SELECT CAST(a AS BIGNT)", "did you mean");
}

#[test]
fn type_problems() {
    reject_expr("CAST(a AS DECIMAL(0))", "not a valid type", "DECIMAL(0)");
    reject_expr(
        "CAST(a AS DECIMAL(99, 2))",
        "not a valid type",
        "DECIMAL(99, 2)",
    );
    reject_expr(
        "CAST(a AS DECIMAL(5, 9))",
        "not a valid type",
        "DECIMAL(5, 9)",
    );
    reject_expr(
        "CAST(a AS TIMESTAMP(7))",
        "not a timestamp resolution",
        "TIMESTAMP(7)",
    );
    reject_expr(
        "CAST(a AS DECIMAL(1, 2, 3))",
        "at most two arguments",
        "DECIMAL(1, 2, 3)",
    );
}

#[test]
fn ddl_problems() {
    reject("CREATE t (a INT32)", "expected TABLE or VIEW", "t");
    reject("CREATE TABLE t", "expected `(`", "");
    reject("CREATE TABLE t (a)", "expected a type name", ")");
    reject("CREATE TABLE t (a INT32", "expected `)`", "");
    reject("DROP t", "expected TABLE", "t");
    reject("INSERT t VALUES (1)", "expected INTO", "t");
    reject("INSERT INTO t", "expected SELECT", "");
    reject("CREATE VIEW v SELECT 1", "expected AS", "SELECT");
    reject("WITH c SELECT 1", "expected AS", "SELECT");
    reject("WITH c AS SELECT 1", "expected `(`", "SELECT");
    hint("CREATE OR REPLACE TABLE t (a INT32)", "views");
    hint("CREATE TABLE t", "AS SELECT");
}

#[test]
fn a_second_statement_needs_a_separator() {
    reject("SELECT 1 SELECT 2", "expected end of statement", "SELECT");
    hint("SELECT 1 SELECT 2", "parse_many");
}

#[test]
fn runaway_nesting_is_an_error_not_a_crash() {
    let deep = format!("SELECT {}1{}", "(".repeat(1000), ")".repeat(1000));
    let err = parse(&deep).unwrap_err();
    assert!(err.message().contains("nests more than"), "{err}");

    let comments = "/*".repeat(1000);
    let err = tokenize(&comments).unwrap_err();
    assert!(err.message().contains("nested more than"), "{err}");
}

#[test]
fn every_rejection_carries_a_span_inside_the_input() {
    // A blanket invariant: no diagnostic may point outside the text it came
    // from, or `render` would produce nonsense.
    for sql in [
        "",
        "SELECT",
        "SELECT * FORM t",
        "SELECT 'abc",
        "SELECT a ! b",
        "CREATE TABLE t (a)",
        "SELECT * FROM a JOIN b",
        "SELECT ?, $2",
        &"(".repeat(1000),
    ] {
        let err = parse(sql).unwrap_err();
        assert!(
            err.span().end <= sql.len(),
            "`{sql}` produced a span past the end of the input: {err}"
        );
        assert!(err.span().start <= err.span().end);
        // `render` must not panic and must mention the position.
        assert!(err.render(sql).contains("-->") || err.render(sql) == err.to_string());
    }
}

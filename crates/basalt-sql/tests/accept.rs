//! The accepted-statement corpus.
//!
//! Every statement here is parsed, printed, re-parsed and compared. That single
//! loop tests three things at once: that the grammar accepts the statement, that
//! the printer produces something the grammar still accepts, and that the
//! printer preserves meaning. A parser test that only checks `is_ok()` would
//! pass for a parser that threw the tree away.

#![allow(clippy::unwrap_used)]

use basalt_sql::ast::{Expr, ExprKind, SelectItem, SetExpr, Statement};
use basalt_sql::{parse, parse_expression};

/// Parse, print, re-parse; assert the trees agree and that printing is
/// idempotent (a second round changes nothing).
#[track_caller]
fn accept(sql: &str) -> Statement {
    let first =
        parse(sql).unwrap_or_else(|e| panic!("failed to parse `{sql}`:\n{}", e.render(sql)));
    let printed = first.to_string();
    let second = parse(&printed)
        .unwrap_or_else(|e| panic!("printer produced unparsable SQL `{printed}`:\n{e}"));
    assert_eq!(first, second, "`{sql}` printed as `{printed}`");
    assert_eq!(printed, second.to_string(), "printing is not idempotent");
    first
}

#[track_caller]
fn accept_expr(sql: &str) {
    let first = parse_expression(sql)
        .unwrap_or_else(|e| panic!("failed to parse `{sql}`:\n{}", e.render(sql)));
    let printed = first.to_string();
    let second = parse_expression(&printed)
        .unwrap_or_else(|e| panic!("printer produced unparsable SQL `{printed}`:\n{e}"));
    assert_eq!(first, second, "`{sql}` printed as `{printed}`");
    assert_eq!(printed, second.to_string(), "printing is not idempotent");
}

#[test]
fn simple_selects() {
    for sql in [
        "SELECT 1",
        "SELECT 1, 2, 3",
        "SELECT *",
        "SELECT * FROM t",
        "SELECT ALL a FROM t",
        "SELECT DISTINCT a FROM t",
        "SELECT a AS x, b y FROM t",
        "SELECT t.*, u.* FROM t, u",
        "SELECT s.t.a FROM s.t",
        "select a from t where b > 1",
        "SELECT a FROM t WHERE a > 1 AND b < 2 OR c = 3",
        "SELECT a FROM t ORDER BY a",
        "SELECT a FROM t ORDER BY 1 DESC, 2 ASC NULLS FIRST",
        "SELECT a FROM t LIMIT 10",
        "SELECT a FROM t OFFSET 5",
        "SELECT a FROM t LIMIT 10 OFFSET 5",
        "SELECT a FROM t OFFSET 5 LIMIT 10",
        "SELECT a FROM t GROUP BY a",
        "SELECT a, count(*) FROM t GROUP BY a HAVING count(*) > 1",
        "SELECT \"Odd Name\", \"select\" FROM \"My Table\"",
        "SELECT a FROM t -- trailing comment",
        "SELECT /* inline /* nested */ */ a FROM t",
        "SELECT a FROM t;",
    ] {
        accept(sql);
    }
}

#[test]
fn every_join_form() {
    for sql in [
        "SELECT * FROM a JOIN b ON a.x = b.x",
        "SELECT * FROM a INNER JOIN b ON a.x = b.x",
        "SELECT * FROM a LEFT JOIN b ON a.x = b.x",
        "SELECT * FROM a LEFT OUTER JOIN b ON a.x = b.x",
        "SELECT * FROM a RIGHT JOIN b ON a.x = b.x",
        "SELECT * FROM a FULL OUTER JOIN b ON a.x = b.x",
        "SELECT * FROM a CROSS JOIN b",
        "SELECT * FROM a NATURAL JOIN b",
        "SELECT * FROM a NATURAL LEFT JOIN b",
        "SELECT * FROM a JOIN b USING (x, y)",
        "SELECT * FROM a, b, c",
        "SELECT * FROM a JOIN b ON a.x = b.x JOIN c ON b.y = c.y",
        "SELECT * FROM a JOIN (b JOIN c ON b.y = c.y) ON a.x = b.x",
        "SELECT * FROM (SELECT 1 AS x) AS s JOIN t ON s.x = t.x",
        "SELECT * FROM a AS x (p, q)",
        "SELECT * FROM a JOIN b ON a.x = b.x AND a.y > b.y OR a.z IS NULL",
    ] {
        accept(sql);
    }
}

#[test]
fn set_operations_and_ctes() {
    for sql in [
        "SELECT 1 UNION SELECT 2",
        "SELECT 1 UNION ALL SELECT 2",
        "SELECT 1 UNION DISTINCT SELECT 2",
        "SELECT 1 EXCEPT SELECT 2",
        "SELECT 1 INTERSECT SELECT 2",
        "SELECT 1 UNION SELECT 2 INTERSECT SELECT 3",
        "(SELECT 1 UNION SELECT 2) INTERSECT SELECT 3",
        "SELECT 1 UNION (SELECT 2 ORDER BY 1 LIMIT 1)",
        "VALUES (1, 2), (3, 4)",
        "SELECT 1 UNION VALUES (2)",
        "WITH c AS (SELECT 1) SELECT * FROM c",
        "WITH c (x) AS (SELECT 1), d AS (SELECT 2 FROM c) SELECT * FROM d",
        "WITH RECURSIVE c AS (SELECT 1) SELECT * FROM c",
        "WITH c AS (SELECT 1) SELECT * FROM c UNION SELECT * FROM c",
    ] {
        accept(sql);
    }
}

#[test]
fn subqueries_and_correlation() {
    for sql in [
        "SELECT (SELECT 1) AS x",
        "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.a = t.a)",
        "SELECT * FROM t WHERE NOT EXISTS (SELECT 1 FROM u)",
        "SELECT * FROM t WHERE a IN (SELECT b FROM u)",
        "SELECT * FROM t WHERE a NOT IN (SELECT b FROM u WHERE u.c = t.c)",
        "SELECT (SELECT max(u.b) FROM u WHERE u.a = t.a) FROM t",
        "SELECT * FROM (SELECT * FROM (SELECT 1 AS x) AS i) AS o",
        "SELECT * FROM t WHERE a > (SELECT avg(b) FROM u WHERE u.k = t.k)",
        "SELECT * FROM ((SELECT 1)) AS s",
    ] {
        accept(sql);
    }
}

#[test]
fn window_functions() {
    for sql in [
        "SELECT row_number() OVER () FROM t",
        "SELECT rank() OVER (PARTITION BY a) FROM t",
        "SELECT rank() OVER (ORDER BY a DESC) FROM t",
        "SELECT rank() OVER (PARTITION BY a, b ORDER BY c ASC NULLS LAST) FROM t",
        "SELECT sum(x) OVER (ROWS UNBOUNDED PRECEDING) FROM t",
        "SELECT sum(x) OVER (ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM t",
        "SELECT sum(x) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
        "SELECT sum(x) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM t",
        "SELECT lag(x, 1) OVER (ORDER BY t) - x FROM t",
    ] {
        accept(sql);
    }
}

#[test]
fn expression_forms() {
    for sql in [
        "a",
        "t.a",
        "s.t.a",
        "1",
        "-1",
        "+1",
        "1.5",
        ".5",
        "1.",
        "1e9",
        "1.5E-3",
        "'text'",
        "'it''s'",
        "NULL",
        "TRUE",
        "FALSE",
        "$1",
        "a + b - c * d / e % f",
        "a || b || c",
        "a = b",
        "a <> b",
        "a < b",
        "a <= b",
        "a > b",
        "a >= b",
        "NOT a",
        "NOT NOT a",
        "a AND b OR c",
        "a IS NULL",
        "a IS NOT NULL",
        "a IS DISTINCT FROM b",
        "a IS NOT DISTINCT FROM b",
        "a IN (1, 2, 3)",
        "a NOT IN (1)",
        "a BETWEEN 1 AND 10",
        "a NOT BETWEEN 1 AND 10",
        "a LIKE 'x%'",
        "a NOT LIKE 'x%'",
        "a ILIKE 'x%'",
        "a LIKE 'x!%' ESCAPE '!'",
        "CASE WHEN a THEN 1 ELSE 2 END",
        "CASE a WHEN 1 THEN 'x' WHEN 2 THEN 'y' ELSE 'z' END",
        "CASE WHEN a IS NULL THEN NULL END",
        "CAST(a AS INT64)",
        "CAST(CAST(a AS UTF8) AS INT64)",
        "EXTRACT(YEAR FROM ts)",
        "EXTRACT(EPOCH FROM ts)",
        "f()",
        "f(a)",
        "f(a, b, c)",
        "count(*)",
        "count(DISTINCT a)",
        "upper(trim(a)) || '-' || lower(b)",
        "(a + b) * (c - d)",
        "a + (b + c)",
        "((a))",
        "- -a",
        "-(a + b)",
        "coalesce(a, b, 0) > 0",
    ] {
        accept_expr(sql);
    }
}

#[test]
fn ddl_and_dml() {
    for sql in [
        "CREATE TABLE t (a INT32)",
        "CREATE TABLE IF NOT EXISTS s.t (a INT32 NOT NULL, b UTF8 NULL, c FLOAT64)",
        "CREATE TABLE t (a INT32 PRIMARY KEY, b INT32 UNIQUE, c INT32 DEFAULT 0)",
        "CREATE TABLE t (a DECIMAL(10, 2), b TIMESTAMP(ns), c DATE32, d BINARY)",
        "CREATE TABLE t AS SELECT a FROM u",
        "DROP TABLE t",
        "DROP TABLE IF EXISTS a, b, c",
        "CREATE VIEW v AS SELECT 1",
        "CREATE OR REPLACE VIEW v (x, y) AS SELECT 1, 2",
        "DROP VIEW IF EXISTS v",
        "INSERT INTO t VALUES (1, 2)",
        "INSERT INTO t (a, b) VALUES (1, 2), (3, 4)",
        "INSERT INTO t SELECT * FROM u",
        "INSERT INTO t (a) WITH c AS (SELECT 1) SELECT * FROM c",
        "EXPLAIN SELECT 1",
        "EXPLAIN ANALYZE SELECT 1",
        "EXPLAIN VERBOSE SELECT 1",
        "EXPLAIN ANALYZE VERBOSE INSERT INTO t VALUES (1)",
    ] {
        accept(sql);
    }
}

#[test]
fn gnarly_nesting() {
    accept(
        "WITH recent AS (
             SELECT o.customer_id, o.total, o.placed_at
             FROM orders AS o
             WHERE o.placed_at >= CAST('2024-01-01' AS DATE32)
         ),
         ranked AS (
             SELECT r.customer_id,
                    r.total,
                    rank() OVER (PARTITION BY r.customer_id ORDER BY r.total DESC) AS rk
             FROM recent AS r
         )
         SELECT c.name,
                ranked.total,
                (SELECT count(*) FROM orders AS o2 WHERE o2.customer_id = c.id) AS order_count,
                CASE WHEN ranked.total > 1000 THEN 'large'
                     WHEN ranked.total > 100 THEN 'medium'
                     ELSE 'small' END AS bucket
         FROM ranked
         JOIN customers AS c ON c.id = ranked.customer_id
         LEFT JOIN regions AS g USING (region_id)
         WHERE ranked.rk = 1
           AND c.name NOT LIKE 'test%'
           AND EXISTS (SELECT 1 FROM payments AS p WHERE p.customer_id = c.id AND p.amount > 0)
         GROUP BY c.name, ranked.total, c.id
         HAVING count(*) BETWEEN 1 AND 100
         ORDER BY ranked.total DESC NULLS LAST, c.name
         LIMIT 50 OFFSET 10",
    );
}

#[test]
fn the_tree_actually_holds_what_was_written() {
    // A round-trip test can be fooled by a printer that drops the same thing
    // the parser dropped, so check some structure directly.
    let Statement::Query(query) = accept("SELECT a AS x, * FROM t WHERE b GROUP BY c HAVING d")
    else {
        panic!("expected a query");
    };
    let SetExpr::Select(select) = &query.body else {
        panic!("expected a select");
    };
    assert_eq!(select.projection.len(), 2);
    assert!(matches!(select.projection[1], SelectItem::Wildcard(_)));
    assert_eq!(select.from.len(), 1);
    assert!(select.selection.is_some());
    assert_eq!(select.group_by.len(), 1);
    assert!(select.having.is_some());

    let expr = parse_expression("a + b * c").unwrap();
    let ExprKind::BinaryOp { op, right, .. } = &expr.kind else {
        panic!("expected a binary op");
    };
    assert_eq!(op.as_str(), "+", "`*` must bind tighter than `+`");
    assert!(matches!(right.kind, ExprKind::BinaryOp { .. }));
}

#[test]
fn spans_cover_the_text_they_describe() {
    let sql = "SELECT alpha + beta FROM t";
    let Statement::Query(query) = parse(sql).unwrap() else {
        panic!("expected a query");
    };
    let SetExpr::Select(select) = &query.body else {
        panic!("expected a select");
    };
    let SelectItem::Expr { expr, .. } = &select.projection[0] else {
        panic!("expected an expression item");
    };
    assert_eq!(expr.span.slice(sql), "alpha + beta");
    let ExprKind::BinaryOp { left, right, .. } = &expr.kind else {
        panic!("expected a binary op");
    };
    assert_eq!(left.span.slice(sql), "alpha");
    assert_eq!(right.span.slice(sql), "beta");
}

#[test]
fn parenthesised_expressions_widen_their_span_but_not_their_shape() {
    let sql = "(a + b)";
    let parenthesised: Expr = parse_expression(sql).unwrap();
    assert_eq!(parenthesised.span.slice(sql), "(a + b)");
    assert_eq!(parenthesised, parse_expression("a + b").unwrap());
}

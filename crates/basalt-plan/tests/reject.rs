//! The rejected-statement corpus for the binder.
//!
//! Each case asserts three things: that the statement is rejected, that the
//! message names the actual problem, and that it names *the thing that is
//! wrong* — the column, the function, the clause, the two types. The last one
//! is the part that rots silently: a binder can drift into reporting every
//! failure as "planning error" and still pass a test that only checks
//! `is_err()`.
//!
//! The corpus is also the honest record of what Basalt does **not** support.
//! Everything in [`parsed_but_not_planned`] is valid SQL that the parser
//! accepts and the binder refuses; those refusals are deliberate, and the
//! README's limitations section is generated from the same list by hand.
//!
//! The style follows `basalt-sql/tests/reject.rs`.

#![allow(clippy::unwrap_used)]

use basalt_common::BasaltError;
use basalt_plan::{bind_statement, Catalog, TableDefinition};
use basalt_types::{DataType, Field, Schema, TimeUnit};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn table(name: &str, columns: &[(&str, DataType, bool)]) -> TableDefinition {
    let fields = columns
        .iter()
        .map(|(name, data_type, nullable)| Field::new(*name, data_type.clone(), *nullable))
        .collect();
    TableDefinition::new(name, Schema::new(fields).unwrap())
}

fn catalog() -> Catalog {
    let mut catalog = Catalog::new();
    catalog
        .create_table(table(
            "orders",
            &[
                ("id", DataType::Int64, false),
                ("customer_id", DataType::Int64, false),
                ("total", DataType::Float64, true),
                ("status", DataType::Utf8, true),
                ("qty", DataType::Int32, false),
                (
                    "placed_at",
                    DataType::Timestamp(TimeUnit::Microsecond),
                    true,
                ),
            ],
        ))
        .unwrap();
    catalog
        .create_table(table(
            "customers",
            &[
                ("id", DataType::Int64, false),
                ("name", DataType::Utf8, false),
                ("region", DataType::Utf8, true),
                ("vip", DataType::Boolean, true),
            ],
        ))
        .unwrap();
    catalog
        .create_table(table(
            "regions",
            &[
                ("region", DataType::Utf8, false),
                ("continent", DataType::Utf8, false),
            ],
        ))
        .unwrap();
    catalog
        .create_table(table(
            "payments",
            &[
                ("order_id", DataType::Int64, false),
                ("amount", DataType::Decimal128(10, 2), true),
                ("currency", DataType::Utf8, false),
            ],
        ))
        .unwrap();
    catalog
}

/// Bind `sql`, expecting failure, and return the error.
#[track_caller]
fn error(sql: &str) -> BasaltError {
    let statement = basalt_sql::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not even parse:\n{}", e.render(sql)));
    match bind_statement(&statement, &catalog()) {
        Err(e) => e,
        Ok(_) => panic!("`{sql}` was accepted"),
    }
}

/// Assert that `sql` is rejected with a message containing every needle.
///
/// Needles are the message *and* what it points at: the phrasing of the rule
/// plus the identifier or type that broke it.
#[track_caller]
fn reject(sql: &str, needles: &[&str]) -> BasaltError {
    let err = error(sql);
    let message = err.to_string();
    for needle in needles {
        assert!(
            message.contains(needle),
            "`{sql}`\nexpected the message to contain `{needle}`, got:\n  {message}"
        );
    }
    err
}

/// As [`reject`], and additionally assert the error is
/// [`BasaltError::Unsupported`] rather than a user mistake — the distinction
/// callers rely on to tell "you asked for something wrong" from "we have not
/// built that yet".
#[track_caller]
fn reject_unsupported(sql: &str, needles: &[&str]) {
    let err = reject(sql, needles);
    assert!(
        matches!(err, BasaltError::Unsupported(_)),
        "`{sql}` should be Unsupported, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

#[test]
fn unknown_names() {
    reject("SELECT nope FROM orders", &["no column named", "nope"]);
    reject(
        "SELECT o.total FROM orders",
        &["no column named", "o.total"],
    );
    reject(
        "SELECT id FROM nosuch",
        &["unknown table", "nosuch", "known tables are", "orders"],
    );
    reject("SELECT id FROM orders GROUP BY nope", &["no column named"]);
    // The catalog names candidates, because "not found" with no candidates is
    // the least useful error in a database.
    let err = error("SELECT id FROM ordres");
    assert!(err.to_string().contains("customers, orders"), "{err}");
}

#[test]
fn ambiguous_columns_are_refused_not_guessed() {
    reject(
        "SELECT id FROM orders, customers",
        &["ambiguous", "id", "orders.id:INT64", "customers.id:INT64"],
    );
    reject(
        "SELECT * FROM orders AS a JOIN orders AS b ON a.id = b.id WHERE total > 1",
        &["ambiguous", "a.total", "b.total"],
    );
    // A qualifier removes the ambiguity.
    assert!(basalt_sql::parse("SELECT orders.id FROM orders, customers")
        .map(|s| bind_statement(&s, &catalog()))
        .unwrap()
        .is_ok());
}

#[test]
fn a_multi_part_qualifier_is_refused() {
    reject_unsupported(
        "SELECT s.t.a FROM orders",
        &["s.t.a", "only relation.column is supported"],
    );
}

#[test]
fn a_qualified_wildcard_must_match_a_relation() {
    reject(
        "SELECT nosuch.* FROM orders",
        &["nosuch.*", "does not match any relation"],
    );
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

#[test]
fn an_unknown_function_suggests_the_closest_name() {
    reject(
        "SELECT lenght(name) FROM customers",
        &["unknown function", "lenght", "did you mean length?"],
    );
    reject(
        "SELECT cont(*) FROM orders",
        &["unknown function", "did you mean count?"],
    );
    // Beyond edit distance two the suggestion is noise, so there is none: a
    // wrong "did you mean" is worse than no hint at all.
    let err = error("SELECT xyzzy(name) FROM customers");
    assert!(err.to_string().contains("unknown function xyzzy"), "{err}");
    assert!(!err.to_string().contains("did you mean"), "{err}");
}

#[test]
fn function_arity_is_checked_against_the_registry() {
    reject(
        "SELECT lower(name, 1) FROM customers",
        &["lower", "expects 1 argument(s), got 2"],
    );
    reject(
        "SELECT substr(name) FROM customers",
        &["substr", "expects 2 or 3 argument(s), got 1"],
    );
    reject(
        "SELECT count(id, qty) FROM orders",
        &["count takes exactly one argument, found 2"],
    );
}

#[test]
fn a_wildcard_argument_belongs_to_count_alone() {
    reject(
        "SELECT sum(*) FROM orders",
        &["sum(*) is not valid", "only count(*) may take a wildcard"],
    );
    reject(
        "SELECT count(*, id) FROM orders",
        &["count(*) does not take other arguments"],
    );
}

#[test]
fn a_function_argument_must_reach_the_declared_type() {
    reject(
        "SELECT sum(status) FROM orders",
        &["sum", "cannot sum UTF8"],
    );
    reject(
        "SELECT avg(name) FROM customers",
        &["avg", "requires a numeric argument, got UTF8"],
    );
    // Text does not reach a number: there is no locale-free reading of '1,5'.
    reject(
        "SELECT sqrt(status) FROM orders",
        &["sqrt", "UTF8 cannot be used where FLOAT64 is required"],
    );
}

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

#[test]
fn a_column_outside_group_by_must_be_aggregated() {
    reject(
        "SELECT status, qty FROM orders GROUP BY status",
        &[
            "orders.qty",
            "must appear in GROUP BY or be used in an aggregate function",
            "currently grouping by status",
        ],
    );
    // With no GROUP BY at all the hint is a different one.
    reject(
        "SELECT id, sum(total) FROM orders",
        &[
            "orders.id",
            "add a GROUP BY clause, or wrap it in an aggregate such as min() or max()",
        ],
    );
    reject(
        "SELECT DISTINCT id FROM orders GROUP BY status",
        &["orders.id", "must appear in GROUP BY"],
    );
}

#[test]
fn aggregates_cannot_be_nested() {
    reject(
        "SELECT sum(count(id)) FROM orders",
        &["sum contains another aggregate", "cannot be nested"],
    );
    reject(
        "SELECT sum(avg(qty)) FROM orders",
        &["sum contains another aggregate"],
    );
}

#[test]
fn an_aggregate_is_illegal_in_the_clauses_that_run_before_it() {
    // The message names the clause, which is the whole point of tracking where
    // the aggregate was written.
    reject(
        "SELECT id FROM orders WHERE count(*) > 1",
        &["aggregate function count is not allowed in WHERE"],
    );
    reject(
        "SELECT id FROM orders GROUP BY count(*)",
        &["aggregate function count is not allowed in GROUP BY"],
    );
    reject(
        "SELECT status, count(*) FROM orders GROUP BY status ORDER BY count(*)",
        &["aggregate function count is not allowed in ORDER BY"],
    );
    reject(
        "SELECT o.id FROM orders AS o JOIN customers AS c ON count(o.id) > 0",
        &["aggregate function count is not allowed in JOIN ON"],
    );
    reject(
        "SELECT sum(total) OVER (PARTITION BY count(id)) FROM orders",
        &["not allowed in a window function's OVER clause"],
    );
}

#[test]
fn having_without_grouping_is_a_where_clause_in_disguise() {
    reject(
        "SELECT id FROM orders HAVING id > 1",
        &[
            "HAVING requires GROUP BY or an aggregate",
            "use WHERE to filter individual rows",
        ],
    );
}

#[test]
fn a_window_function_is_not_an_aggregate() {
    reject(
        "SELECT customer_id, sum(total) OVER (PARTITION BY customer_id) \
         FROM orders GROUP BY customer_id",
        &[
            "sum(orders.total) OVER (...)",
            "must appear in GROUP BY or be used in an aggregate",
        ],
    );
    reject(
        "SELECT rank() FROM orders",
        &["rank requires an OVER clause"],
    );
    reject(
        "SELECT row_number(id) OVER () FROM orders",
        &["row_number takes no arguments"],
    );
    reject(
        "SELECT upper(name) OVER () FROM customers",
        &["upper is a scalar function and cannot take an OVER clause"],
    );
    reject(
        "SELECT sum(total) OVER (ROWS BETWEEN CURRENT ROW AND 1 PRECEDING) FROM orders",
        &["window frame start CURRENT ROW is after its end 1 PRECEDING"],
    );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[test]
fn sql_has_no_truthiness() {
    reject(
        "SELECT * FROM orders WHERE id",
        &["WHERE requires a boolean expression, found INT64"],
    );
    reject(
        "SELECT * FROM orders WHERE qty",
        &["WHERE requires a boolean expression, found INT32"],
    );
    reject(
        "SELECT * FROM orders AS o JOIN customers AS c ON o.id",
        &["JOIN ON requires a boolean expression, found INT64"],
    );
    reject(
        "SELECT status, count(*) FROM orders GROUP BY status HAVING count(*)",
        &["HAVING requires a boolean expression, found INT64"],
    );
}

#[test]
fn comparisons_between_unrelated_types_are_refused() {
    reject(
        "SELECT id FROM orders WHERE status > 3",
        &["> is not defined between UTF8 and INT64"],
    );
    reject(
        "SELECT id FROM orders WHERE placed_at > 5",
        &["> is not defined between TIMESTAMP(us) and INT64"],
    );
    // Implicit `true -> 1` hides real bugs, so booleans do not meet numbers.
    reject(
        "SELECT vip = 1 FROM customers",
        &["= is not defined between BOOLEAN and INT64"],
    );
    reject(
        "SELECT total + status FROM orders",
        &["operator + is not defined for FLOAT64 and UTF8"],
    );
}

#[test]
fn a_decimal_never_silently_becomes_a_float() {
    // This is the coercion lattice's most opinionated refusal: binary floating
    // point and decimal have no common exact type, and promoting to double
    // destroys the exactness that made someone choose DECIMAL. Note that a
    // fractional literal *is* a decimal, so this catches the innocent-looking
    // `total > 10.5` against a FLOAT64 column.
    reject(
        "SELECT total FROM orders WHERE total > 10.5",
        &["> is not defined between FLOAT64 and DECIMAL(3, 1)"],
    );
    reject(
        "SELECT amount + total FROM payments, orders",
        &["operator + is not defined for DECIMAL(10, 2) and FLOAT64"],
    );
}

#[test]
fn an_in_list_must_have_one_common_type() {
    reject(
        "SELECT id FROM orders WHERE id IN ('a', 'b')",
        &["IN list mixes INT64 with UTF8"],
    );
    reject(
        "SELECT id FROM orders WHERE id IN (SELECT id, qty FROM orders)",
        &["IN (subquery) requires exactly one column, found 2"],
    );
}

#[test]
fn case_branches_must_agree() {
    reject(
        "SELECT CASE WHEN vip THEN 1 ELSE 'x' END FROM customers",
        &["CASE ELSE produces UTF8 which is incompatible with INT64"],
    );
    reject(
        "SELECT CASE WHEN vip THEN 1 WHEN vip THEN 'x' END FROM customers",
        &["CASE branches produce incompatible types INT64 and UTF8"],
    );
    reject(
        "SELECT CASE WHEN id THEN 1 END FROM orders",
        &["CASE WHEN requires a boolean expression, found INT64"],
    );
}

#[test]
fn extract_needs_a_temporal_operand() {
    reject(
        "SELECT extract(YEAR FROM total) FROM orders",
        &["EXTRACT requires a DATE or TIMESTAMP, found FLOAT64"],
    );
}

// ---------------------------------------------------------------------------
// Set operations
// ---------------------------------------------------------------------------

#[test]
fn union_arms_must_line_up() {
    reject(
        "SELECT id FROM orders UNION SELECT id, qty FROM orders",
        &["UNION arms have different widths: 1 and 2 columns"],
    );
    reject(
        "SELECT status FROM orders UNION SELECT id FROM orders",
        &["UNION column 1 has incompatible types UTF8 and INT64"],
    );
    reject(
        "SELECT id, status FROM orders UNION SELECT id, qty FROM orders",
        &["UNION column 2 has incompatible types UTF8 and INT32"],
    );
}

#[test]
fn values_rows_must_line_up() {
    reject(
        "VALUES (1, 2), (3)",
        &["VALUES row 2 has 1 expressions but row 1 has 2"],
    );
    reject(
        "VALUES (1), ('x')",
        &["VALUES column 1 mixes incompatible types INT64 and UTF8"],
    );
}

// ---------------------------------------------------------------------------
// Joins
// ---------------------------------------------------------------------------

#[test]
fn using_names_a_column_that_must_exist_on_both_sides() {
    reject(
        "SELECT * FROM customers JOIN regions USING (nosuch)",
        &["USING column nosuch does not exist on the left side of the join"],
    );
    reject(
        "SELECT * FROM customers JOIN regions USING (region, id)",
        &["USING column id does not exist on the right side of the join"],
    );
    reject(
        "SELECT * FROM orders JOIN customers USING (id, status)",
        &["USING column status does not exist on the right side"],
    );
}

#[test]
fn a_natural_join_with_nothing_in_common_is_a_mistake() {
    reject(
        "SELECT * FROM orders NATURAL JOIN regions",
        &[
            "NATURAL join has no column names in common",
            "write an explicit ON clause",
        ],
    );
}

// ---------------------------------------------------------------------------
// FROM, LIMIT and other clause-level rules
// ---------------------------------------------------------------------------

#[test]
fn a_subquery_in_from_needs_an_alias() {
    reject(
        "SELECT * FROM (SELECT id FROM orders)",
        &[
            "a subquery in FROM must have an alias",
            "its columns need a name to be qualified by",
        ],
    );
}

#[test]
fn a_column_alias_list_must_match_the_relation_width() {
    reject(
        "SELECT * FROM orders AS o (a, b)",
        &["alias o names 2 columns but orders has 6"],
    );
    reject(
        "SELECT * FROM (SELECT id FROM orders) AS s (a, b)",
        &["alias s names 2 columns but its subquery produces 1"],
    );
    reject(
        "WITH c (x, y) AS (SELECT id FROM orders) SELECT * FROM c",
        &["CTE c names 2 columns but its query produces 1"],
    );
}

#[test]
fn limit_takes_a_constant_and_nothing_else() {
    reject("SELECT id FROM orders LIMIT qty", &["no column named qty"]);
    reject(
        "SELECT id FROM orders LIMIT 1 + 1",
        &["LIMIT requires a constant, found an expression"],
    );
    reject(
        "SELECT id FROM orders LIMIT -1",
        &["LIMIT requires a constant, found an expression"],
    );
    reject(
        "SELECT id FROM orders LIMIT 1.5",
        &["LIMIT requires an integer, found DECIMAL(2, 1)"],
    );
    reject(
        "SELECT id FROM orders LIMIT 'x'",
        &["LIMIT requires an integer, found UTF8"],
    );
    reject(
        "SELECT id FROM orders OFFSET 'x'",
        &["OFFSET requires an integer, found UTF8"],
    );
}

#[test]
fn an_order_by_ordinal_must_be_in_range() {
    reject(
        "SELECT id FROM orders ORDER BY 0",
        &["ORDER BY position 0 is out of range", "returns 1 columns"],
    );
    reject(
        "SELECT id FROM orders ORDER BY 9",
        &["ORDER BY position 9 is out of range"],
    );
    // ORDER BY sees the query's output, so an unprojected input column is not
    // in scope. Basalt rejects it rather than silently re-reading the input.
    reject(
        "SELECT id FROM orders ORDER BY total",
        &["no column named total"],
    );
}

#[test]
fn a_wildcard_needs_something_to_expand() {
    reject("SELECT *", &["SELECT * requires a FROM clause"]);
}

#[test]
fn insert_checks_the_target_columns() {
    reject(
        "INSERT INTO orders (id, nope) VALUES (1, 2)",
        &["table orders has no column nope"],
    );
    reject(
        "INSERT INTO orders (id, id) VALUES (1, 2)",
        &["column id is named twice in the INSERT column list"],
    );
    reject(
        "INSERT INTO orders (id) VALUES (1, 2)",
        &["INSERT INTO orders targets 1 columns but its source produces 2"],
    );
    reject(
        "INSERT INTO orders (id, status) VALUES (1, 2)",
        &["cannot insert INT64 into column status of type UTF8"],
    );
}

#[test]
fn create_table_rejects_a_duplicate_column_name() {
    reject(
        "CREATE TABLE t (a INT32, a UTF8)",
        &["duplicate column name `a` at positions 0 and 1"],
    );
}

// ---------------------------------------------------------------------------
// Parsed but not planned
// ---------------------------------------------------------------------------

/// The honest list: valid SQL that the parser accepts, the binder understands
/// well enough to name, and Basalt has not implemented. Every one of these is
/// [`BasaltError::Unsupported`], never a silently wrong plan.
#[test]
fn parsed_but_not_planned() {
    reject_unsupported(
        "SELECT (SELECT 1) AS x",
        &[
            "scalar subqueries are parsed but not yet planned",
            "rewrite as a join",
        ],
    );
    reject_unsupported(
        "SELECT id FROM orders EXCEPT SELECT id FROM orders",
        &[
            "Except is parsed but not yet planned",
            "only UNION is implemented",
        ],
    );
    reject_unsupported(
        "SELECT id FROM orders INTERSECT SELECT id FROM orders",
        &["Intersect is parsed but not yet planned"],
    );
    reject_unsupported(
        "WITH RECURSIVE c AS (SELECT 1) SELECT * FROM c",
        &["WITH RECURSIVE is parsed but not yet planned"],
    );
    reject_unsupported(
        "SELECT $1",
        &[
            "query parameters are parsed but not yet bound",
            "inline the value",
        ],
    );
    reject_unsupported(
        "CREATE VIEW v AS SELECT 1",
        &["views are parsed but not yet planned"],
    );
    reject_unsupported("DROP VIEW v", &["views are parsed but not yet planned"]);
    reject_unsupported(
        "EXPLAIN CREATE TABLE t (a INT32)",
        &[
            "EXPLAIN of CREATE TABLE is not supported",
            "only queries can be explained",
        ],
    );
}

// ---------------------------------------------------------------------------
// Blanket invariants
// ---------------------------------------------------------------------------

#[test]
fn no_rejection_is_an_internal_error() {
    // An `Internal` error means Basalt has a bug, so no user mistake may
    // produce one. This is the blanket version of every case above.
    for sql in [
        "SELECT nope FROM orders",
        "SELECT id FROM orders, customers",
        "SELECT status, qty FROM orders GROUP BY status",
        "SELECT sum(count(id)) FROM orders",
        "SELECT id FROM orders WHERE count(*) > 1",
        "SELECT id FROM orders HAVING id > 1",
        "SELECT id FROM orders WHERE status > 3",
        "SELECT id FROM orders UNION SELECT id, qty FROM orders",
        "SELECT * FROM (SELECT id FROM orders)",
        "SELECT id FROM orders LIMIT 1.5",
        "SELECT (SELECT 1)",
        "SELECT $1",
        "SELECT * FROM customers JOIN regions USING (nosuch)",
        "SELECT id FROM orders ORDER BY 0",
        "VALUES (1), ('x')",
    ] {
        let err = error(sql);
        assert!(
            !matches!(err, BasaltError::Internal { .. }),
            "`{sql}` produced an internal error: {err}"
        );
        assert!(
            !err.to_string().is_empty(),
            "`{sql}` produced an empty message"
        );
    }
}

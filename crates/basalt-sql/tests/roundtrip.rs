//! Property tests for the printer.
//!
//! The corpus in `accept.rs` proves the round trip for statements a human
//! thought of. These strategies generate statements nobody thought of — deeply
//! nested mixed-precedence expressions in particular, which is exactly where a
//! hand-written parenthesiser goes wrong.

#![allow(clippy::unwrap_used)]

use basalt_sql::{parse, parse_expression};
use proptest::prelude::*;

/// Leaf expressions: one of each interesting literal shape plus a couple of
/// column references.
fn leaf() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("a".to_string()),
        Just("b".to_string()),
        Just("t.c".to_string()),
        Just("\"Odd Name\"".to_string()),
        Just("1".to_string()),
        Just("1.50".to_string()),
        Just("1e9".to_string()),
        Just("'x''y'".to_string()),
        Just("NULL".to_string()),
        Just("TRUE".to_string()),
        Just("$1".to_string()),
    ]
}

/// Arbitrary expressions, built from every infix and prefix form the grammar
/// has. The generator emits *text*, not trees, so the parser is exercised too.
fn arbitrary_expression() -> impl Strategy<Value = String> {
    leaf().prop_recursive(6, 96, 4, |inner| {
        let binary_op = prop_oneof![
            Just("OR"),
            Just("AND"),
            Just("="),
            Just("<>"),
            Just("<"),
            Just(">="),
            Just("||"),
            Just("+"),
            Just("-"),
            Just("*"),
            Just("/"),
            Just("%"),
        ];
        prop_oneof![
            8 => (inner.clone(), binary_op, inner.clone())
                .prop_map(|(l, op, r)| format!("{l} {op} {r}")),
            2 => inner.clone().prop_map(|e| format!("({e})")),
            2 => inner.clone().prop_map(|e| format!("NOT {e}")),
            // Parenthesised so that a nested negation cannot print as `--`,
            // which every SQL dialect lexes as a line comment. This is a
            // property of the concrete syntax, not of the parser.
            2 => inner.clone().prop_map(|e| format!("-({e})")),
            1 => inner.clone().prop_map(|e| format!("{e} IS NULL")),
            1 => inner.clone().prop_map(|e| format!("{e} IS NOT NULL")),
            1 => (inner.clone(), inner.clone())
                .prop_map(|(l, r)| format!("{l} IS DISTINCT FROM ({r})")),
            // The bounds are parenthesised because SQL parses them at a
            // precedence above AND: `a BETWEEN b IS NULL AND c` is a syntax
            // error in every conforming implementation, not a Basalt
            // limitation. Parenthesising costs the test nothing -- the parser
            // discards parentheses, so the printer still has to decide from the
            // tree alone where they are needed.
            1 => (inner.clone(), inner.clone(), inner.clone())
                .prop_map(|(e, lo, hi)| format!("{e} BETWEEN ({lo}) AND ({hi})")),
            1 => (inner.clone(), inner.clone(), inner.clone())
                .prop_map(|(e, x, y)| format!("{e} NOT IN ({x}, {y})")),
            1 => (inner.clone(), inner.clone())
                .prop_map(|(e, p)| format!("{e} LIKE ({p})")),
            1 => (inner.clone(), inner.clone(), inner.clone())
                .prop_map(|(c, t, e)| format!("CASE WHEN {c} THEN {t} ELSE {e} END")),
            1 => inner.clone().prop_map(|e| format!("CAST({e} AS INT64)")),
            1 => (inner.clone(), inner.clone()).prop_map(|(x, y)| format!("f({x}, {y})")),
            1 => inner.prop_map(|e| format!("sum({e}) OVER (PARTITION BY a ORDER BY b)")),
        ]
    })
}

/// Arbitrary single-block queries around an arbitrary expression.
fn arbitrary_query() -> impl Strategy<Value = String> {
    (
        arbitrary_expression(),
        arbitrary_expression(),
        prop::option::of(Just("DISTINCT")),
        prop::option::of(prop_oneof![
            Just("JOIN u ON t.c = u.c"),
            Just("LEFT JOIN u USING (c)"),
            Just("NATURAL FULL JOIN u"),
            Just("CROSS JOIN u"),
            Just(", u"),
        ]),
        prop::option::of(prop_oneof![Just("ASC"), Just("DESC NULLS FIRST")]),
        prop::option::of(0u32..100),
    )
        .prop_map(
            |(projection, predicate, distinct, join, direction, limit)| {
                let mut sql = String::from("SELECT ");
                if let Some(d) = distinct {
                    sql.push_str(d);
                    sql.push(' ');
                }
                sql.push_str(&projection);
                sql.push_str(" FROM t");
                if let Some(join) = join {
                    sql.push(' ');
                    sql.push_str(join);
                }
                sql.push_str(" WHERE ");
                sql.push_str(&predicate);
                if let Some(direction) = direction {
                    sql.push_str(" ORDER BY 1 ");
                    sql.push_str(direction);
                }
                if let Some(limit) = limit {
                    sql.push_str(&format!(" LIMIT {limit}"));
                }
                sql
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// The headline property: printing does not change meaning.
    #[test]
    fn expressions_round_trip(sql in arbitrary_expression()) {
        let Ok(first) = parse_expression(&sql) else {
            // Generated text is always valid, so a rejection is a real failure.
            return Err(TestCaseError::fail(format!("generated `{sql}` was rejected")));
        };
        let printed = first.to_string();
        let second = parse_expression(&printed)
            .map_err(|e| TestCaseError::fail(format!("`{printed}` was rejected: {e}")))?;
        prop_assert_eq!(&first, &second, "printed as `{}`", printed);
    }

    /// Printing is idempotent: the printed form is a fixed point.
    #[test]
    fn printing_is_a_fixed_point(sql in arbitrary_expression()) {
        let printed = parse_expression(&sql)
            .map_err(|e| TestCaseError::fail(format!("generated `{sql}` was rejected: {e}")))?
            .to_string();
        let reprinted = parse_expression(&printed)
            .map_err(|e| TestCaseError::fail(format!("printed `{printed}` was rejected: {e}")))?
            .to_string();
        prop_assert_eq!(printed, reprinted);
    }

    #[test]
    fn queries_round_trip(sql in arbitrary_query()) {
        let Ok(first) = parse(&sql) else {
            return Err(TestCaseError::fail(format!("generated `{sql}` was rejected")));
        };
        let printed = first.to_string();
        let second = parse(&printed)
            .map_err(|e| TestCaseError::fail(format!("`{printed}` was rejected: {e}")))?;
        prop_assert_eq!(&first, &second, "printed as `{}`", printed);
    }

    /// Whitespace and comments between tokens never change the tree.
    #[test]
    fn trivia_is_insignificant(sql in arbitrary_expression()) {
        let plain = parse_expression(&sql)
            .map_err(|e| TestCaseError::fail(format!("generated `{sql}` was rejected: {e}")))?;
        let padded = format!("/* x */ {} -- y\n", sql.replace(" IS ", " /**/ IS "));
        let spaced = parse_expression(&padded)
            .map_err(|e| TestCaseError::fail(format!("padded `{padded}` was rejected: {e}")))?;
        prop_assert_eq!(plain, spaced);
    }
}

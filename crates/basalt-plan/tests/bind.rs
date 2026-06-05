//! The accepted-statement corpus for the binder.
//!
//! Every case here asserts three things: that the statement binds, that the
//! resulting plan passes [`LogicalPlan::validate`], and that the *resolved*
//! detail is what it should be — the type and nullability of every output
//! column, and the casts the binder inserted. The last one is the part that
//! rots silently: a binder that forgets a cast still produces a plan that looks
//! right in `EXPLAIN` and only disagrees with itself once an executor exists.
//!
//! The style follows `basalt-sql/tests/accept.rs`: a helper that does the whole
//! round trip, and assertions on structure rather than on `is_ok()`.

#![allow(clippy::unwrap_used)]

use basalt_plan::{
    bind, bind_statement, AggregateExpr, BoundStatement, Catalog, Expr, JoinType, LogicalPlan,
    PlanSchema, TableDefinition, WindowExpr,
};
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

/// Four tables chosen so that every rule the binder implements has something to
/// bite on: a mixed-width integer pair (`id`/`qty`), a nullable float, a
/// decimal, a timestamp, and two relations sharing exactly one column name
/// (`region`) so `USING` and `NATURAL` have a subject.
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

/// Parse, bind and validate. Validation runs on *every* accepted statement:
/// the binder's contract is that it only ever emits valid plans, so a test that
/// skipped it would let a schema/expression mismatch through unnoticed.
#[track_caller]
fn plan(sql: &str) -> LogicalPlan {
    let statement = basalt_sql::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse:\n{}", e.render(sql)));
    let plan = bind(&statement, &catalog()).unwrap_or_else(|e| panic!("`{sql}` was rejected: {e}"));
    plan.validate()
        .unwrap_or_else(|e| panic!("`{sql}` produced an invalid plan: {e}\n{plan}"));
    plan
}

/// The output schema as `(name, type, nullable)` triples.
#[track_caller]
fn output(sql: &str) -> Vec<(String, DataType, bool)> {
    schema_triples(plan(sql).schema())
}

fn schema_triples(schema: &PlanSchema) -> Vec<(String, DataType, bool)> {
    schema
        .fields()
        .iter()
        .map(|f| (f.name.clone(), f.data_type.clone(), f.nullable))
        .collect()
}

fn triple(name: &str, data_type: DataType, nullable: bool) -> (String, DataType, bool) {
    (name.to_string(), data_type, nullable)
}

/// The single expression a `Projection` computes for output column `index`.
#[track_caller]
fn projection_expr(plan: &LogicalPlan, index: usize) -> Expr {
    let LogicalPlan::Projection { exprs, .. } = plan else {
        panic!("expected a projection at the top of the plan, found {plan}");
    };
    exprs[index].0.clone()
}

/// Depth-first search for the first node satisfying `predicate`.
fn find<'p>(plan: &'p LogicalPlan, predicate: &impl Fn(&LogicalPlan) -> bool) -> &'p LogicalPlan {
    fn search<'p>(
        plan: &'p LogicalPlan,
        predicate: &impl Fn(&LogicalPlan) -> bool,
    ) -> Option<&'p LogicalPlan> {
        if predicate(plan) {
            return Some(plan);
        }
        plan.children()
            .into_iter()
            .find_map(|child| search(child, predicate))
    }
    search(plan, predicate).unwrap_or_else(|| panic!("no matching node in:\n{plan}"))
}

/// Every outer (correlated) column reference anywhere in a plan, rendered.
fn outer_references(plan: &LogicalPlan) -> Vec<String> {
    let mut found = Vec::new();
    collect_outer_references(plan, &mut found);
    found
}

fn collect_outer_references(plan: &LogicalPlan, found: &mut Vec<String>) {
    for expr in plan.expressions() {
        expr.visit(&mut |node| {
            if let Expr::OuterColumn(reference) = node {
                found.push(reference.to_string());
            }
            true
        });
    }
    for child in plan.children() {
        collect_outer_references(child, found);
    }
}

// ---------------------------------------------------------------------------
// Projection, types and nullability
// ---------------------------------------------------------------------------

#[test]
fn a_projection_carries_the_column_types_through() {
    assert_eq!(
        output("SELECT id, total, status FROM orders"),
        vec![
            triple("id", DataType::Int64, false),
            triple("total", DataType::Float64, true),
            triple("status", DataType::Utf8, true),
        ]
    );
    // `SELECT *` expands in declaration order and keeps each column's relation
    // qualifier, so `SELECT *` over a join still says which side a column is
    // from. An explicitly written expression has no qualifier.
    let all = plan("SELECT * FROM orders");
    assert_eq!(all.schema().len(), 6);
    assert_eq!(
        all.schema().field(0).unwrap().qualifier.as_deref(),
        Some("orders")
    );
    assert_eq!(
        plan("SELECT id FROM orders")
            .schema()
            .field(0)
            .unwrap()
            .qualifier,
        None
    );
}

#[test]
fn an_alias_renames_but_does_not_retype() {
    assert_eq!(
        output("SELECT total AS revenue FROM orders"),
        vec![triple("revenue", DataType::Float64, true)]
    );
}

#[test]
fn a_select_without_from_is_one_row_of_constants() {
    let plan = plan("SELECT 1, 'x', NULL");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![
            triple("1", DataType::Int64, false),
            triple("'x'", DataType::Utf8, false),
            triple("NULL", DataType::Null, true),
        ]
    );
    assert!(matches!(
        find(&plan, &|node| matches!(
            node,
            LogicalPlan::EmptyRelation { .. }
        )),
        LogicalPlan::EmptyRelation {
            produce_one_row: true,
            ..
        }
    ));
}

#[test]
fn values_takes_the_common_type_of_each_column() {
    assert_eq!(
        output("VALUES (1, 'a'), (2, 'b')"),
        vec![
            triple("column1", DataType::Int64, false),
            triple("column2", DataType::Utf8, false),
        ]
    );
}

// ---------------------------------------------------------------------------
// Inserted casts
// ---------------------------------------------------------------------------

#[test]
fn mixed_width_arithmetic_casts_the_narrower_operand() {
    let plan = plan("SELECT id + qty AS n FROM orders");
    assert_eq!(
        projection_expr(&plan, 0).to_string(),
        "(orders.id + CAST(orders.qty AS INT64))"
    );
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple("n", DataType::Int64, false)]
    );
}

#[test]
fn division_always_widens_to_float_and_casts_both_sides() {
    let plan = plan("SELECT total / qty FROM orders");
    assert_eq!(
        projection_expr(&plan, 0).to_string(),
        "(orders.total / CAST(orders.qty AS FLOAT64))"
    );
    assert_eq!(plan.schema().field(0).unwrap().data_type, DataType::Float64);

    // Even integer / integer: `5 / 2` is 2.5, deliberately.
    let integers = plan_of("SELECT id / customer_id FROM orders");
    assert_eq!(
        projection_expr(&integers, 0).to_string(),
        "(CAST(orders.id AS FLOAT64) / CAST(orders.customer_id AS FLOAT64))"
    );
}

/// [`plan`] under another name, for cases that already shadow `plan` locally.
#[track_caller]
fn plan_of(sql: &str) -> LogicalPlan {
    plan(sql)
}

#[test]
fn a_literal_is_cast_to_the_column_type_not_the_other_way_round() {
    let plan = plan("SELECT id FROM orders WHERE total > 10");
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(
        predicate.to_string(),
        "(orders.total > CAST(10 AS FLOAT64))"
    );
}

#[test]
fn comparing_a_decimal_to_an_integer_widens_both_to_a_decimal() {
    // `DECIMAL(10, 2)` with `INT64` (which needs 19 integral digits) is
    // `DECIMAL(21, 2)`: every integral digit of both sides is kept, and so is
    // the finer scale. Neither side is allowed to become a float.
    let plan = plan("SELECT amount FROM payments WHERE amount > 3");
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(
        predicate.to_string(),
        "(CAST(payments.amount AS DECIMAL(21, 2)) > CAST(3 AS DECIMAL(21, 2)))"
    );
}

#[test]
fn between_is_desugared_into_two_range_predicates() {
    let plan = plan("SELECT id FROM orders WHERE qty BETWEEN 1 AND 10");
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(
        predicate.to_string(),
        "((CAST(orders.qty AS INT64) >= 1) AND (CAST(orders.qty AS INT64) <= 10))"
    );
    assert_eq!(predicate.clone().split_conjuncts().len(), 2);
}

#[test]
fn a_long_in_list_of_constants_becomes_a_hash_set() {
    let plan = plan("SELECT id FROM orders WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)");
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(predicate.to_string(), "(orders.id IN <set of 10>)");

    // A short list stays a list; the comparison chain is cheaper than hashing.
    let short = plan_of("SELECT id FROM orders WHERE id IN (1, 2)");
    let LogicalPlan::Filter { predicate, .. } =
        find(&short, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(predicate.to_string(), "(orders.id IN [1, 2])");
}

#[test]
fn scalar_function_arguments_are_cast_to_the_declared_types() {
    let plan = plan("SELECT nullif(qty, 0) FROM orders");
    assert_eq!(
        projection_expr(&plan, 0).to_string(),
        "nullif(CAST(orders.qty AS INT64), 0)"
    );
    // `nullif` can turn two non-null arguments into NULL, so the output is
    // nullable even though `qty` is not.
    assert!(plan.schema().field(0).unwrap().nullable);

    // `coalesce` is the opposite: a non-null last argument makes it non-null.
    let coalesce = plan_of("SELECT coalesce(status, 'none') FROM orders");
    assert!(!coalesce.schema().field(0).unwrap().nullable);
}

// ---------------------------------------------------------------------------
// Joins
// ---------------------------------------------------------------------------

/// `(join type, nullability of the ten output columns)`.
#[track_caller]
fn join_nullability(sql: &str) -> Vec<bool> {
    plan(sql)
        .schema()
        .fields()
        .iter()
        .map(|f| f.nullable)
        .collect()
}

#[test]
fn every_join_type_applies_its_own_nullability_rule() {
    // customers: id NOT NULL, name NOT NULL, region NULL, vip NULL
    // orders:    id NOT NULL, customer_id NOT NULL, total NULL, status NULL,
    //            qty NOT NULL, placed_at NULL
    let inner = "SELECT * FROM customers JOIN orders ON customers.id = orders.customer_id";
    assert_eq!(
        join_nullability(inner),
        vec![false, false, true, true, false, false, true, true, false, true],
        "an inner join changes nothing"
    );

    let left = "SELECT * FROM customers LEFT JOIN orders ON customers.id = orders.customer_id";
    assert_eq!(
        join_nullability(left),
        vec![false, false, true, true, true, true, true, true, true, true],
        "a left join null-extends the right side and only the right side"
    );

    let right = "SELECT * FROM customers RIGHT JOIN orders ON customers.id = orders.customer_id";
    assert_eq!(
        join_nullability(right),
        vec![true, true, true, true, false, false, true, true, false, true],
        "a right join null-extends the left side"
    );

    let full =
        "SELECT * FROM customers FULL OUTER JOIN orders ON customers.id = orders.customer_id";
    assert_eq!(
        join_nullability(full),
        vec![true; 10],
        "a full join null-extends both"
    );

    let cross = "SELECT * FROM customers CROSS JOIN regions";
    assert_eq!(
        join_nullability(cross),
        vec![false, false, true, true, false, false],
        "a cross join changes nothing"
    );
}

#[test]
fn a_comma_separated_from_list_is_a_cross_join() {
    let plan = plan("SELECT customers.id, regions.region FROM customers, regions");
    let join = find(&plan, &|node| matches!(node, LogicalPlan::Join { .. }));
    let LogicalPlan::Join {
        join_type,
        on,
        filter,
        ..
    } = join
    else {
        unreachable!()
    };
    assert_eq!(*join_type, JoinType::Cross);
    assert!(on.is_empty() && filter.is_none());
}

#[test]
fn an_equality_condition_becomes_a_join_key_and_the_rest_a_filter() {
    let plan = plan(
        "SELECT o.id FROM orders AS o JOIN customers AS c \
         ON o.customer_id = c.id AND o.total > 5",
    );
    let LogicalPlan::Join { on, filter, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Join { .. }))
    else {
        unreachable!()
    };
    assert_eq!(on.len(), 1, "the equality is a hash-join key");
    assert_eq!(on[0].0.to_string(), "o.customer_id");
    assert_eq!(on[0].1.to_string(), "c.id");
    assert_eq!(
        filter.as_ref().map(ToString::to_string),
        Some("(o.total > CAST(5 AS FLOAT64))".to_string()),
        "the inequality stays as a residual predicate"
    );
}

#[test]
fn a_key_written_right_to_left_is_normalised() {
    let plan = plan("SELECT o.id FROM orders AS o JOIN customers AS c ON c.id = o.customer_id");
    let LogicalPlan::Join { on, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Join { .. }))
    else {
        unreachable!()
    };
    assert_eq!(on[0].0.to_string(), "o.customer_id", "left key first");
    assert_eq!(on[0].1.to_string(), "c.id");
}

#[test]
fn using_merges_the_named_column_and_hides_the_right_hand_copy() {
    // Both `region` columns exist in the join's output schema, but only one is
    // visible to name resolution — so `SELECT *` yields five columns, not six.
    let plan = plan("SELECT * FROM customers JOIN regions USING (region)");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![
            triple("id", DataType::Int64, false),
            triple("name", DataType::Utf8, false),
            triple("region", DataType::Utf8, true),
            triple("vip", DataType::Boolean, true),
            triple("continent", DataType::Utf8, false),
        ]
    );
    let LogicalPlan::Join { on, schema, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Join { .. }))
    else {
        unreachable!()
    };
    assert_eq!(on.len(), 1);
    assert_eq!(on[0].0.to_string(), "customers.region");
    assert_eq!(on[0].1.to_string(), "regions.region");
    assert_eq!(
        schema.len(),
        6,
        "the join node itself still produces both copies; only the visible schema merges them"
    );
    // The surviving copy is the left one, so under an outer join it carries the
    // left side's nullability rather than a coalesce of the two.
    assert_eq!(
        join_nullability("SELECT * FROM customers LEFT JOIN regions USING (region)"),
        vec![false, false, true, true, true]
    );
}

#[test]
fn natural_join_equates_every_shared_name() {
    let natural = plan("SELECT * FROM customers NATURAL JOIN regions");
    let explicit = plan("SELECT * FROM customers JOIN regions USING (region)");
    assert_eq!(
        schema_triples(natural.schema()),
        schema_triples(explicit.schema()),
        "NATURAL is USING over the common names"
    );
    let LogicalPlan::Join { on, join_type, .. } =
        find(&natural, &|node| matches!(node, LogicalPlan::Join { .. }))
    else {
        unreachable!()
    };
    assert_eq!(*join_type, JoinType::Inner);
    assert_eq!(on.len(), 1);
}

#[test]
fn a_qualified_reference_survives_a_self_join() {
    let plan = plan("SELECT a.id, b.id FROM orders AS a JOIN orders AS b ON a.id = b.customer_id");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![
            triple("id", DataType::Int64, false),
            triple("id", DataType::Int64, false),
        ]
    );
    // Two columns with the same name, but never the same identity.
    let ids: Vec<_> = plan.schema().fields().iter().map(|f| f.id).collect();
    assert_ne!(ids[0], ids[1]);
}

// ---------------------------------------------------------------------------
// Subqueries and correlation
// ---------------------------------------------------------------------------

#[test]
fn an_uncorrelated_in_subquery_has_no_outer_references() {
    let plan = plan("SELECT id FROM orders WHERE customer_id IN (SELECT id FROM customers)");
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    let Expr::InSubquery { subquery, .. } = predicate else {
        panic!("expected an IN (subquery), found {predicate}");
    };
    assert!(outer_references(subquery).is_empty());
}

#[test]
fn a_correlated_in_subquery_produces_an_outer_column() {
    let plan = plan(
        "SELECT id FROM orders AS o \
         WHERE customer_id IN (SELECT c.id FROM customers AS c WHERE c.region = o.status)",
    );
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    let Expr::InSubquery { subquery, .. } = predicate else {
        panic!("expected an IN (subquery), found {predicate}");
    };
    assert_eq!(
        outer_references(subquery),
        vec!["o.status".to_string()],
        "the reference to the enclosing block is an OuterColumn, which is what marks the \
         subquery as correlated"
    );
    // The outer column belongs to the *enclosing* block, so it must not appear
    // in `columns()` — column pruning would otherwise ask the subquery's input
    // for a column it does not have.
    let inner_filter = find(subquery, &|node| matches!(node, LogicalPlan::Filter { .. }));
    let LogicalPlan::Filter { predicate, .. } = inner_filter else {
        unreachable!()
    };
    assert_eq!(predicate.columns().len(), 1);
    assert_eq!(predicate.outer_columns().len(), 1);
}

#[test]
fn a_correlated_exists_produces_an_outer_column() {
    // EXISTS binds in the select list, where the aggregate sink is collecting.
    let plan = plan(
        "SELECT EXISTS (SELECT 1 FROM orders AS o WHERE o.customer_id = c.id) AS has \
         FROM customers AS c",
    );
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple("has", DataType::Boolean, false)],
        "EXISTS is never null"
    );
    let Expr::Exists { subquery, .. } = projection_expr(&plan, 0) else {
        panic!("expected an EXISTS");
    };
    assert_eq!(outer_references(&subquery), vec!["c.id".to_string()]);
}

#[test]
fn a_subquery_in_from_is_aliased_and_requalified() {
    let plan = plan("SELECT s.id FROM (SELECT id FROM orders) AS s");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple("id", DataType::Int64, false)]
    );
    let alias = find(&plan, &|node| {
        matches!(node, LogicalPlan::SubqueryAlias { .. })
    });
    let LogicalPlan::SubqueryAlias { alias, schema, .. } = alias else {
        unreachable!()
    };
    assert_eq!(alias, "s");
    assert_eq!(schema.field(0).unwrap().qualifier.as_deref(), Some("s"));
}

#[test]
fn a_derived_table_may_rename_its_columns() {
    assert_eq!(
        output("SELECT x FROM (SELECT id, total FROM orders) AS s (x, y)"),
        vec![triple("x", DataType::Int64, false)]
    );
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

#[test]
fn group_by_puts_keys_before_aggregates_and_types_them() {
    let plan = plan("SELECT customer_id, count(*), sum(total) FROM orders GROUP BY customer_id");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![
            triple("customer_id", DataType::Int64, false),
            triple("count(*)", DataType::Int64, false),
            triple("sum(orders.total)", DataType::Float64, true),
        ],
        "count is never null; sum of an empty group is"
    );
    let LogicalPlan::Aggregate {
        group_by,
        aggregates,
        schema,
        ..
    } = find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    assert_eq!(group_by.len(), 1);
    assert_eq!(aggregates.len(), 2);
    assert_eq!(schema.len(), 3, "group keys then aggregates");
}

#[test]
fn an_aggregate_argument_is_cast_before_accumulation() {
    let plan = plan("SELECT avg(qty), sum(qty) FROM orders");
    let LogicalPlan::Aggregate { aggregates, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    let rendered: Vec<String> = aggregates.iter().map(AggregateExpr::to_string).collect();
    assert_eq!(
        rendered,
        vec![
            "avg(CAST(orders.qty AS FLOAT64)) AS avg(CAST(orders.qty AS FLOAT64))".to_string(),
            "sum(CAST(orders.qty AS INT64)) AS sum(CAST(orders.qty AS INT64))".to_string(),
        ],
        "avg accumulates in FLOAT64 and sum in INT64, decided once at bind time"
    );
}

#[test]
fn the_same_aggregate_written_twice_is_computed_once() {
    // Not only an optimisation: the select list and HAVING must agree, and
    // sharing the output column makes that true by construction.
    let plan =
        plan("SELECT customer_id, count(*) FROM orders GROUP BY customer_id HAVING count(*) > 2");
    let LogicalPlan::Aggregate { aggregates, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    assert_eq!(aggregates.len(), 1);
    // HAVING became a Filter above the Aggregate, reading its output column.
    let LogicalPlan::Filter { predicate, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Filter { .. }))
    else {
        unreachable!()
    };
    assert_eq!(predicate.to_string(), "(count(*) > 2)");
    assert_eq!(predicate.columns(), [aggregates[0].output.id].into());
}

#[test]
fn having_may_use_an_aggregate_the_select_list_does_not() {
    let plan = plan("SELECT status, count(*) FROM orders GROUP BY status HAVING sum(total) > 100");
    let LogicalPlan::Aggregate { aggregates, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    assert_eq!(aggregates.len(), 2, "count for the output, sum for HAVING");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![
            triple("status", DataType::Utf8, true),
            triple("count(*)", DataType::Int64, false),
        ],
        "the HAVING aggregate is not projected"
    );
}

#[test]
fn a_group_key_expression_is_matched_structurally_in_the_select_list() {
    // `GROUP BY qty + 1` makes `qty + 1` legal above the aggregate; the binder
    // substitutes it for a reference to the group column rather than trying to
    // re-evaluate it.
    let plan = plan("SELECT qty + 1 FROM orders GROUP BY qty + 1");
    let LogicalPlan::Aggregate { group_by, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    assert_eq!(group_by.len(), 1);
    assert_eq!(group_by[0].0.to_string(), "(CAST(orders.qty AS INT64) + 1)");
    assert_eq!(
        projection_expr(&plan, 0),
        Expr::Column(basalt_plan::ColumnRef {
            id: group_by[0].1.id,
            qualifier: group_by[0].1.qualifier.clone(),
            name: group_by[0].1.name.clone(),
            data_type: group_by[0].1.data_type.clone(),
            nullable: group_by[0].1.nullable,
        })
    );
}

#[test]
fn a_global_aggregate_has_no_group_keys() {
    let plan = plan("SELECT count(DISTINCT status) FROM orders");
    let LogicalPlan::Aggregate {
        group_by,
        aggregates,
        ..
    } = find(&plan, &|node| matches!(node, LogicalPlan::Aggregate { .. }))
    else {
        unreachable!()
    };
    assert!(group_by.is_empty());
    assert!(aggregates[0].distinct);
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple(
            "count(DISTINCT orders.status)",
            DataType::Int64,
            false
        )]
    );
}

#[test]
fn min_and_max_keep_the_input_type() {
    assert_eq!(
        output("SELECT min(placed_at), max(name) FROM orders, customers"),
        vec![
            triple(
                "min(orders.placed_at)",
                DataType::Timestamp(TimeUnit::Microsecond),
                true
            ),
            triple("max(customers.name)", DataType::Utf8, true),
        ]
    );
}

// ---------------------------------------------------------------------------
// ORDER BY, LIMIT and DISTINCT
// ---------------------------------------------------------------------------

#[test]
fn order_by_resolves_an_output_alias_before_an_input_column() {
    let plan = plan("SELECT id AS x FROM orders ORDER BY x DESC");
    let LogicalPlan::Sort { keys, .. } = &plan else {
        panic!("expected a sort at the top, found {plan}");
    };
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].expr.to_string(), "x");
    assert!(!keys[0].asc);
    assert!(
        keys[0].nulls_first,
        "descending sorts put nulls first, so reversing a sort reverses the whole row order"
    );
}

#[test]
fn order_by_accepts_an_output_ordinal() {
    let plan = plan("SELECT id, total FROM orders ORDER BY 2 DESC, 1");
    let LogicalPlan::Sort { keys, .. } = &plan else {
        panic!("expected a sort, found {plan}");
    };
    assert_eq!(keys[0].expr.to_string(), "total");
    assert_eq!(keys[1].expr.to_string(), "id");
    assert!(!keys[0].asc && keys[1].asc);
    assert!(!keys[1].nulls_first, "ascending sorts put nulls last");
}

#[test]
fn order_by_sees_the_query_output_and_only_the_query_output() {
    // A key that is not an alias and not an ordinal is bound against the
    // *output* schema, not the FROM relation: `ORDER BY status` works because
    // `status` is projected. See `reject.rs` for the unprojected case, which is
    // rejected rather than silently re-reading the input.
    let plan = plan("SELECT status, id FROM orders ORDER BY status");
    let LogicalPlan::Sort { keys, .. } = &plan else {
        panic!("expected a sort, found {plan}");
    };
    assert_eq!(keys[0].expr.to_string(), "status");
    assert_eq!(keys[0].expr.columns().len(), 1);
}

#[test]
fn limit_and_offset_are_folded_into_one_node() {
    let plan = plan("SELECT id FROM orders LIMIT 10 OFFSET 5");
    let LogicalPlan::Limit { skip, fetch, .. } = &plan else {
        panic!("expected a limit, found {plan}");
    };
    assert_eq!((*skip, *fetch), (5, Some(10)));

    let offset_only = plan_of("SELECT id FROM orders OFFSET 5");
    let LogicalPlan::Limit { skip, fetch, .. } = &offset_only else {
        panic!("expected a limit");
    };
    assert_eq!((*skip, *fetch), (5, None));

    // `LIMIT 0` fetches nothing but is still a node; `OFFSET 0` alone is not.
    assert!(matches!(
        plan_of("SELECT id FROM orders LIMIT 0"),
        LogicalPlan::Limit { fetch: Some(0), .. }
    ));
    assert!(matches!(
        plan_of("SELECT id FROM orders OFFSET 0"),
        LogicalPlan::Projection { .. }
    ));
}

#[test]
fn distinct_wraps_the_projection() {
    assert!(matches!(
        plan("SELECT DISTINCT status FROM orders"),
        LogicalPlan::Distinct { .. }
    ));
}

// ---------------------------------------------------------------------------
// Set operations
// ---------------------------------------------------------------------------

#[test]
fn union_coerces_both_arms_to_a_common_type() {
    let plan = plan("SELECT total FROM orders UNION SELECT qty FROM orders");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple("total", DataType::Float64, true)],
        "the first arm names the column, the lattice types it"
    );
    let LogicalPlan::Union { inputs, all, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Union { .. }))
    else {
        unreachable!()
    };
    assert!(!all);
    assert_eq!(inputs.len(), 2);
    for input in inputs {
        assert_eq!(
            input.schema().field(0).unwrap().data_type,
            DataType::Float64,
            "each arm is projected to the union's type, so the executor never reconciles types"
        );
    }
    // `UNION` (without ALL) deduplicates, which is a Distinct on top.
    assert!(matches!(plan, LogicalPlan::Distinct { .. }));
}

#[test]
fn union_all_does_not_deduplicate() {
    let plan = plan("SELECT id FROM orders UNION ALL SELECT id FROM orders");
    let LogicalPlan::Union { all, .. } = &plan else {
        panic!("expected a union at the top, found {plan}");
    };
    assert!(*all);
}

#[test]
fn a_null_arm_takes_the_other_arms_type_but_makes_it_nullable() {
    assert_eq!(
        output("SELECT id FROM orders UNION ALL SELECT NULL"),
        vec![triple("id", DataType::Int64, true)]
    );
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

#[test]
fn a_window_function_appends_a_column_and_gets_the_ordered_default_frame() {
    let plan = plan(
        "SELECT id, row_number() OVER (PARTITION BY customer_id ORDER BY total DESC) FROM orders",
    );
    let LogicalPlan::Window {
        windows, schema, ..
    } = find(&plan, &|node| matches!(node, LogicalPlan::Window { .. }))
    else {
        unreachable!()
    };
    assert_eq!(windows.len(), 1);
    let window: &WindowExpr = &windows[0];
    assert_eq!(window.partition_by.len(), 1);
    assert_eq!(window.order_by.len(), 1);
    assert_eq!(
        window.frame.to_string(),
        "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "the SQL default for an ordered window, which is what makes a running total run"
    );
    assert_eq!(
        schema.len(),
        7,
        "the window node adds its column to the input schema"
    );
    assert_eq!(
        plan.schema().field(1).unwrap().data_type,
        DataType::Int64,
        "row_number is an INT64"
    );
}

#[test]
fn an_unordered_window_defaults_to_the_whole_partition() {
    let plan = plan("SELECT sum(total) OVER () FROM orders");
    let LogicalPlan::Window { windows, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Window { .. }))
    else {
        unreachable!()
    };
    assert_eq!(
        windows[0].frame.to_string(),
        "RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
    );
    assert_eq!(plan.schema().field(0).unwrap().data_type, DataType::Float64);
}

#[test]
fn an_explicit_frame_is_carried_through() {
    let plan = plan(
        "SELECT sum(total) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) FROM orders",
    );
    let LogicalPlan::Window { windows, .. } =
        find(&plan, &|node| matches!(node, LogicalPlan::Window { .. }))
    else {
        unreachable!()
    };
    assert_eq!(
        windows[0].frame.to_string(),
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING"
    );
}

// ---------------------------------------------------------------------------
// CTEs
// ---------------------------------------------------------------------------

#[test]
fn a_cte_is_bound_once_and_referenced_by_name() {
    assert_eq!(
        output("WITH recent AS (SELECT id, total FROM orders) SELECT total FROM recent"),
        vec![triple("total", DataType::Float64, true)]
    );
}

#[test]
fn a_cte_column_list_renames_the_query_output() {
    assert_eq!(
        output("WITH c (a, b) AS (SELECT id, total FROM orders) SELECT a, b FROM c"),
        vec![
            triple("a", DataType::Int64, false),
            triple("b", DataType::Float64, true),
        ]
    );
}

#[test]
fn two_references_to_one_cte_do_not_share_column_identities() {
    // If they did, a predicate pushed into one reference would land in both.
    let plan = plan("WITH c AS (SELECT id FROM orders) SELECT x.id, y.id FROM c AS x JOIN c AS y ON x.id = y.id");
    let ids: Vec<_> = plan.schema().fields().iter().map(|f| f.id).collect();
    assert_ne!(ids[0], ids[1]);
}

#[test]
fn a_cte_may_refer_to_an_earlier_one() {
    assert_eq!(
        output("WITH a AS (SELECT id FROM orders), b AS (SELECT id FROM a) SELECT id FROM b"),
        vec![triple("id", DataType::Int64, false)]
    );
}

// ---------------------------------------------------------------------------
// Statements other than queries
// ---------------------------------------------------------------------------

#[track_caller]
fn bound(sql: &str) -> BoundStatement {
    let statement = basalt_sql::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse:\n{}", e.render(sql)));
    bind_statement(&statement, &catalog()).unwrap_or_else(|e| panic!("`{sql}` was rejected: {e}"))
}

#[test]
fn create_table_takes_nullability_from_the_column_options() {
    let BoundStatement::CreateTable { name, schema, .. } =
        bound("CREATE TABLE t (a INT32 PRIMARY KEY, b UTF8, c FLOAT64 NOT NULL)")
    else {
        panic!("expected a CREATE TABLE");
    };
    assert_eq!(name, "t");
    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|f| (f.name().to_string(), f.is_nullable()))
            .collect::<Vec<_>>(),
        vec![
            ("a".to_string(), false),
            ("b".to_string(), true),
            ("c".to_string(), false),
        ],
        "PRIMARY KEY implies NOT NULL; a bare column is nullable"
    );
}

#[test]
fn create_table_as_select_takes_its_schema_from_the_query() {
    let BoundStatement::CreateTable { schema, source, .. } =
        bound("CREATE TABLE t AS SELECT id, total FROM orders")
    else {
        panic!("expected a CREATE TABLE");
    };
    assert_eq!(schema.len(), 2);
    assert_eq!(schema.field(1).data_type(), &DataType::Float64);
    let source = source.expect("CTAS carries its query");
    source.validate().unwrap();
}

#[test]
fn insert_casts_the_source_to_the_target_column_types() {
    let BoundStatement::Insert {
        table,
        columns,
        source,
    } = bound("INSERT INTO orders (id, total) SELECT id, qty FROM orders")
    else {
        panic!("expected an INSERT");
    };
    assert_eq!(table.name, "orders");
    assert_eq!(columns, vec![0, 2]);
    source.validate().unwrap();
    assert_eq!(
        projection_expr(&source, 1).to_string(),
        "CAST(qty AS FLOAT64)",
        "the cast is inserted at bind time, so an impossible conversion is a bind error \
         rather than a half-written table"
    );
}

#[test]
fn explain_wraps_the_plan_and_returns_one_text_column() {
    let plan = plan("EXPLAIN SELECT id FROM orders");
    assert_eq!(
        schema_triples(plan.schema()),
        vec![triple("plan", DataType::Utf8, false)]
    );
    assert!(matches!(plan, LogicalPlan::Explain { analyze: false, .. }));
    assert!(matches!(
        plan_of("EXPLAIN ANALYZE SELECT id FROM orders"),
        LogicalPlan::Explain { analyze: true, .. }
    ));
}

#[test]
fn drop_table_lists_its_targets() {
    let BoundStatement::DropTable { names, if_exists } = bound("DROP TABLE IF EXISTS a, b") else {
        panic!("expected a DROP TABLE");
    };
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    assert!(if_exists);
}

/// `EXISTS` is a predicate and binds wherever a boolean expression is legal.
///
/// It is pinned in every clause because an earlier revision routed `EXISTS`
/// through the same gate that forbids aggregates, which made `WHERE EXISTS
/// (...)` -- the most common form of it -- a bind error.
#[test]
fn exists_binds_in_every_boolean_position() {
    for sql in [
        "SELECT id FROM orders WHERE EXISTS (SELECT 1 FROM customers)",
        "SELECT id FROM orders WHERE NOT EXISTS (SELECT 1 FROM customers)",
        "SELECT o.id FROM orders AS o JOIN customers AS c ON EXISTS (SELECT 1 FROM customers)",
        "SELECT EXISTS (SELECT 1 FROM customers) FROM orders",
        "SELECT id FROM orders o WHERE EXISTS (SELECT 1 FROM customers c WHERE c.id = o.customer_id)",
    ] {
        let plan = plan(sql);
        plan.validate()
            .unwrap_or_else(|e| panic!("`{sql}` produced an invalid plan: {e}"));
    }
}

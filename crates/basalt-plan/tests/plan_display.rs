//! `EXPLAIN`-style rendering, and what `validate` catches.
//!
//! The rendering tests are snapshots of `Display for LogicalPlan`. They are
//! verbose on purpose: the plan text is the primary debugging interface of an
//! engine with no executor yet, and a snapshot is the only kind of test that
//! notices when an operator quietly stops printing its join keys.
//!
//! The validation tests go the other way. They hand-build plans that the binder
//! would never emit — a filter reading a column its input does not produce, a
//! union whose arms disagree, a join whose schema contradicts its inputs — and
//! assert that `validate` names the problem. Those are exactly the plans a
//! buggy optimizer rule will one day produce.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use basalt_plan::logical::join_schema;
use basalt_plan::{
    bind, BinaryOp, Catalog, ColumnId, ColumnRef, Expr, JoinType, LogicalPlan, PlanField,
    PlanSchema, TableDefinition,
};
use basalt_types::{DataType, Field, Schema, TimeUnit, Value};

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
            ],
        ))
        .unwrap();
    catalog
}

#[track_caller]
fn plan(sql: &str) -> LogicalPlan {
    let statement = basalt_sql::parse(sql)
        .unwrap_or_else(|e| panic!("`{sql}` did not parse:\n{}", e.render(sql)));
    bind(&statement, &catalog()).unwrap_or_else(|e| panic!("`{sql}` was rejected: {e}"))
}

/// Assert the whole rendering, and that the plan being rendered is valid.
#[track_caller]
fn renders(sql: &str, expected: &str) {
    let plan = plan(sql);
    plan.validate()
        .unwrap_or_else(|e| panic!("`{sql}` produced an invalid plan: {e}"));
    assert_eq!(
        plan.to_string(),
        expected,
        "\n`{sql}` rendered as:\n{plan}\n"
    );
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

#[test]
fn a_scan_prints_its_projection() {
    renders(
        "SELECT id, total FROM orders",
        "Projection: orders.id, orders.total
  TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn an_alias_is_printed_only_when_it_differs_from_the_table_name() {
    renders(
        "SELECT o.id FROM orders AS o",
        "Projection: o.id
  TableScan: orders AS o projection=[id, customer_id, total, status, qty, placed_at]",
    );
    renders(
        "SELECT orders.id FROM orders AS orders",
        "Projection: orders.id
  TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_filter_prints_the_predicate_with_its_inserted_casts() {
    renders(
        "SELECT id FROM orders WHERE status = 'open' AND qty > 2",
        "Projection: orders.id
  Filter: ((orders.status = 'open') AND (CAST(orders.qty AS INT64) > 2))
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn an_expression_column_prints_its_alias() {
    renders(
        "SELECT id + qty AS n, total FROM orders",
        "Projection: (orders.id + CAST(orders.qty AS INT64)) AS n, orders.total
  TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_join_prints_its_keys_and_its_residual_filter() {
    renders(
        "SELECT o.id, c.name FROM orders AS o JOIN customers AS c \
         ON o.customer_id = c.id AND o.total > 5",
        "Projection: o.id, c.name
  InnerJoin: on=[o.customer_id = c.id] filter=(o.total > CAST(5 AS FLOAT64))
    TableScan: orders AS o projection=[id, customer_id, total, status, qty, placed_at]
    TableScan: customers AS c projection=[id, name, region]",
    );
}

#[test]
fn a_cross_join_prints_no_condition() {
    renders(
        "SELECT orders.id, customers.name FROM orders, customers",
        "Projection: orders.id, customers.name
  CrossJoin:
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]
    TableScan: customers projection=[id, name, region]",
    );
}

#[test]
fn an_aggregate_prints_its_groups_and_its_aggregates() {
    renders(
        "SELECT customer_id, count(*), sum(total) FROM orders GROUP BY customer_id",
        "Projection: customer_id, count(*), sum(orders.total)
  Aggregate: groupBy=[orders.customer_id] aggregates=[count(*) AS count(*), \
sum(orders.total) AS sum(orders.total)]
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn having_becomes_a_filter_above_the_aggregate() {
    renders(
        "SELECT status, count(*) AS n FROM orders GROUP BY status HAVING count(*) > 2",
        "Projection: status, count(*) AS n
  Filter: (count(*) > 2)
    Aggregate: groupBy=[orders.status] aggregates=[count(*) AS count(*)]
      TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn sort_and_limit_stack_above_the_projection() {
    renders(
        "SELECT id, total FROM orders ORDER BY 2 DESC LIMIT 10 OFFSET 5",
        "Limit: skip=5 fetch=10
  Sort: total DESC NULLS FIRST
    Projection: orders.id, orders.total
      TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_union_prints_each_arm_and_the_cast_that_reconciles_them() {
    renders(
        "SELECT total FROM orders UNION SELECT qty FROM orders",
        "Distinct
  Union
    Projection: orders.total
      TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]
    Projection: CAST(qty AS FLOAT64) AS total
      Projection: orders.qty
        TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
    renders(
        "SELECT id FROM orders UNION ALL SELECT id FROM orders",
        "Union All
  Projection: orders.id
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]
  Projection: orders.id
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_window_prints_its_partition_order_and_frame() {
    renders(
        "SELECT id, row_number() OVER (PARTITION BY customer_id ORDER BY total DESC) AS rk \
         FROM orders",
        "Projection: orders.id, row_number() OVER (...) AS rk
  Window: row_number() OVER (PARTITION BY orders.customer_id ORDER BY orders.total DESC \
NULLS FIRST RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS row_number() OVER (...)
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_cte_reference_prints_as_a_subquery_alias() {
    renders(
        "WITH recent AS (SELECT id FROM orders) SELECT id FROM recent",
        "Projection: recent.id
  SubqueryAlias: recent
    Projection: recent.id
      SubqueryAlias: recent
        Projection: orders.id
          TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn a_subquery_predicate_prints_as_an_opaque_marker() {
    // The subquery's own plan is a separate tree; printing it inline would make
    // the operator line unreadable, and it is reachable from the expression.
    renders(
        "SELECT id FROM orders WHERE customer_id IN (SELECT id FROM customers)",
        "Projection: orders.id
  Filter: (orders.customer_id IN <subquery>)
    TableScan: orders projection=[id, customer_id, total, status, qty, placed_at]",
    );
}

#[test]
fn explain_and_the_constant_relations_print() {
    renders(
        "EXPLAIN SELECT 1",
        "Explain
  Projection: 1
    EmptyRelation: one row",
    );
    renders("VALUES (1, 2), (3, 4)", "Values: 2 rows");
}

#[test]
fn a_deep_plan_indents_two_spaces_per_level() {
    let plan = plan(
        "SELECT c.region, count(*) AS n \
         FROM customers AS c JOIN orders AS o ON o.customer_id = c.id \
         WHERE o.total > 0 GROUP BY c.region HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 5",
    );
    plan.validate().unwrap();
    assert_eq!(
        plan.to_string(),
        "Limit: skip=0 fetch=5
  Sort: n DESC NULLS FIRST
    Projection: region, count(*) AS n
      Filter: (count(*) > 1)
        Aggregate: groupBy=[c.region] aggregates=[count(*) AS count(*)]
          Filter: (o.total > CAST(0 AS FLOAT64))
            InnerJoin: on=[c.id = o.customer_id]
              TableScan: customers AS c projection=[id, name, region]
              TableScan: orders AS o projection=[id, customer_id, total, status, qty, placed_at]"
    );
}

#[test]
fn every_operator_kind_has_a_name() {
    // `name()` feeds both EXPLAIN and the validation messages, so an operator
    // added without one would produce diagnostics that say nothing.
    for sql in [
        "SELECT id FROM orders",
        "SELECT id FROM orders WHERE id > 1",
        "SELECT count(*) FROM orders",
        "SELECT id FROM orders ORDER BY 1",
        "SELECT id FROM orders LIMIT 1",
        "SELECT DISTINCT id FROM orders",
        "SELECT id FROM orders UNION ALL SELECT id FROM orders",
        "SELECT o.id FROM orders AS o JOIN customers AS c ON o.customer_id = c.id",
        "SELECT row_number() OVER () FROM orders",
        "SELECT s.id FROM (SELECT id FROM orders) AS s",
        "VALUES (1)",
        "SELECT 1",
        "EXPLAIN SELECT 1",
    ] {
        let plan = plan(sql);
        let mut stack = vec![&plan];
        while let Some(node) = stack.pop() {
            assert!(!node.name().is_empty(), "{sql}");
            assert!(
                plan.to_string().contains(node.name()),
                "`{sql}`: {} does not appear in its own rendering",
                node.name()
            );
            stack.extend(node.children().into_iter().map(Arc::as_ref));
        }
    }
}

// ---------------------------------------------------------------------------
// Validation of hand-built plans
// ---------------------------------------------------------------------------

fn orders() -> Arc<TableDefinition> {
    catalog().table("orders").unwrap()
}

/// A scan of `orders` with fresh identities starting at `first_id`.
fn scan(first_id: u32) -> LogicalPlan {
    let table = orders();
    let schema = PlanSchema::new(
        table
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| {
                PlanField::qualified(
                    ColumnId(first_id + index as u32),
                    "orders",
                    field.name().to_string(),
                    field.data_type().clone(),
                    field.is_nullable(),
                )
            })
            .collect(),
    );
    LogicalPlan::TableScan {
        table,
        alias: None,
        projection: (0..6).collect(),
        filters: Vec::new(),
        schema,
    }
}

fn column(plan: &LogicalPlan, index: usize) -> Expr {
    let field = plan.schema().field(index).unwrap();
    Expr::Column(ColumnRef {
        id: field.id,
        qualifier: field.qualifier.clone(),
        name: field.name.clone(),
        data_type: field.data_type.clone(),
        nullable: field.nullable,
    })
}

#[track_caller]
fn invalid(plan: &LogicalPlan, needles: &[&str]) {
    let message = match plan.validate() {
        Err(e) => e.to_string(),
        Ok(()) => panic!("this plan should not have validated:\n{plan}"),
    };
    for needle in needles {
        assert!(
            message.contains(needle),
            "expected `{needle}` in:\n  {message}\nplan:\n{plan}"
        );
    }
}

#[test]
fn a_filter_may_not_read_a_column_its_input_does_not_produce() {
    let input = scan(0);
    let plan = LogicalPlan::Filter {
        input: Arc::new(input),
        predicate: Expr::binary(
            Expr::Column(ColumnRef {
                id: ColumnId(99),
                qualifier: Some("elsewhere".to_string()),
                name: "ghost".to_string(),
                data_type: DataType::Int64,
                nullable: false,
            }),
            BinaryOp::Gt,
            Expr::Literal(Value::Int64(1)),
        ),
    };
    invalid(
        &plan,
        &[
            "Filter",
            "references column #99",
            "which its input does not produce",
        ],
    );
}

#[test]
fn a_filter_predicate_must_be_boolean() {
    let input = scan(0);
    let predicate = column(&input, 0);
    let plan = LogicalPlan::Filter {
        input: Arc::new(input),
        predicate,
    };
    invalid(
        &plan,
        &["filter predicate", "orders.id is INT64, not BOOLEAN"],
    );
}

#[test]
fn a_union_arm_must_have_the_declared_arity() {
    let wide = scan(0);
    let narrow = LogicalPlan::Projection {
        input: Arc::new(scan(100)),
        exprs: vec![(
            column(&scan(100), 0),
            PlanField::new(ColumnId(200), "id", DataType::Int64, false),
        )],
        schema: PlanSchema::new(vec![PlanField::new(
            ColumnId(200),
            "id",
            DataType::Int64,
            false,
        )]),
    };
    let schema = wide.schema().clone();
    let plan = LogicalPlan::Union {
        inputs: vec![Arc::new(wide), Arc::new(narrow)],
        all: true,
        schema,
    };
    invalid(&plan, &["union arm 1 has 1 columns but the union has 6"]);
}

#[test]
fn a_union_arm_must_have_the_declared_types() {
    // The binder projects every arm to the union's type; an optimizer rule that
    // drops one of those projections produces exactly this plan.
    let left = scan(0);
    let right = scan(100);
    let schema = PlanSchema::new(
        left.schema()
            .fields()
            .iter()
            .map(|field| {
                let mut field = field.clone();
                if field.name == "qty" {
                    field.data_type = DataType::Int64;
                }
                field
            })
            .collect(),
    );
    let plan = LogicalPlan::Union {
        inputs: vec![Arc::new(left), Arc::new(right)],
        all: true,
        schema,
    };
    invalid(
        &plan,
        &[
            "union arm 0 column 4 is INT32 but the union declares INT64",
            "the binder must insert a cast",
        ],
    );
}

#[test]
fn a_union_needs_at_least_two_arms() {
    let only = scan(0);
    let schema = only.schema().clone();
    let plan = LogicalPlan::Union {
        inputs: vec![Arc::new(only)],
        all: true,
        schema,
    };
    invalid(&plan, &["union has 1 arms; it needs at least two"]);
}

#[test]
fn a_join_schema_must_agree_with_its_inputs() {
    let left = scan(0);
    let right = scan(100);
    // A left join's right side is null-extended. Claiming the inner-join schema
    // is the classic outer-join nullability bug, and it is caught here.
    let schema = join_schema(JoinType::Inner, left.schema(), right.schema());
    let plan = LogicalPlan::Join {
        left: Arc::new(left),
        right: Arc::new(right),
        join_type: JoinType::Left,
        on: Vec::new(),
        filter: None,
        schema,
    };
    invalid(
        &plan,
        &[
            "Left join schema",
            "does not match the one implied by its inputs",
        ],
    );
}

#[test]
fn a_join_key_must_come_from_the_side_it_claims() {
    let left = scan(0);
    let right = scan(100);
    let schema = join_schema(JoinType::Inner, left.schema(), right.schema());
    // Both halves of the key read the left input.
    let key = (column(&left, 0), column(&left, 1));
    let plan = LogicalPlan::Join {
        left: Arc::new(left),
        right: Arc::new(right),
        join_type: JoinType::Inner,
        on: vec![key],
        filter: None,
        schema,
    };
    invalid(
        &plan,
        &["join key orders.customer_id references columns that are not on the right side"],
    );
}

#[test]
fn join_keys_must_have_matching_types() {
    let left = scan(0);
    let right = scan(100);
    let schema = join_schema(JoinType::Inner, left.schema(), right.schema());
    let key = (column(&left, 0), column(&right, 4)); // INT64 = INT32
    let plan = LogicalPlan::Join {
        left: Arc::new(left),
        right: Arc::new(right),
        join_type: JoinType::Inner,
        on: vec![key],
        filter: None,
        schema,
    };
    invalid(
        &plan,
        &[
            "have different types",
            "the binder must insert a cast",
            "INT64",
            "INT32",
        ],
    );
}

#[test]
fn a_cross_join_may_not_carry_a_condition() {
    let left = scan(0);
    let right = scan(100);
    let schema = join_schema(JoinType::Cross, left.schema(), right.schema());
    let key = (column(&left, 0), column(&right, 0));
    let plan = LogicalPlan::Join {
        left: Arc::new(left),
        right: Arc::new(right),
        join_type: JoinType::Cross,
        on: vec![key],
        filter: None,
        schema,
    };
    invalid(
        &plan,
        &["cross join carries a condition; it should be an inner join"],
    );
}

#[test]
fn a_projection_column_must_have_the_type_it_declares() {
    let input = scan(0);
    let expr = column(&input, 0); // INT64
    let plan = LogicalPlan::Projection {
        input: Arc::new(input),
        exprs: vec![(
            expr,
            PlanField::new(ColumnId(200), "id", DataType::Utf8, false),
        )],
        schema: PlanSchema::new(vec![PlanField::new(
            ColumnId(200),
            "id",
            DataType::Utf8,
            false,
        )]),
    };
    invalid(
        &plan,
        &["projection column id is declared UTF8", "is INT64"],
    );
}

#[test]
fn a_projection_must_have_one_expression_per_output_column() {
    let input = scan(0);
    let expr = column(&input, 0);
    let plan = LogicalPlan::Projection {
        input: Arc::new(input),
        exprs: vec![(
            expr,
            PlanField::new(ColumnId(200), "id", DataType::Int64, false),
        )],
        schema: PlanSchema::empty(),
    };
    invalid(
        &plan,
        &["projection computes 1 expressions but its schema has 0 columns"],
    );
}

#[test]
fn a_table_scan_projection_must_be_in_range() {
    let table = orders();
    let plan = LogicalPlan::TableScan {
        table,
        alias: None,
        projection: vec![42],
        filters: Vec::new(),
        schema: PlanSchema::new(vec![PlanField::new(
            ColumnId(0),
            "id",
            DataType::Int64,
            false,
        )]),
    };
    invalid(
        &plan,
        &["table scan of orders projects column 42, but the table has 6"],
    );
}

#[test]
fn validation_recurses_into_children() {
    // The invalid node is three levels down, under nodes that are themselves
    // fine. `validate` checks children first, so the message is about the real
    // problem rather than about its parent.
    let broken = LogicalPlan::Filter {
        input: Arc::new(scan(0)),
        predicate: Expr::Column(ColumnRef {
            id: ColumnId(99),
            qualifier: None,
            name: "ghost".to_string(),
            data_type: DataType::Boolean,
            nullable: false,
        }),
    };
    let plan = LogicalPlan::Limit {
        input: Arc::new(LogicalPlan::Distinct {
            input: Arc::new(LogicalPlan::Sort {
                input: Arc::new(broken),
                keys: Vec::new(),
            }),
        }),
        skip: 0,
        fetch: Some(1),
    };
    invalid(&plan, &["Filter references column #99"]);
}

#[test]
fn every_bound_plan_validates() {
    // The binder's contract, stated once: whatever it emits is valid.
    for sql in [
        "SELECT 1",
        "SELECT * FROM orders",
        "SELECT id FROM orders WHERE total > 1 AND status IS NOT NULL",
        "SELECT customer_id, count(*), avg(qty) FROM orders GROUP BY customer_id HAVING count(*) > 1",
        "SELECT o.id, c.name FROM orders AS o LEFT JOIN customers AS c ON o.customer_id = c.id",
        "SELECT * FROM customers NATURAL JOIN orders",
        "SELECT id FROM orders UNION SELECT qty FROM orders",
        "SELECT sum(total) OVER (PARTITION BY status) FROM orders",
        "WITH c AS (SELECT id FROM orders) SELECT x.id FROM c AS x JOIN c AS y ON x.id = y.id",
        "SELECT s.id FROM (SELECT id FROM orders) AS s ORDER BY 1 LIMIT 3 OFFSET 1",
        "SELECT id FROM orders WHERE customer_id IN (SELECT id FROM customers)",
        "VALUES (1, 'a'), (2, 'b')",
        "EXPLAIN ANALYZE SELECT DISTINCT status FROM orders",
    ] {
        let plan = plan(sql);
        plan.validate()
            .unwrap_or_else(|e| panic!("`{sql}` produced an invalid plan: {e}\n{plan}"));
        // And rebuilding it with its own children must not change that.
        let rebuilt = plan
            .with_new_children(plan.children().into_iter().cloned().collect())
            .unwrap();
        rebuilt
            .validate()
            .unwrap_or_else(|e| panic!("`{sql}` broke on rebuild: {e}"));
        assert_eq!(rebuilt.schema(), plan.schema());
    }
}

//! Catalog, name resolution and the logical plan.
//!
//! `basalt-plan` sits between the syntax tree produced by
//! [`basalt-sql`](https://docs.rs/basalt-sql) and the optimizer/executor. It
//! answers the questions the parser deliberately refuses to answer: *what table
//! is that*, *what type is this expression*, *can it be null*.
//!
//! # The four pieces
//!
//! * [`catalog`] — tables, their schemas and their **statistics**. The
//!   statistics are the only input the cost model has, so they live here rather
//!   than in the optimizer.
//! * [`expr`] — the *resolved* expression tree. Every column reference is a
//!   [`ColumnId`], not a name; every node knows its type and its
//!   nullability; there are no ambiguities left.
//! * [`function`] — the function registry. It is a **contract**: the registry
//!   declares each function's argument coercions and return type, and the
//!   executor promises a kernel for every variant. Because the registry is a
//!   closed enum rather than a bag of trait objects, "every function has a
//!   kernel" is checked by the compiler's exhaustiveness analysis rather than by
//!   a runtime lookup that can fail.
//! * [`logical`] — the plan nodes, each of which computes its own output schema
//!   from its inputs.
//!
//! # Why column *identities* rather than positions
//!
//! A column reference could be a position into the operator's input schema
//! (cheap to execute) or a unique identity (cheap to rewrite). Basalt uses
//! identities, because the optimizer's most valuable rules — column pruning and
//! join reordering — change the shape and width of intermediate schemas. With
//! positions, every one of those rewrites has to renumber every expression in
//! the subtree, and a single missed renumbering is a silently wrong answer.
//! With identities the rewrites are structural and the executor pays one
//! `id -> index` lookup per expression at compile time, never per row.
//!
//! ```
//! use basalt_plan::{bind, Catalog, TableDefinition};
//! use basalt_types::{DataType, Field, Schema};
//!
//! let mut catalog = Catalog::new();
//! catalog.create_table(TableDefinition::new(
//!     "orders",
//!     Schema::new(vec![
//!         Field::new("id", DataType::Int64, false),
//!         Field::new("total", DataType::Float64, true),
//!     ])?,
//! ))?;
//!
//! let statement = basalt_sql::parse("SELECT id FROM orders WHERE total > 10")?;
//! let plan = bind(&statement, &catalog)?;
//! assert_eq!(plan.schema().len(), 1);
//! assert_eq!(plan.schema().field(0).map(|f| f.name.as_str()), Some("id"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![forbid(unsafe_code)]

pub mod binder;
pub mod catalog;
pub mod expr;
pub mod function;
pub mod logical;

pub use binder::{bind, bind_statement, BoundStatement};
pub use catalog::{Catalog, ColumnStatistics, TableDefinition, TableStatistics};
pub use expr::{BinaryOp, ColumnId, ColumnRef, Expr, SortExpr};
pub use function::{AggregateFunc, FunctionRegistry, ScalarFunc, WindowFunc};
pub use logical::{
    AggregateExpr, JoinType, LogicalPlan, PlanField, PlanSchema, WindowExpr, WindowFrameSpec,
};

//! The logical plan: what the query means, not how it will be run.
//!
//! A logical plan is a tree of relational operators. Each node knows the schema
//! it produces and nothing about how it will produce it -- no join algorithms,
//! no batch sizes, no parallelism. That separation is what lets the optimizer
//! rewrite the tree freely: every rule is a tree-to-tree transformation whose
//! only obligation is to preserve the output schema and the multiset of rows.
//!
//! # Schemas carry identities, not positions
//!
//! Every output column has a [`ColumnId`] that is unique within a plan and
//! stable across rewrites. Column pruning, join reordering and predicate
//! pushdown all change the width and order of intermediate schemas; with
//! positional references each of those rewrites would have to renumber every
//! expression beneath it, and a single missed renumbering is a silently wrong
//! answer rather than a crash. See the crate documentation for the full
//! argument.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use basalt_common::{BasaltError, Result};
use basalt_types::{DataType, Field, Schema};

use crate::catalog::TableDefinition;
use crate::expr::{ColumnId, Expr, SortExpr, WindowCall};
use crate::function::AggregateFunc;

/// One column of a plan node's output.
///
/// This is the plan-level counterpart of [`basalt_types::Field`]: the same
/// name/type/nullability triple plus the identity and the relation qualifier
/// that name resolution needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanField {
    /// Stable identity, unique within a plan.
    pub id: ColumnId,
    /// The relation alias this column came from, if any. Display and name
    /// resolution only; two fields with the same qualifier and name in one
    /// schema are an ambiguity, not an error, until someone references them.
    pub qualifier: Option<String>,
    /// The user-visible column name.
    pub name: String,
    /// The column's type.
    pub data_type: DataType,
    /// Whether the column can be null.
    pub nullable: bool,
}

impl PlanField {
    /// A field with no relation qualifier.
    pub fn new(
        id: ColumnId,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
    ) -> PlanField {
        PlanField {
            id,
            qualifier: None,
            name: name.into(),
            data_type,
            nullable,
        }
    }

    /// A field qualified by a relation alias.
    pub fn qualified(
        id: ColumnId,
        qualifier: impl Into<String>,
        name: impl Into<String>,
        data_type: DataType,
        nullable: bool,
    ) -> PlanField {
        PlanField {
            id,
            qualifier: Some(qualifier.into()),
            name: name.into(),
            data_type,
            nullable,
        }
    }

    /// The same field made nullable, as an outer join's null-extended side is.
    #[must_use]
    pub fn into_nullable(mut self) -> PlanField {
        self.nullable = true;
        self
    }

    /// The same field carrying a different qualifier, as `SubqueryAlias` does.
    #[must_use]
    pub fn with_qualifier(mut self, qualifier: Option<String>) -> PlanField {
        self.qualifier = qualifier;
        self
    }
}

impl fmt::Display for PlanField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.qualifier {
            Some(q) => write!(f, "{q}.{}:{}", self.name, self.data_type),
            None => write!(f, "{}:{}", self.name, self.data_type),
        }
    }
}

/// The ordered output columns of a plan node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanSchema {
    fields: Vec<PlanField>,
}

impl PlanSchema {
    /// A schema from an ordered field list.
    pub fn new(fields: Vec<PlanField>) -> PlanSchema {
        PlanSchema { fields }
    }

    /// The empty schema, produced by a zero-column relation.
    pub fn empty() -> PlanSchema {
        PlanSchema { fields: Vec::new() }
    }

    /// The fields, in output order.
    pub fn fields(&self) -> &[PlanField] {
        &self.fields
    }

    /// The number of columns.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no columns.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field at a position.
    pub fn field(&self, index: usize) -> Option<&PlanField> {
        self.fields.get(index)
    }

    /// The position of a column identity, for the executor's compile step.
    ///
    /// This is a linear scan. Schemas are tens of columns wide at most and the
    /// lookup happens once per expression at plan-compile time, never per row,
    /// so an index would cost more in complexity than it saves.
    pub fn index_of(&self, id: ColumnId) -> Option<usize> {
        self.fields.iter().position(|f| f.id == id)
    }

    /// The field with a given identity.
    pub fn column(&self, id: ColumnId) -> Option<&PlanField> {
        self.fields.iter().find(|f| f.id == id)
    }

    /// Resolve an optionally-qualified name to exactly one field.
    ///
    /// Returns an error rather than a choice when a bare name matches more than
    /// one relation: guessing which `id` the user meant in `a JOIN b` is how a
    /// query silently returns the wrong column.
    pub fn resolve(&self, qualifier: Option<&str>, name: &str) -> Result<Option<&PlanField>> {
        let mut found: Option<&PlanField> = None;
        for field in &self.fields {
            let name_matches = field.name.eq_ignore_ascii_case(name);
            if !name_matches {
                continue;
            }
            if let Some(q) = qualifier {
                match &field.qualifier {
                    Some(fq) if fq.eq_ignore_ascii_case(q) => {}
                    _ => continue,
                }
            }
            if let Some(previous) = found {
                if previous.id != field.id {
                    return Err(BasaltError::schema(format!(
                        "column reference {} is ambiguous; it matches {} and {}",
                        display_reference(qualifier, name),
                        previous,
                        field
                    )));
                }
            }
            found = Some(field);
        }
        Ok(found)
    }

    /// Every identity in the schema.
    pub fn column_ids(&self) -> BTreeSet<ColumnId> {
        self.fields.iter().map(|f| f.id).collect()
    }

    /// The physical schema the executor materialises.
    pub fn to_types_schema(&self) -> Result<Schema> {
        Schema::new(
            self.fields
                .iter()
                .map(|f| Field::new(f.name.clone(), f.data_type.clone(), f.nullable))
                .collect(),
        )
    }

    /// This schema with every field made nullable, for an outer join's
    /// null-extended side.
    #[must_use]
    pub fn into_nullable(self) -> PlanSchema {
        PlanSchema {
            fields: self
                .fields
                .into_iter()
                .map(PlanField::into_nullable)
                .collect(),
        }
    }

    /// The concatenation of two schemas, as a join produces.
    #[must_use]
    pub fn concat(mut self, other: PlanSchema) -> PlanSchema {
        self.fields.extend(other.fields);
        self
    }
}

fn display_reference(qualifier: Option<&str>, name: &str) -> String {
    match qualifier {
        Some(q) => format!("{q}.{name}"),
        None => name.to_string(),
    }
}

impl fmt::Display for PlanSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{field}")?;
        }
        f.write_str("]")
    }
}

/// How a join combines its two inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Rows that match on both sides.
    Inner,
    /// Every left row, null-extended when the right has no match.
    Left,
    /// Every right row, null-extended when the left has no match.
    Right,
    /// The union of `Left` and `Right`.
    Full,
    /// The Cartesian product; the condition list is empty by construction.
    Cross,
    /// Left rows having at least one match, each emitted once.
    Semi,
    /// Left rows having no match.
    Anti,
}

impl JoinType {
    /// Whether the right input contributes columns to the output.
    ///
    /// Semi and anti joins are filters wearing a join's clothing: they use the
    /// right side to decide which left rows survive and then discard it.
    pub fn projects_right(self) -> bool {
        !matches!(self, JoinType::Semi | JoinType::Anti)
    }

    /// Whether left rows may be null-extended.
    pub fn left_nullable(self) -> bool {
        matches!(self, JoinType::Right | JoinType::Full)
    }

    /// Whether right rows may be null-extended.
    pub fn right_nullable(self) -> bool {
        matches!(self, JoinType::Left | JoinType::Full)
    }

    /// The name used by `EXPLAIN`.
    pub fn name(self) -> &'static str {
        match self {
            JoinType::Inner => "Inner",
            JoinType::Left => "Left",
            JoinType::Right => "Right",
            JoinType::Full => "Full",
            JoinType::Cross => "Cross",
            JoinType::Semi => "Semi",
            JoinType::Anti => "Anti",
        }
    }
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One aggregate in an [`LogicalPlan::Aggregate`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    /// Which aggregate.
    pub func: AggregateFunc,
    /// Its argument. `None` is `count(*)`, the only argument-free aggregate.
    pub arg: Option<Expr>,
    /// Whether duplicate argument values are collapsed before aggregating.
    pub distinct: bool,
    /// The output column this aggregate produces.
    pub output: PlanField,
}

impl fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.func.name())?;
        if self.distinct {
            f.write_str("DISTINCT ")?;
        }
        match &self.arg {
            Some(arg) => write!(f, "{arg}")?,
            None => f.write_str("*")?,
        }
        write!(f, ") AS {}", self.output.name)
    }
}

/// Whether a window frame counts rows or compares ordering values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameUnits {
    /// Physical offsets: the third row before this one.
    Rows,
    /// Logical offsets: every peer with the same `ORDER BY` value.
    Range,
}

/// One end of a window frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameBound {
    /// The start of the partition.
    UnboundedPreceding,
    /// `n` units before the current row.
    Preceding(u64),
    /// The current row, or its peer group under `RANGE`.
    CurrentRow,
    /// `n` units after the current row.
    Following(u64),
    /// The end of the partition.
    UnboundedFollowing,
}

impl fmt::Display for WindowFrameBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WindowFrameBound::UnboundedPreceding => f.write_str("UNBOUNDED PRECEDING"),
            WindowFrameBound::Preceding(n) => write!(f, "{n} PRECEDING"),
            WindowFrameBound::CurrentRow => f.write_str("CURRENT ROW"),
            WindowFrameBound::Following(n) => write!(f, "{n} FOLLOWING"),
            WindowFrameBound::UnboundedFollowing => f.write_str("UNBOUNDED FOLLOWING"),
        }
    }
}

/// The rows a window function sees for each current row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowFrameSpec {
    /// Physical or logical offsets.
    pub units: WindowFrameUnits,
    /// The frame's lower bound.
    pub start: WindowFrameBound,
    /// The frame's upper bound.
    pub end: WindowFrameBound,
}

impl WindowFrameSpec {
    /// The SQL default frame for a window that has an `ORDER BY`:
    /// `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which is what makes
    /// `sum(x) OVER (ORDER BY t)` a running total rather than a partition total.
    pub fn default_ordered() -> WindowFrameSpec {
        WindowFrameSpec {
            units: WindowFrameUnits::Range,
            start: WindowFrameBound::UnboundedPreceding,
            end: WindowFrameBound::CurrentRow,
        }
    }

    /// The default frame for a window with no `ORDER BY`: the whole partition.
    pub fn whole_partition() -> WindowFrameSpec {
        WindowFrameSpec {
            units: WindowFrameUnits::Range,
            start: WindowFrameBound::UnboundedPreceding,
            end: WindowFrameBound::UnboundedFollowing,
        }
    }

    /// Reject frames whose bounds are in the wrong order.
    pub fn validate(&self) -> Result<()> {
        let rank = |b: &WindowFrameBound| match b {
            WindowFrameBound::UnboundedPreceding => (0i8, 0u64),
            WindowFrameBound::Preceding(n) => (1, u64::MAX - *n),
            WindowFrameBound::CurrentRow => (2, 0),
            WindowFrameBound::Following(n) => (3, *n),
            WindowFrameBound::UnboundedFollowing => (4, 0),
        };
        if rank(&self.start) > rank(&self.end) {
            return Err(BasaltError::plan(format!(
                "window frame start {} is after its end {}",
                self.start, self.end
            )));
        }
        Ok(())
    }
}

impl fmt::Display for WindowFrameSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let units = match self.units {
            WindowFrameUnits::Rows => "ROWS",
            WindowFrameUnits::Range => "RANGE",
        };
        write!(f, "{units} BETWEEN {} AND {}", self.start, self.end)
    }
}

/// One window function in an [`LogicalPlan::Window`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExpr {
    /// The function and its argument.
    pub call: WindowCall,
    /// The partition keys; an empty list means one partition.
    pub partition_by: Vec<Expr>,
    /// The ordering within each partition.
    pub order_by: Vec<SortExpr>,
    /// The frame.
    pub frame: WindowFrameSpec,
    /// The output column this window function produces.
    pub output: PlanField,
}

impl fmt::Display for WindowExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.call.func.name())?;
        if let Some(arg) = &self.call.arg {
            write!(f, "{arg}")?;
        }
        f.write_str(") OVER (")?;
        if !self.partition_by.is_empty() {
            f.write_str("PARTITION BY ")?;
            for (i, e) in self.partition_by.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{e}")?;
            }
        }
        if !self.order_by.is_empty() {
            if !self.partition_by.is_empty() {
                f.write_str(" ")?;
            }
            f.write_str("ORDER BY ")?;
            for (i, e) in self.order_by.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{e}")?;
            }
        }
        write!(f, " {}) AS {}", self.frame, self.output.name)
    }
}

/// A node of the logical plan.
///
/// Children are held behind [`Arc`] so that a rewrite which leaves a subtree
/// untouched shares it instead of cloning it. Optimizer rules rebuild the spine
/// of the tree many times; without sharing, a ten-rule fixpoint over a deep
/// plan would copy every leaf ten times over.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Read a base table.
    TableScan {
        /// The catalog entry.
        table: Arc<TableDefinition>,
        /// The alias the query gave it, which is what its columns are qualified
        /// by.
        alias: Option<String>,
        /// Indexes into the table's schema, in output order. Column pruning
        /// narrows this rather than inserting a projection, so the scan reads
        /// only what is needed.
        projection: Vec<usize>,
        /// Predicates pushed into the scan. They are a conjunction, and the
        /// executor may evaluate them against zone maps before reading data.
        filters: Vec<Expr>,
        /// The output schema implied by `projection`.
        schema: PlanSchema,
    },
    /// A literal relation, as in `INSERT ... VALUES` or `SELECT` with no `FROM`.
    Values {
        /// Rows of constant expressions, all the same width.
        rows: Vec<Vec<Expr>>,
        /// The output schema.
        schema: PlanSchema,
    },
    /// A relation with no rows, or exactly one row of no columns.
    ///
    /// `produce_one_row` distinguishes `SELECT 1` (one row, so the projection
    /// above it emits a row) from a relation the optimizer proved empty.
    EmptyRelation {
        /// Whether the single empty row is produced.
        produce_one_row: bool,
        /// The output schema.
        schema: PlanSchema,
    },
    /// Keep the rows for which the predicate is true.
    ///
    /// SQL's three-valued logic matters here: a predicate evaluating to NULL
    /// removes the row, exactly as `false` does.
    Filter {
        /// The input.
        input: Arc<LogicalPlan>,
        /// A boolean predicate.
        predicate: Expr,
    },
    /// Compute a new tuple shape.
    Projection {
        /// The input.
        input: Arc<LogicalPlan>,
        /// Each output column and the expression that produces it.
        exprs: Vec<(Expr, PlanField)>,
        /// The output schema, mirroring `exprs`.
        schema: PlanSchema,
    },
    /// Group and aggregate.
    Aggregate {
        /// The input.
        input: Arc<LogicalPlan>,
        /// The grouping keys and the columns they become. An empty list is a
        /// global aggregate, which produces exactly one row even over no input.
        group_by: Vec<(Expr, PlanField)>,
        /// The aggregates.
        aggregates: Vec<AggregateExpr>,
        /// Group columns followed by aggregate columns.
        schema: PlanSchema,
    },
    /// Order the rows.
    Sort {
        /// The input.
        input: Arc<LogicalPlan>,
        /// The sort keys, most significant first.
        keys: Vec<SortExpr>,
    },
    /// Skip and take.
    Limit {
        /// The input.
        input: Arc<LogicalPlan>,
        /// Rows to discard first.
        skip: usize,
        /// Rows to emit, or all of them.
        fetch: Option<usize>,
    },
    /// Combine two relations.
    Join {
        /// The left input.
        left: Arc<LogicalPlan>,
        /// The right input.
        right: Arc<LogicalPlan>,
        /// How the inputs combine.
        join_type: JoinType,
        /// Equality conditions, split into `(left_expr, right_expr)` pairs so a
        /// hash join can read them directly without re-analysing a predicate.
        on: Vec<(Expr, Expr)>,
        /// Any remaining non-equi condition.
        filter: Option<Expr>,
        /// The output schema, with outer-join nullability applied.
        schema: PlanSchema,
    },
    /// Concatenate compatible relations.
    Union {
        /// The arms, all with the same arity and compatible types.
        inputs: Vec<Arc<LogicalPlan>>,
        /// `false` for `UNION`, which deduplicates.
        all: bool,
        /// The output schema, using the first arm's names and the coerced types.
        schema: PlanSchema,
    },
    /// Remove duplicate rows.
    Distinct {
        /// The input.
        input: Arc<LogicalPlan>,
    },
    /// Append window function results to each row.
    Window {
        /// The input.
        input: Arc<LogicalPlan>,
        /// The window functions.
        windows: Vec<WindowExpr>,
        /// The input schema followed by the window columns.
        schema: PlanSchema,
    },
    /// Rename a subquery's relation, requalifying its columns.
    SubqueryAlias {
        /// The input.
        input: Arc<LogicalPlan>,
        /// The new relation name.
        alias: String,
        /// The input schema requalified.
        schema: PlanSchema,
    },
    /// Render a plan instead of running it.
    Explain {
        /// The plan being explained.
        plan: Arc<LogicalPlan>,
        /// Whether to execute and report actual statistics.
        analyze: bool,
        /// A single text column.
        schema: PlanSchema,
    },
}

impl LogicalPlan {
    /// The node's output schema.
    pub fn schema(&self) -> &PlanSchema {
        match self {
            LogicalPlan::TableScan { schema, .. }
            | LogicalPlan::Values { schema, .. }
            | LogicalPlan::EmptyRelation { schema, .. }
            | LogicalPlan::Projection { schema, .. }
            | LogicalPlan::Aggregate { schema, .. }
            | LogicalPlan::Join { schema, .. }
            | LogicalPlan::Union { schema, .. }
            | LogicalPlan::Window { schema, .. }
            | LogicalPlan::SubqueryAlias { schema, .. }
            | LogicalPlan::Explain { schema, .. } => schema,
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => input.schema(),
        }
    }

    /// The operator's name, as `EXPLAIN` prints it.
    pub fn name(&self) -> &'static str {
        match self {
            LogicalPlan::TableScan { .. } => "TableScan",
            LogicalPlan::Values { .. } => "Values",
            LogicalPlan::EmptyRelation { .. } => "EmptyRelation",
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Projection { .. } => "Projection",
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::Sort { .. } => "Sort",
            LogicalPlan::Limit { .. } => "Limit",
            LogicalPlan::Join { .. } => "Join",
            LogicalPlan::Union { .. } => "Union",
            LogicalPlan::Distinct { .. } => "Distinct",
            LogicalPlan::Window { .. } => "Window",
            LogicalPlan::SubqueryAlias { .. } => "SubqueryAlias",
            LogicalPlan::Explain { .. } => "Explain",
        }
    }

    /// The node's inputs, in evaluation order.
    pub fn children(&self) -> Vec<&Arc<LogicalPlan>> {
        match self {
            LogicalPlan::TableScan { .. }
            | LogicalPlan::Values { .. }
            | LogicalPlan::EmptyRelation { .. } => Vec::new(),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::SubqueryAlias { input, .. } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
            LogicalPlan::Union { inputs, .. } => inputs.iter().collect(),
            LogicalPlan::Explain { plan, .. } => vec![plan],
        }
    }

    /// The same node with new inputs.
    ///
    /// Schemas are recomputed for the nodes whose output depends on their
    /// input's shape, so a rule that rewrites a child cannot leave a stale
    /// schema behind. The nodes that own an explicit schema keep it: replacing
    /// a child under a `Projection` must not silently change what the query
    /// returns, and if a rule does that it is the rule that is wrong.
    pub fn with_new_children(&self, mut children: Vec<Arc<LogicalPlan>>) -> Result<LogicalPlan> {
        let expected = self.children().len();
        if children.len() != expected {
            return Err(BasaltError::internal(
                "logical::with_new_children",
                format!(
                    "{} expects {expected} children, got {}",
                    self.name(),
                    children.len()
                ),
            ));
        }
        Ok(match self {
            LogicalPlan::TableScan { .. }
            | LogicalPlan::Values { .. }
            | LogicalPlan::EmptyRelation { .. } => self.clone(),
            LogicalPlan::Filter { predicate, .. } => LogicalPlan::Filter {
                input: children.remove(0),
                predicate: predicate.clone(),
            },
            LogicalPlan::Projection { exprs, schema, .. } => LogicalPlan::Projection {
                input: children.remove(0),
                exprs: exprs.clone(),
                schema: schema.clone(),
            },
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                schema,
                ..
            } => LogicalPlan::Aggregate {
                input: children.remove(0),
                group_by: group_by.clone(),
                aggregates: aggregates.clone(),
                schema: schema.clone(),
            },
            LogicalPlan::Sort { keys, .. } => LogicalPlan::Sort {
                input: children.remove(0),
                keys: keys.clone(),
            },
            LogicalPlan::Limit { skip, fetch, .. } => LogicalPlan::Limit {
                input: children.remove(0),
                skip: *skip,
                fetch: *fetch,
            },
            LogicalPlan::Distinct { .. } => LogicalPlan::Distinct {
                input: children.remove(0),
            },
            LogicalPlan::Window {
                windows, schema, ..
            } => LogicalPlan::Window {
                input: children.remove(0),
                windows: windows.clone(),
                schema: schema.clone(),
            },
            LogicalPlan::SubqueryAlias { alias, schema, .. } => LogicalPlan::SubqueryAlias {
                input: children.remove(0),
                alias: alias.clone(),
                schema: schema.clone(),
            },
            LogicalPlan::Join {
                join_type,
                on,
                filter,
                ..
            } => {
                let left = children.remove(0);
                let right = children.remove(0);
                let schema = join_schema(*join_type, left.schema(), right.schema());
                LogicalPlan::Join {
                    left,
                    right,
                    join_type: *join_type,
                    on: on.clone(),
                    filter: filter.clone(),
                    schema,
                }
            }
            LogicalPlan::Union { all, schema, .. } => LogicalPlan::Union {
                inputs: children,
                all: *all,
                schema: schema.clone(),
            },
            LogicalPlan::Explain {
                analyze, schema, ..
            } => LogicalPlan::Explain {
                plan: children.remove(0),
                analyze: *analyze,
                schema: schema.clone(),
            },
        })
    }

    /// Rewrite the tree bottom up: children first, then the node itself.
    ///
    /// This is the order almost every optimizer rule wants, because a rule that
    /// inspects its input's shape should see the already-rewritten input.
    pub fn transform_up(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>>,
    ) -> Result<Arc<LogicalPlan>> {
        let children = self.children().into_iter().cloned().collect::<Vec<_>>();
        let node = if children.is_empty() {
            self
        } else {
            let mut rewritten = Vec::with_capacity(children.len());
            for child in children {
                rewritten.push(child.transform_up(f)?);
            }
            Arc::new(self.with_new_children(rewritten)?)
        };
        f(node)
    }

    /// Rewrite the tree top down: the node first, then its children.
    ///
    /// Pushdown rules want this order, since pushing a predicate through a node
    /// changes which children exist to push it into.
    pub fn transform_down(
        self: Arc<Self>,
        f: &mut impl FnMut(Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>>,
    ) -> Result<Arc<LogicalPlan>> {
        let node = f(self)?;
        let children = node.children().into_iter().cloned().collect::<Vec<_>>();
        if children.is_empty() {
            return Ok(node);
        }
        let mut rewritten = Vec::with_capacity(children.len());
        for child in children {
            rewritten.push(child.transform_down(f)?);
        }
        Ok(Arc::new(node.with_new_children(rewritten)?))
    }

    /// The identities this node emits.
    pub fn output_columns(&self) -> BTreeSet<ColumnId> {
        self.schema().column_ids()
    }

    /// Every expression the node evaluates itself, excluding its children's.
    pub fn expressions(&self) -> Vec<&Expr> {
        match self {
            LogicalPlan::TableScan { filters, .. } => filters.iter().collect(),
            LogicalPlan::Values { rows, .. } => rows.iter().flatten().collect(),
            LogicalPlan::EmptyRelation { .. }
            | LogicalPlan::Distinct { .. }
            | LogicalPlan::Limit { .. }
            | LogicalPlan::SubqueryAlias { .. }
            | LogicalPlan::Explain { .. } => Vec::new(),
            LogicalPlan::Filter { predicate, .. } => vec![predicate],
            LogicalPlan::Projection { exprs, .. } => exprs.iter().map(|(e, _)| e).collect(),
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => group_by
                .iter()
                .map(|(e, _)| e)
                .chain(aggregates.iter().filter_map(|a| a.arg.as_ref()))
                .collect(),
            LogicalPlan::Sort { keys, .. } => keys.iter().map(|k| &k.expr).collect(),
            LogicalPlan::Join { on, filter, .. } => on
                .iter()
                .flat_map(|(l, r)| [l, r])
                .chain(filter.iter())
                .collect(),
            LogicalPlan::Union { .. } => Vec::new(),
            LogicalPlan::Window { windows, .. } => windows
                .iter()
                .flat_map(|w| {
                    w.call
                        .arg
                        .iter()
                        .chain(w.partition_by.iter())
                        .chain(w.order_by.iter().map(|s| &s.expr))
                })
                .collect(),
        }
    }
}

/// The schema a join produces, with outer-join nullability applied.
///
/// This is the single place the rule lives. It was tempting to inline it into
/// the binder, but the optimizer rebuilds joins during reordering and must
/// arrive at exactly the same answer; two copies of this rule would eventually
/// disagree, and the symptom would be a NOT NULL column that is sometimes null.
pub fn join_schema(join_type: JoinType, left: &PlanSchema, right: &PlanSchema) -> PlanSchema {
    let left = if join_type.left_nullable() {
        left.clone().into_nullable()
    } else {
        left.clone()
    };
    if !join_type.projects_right() {
        return left;
    }
    let right = if join_type.right_nullable() {
        right.clone().into_nullable()
    } else {
        right.clone()
    };
    left.concat(right)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

impl LogicalPlan {
    /// Check that the tree is well-typed and internally consistent.
    ///
    /// The binder is supposed to produce only valid plans and every optimizer
    /// rule is supposed to preserve validity, so in a correct build this never
    /// fires. It exists because "supposed to" is not a guarantee: a rule that
    /// drops a column its parent still references produces a plan that looks
    /// fine and executes into a panic deep in the executor. Running `validate`
    /// after every rule in debug builds turns that into an error naming the
    /// rule, at the moment it happened.
    // One arm per node kind, and every arm is a short list of checks. Splitting
    // it into per-node functions would scatter the invariants across the file
    // and make it harder to see, in one screen, everything the plan promises.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<()> {
        for child in self.children() {
            child.validate()?;
        }
        let in_scope = self.input_scope();
        for expr in self.expressions() {
            check_columns_in_scope(expr, &in_scope, self.name())?;
        }
        match self {
            LogicalPlan::Filter { predicate, .. } => {
                expect_boolean(predicate, "filter predicate")?;
            }
            LogicalPlan::TableScan {
                table,
                projection,
                schema,
                filters,
                ..
            } => {
                if projection.len() != schema.len() {
                    return Err(BasaltError::plan(format!(
                        "table scan of {} projects {} columns but its schema has {}",
                        table.name,
                        projection.len(),
                        schema.len()
                    )));
                }
                for &index in projection {
                    if index >= table.schema.len() {
                        return Err(BasaltError::plan(format!(
                            "table scan of {} projects column {index}, but the table has {}",
                            table.name,
                            table.schema.len()
                        )));
                    }
                }
                for filter in filters {
                    expect_boolean(filter, "pushed-down filter")?;
                }
            }
            LogicalPlan::Values { rows, schema, .. } => {
                for (i, row) in rows.iter().enumerate() {
                    if row.len() != schema.len() {
                        return Err(BasaltError::plan(format!(
                            "VALUES row {i} has {} expressions but the schema has {} columns",
                            row.len(),
                            schema.len()
                        )));
                    }
                }
            }
            LogicalPlan::Projection { exprs, schema, .. } => {
                if exprs.len() != schema.len() {
                    return Err(BasaltError::plan(format!(
                        "projection computes {} expressions but its schema has {} columns",
                        exprs.len(),
                        schema.len()
                    )));
                }
                for (expr, field) in exprs {
                    if expr.data_type() != field.data_type {
                        return Err(BasaltError::plan(format!(
                            "projection column {} is declared {} but its expression {expr} is {}",
                            field.name,
                            field.data_type,
                            expr.data_type()
                        )));
                    }
                }
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                schema,
                ..
            } => {
                if group_by.len() + aggregates.len() != schema.len() {
                    return Err(BasaltError::plan(format!(
                        "aggregate produces {} groups and {} aggregates but its schema has {} columns",
                        group_by.len(),
                        aggregates.len(),
                        schema.len()
                    )));
                }
                for aggregate in aggregates {
                    if let Some(arg) = &aggregate.arg {
                        if contains_aggregate_reference(arg, aggregates) {
                            return Err(BasaltError::plan(format!(
                                "aggregate {} takes another aggregate's output as its argument",
                                aggregate.func.name()
                            )));
                        }
                    } else if aggregate.func != AggregateFunc::Count {
                        return Err(BasaltError::plan(format!(
                            "{} requires an argument; only count(*) may omit one",
                            aggregate.func.name()
                        )));
                    }
                }
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
                schema,
            } => {
                let expected = join_schema(*join_type, left.schema(), right.schema());
                if expected != *schema {
                    return Err(BasaltError::plan(format!(
                        "{join_type} join schema {schema} does not match the one implied by its inputs {expected}"
                    )));
                }
                if *join_type == JoinType::Cross && (!on.is_empty() || filter.is_some()) {
                    return Err(BasaltError::plan(
                        "cross join carries a condition; it should be an inner join".to_string(),
                    ));
                }
                let left_ids = left.schema().column_ids();
                let right_ids = right.schema().column_ids();
                for (l, r) in on {
                    if !l.columns().is_subset(&left_ids) {
                        return Err(BasaltError::plan(format!(
                            "join key {l} references columns that are not on the left side"
                        )));
                    }
                    if !r.columns().is_subset(&right_ids) {
                        return Err(BasaltError::plan(format!(
                            "join key {r} references columns that are not on the right side"
                        )));
                    }
                    if l.data_type() != r.data_type() {
                        return Err(BasaltError::plan(format!(
                            "join keys {l} ({}) and {r} ({}) have different types; the binder must insert a cast",
                            l.data_type(),
                            r.data_type()
                        )));
                    }
                }
                if let Some(filter) = filter {
                    expect_boolean(filter, "join filter")?;
                }
            }
            LogicalPlan::Union { inputs, schema, .. } => {
                if inputs.len() < 2 {
                    return Err(BasaltError::plan(format!(
                        "union has {} arms; it needs at least two",
                        inputs.len()
                    )));
                }
                for (i, input) in inputs.iter().enumerate() {
                    if input.schema().len() != schema.len() {
                        return Err(BasaltError::plan(format!(
                            "union arm {i} has {} columns but the union has {}",
                            input.schema().len(),
                            schema.len()
                        )));
                    }
                    for (j, field) in input.schema().fields().iter().enumerate() {
                        let Some(target) = schema.field(j) else {
                            continue;
                        };
                        if field.data_type != target.data_type {
                            return Err(BasaltError::plan(format!(
                                "union arm {i} column {j} is {} but the union declares {}; \
                                 the binder must insert a cast",
                                field.data_type, target.data_type
                            )));
                        }
                    }
                }
            }
            LogicalPlan::Window {
                input,
                windows,
                schema,
            } => {
                if input.schema().len() + windows.len() != schema.len() {
                    return Err(BasaltError::plan(format!(
                        "window node adds {} columns to a {}-column input but its schema has {}",
                        windows.len(),
                        input.schema().len(),
                        schema.len()
                    )));
                }
                for window in windows {
                    window.frame.validate()?;
                }
            }
            LogicalPlan::Limit { .. }
            | LogicalPlan::Sort { .. }
            | LogicalPlan::Distinct { .. }
            | LogicalPlan::SubqueryAlias { .. }
            | LogicalPlan::EmptyRelation { .. }
            | LogicalPlan::Explain { .. } => {}
        }
        Ok(())
    }

    /// The identities a node's own expressions may reference.
    ///
    /// For most nodes that is simply the input's output. A join is the
    /// exception: its conditions see both sides, which is precisely the scope
    /// its output schema does *not* have for a semi or anti join.
    fn input_scope(&self) -> BTreeSet<ColumnId> {
        match self {
            LogicalPlan::Join { left, right, .. } => {
                let mut ids = left.schema().column_ids();
                ids.extend(right.schema().column_ids());
                ids
            }
            LogicalPlan::Window { input, .. } => input.schema().column_ids(),
            _ => {
                let mut ids = BTreeSet::new();
                for child in self.children() {
                    ids.extend(child.schema().column_ids());
                }
                ids
            }
        }
    }
}

fn expect_boolean(expr: &Expr, what: &str) -> Result<()> {
    if expr.data_type() != DataType::Boolean {
        return Err(BasaltError::type_error(format!(
            "{what} {expr} is {}, not BOOLEAN",
            expr.data_type()
        )));
    }
    Ok(())
}

fn check_columns_in_scope(expr: &Expr, scope: &BTreeSet<ColumnId>, node: &str) -> Result<()> {
    for id in expr.columns() {
        if !scope.contains(&id) {
            return Err(BasaltError::plan(format!(
                "{node} references column {id} which its input does not produce"
            )));
        }
    }
    Ok(())
}

fn contains_aggregate_reference(expr: &Expr, aggregates: &[AggregateExpr]) -> bool {
    let outputs: BTreeSet<ColumnId> = aggregates.iter().map(|a| a.output.id).collect();
    expr.columns().iter().any(|id| outputs.contains(id))
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for LogicalPlan {
    /// Renders the plan the way `EXPLAIN` should: one operator per line, two
    /// spaces of indentation per level, and enough detail on each line to
    /// answer "why is this slow" without a second command.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, 0)
    }
}

impl LogicalPlan {
    // A single exhaustive match over the node kinds; see `validate` for why it
    // is kept whole.
    #[allow(clippy::too_many_lines)]
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        for _ in 0..depth {
            f.write_str("  ")?;
        }
        match self {
            LogicalPlan::TableScan {
                table,
                alias,
                projection,
                filters,
                ..
            } => {
                write!(f, "TableScan: {}", table.name)?;
                if let Some(alias) = alias {
                    if !alias.eq_ignore_ascii_case(&table.name) {
                        write!(f, " AS {alias}")?;
                    }
                }
                let names: Vec<&str> = projection
                    .iter()
                    .filter_map(|&i| table.schema.fields().get(i).map(basalt_types::Field::name))
                    .collect();
                write!(f, " projection=[{}]", names.join(", "))?;
                if !filters.is_empty() {
                    write!(f, " filters=[{}]", join_display(filters))?;
                }
            }
            LogicalPlan::Values { rows, .. } => write!(f, "Values: {} rows", rows.len())?,
            LogicalPlan::EmptyRelation {
                produce_one_row, ..
            } => write!(
                f,
                "EmptyRelation: {}",
                if *produce_one_row {
                    "one row"
                } else {
                    "no rows"
                }
            )?,
            LogicalPlan::Filter { predicate, .. } => write!(f, "Filter: {predicate}")?,
            LogicalPlan::Projection { exprs, .. } => {
                let rendered: Vec<String> = exprs
                    .iter()
                    .map(|(e, field)| {
                        if e.output_name() == field.name {
                            format!("{e}")
                        } else {
                            format!("{e} AS {}", field.name)
                        }
                    })
                    .collect();
                write!(f, "Projection: {}", rendered.join(", "))?;
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                ..
            } => {
                let groups: Vec<String> = group_by.iter().map(|(e, _)| format!("{e}")).collect();
                let aggs: Vec<String> = aggregates.iter().map(|a| format!("{a}")).collect();
                write!(
                    f,
                    "Aggregate: groupBy=[{}] aggregates=[{}]",
                    groups.join(", "),
                    aggs.join(", ")
                )?;
            }
            LogicalPlan::Sort { keys, .. } => {
                let rendered: Vec<String> = keys.iter().map(|k| format!("{k}")).collect();
                write!(f, "Sort: {}", rendered.join(", "))?;
            }
            LogicalPlan::Limit { skip, fetch, .. } => {
                write!(f, "Limit: skip={skip}")?;
                match fetch {
                    Some(n) => write!(f, " fetch={n}")?,
                    None => write!(f, " fetch=all")?,
                }
            }
            LogicalPlan::Join {
                join_type,
                on,
                filter,
                ..
            } => {
                write!(f, "{join_type}Join:")?;
                if !on.is_empty() {
                    let keys: Vec<String> = on.iter().map(|(l, r)| format!("{l} = {r}")).collect();
                    write!(f, " on=[{}]", keys.join(", "))?;
                }
                if let Some(filter) = filter {
                    write!(f, " filter={filter}")?;
                }
            }
            LogicalPlan::Union { all, .. } => {
                write!(f, "Union{}", if *all { " All" } else { "" })?;
            }
            LogicalPlan::Distinct { .. } => f.write_str("Distinct")?,
            LogicalPlan::Window { windows, .. } => {
                let rendered: Vec<String> = windows.iter().map(|w| format!("{w}")).collect();
                write!(f, "Window: {}", rendered.join(", "))?;
            }
            LogicalPlan::SubqueryAlias { alias, .. } => write!(f, "SubqueryAlias: {alias}")?,
            LogicalPlan::Explain { analyze, .. } => {
                write!(f, "Explain{}", if *analyze { " Analyze" } else { "" })?;
            }
        }
        for child in self.children() {
            f.write_str("\n")?;
            child.fmt_indented(f, depth + 1)?;
        }
        Ok(())
    }
}

fn join_display(exprs: &[Expr]) -> String {
    exprs
        .iter()
        .map(|e| format!("{e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

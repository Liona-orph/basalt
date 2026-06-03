//! Semantic analysis: turning a syntax tree into a logical plan.
//!
//! The parser's job ends at "this is grammatical". The binder's job is
//! everything that requires knowing what the names mean: which table `o` is,
//! which column `total` is, what type `a + b` has, whether `count(x)` is legal
//! where it was written. Everything the binder emits is fully resolved --
//! column identities instead of names, explicit casts instead of implicit ones,
//! a known type and nullability on every node -- so that no stage downstream
//! ever has to guess.
//!
//! # Why casts are inserted here and nowhere else
//!
//! SQL's implicit conversions are a source of quiet wrongness: an engine that
//! compares an `INT64` to a `FLOAT64` by promoting at execution time will do it
//! slightly differently in the join operator, the filter kernel and the
//! aggregate, and the three will eventually disagree. Basalt resolves every
//! conversion once, at bind time, into an explicit [`Expr::Cast`]. The executor
//! implements exactly one rule: operands of a binary operator have identical
//! types. If they do not, that is a binder bug, and `LogicalPlan::validate`
//! reports it rather than letting the executor improvise.
//!
//! # Scopes
//!
//! Name resolution walks a chain of `Scope`s. The innermost scope holds the
//! columns of the current query block; its parent holds the enclosing block's,
//! and so on. A reference that resolves in an ancestor rather than the current
//! scope becomes an [`Expr::OuterColumn`], which is what marks a subquery as
//! correlated -- information the optimizer needs in order to decide whether it
//! can be decorrelated into a join.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use basalt_common::{BasaltError, Result};
use basalt_sql::ast;
use basalt_types::{coerce, DataType, Field, Schema, Value};

use crate::catalog::{Catalog, TableDefinition};
use crate::expr::{
    arithmetic_type, BinaryOp, ColumnId, ColumnIdGenerator, ColumnRef, Expr, InValueSet, SortExpr,
    WindowCall,
};
use crate::function::{AggregateFunc, FunctionRegistry, ResolvedFunction, ScalarFunc, WindowFunc};
use crate::logical::{
    join_schema, AggregateExpr, JoinType, LogicalPlan, PlanField, PlanSchema, WindowExpr,
    WindowFrameBound, WindowFrameSpec, WindowFrameUnits,
};

/// A statement that has been checked against the catalog.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundStatement {
    /// A query, ready to optimize and execute.
    Query(LogicalPlan),
    /// `CREATE TABLE`.
    CreateTable {
        /// The table name.
        name: String,
        /// Its schema.
        schema: Schema,
        /// Whether an existing table is tolerated.
        if_not_exists: bool,
        /// The plan feeding a `CREATE TABLE AS SELECT`.
        source: Option<LogicalPlan>,
    },
    /// `DROP TABLE`.
    DropTable {
        /// The tables to drop.
        names: Vec<String>,
        /// Whether a missing table is tolerated.
        if_exists: bool,
    },
    /// `INSERT INTO ... SELECT` or `... VALUES`.
    Insert {
        /// The target table.
        table: Arc<TableDefinition>,
        /// Target column positions, in the order the source produces them.
        columns: Vec<usize>,
        /// The rows to insert, already cast to the target column types.
        source: LogicalPlan,
    },
    /// `EXPLAIN`.
    Explain {
        /// The plan being explained.
        plan: LogicalPlan,
        /// Whether to execute and report actual statistics.
        analyze: bool,
    },
}

/// Bind a statement that is expected to be a query.
///
/// # Errors
///
/// If the statement is not a query, or if semantic analysis rejects it.
pub fn bind(statement: &ast::Statement, catalog: &Catalog) -> Result<LogicalPlan> {
    match bind_statement(statement, catalog)? {
        BoundStatement::Query(plan) => Ok(plan),
        BoundStatement::Explain { plan, analyze } => Ok(LogicalPlan::Explain {
            schema: explain_schema(&mut ColumnIdGenerator::new()),
            plan: Arc::new(plan),
            analyze,
        }),
        other => Err(BasaltError::plan(format!(
            "expected a query, found {}",
            statement_kind(&other)
        ))),
    }
}

/// Bind any statement.
///
/// # Errors
///
/// If a name does not resolve, a type does not check, or the statement uses a
/// construct Basalt does not implement.
pub fn bind_statement(statement: &ast::Statement, catalog: &Catalog) -> Result<BoundStatement> {
    Binder::new(catalog).bind_statement(statement)
}

fn statement_kind(bound: &BoundStatement) -> &'static str {
    match bound {
        BoundStatement::Query(_) => "a query",
        BoundStatement::CreateTable { .. } => "CREATE TABLE",
        BoundStatement::DropTable { .. } => "DROP TABLE",
        BoundStatement::Insert { .. } => "INSERT",
        BoundStatement::Explain { .. } => "EXPLAIN",
    }
}

fn explain_schema(ids: &mut ColumnIdGenerator) -> PlanSchema {
    PlanSchema::new(vec![PlanField::new(
        ids.next(),
        "plan",
        DataType::Utf8,
        false,
    )])
}

/// A chain of visible relations.
///
/// Held by reference rather than cloned so that binding a deeply nested
/// correlated subquery costs one pointer per level instead of one schema copy.
struct Scope<'p> {
    schema: PlanSchema,
    parent: Option<&'p Scope<'p>>,
}

impl<'p> Scope<'p> {
    fn root(schema: PlanSchema) -> Scope<'p> {
        Scope {
            schema,
            parent: None,
        }
    }

    fn child(&'p self, schema: PlanSchema) -> Scope<'p> {
        Scope {
            schema,
            parent: Some(self),
        }
    }

    /// Resolve a name, reporting how far up the chain it was found.
    ///
    /// Depth 0 is the current query block. Anything deeper is a correlated
    /// reference.
    fn resolve(&self, qualifier: Option<&str>, name: &str) -> Result<Option<(&PlanField, usize)>> {
        let mut scope = Some(self);
        let mut depth = 0usize;
        while let Some(current) = scope {
            if let Some(field) = current.schema.resolve(qualifier, name)? {
                return Ok(Some((field, depth)));
            }
            scope = current.parent;
            depth += 1;
        }
        Ok(None)
    }
}

/// The binder's mutable state for one statement.
struct Binder<'a> {
    catalog: &'a Catalog,
    ids: ColumnIdGenerator,
    /// Common table expressions in scope, already bound. A CTE referenced twice
    /// is bound once and shared; the optimizer decides whether to inline it.
    ctes: HashMap<String, Arc<LogicalPlan>>,
}

impl<'a> Binder<'a> {
    fn new(catalog: &'a Catalog) -> Binder<'a> {
        Binder {
            catalog,
            ids: ColumnIdGenerator::new(),
            ctes: HashMap::new(),
        }
    }

    fn bind_statement(&mut self, statement: &ast::Statement) -> Result<BoundStatement> {
        match statement {
            ast::Statement::Query(query) => {
                let plan = self.bind_query(query, None)?;
                Ok(BoundStatement::Query(plan))
            }
            ast::Statement::CreateTable(create) => self.bind_create_table(create),
            ast::Statement::DropTable(drop) => Ok(BoundStatement::DropTable {
                names: drop.names.iter().map(|n| n.base().to_string()).collect(),
                if_exists: drop.if_exists,
            }),
            ast::Statement::Insert(insert) => self.bind_insert(insert),
            ast::Statement::Explain(explain) => {
                let inner = self.bind_statement(&explain.statement)?;
                match inner {
                    BoundStatement::Query(plan) => Ok(BoundStatement::Explain {
                        plan,
                        analyze: explain.analyze,
                    }),
                    other => Err(BasaltError::unsupported(format!(
                        "EXPLAIN of {} is not supported; only queries can be explained",
                        statement_kind(&other)
                    ))),
                }
            }
            ast::Statement::CreateView(_) | ast::Statement::DropView(_) => Err(
                BasaltError::unsupported("views are parsed but not yet planned".to_string()),
            ),
        }
    }

    // -----------------------------------------------------------------------
    // DDL and DML
    // -----------------------------------------------------------------------

    fn bind_create_table(&mut self, create: &ast::CreateTable) -> Result<BoundStatement> {
        let name = create.name.base().to_string();

        if let Some(query) = &create.query {
            // CREATE TABLE AS SELECT: the schema comes from the query, and a
            // column list, if written, renames rather than retypes.
            let plan = self.bind_query(query, None)?;
            let mut fields = Vec::with_capacity(plan.schema().len());
            for (index, field) in plan.schema().fields().iter().enumerate() {
                let column_name = match create.columns.get(index) {
                    Some(def) => def.name.value.clone(),
                    None => field.name.clone(),
                };
                fields.push(Field::new(
                    column_name,
                    field.data_type.clone(),
                    field.nullable,
                ));
            }
            if !create.columns.is_empty() && create.columns.len() != plan.schema().len() {
                return Err(BasaltError::plan(format!(
                    "CREATE TABLE {name} names {} columns but its query produces {}",
                    create.columns.len(),
                    plan.schema().len()
                )));
            }
            return Ok(BoundStatement::CreateTable {
                name,
                schema: Schema::new(fields)?,
                if_not_exists: create.if_not_exists,
                source: Some(plan),
            });
        }

        if create.columns.is_empty() {
            return Err(BasaltError::plan(format!(
                "CREATE TABLE {name} declares no columns"
            )));
        }
        let mut fields = Vec::with_capacity(create.columns.len());
        for column in &create.columns {
            // A column is nullable unless something says otherwise. PRIMARY KEY
            // implies NOT NULL, which is why the option list is scanned rather
            // than only checked for the explicit constraint.
            let mut nullable = true;
            for option in &column.options {
                match option {
                    ast::ColumnOption::NotNull | ast::ColumnOption::PrimaryKey => nullable = false,
                    ast::ColumnOption::Null => nullable = true,
                    ast::ColumnOption::Unique | ast::ColumnOption::Default(_) => {}
                }
            }
            fields.push(Field::new(
                column.name.value.clone(),
                column.data_type.clone(),
                nullable,
            ));
        }
        Ok(BoundStatement::CreateTable {
            name,
            schema: Schema::new(fields)?,
            if_not_exists: create.if_not_exists,
            source: None,
        })
    }

    fn bind_insert(&mut self, insert: &ast::Insert) -> Result<BoundStatement> {
        let table = self.catalog.table(insert.table.base())?;
        let source = self.bind_query(&insert.source, None)?;

        // Resolve the target column list, defaulting to every column in
        // declaration order.
        let targets: Vec<usize> = if insert.columns.is_empty() {
            (0..table.schema.len()).collect()
        } else {
            let mut seen = HashSet::new();
            let mut indexes = Vec::with_capacity(insert.columns.len());
            for ident in &insert.columns {
                let index = table.schema.index_of(&ident.value).map_err(|_| {
                    BasaltError::schema(format!(
                        "table {} has no column {}",
                        table.name, ident.value
                    ))
                })?;
                if !seen.insert(index) {
                    return Err(BasaltError::plan(format!(
                        "column {} is named twice in the INSERT column list",
                        ident.value
                    )));
                }
                indexes.push(index);
            }
            indexes
        };

        if targets.len() != source.schema().len() {
            return Err(BasaltError::plan(format!(
                "INSERT INTO {} targets {} columns but its source produces {}",
                table.name,
                targets.len(),
                source.schema().len()
            )));
        }

        // Cast the source to the target types here rather than at execution
        // time, so that an impossible conversion is a bind error naming the
        // column rather than a partially-written table.
        let mut exprs = Vec::with_capacity(targets.len());
        let mut fields = Vec::with_capacity(targets.len());
        for (position, &target) in targets.iter().enumerate() {
            let column = table.schema.field(target);
            let field = &source.schema().fields()[position];
            let mut expr = Expr::Column(ColumnRef {
                id: field.id,
                qualifier: field.qualifier.clone(),
                name: field.name.clone(),
                data_type: field.data_type.clone(),
                nullable: field.nullable,
            });
            if field.data_type != *column.data_type() {
                if coerce(&field.data_type, column.data_type()).as_ref() != Some(column.data_type())
                {
                    return Err(BasaltError::type_error(format!(
                        "cannot insert {} into column {} of type {}",
                        field.data_type,
                        column.name(),
                        column.data_type()
                    )));
                }
                expr = Expr::Cast {
                    expr: Box::new(expr),
                    data_type: column.data_type().clone(),
                };
            }
            if !column.is_nullable() && field.nullable {
                // Not an error: the row-level check belongs to the writer,
                // which is the only place that sees actual values. Recorded
                // here so the writer knows which columns to police.
            }
            let output = PlanField::new(
                self.ids.next(),
                column.name().to_string(),
                column.data_type().clone(),
                column.is_nullable(),
            );
            fields.push(output.clone());
            exprs.push((expr, output));
        }

        let projected = LogicalPlan::Projection {
            input: Arc::new(source),
            exprs,
            schema: PlanSchema::new(fields),
        };
        Ok(BoundStatement::Insert {
            table,
            columns: targets,
            source: projected,
        })
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    fn bind_query(&mut self, query: &ast::Query, outer: Option<&Scope<'_>>) -> Result<LogicalPlan> {
        // CTEs are bound before the body so the body can see them, and are
        // removed afterwards so a sibling query block cannot.
        let mut introduced = Vec::new();
        if let Some(with) = &query.with {
            if with.recursive {
                return Err(BasaltError::unsupported(
                    "WITH RECURSIVE is parsed but not yet planned".to_string(),
                ));
            }
            for cte in &with.ctes {
                let plan = self.bind_query(&cte.query, outer)?;
                let plan = self.apply_cte_alias(plan, &cte.name.value, &cte.columns)?;
                let key = cte.name.value.to_ascii_lowercase();
                self.ctes.insert(key.clone(), Arc::new(plan));
                introduced.push(key);
            }
        }

        let result = self.bind_query_body(query, outer);

        for key in introduced {
            self.ctes.remove(&key);
        }
        result
    }

    fn bind_query_body(
        &mut self,
        query: &ast::Query,
        outer: Option<&Scope<'_>>,
    ) -> Result<LogicalPlan> {
        let (mut plan, projection_aliases) = self.bind_set_expr(&query.body, outer)?;

        if !query.order_by.is_empty() {
            let scope = match outer {
                Some(parent) => parent.child(plan.schema().clone()),
                None => Scope::root(plan.schema().clone()),
            };
            let mut keys = Vec::with_capacity(query.order_by.len());
            for item in &query.order_by {
                let expr = self.bind_order_key(item, &scope, &projection_aliases, plan.schema())?;
                keys.push(SortExpr::new(
                    expr,
                    item.asc.unwrap_or(true),
                    // SQL's default is engine-defined; Basalt follows the
                    // common convention that nulls sort last ascending and
                    // first descending, so reversing a sort reverses the whole
                    // row order rather than only the non-null part.
                    item.nulls_first.unwrap_or(!item.asc.unwrap_or(true)),
                ));
            }
            plan = LogicalPlan::Sort {
                input: Arc::new(plan),
                keys,
            };
        }

        let skip = match &query.offset {
            Some(expr) => self.bind_row_count(expr, "OFFSET")?,
            None => 0,
        };
        let fetch = match &query.limit {
            Some(expr) => Some(self.bind_row_count(expr, "LIMIT")?),
            None => None,
        };
        if skip != 0 || fetch.is_some() {
            plan = LogicalPlan::Limit {
                input: Arc::new(plan),
                skip,
                fetch,
            };
        }
        Ok(plan)
    }

    /// `LIMIT` and `OFFSET` take a constant, not an expression over the rows
    /// being limited -- so this evaluates at bind time and rejects anything
    /// that is not a non-negative integer literal.
    fn bind_row_count(&mut self, expr: &ast::Expr, clause: &str) -> Result<usize> {
        let bound = self.bind_expr(
            expr,
            &Scope::root(PlanSchema::empty()),
            &mut AggregateSink::forbidden(clause),
        )?;
        let Expr::Literal(value) = bound else {
            return Err(BasaltError::plan(format!(
                "{clause} requires a constant, found an expression"
            )));
        };
        match value.as_i128() {
            Some(n) if n >= 0 => usize::try_from(n).map_err(|_| {
                BasaltError::plan(format!(
                    "{clause} value {n} is larger than this platform allows"
                ))
            }),
            Some(n) => Err(BasaltError::plan(format!(
                "{clause} must not be negative, found {n}"
            ))),
            None => Err(BasaltError::plan(format!(
                "{clause} requires an integer, found {}",
                value.data_type()
            ))),
        }
    }

    fn apply_cte_alias(
        &mut self,
        plan: LogicalPlan,
        alias: &str,
        columns: &[ast::Ident],
    ) -> Result<LogicalPlan> {
        let plan = if columns.is_empty() {
            plan
        } else {
            if columns.len() != plan.schema().len() {
                return Err(BasaltError::plan(format!(
                    "CTE {alias} names {} columns but its query produces {}",
                    columns.len(),
                    plan.schema().len()
                )));
            }
            self.rename_columns(plan, columns)?
        };
        let schema = PlanSchema::new(
            plan.schema()
                .fields()
                .iter()
                .cloned()
                .map(|f| f.with_qualifier(Some(alias.to_string())))
                .collect(),
        );
        Ok(LogicalPlan::SubqueryAlias {
            input: Arc::new(plan),
            alias: alias.to_string(),
            schema,
        })
    }

    /// Rename a relation's output columns by wrapping it in a projection.
    ///
    /// A projection rather than an in-place schema edit, because the identities
    /// must change too: two CTEs over the same subquery must not share column
    /// ids, or a predicate pushed into one would land in both.
    // Infallible today, but it returns `Result` because every other plan
    // constructor in this file does; a caller should not have to remember which
    // ones can fail, and a future column-count check here would otherwise be a
    // breaking change.
    #[allow(clippy::unnecessary_wraps)]
    fn rename_columns(&mut self, plan: LogicalPlan, names: &[ast::Ident]) -> Result<LogicalPlan> {
        let mut exprs = Vec::with_capacity(names.len());
        let mut fields = Vec::with_capacity(names.len());
        for (field, ident) in plan.schema().fields().iter().zip(names) {
            let output = PlanField::new(
                self.ids.next(),
                ident.value.clone(),
                field.data_type.clone(),
                field.nullable,
            );
            fields.push(output.clone());
            exprs.push((column_expr(field), output));
        }
        Ok(LogicalPlan::Projection {
            input: Arc::new(plan),
            exprs,
            schema: PlanSchema::new(fields),
        })
    }
}

/// Build a resolved column reference from a schema field.
fn column_expr(field: &PlanField) -> Expr {
    Expr::Column(ColumnRef {
        id: field.id,
        qualifier: field.qualifier.clone(),
        name: field.name.clone(),
        data_type: field.data_type.clone(),
        nullable: field.nullable,
    })
}

// ---------------------------------------------------------------------------
// Set expressions and SELECT
// ---------------------------------------------------------------------------

/// The output aliases of a `SELECT`, so that `ORDER BY` can refer to them.
type ProjectionAliases = Vec<(String, PlanField)>;

impl Binder<'_> {
    fn bind_set_expr(
        &mut self,
        body: &ast::SetExpr,
        outer: Option<&Scope<'_>>,
    ) -> Result<(LogicalPlan, ProjectionAliases)> {
        match body {
            ast::SetExpr::Select(select) => self.bind_select(select, outer),
            ast::SetExpr::Query(query) => Ok((self.bind_query(query, outer)?, Vec::new())),
            ast::SetExpr::Values(values) => Ok((self.bind_values(values, outer)?, Vec::new())),
            ast::SetExpr::SetOperation {
                op,
                quantifier,
                left,
                right,
            } => {
                let (left_plan, _) = self.bind_set_expr(left, outer)?;
                let (right_plan, _) = self.bind_set_expr(right, outer)?;
                let plan = self.bind_set_operation(*op, *quantifier, left_plan, right_plan)?;
                Ok((plan, Vec::new()))
            }
        }
    }

    /// `UNION`, `EXCEPT` and `INTERSECT`.
    ///
    /// Only `UNION` is planned today. `EXCEPT` and `INTERSECT` are anti- and
    /// semi-joins on every column with null-safe equality; they are rejected
    /// explicitly rather than approximated, because an approximation that gets
    /// NULL handling wrong is worse than a clear refusal.
    fn bind_set_operation(
        &mut self,
        op: ast::SetOperator,
        quantifier: ast::SetQuantifier,
        left: LogicalPlan,
        right: LogicalPlan,
    ) -> Result<LogicalPlan> {
        if op != ast::SetOperator::Union {
            return Err(BasaltError::unsupported(format!(
                "{op:?} is parsed but not yet planned; only UNION is implemented"
            )));
        }
        if left.schema().len() != right.schema().len() {
            return Err(BasaltError::plan(format!(
                "UNION arms have different widths: {} and {} columns",
                left.schema().len(),
                right.schema().len()
            )));
        }

        // The result type of each column is the common type of the arms. Both
        // arms are then projected to it, so the executor never has to reconcile
        // types across a union.
        let mut target = Vec::with_capacity(left.schema().len());
        for (index, (l, r)) in left
            .schema()
            .fields()
            .iter()
            .zip(right.schema().fields())
            .enumerate()
        {
            let Some(common) = coerce(&l.data_type, &r.data_type) else {
                return Err(BasaltError::type_error(format!(
                    "UNION column {} has incompatible types {} and {}",
                    index + 1,
                    l.data_type,
                    r.data_type
                )));
            };
            target.push((l.name.clone(), common, l.nullable || r.nullable));
        }

        let left = self.cast_to_union_types(left, &target)?;
        let right = self.cast_to_union_types(right, &target)?;
        let schema = PlanSchema::new(
            left.schema()
                .fields()
                .iter()
                .zip(&target)
                .map(|(field, (name, ty, nullable))| {
                    PlanField::new(field.id, name.clone(), ty.clone(), *nullable)
                })
                .collect(),
        );

        let all = quantifier == ast::SetQuantifier::All;
        let union = LogicalPlan::Union {
            inputs: vec![Arc::new(left), Arc::new(right)],
            all,
            schema,
        };
        Ok(if all {
            union
        } else {
            LogicalPlan::Distinct {
                input: Arc::new(union),
            }
        })
    }

    #[allow(clippy::unnecessary_wraps)] // See `rename_columns`.
    fn cast_to_union_types(
        &mut self,
        plan: LogicalPlan,
        target: &[(String, DataType, bool)],
    ) -> Result<LogicalPlan> {
        if plan
            .schema()
            .fields()
            .iter()
            .zip(target)
            .all(|(f, (_, ty, _))| f.data_type == *ty)
        {
            return Ok(plan);
        }
        let mut exprs = Vec::with_capacity(target.len());
        let mut fields = Vec::with_capacity(target.len());
        for (field, (name, ty, nullable)) in plan.schema().fields().iter().zip(target) {
            let mut expr = column_expr(field);
            if field.data_type != *ty {
                expr = Expr::Cast {
                    expr: Box::new(expr),
                    data_type: ty.clone(),
                };
            }
            let output = PlanField::new(self.ids.next(), name.clone(), ty.clone(), *nullable);
            fields.push(output.clone());
            exprs.push((expr, output));
        }
        Ok(LogicalPlan::Projection {
            input: Arc::new(plan),
            exprs,
            schema: PlanSchema::new(fields),
        })
    }

    fn bind_values(
        &mut self,
        values: &ast::Values,
        outer: Option<&Scope<'_>>,
    ) -> Result<LogicalPlan> {
        if values.rows.is_empty() {
            return Err(BasaltError::plan("VALUES has no rows".to_string()));
        }
        let width = values.rows[0].len();
        let empty = PlanSchema::empty();
        let scope = match outer {
            Some(parent) => parent.child(empty),
            None => Scope::root(empty),
        };

        let mut rows: Vec<Vec<Expr>> = Vec::with_capacity(values.rows.len());
        for (index, row) in values.rows.iter().enumerate() {
            if row.len() != width {
                return Err(BasaltError::plan(format!(
                    "VALUES row {} has {} expressions but row 1 has {width}",
                    index + 1,
                    row.len()
                )));
            }
            let mut bound = Vec::with_capacity(width);
            for expr in row {
                bound.push(self.bind_expr(
                    expr,
                    &scope,
                    &mut AggregateSink::forbidden("VALUES"),
                )?);
            }
            rows.push(bound);
        }

        // Every row must agree on a column's type, so the column type is the
        // common type of the whole column and every row is cast to it.
        let mut types = Vec::with_capacity(width);
        for column in 0..width {
            let mut current = rows[0][column].data_type();
            for row in &rows[1..] {
                let candidate = row[column].data_type();
                current = coerce(&current, &candidate).ok_or_else(|| {
                    BasaltError::type_error(format!(
                        "VALUES column {} mixes incompatible types {current} and {candidate}",
                        column + 1
                    ))
                })?;
            }
            types.push(current);
        }
        for row in &mut rows {
            for (column, expr) in row.iter_mut().enumerate() {
                if expr.data_type() != types[column] {
                    let inner = std::mem::replace(expr, Expr::Literal(Value::Null));
                    *expr = Expr::Cast {
                        expr: Box::new(inner),
                        data_type: types[column].clone(),
                    };
                }
            }
        }

        let schema = PlanSchema::new(
            types
                .iter()
                .enumerate()
                .map(|(index, ty)| {
                    let nullable = rows.iter().any(|row| row[index].nullable());
                    PlanField::new(
                        self.ids.next(),
                        format!("column{}", index + 1),
                        ty.clone(),
                        nullable,
                    )
                })
                .collect(),
        );
        Ok(LogicalPlan::Values { rows, schema })
    }

    #[allow(clippy::too_many_lines)] // The SELECT pipeline reads best in order.
    fn bind_select(
        &mut self,
        select: &ast::Select,
        outer: Option<&Scope<'_>>,
    ) -> Result<(LogicalPlan, ProjectionAliases)> {
        let (mut plan, visible) = self.bind_from(&select.from, outer)?;
        let scope = match outer {
            Some(parent) => parent.child(visible.clone()),
            None => Scope::root(visible.clone()),
        };

        if let Some(predicate) = &select.selection {
            let expr = self.bind_expr(predicate, &scope, &mut AggregateSink::forbidden("WHERE"))?;
            let expr = coerce_to_boolean(expr, "WHERE")?;
            plan = LogicalPlan::Filter {
                input: Arc::new(plan),
                predicate: expr,
            };
        }

        // Group keys are bound first so that projection and HAVING expressions
        // can be matched against them structurally.
        let mut group_keys: Vec<(Expr, PlanField)> = Vec::with_capacity(select.group_by.len());
        for key in &select.group_by {
            let expr = self.bind_expr(key, &scope, &mut AggregateSink::forbidden("GROUP BY"))?;
            let output = PlanField::new(
                self.ids.next(),
                expr.output_name(),
                expr.data_type(),
                expr.nullable(),
            );
            group_keys.push((expr, output));
        }

        let mut sink = AggregateSink::collecting();
        let mut items = self.bind_projection(select, &scope, &mut sink)?;
        let having = match &select.having {
            Some(expr) => Some(coerce_to_boolean(
                self.bind_expr(expr, &scope, &mut sink)?,
                "HAVING",
            )?),
            None => None,
        };
        let aggregates = sink.take_aggregates();
        let mut windows = sink.take_windows();

        let aggregating = !group_keys.is_empty() || !aggregates.is_empty();
        if aggregating {
            plan = LogicalPlan::Aggregate {
                input: Arc::new(plan),
                group_by: group_keys.clone(),
                aggregates: aggregates.clone(),
                schema: aggregate_schema(&group_keys, &aggregates),
            };
            let allowed = plan.schema().column_ids();
            for item in &mut items {
                item.expr = substitute_group_keys(
                    std::mem::replace(&mut item.expr, Expr::Literal(Value::Null)),
                    &group_keys,
                )?;
                check_grouped(&item.expr, &allowed, &group_keys)?;
            }
        } else if having.is_some() {
            return Err(BasaltError::plan(
                "HAVING requires GROUP BY or an aggregate; use WHERE to filter individual rows"
                    .to_string(),
            ));
        }

        if let Some(having) = having {
            let having = substitute_group_keys(having, &group_keys)?;
            check_grouped(&having, &plan.schema().column_ids(), &group_keys)?;
            plan = LogicalPlan::Filter {
                input: Arc::new(plan),
                predicate: having,
            };
        }

        // Window functions run after aggregation, so the expressions inside
        // their OVER clauses are rewritten against the aggregate's output
        // before the Window node is built.
        for window in &mut windows {
            rewrite_window(window, &group_keys)?;
        }
        if !windows.is_empty() {
            let mut fields = plan.schema().fields().to_vec();
            fields.extend(windows.iter().map(|w| w.output.clone()));
            plan = LogicalPlan::Window {
                input: Arc::new(plan),
                windows,
                schema: PlanSchema::new(fields),
            };
        }

        let mut exprs = Vec::with_capacity(items.len());
        let mut fields = Vec::with_capacity(items.len());
        let mut aliases = Vec::with_capacity(items.len());
        for item in items {
            let output = PlanField {
                id: self.ids.next(),
                qualifier: item.qualifier,
                name: item.name.clone(),
                data_type: item.expr.data_type(),
                nullable: item.expr.nullable(),
            };
            aliases.push((item.name, output.clone()));
            fields.push(output.clone());
            exprs.push((item.expr, output));
        }
        plan = LogicalPlan::Projection {
            input: Arc::new(plan),
            exprs,
            schema: PlanSchema::new(fields),
        };

        if select.distinct {
            plan = LogicalPlan::Distinct {
                input: Arc::new(plan),
            };
        }
        Ok((plan, aliases))
    }
}

/// One column of a `SELECT` list, after wildcard expansion.
struct ProjectionItem {
    expr: Expr,
    name: String,
    qualifier: Option<String>,
}

/// The schema an `Aggregate` node produces: group keys then aggregates.
fn aggregate_schema(group_by: &[(Expr, PlanField)], aggregates: &[AggregateExpr]) -> PlanSchema {
    PlanSchema::new(
        group_by
            .iter()
            .map(|(_, field)| field.clone())
            .chain(aggregates.iter().map(|a| a.output.clone()))
            .collect(),
    )
}

/// Replace any subexpression equal to a group key with a reference to the
/// aggregate's output column.
///
/// Structural equality is the right test here and it is what the standard
/// describes: `GROUP BY a + b` makes `a + b` -- and nothing else -- legal in the
/// select list. Basalt does not try to prove that `b + a` is the same thing,
/// because a rule that is clever in some cases and not others is harder to
/// explain than one that is simple and predictable.
fn substitute_group_keys(expr: Expr, group_by: &[(Expr, PlanField)]) -> Result<Expr> {
    if group_by.is_empty() {
        return Ok(expr);
    }
    expr.transform_up(&mut |node| {
        for (key, field) in group_by {
            if node == *key {
                return Ok(column_expr(field));
            }
        }
        Ok(node)
    })
}

/// Reject a select-list or HAVING expression that reads a column the aggregate
/// did not produce.
fn check_grouped(
    expr: &Expr,
    allowed: &std::collections::BTreeSet<ColumnId>,
    group_by: &[(Expr, PlanField)],
) -> Result<()> {
    for id in expr.columns() {
        if allowed.contains(&id) {
            continue;
        }
        let name = find_column_name(expr, id).unwrap_or_else(|| id.to_string());
        let hint = if group_by.is_empty() {
            "add a GROUP BY clause, or wrap it in an aggregate such as min() or max()".to_string()
        } else {
            let keys: Vec<String> = group_by.iter().map(|(_, f)| f.name.clone()).collect();
            format!(
                "add it to GROUP BY (currently grouping by {}), or wrap it in an aggregate",
                keys.join(", ")
            )
        };
        return Err(BasaltError::plan(format!(
            "column {name} must appear in GROUP BY or be used in an aggregate function; {hint}"
        )));
    }
    Ok(())
}

fn find_column_name(expr: &Expr, id: ColumnId) -> Option<String> {
    let mut found = None;
    expr.visit(&mut |node| {
        if let Expr::Column(reference) = node {
            if reference.id == id {
                found = Some(reference.to_string());
                return false;
            }
        }
        true
    });
    found
}

/// Turn a non-boolean predicate into a bind error.
///
/// SQL has no truthiness: `WHERE 1` is a type error, not a shorthand. Saying so
/// at bind time is kinder than evaluating it and returning a surprising answer.
fn coerce_to_boolean(expr: Expr, clause: &str) -> Result<Expr> {
    match expr.data_type() {
        DataType::Boolean => Ok(expr),
        DataType::Null => Ok(Expr::Cast {
            expr: Box::new(expr),
            data_type: DataType::Boolean,
        }),
        other => Err(BasaltError::type_error(format!(
            "{clause} requires a boolean expression, found {other}"
        ))),
    }
}

/// Rewrite a collected window function against the aggregate output beneath it.
fn rewrite_window(window: &mut WindowExpr, group_by: &[(Expr, PlanField)]) -> Result<()> {
    if group_by.is_empty() {
        return Ok(());
    }
    if let Some(arg) = window.call.arg.take() {
        window.call.arg = Some(substitute_group_keys(arg, group_by)?);
    }
    for expr in &mut window.partition_by {
        let taken = std::mem::replace(expr, Expr::Literal(Value::Null));
        *expr = substitute_group_keys(taken, group_by)?;
    }
    for key in &mut window.order_by {
        let taken = std::mem::replace(&mut key.expr, Expr::Literal(Value::Null));
        key.expr = substitute_group_keys(taken, group_by)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FROM
// ---------------------------------------------------------------------------

impl Binder<'_> {
    /// Bind the `FROM` clause.
    ///
    /// Returns the plan and the schema that *name resolution* should see, which
    /// is not always the plan's output schema: `JOIN ... USING (id)` produces
    /// both `id` columns but only one of them is visible as a bare `id`.
    /// Keeping the two apart means the join node stays consistent with
    /// [`join_schema`] -- which `validate` checks -- while `SELECT id` still
    /// resolves.
    fn bind_from(
        &mut self,
        from: &[ast::TableRef],
        outer: Option<&Scope<'_>>,
    ) -> Result<(LogicalPlan, PlanSchema)> {
        if from.is_empty() {
            // `SELECT 1` with no FROM: one row of no columns, so the projection
            // above produces exactly one row.
            return Ok((
                LogicalPlan::EmptyRelation {
                    produce_one_row: true,
                    schema: PlanSchema::empty(),
                },
                PlanSchema::empty(),
            ));
        }
        let (mut plan, mut visible) = self.bind_table_ref(&from[0], outer)?;
        for reference in &from[1..] {
            // A comma-separated FROM list is a cross join. The optimizer turns
            // it into an inner join once it sees the WHERE predicate.
            let (right, right_visible) = self.bind_table_ref(reference, outer)?;
            let schema = join_schema(JoinType::Cross, plan.schema(), right.schema());
            plan = LogicalPlan::Join {
                left: Arc::new(plan),
                right: Arc::new(right),
                join_type: JoinType::Cross,
                on: Vec::new(),
                filter: None,
                schema,
            };
            visible = visible.concat(right_visible);
        }
        Ok((plan, visible))
    }

    fn bind_table_ref(
        &mut self,
        reference: &ast::TableRef,
        outer: Option<&Scope<'_>>,
    ) -> Result<(LogicalPlan, PlanSchema)> {
        let (mut plan, mut visible) = self.bind_table_factor(&reference.relation, outer)?;
        for join in &reference.joins {
            let (right, right_visible) = self.bind_table_factor(&join.relation, outer)?;
            let join_type = match join.operator {
                ast::JoinOperator::Inner => JoinType::Inner,
                ast::JoinOperator::Left => JoinType::Left,
                ast::JoinOperator::Right => JoinType::Right,
                ast::JoinOperator::Full => JoinType::Full,
                ast::JoinOperator::Cross => JoinType::Cross,
            };
            let schema = join_schema(join_type, plan.schema(), right.schema());

            let mut on = Vec::new();
            let mut filter = None;
            let mut hidden: Vec<ColumnId> = Vec::new();

            match &join.constraint {
                ast::JoinConstraint::None => {
                    if join_type != JoinType::Cross {
                        return Err(BasaltError::plan(format!(
                            "{join_type} join needs an ON or USING clause"
                        )));
                    }
                }
                ast::JoinConstraint::On(expr) => {
                    // The condition sees both sides, so it is bound against the
                    // concatenation of the visible schemas regardless of the
                    // join type.
                    let both = visible.clone().concat(right_visible.clone());
                    let scope = match outer {
                        Some(parent) => parent.child(both),
                        None => Scope::root(both),
                    };
                    let bound =
                        self.bind_expr(expr, &scope, &mut AggregateSink::forbidden("JOIN ON"))?;
                    let bound = coerce_to_boolean(bound, "JOIN ON")?;
                    let (keys, rest) = split_equi_conditions(bound, plan.schema(), right.schema());
                    on = keys;
                    filter = rest;
                }
                ast::JoinConstraint::Using(columns) => {
                    let names: Vec<String> =
                        columns.iter().map(|ident| ident.value.clone()).collect();
                    let (keys, dropped) =
                        equate_named_columns(&names, &visible, &right_visible, "USING")?;
                    on = keys;
                    hidden = dropped;
                }
                ast::JoinConstraint::Natural => {
                    let names = common_column_names(&visible, &right_visible);
                    if names.is_empty() {
                        return Err(BasaltError::plan(
                            "NATURAL join has no column names in common; write an explicit ON clause"
                                .to_string(),
                        ));
                    }
                    let (keys, dropped) =
                        equate_named_columns(&names, &visible, &right_visible, "NATURAL join")?;
                    on = keys;
                    hidden = dropped;
                }
            }

            plan = LogicalPlan::Join {
                left: Arc::new(plan),
                right: Arc::new(right),
                join_type,
                on,
                filter,
                schema,
            };
            visible = merge_visible(visible, right_visible, join_type, &hidden);
        }
        Ok((plan, visible))
    }

    fn bind_table_factor(
        &mut self,
        factor: &ast::TableFactor,
        outer: Option<&Scope<'_>>,
    ) -> Result<(LogicalPlan, PlanSchema)> {
        match factor {
            ast::TableFactor::Table { name, alias } => {
                let base = name.base();
                let qualifier = alias
                    .as_ref()
                    .map_or_else(|| base.to_string(), |a| a.name.value.clone());

                if let Some(cte) = self.ctes.get(&base.to_ascii_lowercase()).cloned() {
                    // A CTE referenced twice must not share column identities,
                    // so each reference gets a fresh projection over the shared
                    // subtree.
                    let renamed = self.requalify(cte, &qualifier)?;
                    let visible = renamed.schema().clone();
                    return Ok((renamed, visible));
                }

                let table = self.catalog.table(base)?;
                let projection: Vec<usize> = (0..table.schema.len()).collect();
                let schema = PlanSchema::new(
                    table
                        .schema
                        .fields()
                        .iter()
                        .map(|field| {
                            PlanField::qualified(
                                self.ids.next(),
                                qualifier.clone(),
                                field.name().to_string(),
                                field.data_type().clone(),
                                field.is_nullable(),
                            )
                        })
                        .collect(),
                );
                let plan = LogicalPlan::TableScan {
                    table,
                    alias: alias.as_ref().map(|a| a.name.value.clone()),
                    projection,
                    filters: Vec::new(),
                    schema: schema.clone(),
                };
                // A table alias with a column list renames as well as qualifies.
                if let Some(alias) = alias {
                    if !alias.columns.is_empty() {
                        if alias.columns.len() != schema.len() {
                            return Err(BasaltError::plan(format!(
                                "alias {} names {} columns but {} has {}",
                                alias.name.value,
                                alias.columns.len(),
                                base,
                                schema.len()
                            )));
                        }
                        let renamed = self.rename_columns(plan, &alias.columns)?;
                        let renamed = self.requalify(Arc::new(renamed), &qualifier)?;
                        let visible = renamed.schema().clone();
                        return Ok((renamed, visible));
                    }
                }
                Ok((plan, schema))
            }
            ast::TableFactor::Derived { subquery, alias } => {
                let Some(alias) = alias else {
                    return Err(BasaltError::plan(
                        "a subquery in FROM must have an alias, because its columns need a name to be qualified by"
                            .to_string(),
                    ));
                };
                let plan = self.bind_query(subquery, outer)?;
                let plan = if alias.columns.is_empty() {
                    plan
                } else {
                    if alias.columns.len() != plan.schema().len() {
                        return Err(BasaltError::plan(format!(
                            "alias {} names {} columns but its subquery produces {}",
                            alias.name.value,
                            alias.columns.len(),
                            plan.schema().len()
                        )));
                    }
                    self.rename_columns(plan, &alias.columns)?
                };
                let plan = self.requalify(Arc::new(plan), &alias.name.value)?;
                let visible = plan.schema().clone();
                Ok((plan, visible))
            }
            ast::TableFactor::Nested(inner) => self.bind_table_ref(inner, outer),
        }
    }

    /// Wrap a plan so its columns carry a new qualifier and fresh identities.
    #[allow(clippy::unnecessary_wraps)] // See `rename_columns`.
    fn requalify(&mut self, plan: Arc<LogicalPlan>, alias: &str) -> Result<LogicalPlan> {
        let mut exprs = Vec::with_capacity(plan.schema().len());
        let mut fields = Vec::with_capacity(plan.schema().len());
        for field in plan.schema().fields() {
            let output = PlanField::qualified(
                self.ids.next(),
                alias.to_string(),
                field.name.clone(),
                field.data_type.clone(),
                field.nullable,
            );
            fields.push(output.clone());
            exprs.push((column_expr(field), output));
        }
        let schema = PlanSchema::new(fields);
        Ok(LogicalPlan::SubqueryAlias {
            input: Arc::new(LogicalPlan::Projection {
                input: plan,
                exprs,
                schema: schema.clone(),
            }),
            alias: alias.to_string(),
            schema,
        })
    }
}

/// Split a join condition into hash-join keys and a residual predicate.
///
/// Recognising equalities here rather than in the optimizer means a plan is
/// hash-joinable the moment it is bound, and `EXPLAIN` on an unoptimized plan
/// already shows the join keys -- which is exactly what someone debugging a
/// slow query wants to see.
fn split_equi_conditions(
    condition: Expr,
    left: &PlanSchema,
    right: &PlanSchema,
) -> (Vec<(Expr, Expr)>, Option<Expr>) {
    let left_ids = left.column_ids();
    let right_ids = right.column_ids();
    let mut keys = Vec::new();
    let mut residual = Vec::new();

    for conjunct in condition.split_conjuncts() {
        if let Expr::Binary {
            left: l,
            op,
            right: r,
        } = &conjunct
        {
            if *op == BinaryOp::Eq && l.data_type() == r.data_type() {
                let lc = l.columns();
                let rc = r.columns();
                if !lc.is_empty()
                    && !rc.is_empty()
                    && lc.is_subset(&left_ids)
                    && rc.is_subset(&right_ids)
                {
                    keys.push((l.as_ref().clone(), r.as_ref().clone()));
                    continue;
                }
                if !lc.is_empty()
                    && !rc.is_empty()
                    && lc.is_subset(&right_ids)
                    && rc.is_subset(&left_ids)
                {
                    keys.push((r.as_ref().clone(), l.as_ref().clone()));
                    continue;
                }
            }
        }
        residual.push(conjunct);
    }

    let filter = if residual.is_empty() {
        None
    } else {
        Some(Expr::conjunction(residual))
    };
    (keys, filter)
}

/// Equality keys for a join, plus the right-side columns a `USING` or
/// `NATURAL` clause merges away.
type JoinKeys = (Vec<(Expr, Expr)>, Vec<ColumnId>);

/// Build the equality keys for `USING` and `NATURAL`, and report which
/// right-side columns become invisible as bare names.
fn equate_named_columns(
    names: &[String],
    left: &PlanSchema,
    right: &PlanSchema,
    clause: &str,
) -> Result<JoinKeys> {
    let mut keys = Vec::with_capacity(names.len());
    let mut hidden = Vec::with_capacity(names.len());
    for name in names {
        let left_field = left.resolve(None, name)?.ok_or_else(|| {
            BasaltError::schema(format!(
                "{clause} column {name} does not exist on the left side of the join"
            ))
        })?;
        let right_field = right.resolve(None, name)?.ok_or_else(|| {
            BasaltError::schema(format!(
                "{clause} column {name} does not exist on the right side of the join"
            ))
        })?;
        if left_field.data_type != right_field.data_type {
            return Err(BasaltError::type_error(format!(
                "{clause} column {name} is {} on the left and {} on the right",
                left_field.data_type, right_field.data_type
            )));
        }
        keys.push((column_expr(left_field), column_expr(right_field)));
        hidden.push(right_field.id);
    }
    Ok((keys, hidden))
}

fn common_column_names(left: &PlanSchema, right: &PlanSchema) -> Vec<String> {
    let mut names = Vec::new();
    for field in left.fields() {
        if right
            .fields()
            .iter()
            .any(|other| other.name.eq_ignore_ascii_case(&field.name))
            && !names
                .iter()
                .any(|n: &String| n.eq_ignore_ascii_case(&field.name))
        {
            names.push(field.name.clone());
        }
    }
    names
}

/// Combine the visible schemas of a join's inputs.
fn merge_visible(
    left: PlanSchema,
    right: PlanSchema,
    join_type: JoinType,
    hidden: &[ColumnId],
) -> PlanSchema {
    let left = if join_type.left_nullable() {
        left.into_nullable()
    } else {
        left
    };
    if !join_type.projects_right() {
        return left;
    }
    let right = if join_type.right_nullable() {
        right.into_nullable()
    } else {
        right
    };
    let kept: Vec<PlanField> = right
        .fields()
        .iter()
        .filter(|field| !hidden.contains(&field.id))
        .cloned()
        .collect();
    left.concat(PlanSchema::new(kept))
}

// ---------------------------------------------------------------------------
// Aggregate and window collection
// ---------------------------------------------------------------------------

/// Where an aggregate or window function found inside an expression goes.
///
/// Aggregates cannot be evaluated in place: `sum(x) + 1` is an addition over
/// the *output* of an aggregate node that sits below the projection. Binding
/// therefore hoists every aggregate into this sink and leaves a column
/// reference behind, so the expression that comes back is already written
/// against the aggregate's output.
///
/// Clauses where an aggregate is illegal (`WHERE`, `GROUP BY`, `JOIN ON`) pass
/// a forbidding sink, which turns the mistake into an error naming the clause
/// instead of a confusing plan.
enum AggregateSink {
    Forbidden(String),
    Collecting {
        aggregates: Vec<AggregateExpr>,
        windows: Vec<WindowExpr>,
    },
}

impl AggregateSink {
    fn forbidden(clause: &str) -> AggregateSink {
        AggregateSink::Forbidden(clause.to_string())
    }

    fn collecting() -> AggregateSink {
        AggregateSink::Collecting {
            aggregates: Vec::new(),
            windows: Vec::new(),
        }
    }

    fn push_aggregate(&mut self, aggregate: AggregateExpr) -> Result<Expr> {
        match self {
            AggregateSink::Forbidden(clause) => Err(BasaltError::plan(format!(
                "aggregate function {} is not allowed in {clause}",
                aggregate.func.name()
            ))),
            AggregateSink::Collecting { aggregates, .. } => {
                // An aggregate written twice is computed once. This is not only
                // an optimisation: `sum(x) / count(x)` appearing in both the
                // select list and HAVING must produce identical values, and
                // sharing the column makes that true by construction.
                if let Some(existing) = aggregates.iter().find(|a| {
                    a.func == aggregate.func
                        && a.distinct == aggregate.distinct
                        && a.arg == aggregate.arg
                }) {
                    return Ok(column_expr(&existing.output));
                }
                let reference = column_expr(&aggregate.output);
                aggregates.push(aggregate);
                Ok(reference)
            }
        }
    }

    fn push_window(&mut self, window: WindowExpr) -> Result<Expr> {
        match self {
            AggregateSink::Forbidden(clause) => Err(BasaltError::plan(format!(
                "window function {} is not allowed in {clause}",
                window.call.func.name()
            ))),
            AggregateSink::Collecting { windows, .. } => {
                let reference = column_expr(&window.output);
                windows.push(window);
                Ok(reference)
            }
        }
    }

    fn take_aggregates(&mut self) -> Vec<AggregateExpr> {
        match self {
            AggregateSink::Forbidden(_) => Vec::new(),
            AggregateSink::Collecting { aggregates, .. } => std::mem::take(aggregates),
        }
    }

    fn take_windows(&mut self) -> Vec<WindowExpr> {
        match self {
            AggregateSink::Forbidden(_) => Vec::new(),
            AggregateSink::Collecting { windows, .. } => std::mem::take(windows),
        }
    }
}

// ---------------------------------------------------------------------------
// The select list and ORDER BY
// ---------------------------------------------------------------------------

impl Binder<'_> {
    fn bind_projection(
        &mut self,
        select: &ast::Select,
        scope: &Scope<'_>,
        sink: &mut AggregateSink,
    ) -> Result<Vec<ProjectionItem>> {
        let mut items = Vec::with_capacity(select.projection.len());
        for item in &select.projection {
            match item {
                ast::SelectItem::Wildcard(_) => {
                    if scope.schema.is_empty() {
                        return Err(BasaltError::plan(
                            "SELECT * requires a FROM clause".to_string(),
                        ));
                    }
                    for field in scope.schema.fields() {
                        items.push(ProjectionItem {
                            expr: column_expr(field),
                            name: field.name.clone(),
                            qualifier: field.qualifier.clone(),
                        });
                    }
                }
                ast::SelectItem::QualifiedWildcard(name) => {
                    let relation = name.base();
                    let mut matched = false;
                    for field in scope.schema.fields() {
                        if field
                            .qualifier
                            .as_deref()
                            .is_some_and(|q| q.eq_ignore_ascii_case(relation))
                        {
                            matched = true;
                            items.push(ProjectionItem {
                                expr: column_expr(field),
                                name: field.name.clone(),
                                qualifier: field.qualifier.clone(),
                            });
                        }
                    }
                    if !matched {
                        return Err(BasaltError::schema(format!(
                            "{relation}.* does not match any relation in this query"
                        )));
                    }
                }
                ast::SelectItem::Expr { expr, alias } => {
                    let bound = self.bind_expr(expr, scope, sink)?;
                    let name = match alias {
                        Some(ident) => ident.value.clone(),
                        None => bound.output_name(),
                    };
                    items.push(ProjectionItem {
                        expr: bound,
                        name,
                        qualifier: None,
                    });
                }
            }
        }
        if items.is_empty() {
            return Err(BasaltError::plan(
                "SELECT has no output columns".to_string(),
            ));
        }
        Ok(items)
    }

    /// Bind one `ORDER BY` key.
    ///
    /// `ORDER BY` is the one clause that sees the query's *output*, so a bare
    /// integer is an output column position and a bare name prefers an output
    /// alias over an input column. Both are in the standard and both surprise
    /// people, which is why they are handled explicitly here rather than
    /// falling out of the generic expression path.
    fn bind_order_key(
        &mut self,
        item: &ast::OrderByExpr,
        scope: &Scope<'_>,
        aliases: &ProjectionAliases,
        output: &PlanSchema,
    ) -> Result<Expr> {
        if let ast::ExprKind::Literal(ast::Literal::Number(text)) = &item.expr.kind {
            if let Ok(position) = text.parse::<usize>() {
                if position == 0 || position > output.len() {
                    return Err(BasaltError::plan(format!(
                        "ORDER BY position {position} is out of range; the query returns {} columns",
                        output.len()
                    )));
                }
                let field = output.field(position - 1).ok_or_else(|| {
                    BasaltError::internal("binder::bind_order_key", "position checked but absent")
                })?;
                return Ok(column_expr(field));
            }
        }
        if let ast::ExprKind::Column(name) = &item.expr.kind {
            if name.qualifier().is_empty() {
                if let Some((_, field)) = aliases
                    .iter()
                    .find(|(alias, _)| alias.eq_ignore_ascii_case(name.base()))
                {
                    return Ok(column_expr(field));
                }
            }
        }
        self.bind_expr(&item.expr, scope, &mut AggregateSink::forbidden("ORDER BY"))
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl Binder<'_> {
    // One arm per syntax node. Splitting it up would hide the fact that this is
    // an exhaustive dispatch, which is the property that makes it easy to check
    // that no expression form is silently unhandled.
    #[allow(clippy::too_many_lines)]
    fn bind_expr(
        &mut self,
        expr: &ast::Expr,
        scope: &Scope<'_>,
        sink: &mut AggregateSink,
    ) -> Result<Expr> {
        match &expr.kind {
            ast::ExprKind::Literal(literal) => Ok(Expr::Literal(literal.to_value()?)),

            ast::ExprKind::Parameter(_) => Err(BasaltError::unsupported(
                "query parameters are parsed but not yet bound; inline the value".to_string(),
            )),

            ast::ExprKind::Column(name) => self.bind_column(name, scope),

            ast::ExprKind::UnaryOp { op, expr: operand } => {
                let operand = self.bind_expr(operand, scope, sink)?;
                match op {
                    ast::UnaryOperator::Not => {
                        Ok(Expr::Not(Box::new(coerce_to_boolean(operand, "NOT")?)))
                    }
                    ast::UnaryOperator::Plus => Ok(operand),
                    ast::UnaryOperator::Minus => {
                        if !operand.data_type().is_numeric()
                            && operand.data_type() != DataType::Null
                        {
                            return Err(BasaltError::type_error(format!(
                                "unary minus is not defined for {}",
                                operand.data_type()
                            )));
                        }
                        Ok(Expr::Negate(Box::new(operand)))
                    }
                }
            }

            ast::ExprKind::BinaryOp { left, op, right } => {
                let left = self.bind_expr(left, scope, sink)?;
                let right = self.bind_expr(right, scope, sink)?;
                self.bind_binary(left, *op, right)
            }

            ast::ExprKind::IsNull { expr, negated } => Ok(Expr::IsNull {
                expr: Box::new(self.bind_expr(expr, scope, sink)?),
                negated: *negated,
            }),

            ast::ExprKind::IsDistinctFrom {
                left,
                right,
                negated,
            } => {
                let left = self.bind_expr(left, scope, sink)?;
                let right = self.bind_expr(right, scope, sink)?;
                let (left, right) = align_types(left, right, "IS DISTINCT FROM")?;
                Ok(Expr::IsDistinctFrom {
                    left: Box::new(left),
                    right: Box::new(right),
                    negated: *negated,
                })
            }

            ast::ExprKind::Between {
                expr,
                low,
                high,
                negated,
            } => {
                // BETWEEN is desugared here rather than carried into the plan,
                // so the optimizer and executor have one fewer node to handle
                // and predicate pushdown sees two ordinary range predicates.
                let value = self.bind_expr(expr, scope, sink)?;
                let low = self.bind_expr(low, scope, sink)?;
                let high = self.bind_expr(high, scope, sink)?;
                let (value_low, low) = align_types(value.clone(), low, "BETWEEN")?;
                let (value_high, high) = align_types(value, high, "BETWEEN")?;
                let lower = Expr::binary(value_low, BinaryOp::GtEq, low);
                let upper = Expr::binary(value_high, BinaryOp::LtEq, high);
                let range = lower.and(upper);
                Ok(if *negated {
                    Expr::Not(Box::new(range))
                } else {
                    range
                })
            }

            ast::ExprKind::InList {
                expr,
                list,
                negated,
            } => {
                let probe = self.bind_expr(expr, scope, sink)?;
                let mut bound = Vec::with_capacity(list.len());
                for item in list {
                    bound.push(self.bind_expr(item, scope, sink)?);
                }
                self.bind_in_list(probe, bound, *negated)
            }

            ast::ExprKind::Like {
                expr,
                pattern,
                escape,
                negated,
                case_insensitive,
            } => {
                let subject = self.bind_expr(expr, scope, sink)?;
                let pattern = self.bind_expr(pattern, scope, sink)?;
                let escape = match escape {
                    Some(expr) => {
                        let bound = self.bind_expr(expr, scope, sink)?;
                        match &bound {
                            Expr::Literal(Value::Utf8(text)) if text.chars().count() == 1 => {
                                text.chars().next()
                            }
                            _ => {
                                return Err(BasaltError::plan(
                                    "LIKE ESCAPE requires a single-character string literal"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    None => None,
                };
                Ok(Expr::Like {
                    expr: Box::new(cast_to(subject, &DataType::Utf8, "LIKE")?),
                    pattern: Box::new(cast_to(pattern, &DataType::Utf8, "LIKE pattern")?),
                    escape,
                    negated: *negated,
                    case_insensitive: *case_insensitive,
                })
            }

            ast::ExprKind::Case {
                operand,
                arms,
                else_result,
            } => self.bind_case(
                operand.as_deref(),
                arms,
                else_result.as_deref(),
                scope,
                sink,
            ),

            ast::ExprKind::Cast { expr, data_type } => {
                let operand = self.bind_expr(expr, scope, sink)?;
                if operand.data_type() == *data_type {
                    return Ok(operand);
                }
                Ok(Expr::Cast {
                    expr: Box::new(operand),
                    data_type: data_type.clone(),
                })
            }

            ast::ExprKind::Extract { field, expr } => {
                let operand = self.bind_expr(expr, scope, sink)?;
                match operand.data_type() {
                    DataType::Date32 | DataType::Timestamp(_) | DataType::Null => {}
                    other => {
                        return Err(BasaltError::type_error(format!(
                            "EXTRACT requires a DATE or TIMESTAMP, found {other}"
                        )))
                    }
                }
                Ok(Expr::Extract {
                    field: *field,
                    expr: Box::new(operand),
                })
            }

            ast::ExprKind::Function(function) => self.bind_function(function, scope, sink),

            ast::ExprKind::Exists { subquery, negated } => {
                // No sink check: EXISTS is a predicate, legal wherever a
                // boolean expression is. An earlier revision routed it through
                // the aggregate gate, which made `WHERE EXISTS (...)` -- the
                // single most common use of it -- a bind error.
                let plan = self.bind_query(subquery, Some(scope))?;
                Ok(Expr::Exists {
                    subquery: Arc::new(plan),
                    negated: *negated,
                })
            }

            ast::ExprKind::InSubquery {
                expr,
                subquery,
                negated,
            } => {
                let probe = self.bind_expr(expr, scope, sink)?;
                let plan = self.bind_query(subquery, Some(scope))?;
                if plan.schema().len() != 1 {
                    return Err(BasaltError::plan(format!(
                        "IN (subquery) requires exactly one column, found {}",
                        plan.schema().len()
                    )));
                }
                Ok(Expr::InSubquery {
                    expr: Box::new(probe),
                    subquery: Arc::new(plan),
                    negated: *negated,
                })
            }

            ast::ExprKind::Subquery(_) => Err(BasaltError::unsupported(
                "scalar subqueries are parsed but not yet planned; rewrite as a join".to_string(),
            )),
        }
    }

    #[allow(clippy::unused_self)] // A binder method for call-site symmetry; see the module docs.
    fn bind_column(&mut self, name: &ast::ObjectName, scope: &Scope<'_>) -> Result<Expr> {
        let qualifier = match name.qualifier() {
            [] => None,
            [single] => Some(single.value.as_str()),
            parts => {
                return Err(BasaltError::unsupported(format!(
                    "qualified name {}.{} has {} qualifiers; only relation.column is supported",
                    parts
                        .iter()
                        .map(|p| p.value.as_str())
                        .collect::<Vec<_>>()
                        .join("."),
                    name.base(),
                    parts.len()
                )))
            }
        };
        match scope.resolve(qualifier, name.base())? {
            Some((field, 0)) => Ok(column_expr(field)),
            Some((field, _)) => Ok(Expr::OuterColumn(ColumnRef {
                id: field.id,
                qualifier: field.qualifier.clone(),
                name: field.name.clone(),
                data_type: field.data_type.clone(),
                nullable: field.nullable,
            })),
            None => Err(BasaltError::schema(format!(
                "no column named {}{}",
                match qualifier {
                    Some(q) => format!("{q}."),
                    None => String::new(),
                },
                name.base()
            ))),
        }
    }

    #[allow(clippy::unused_self)] // A binder method for call-site symmetry; see the module docs.
    fn bind_binary(&mut self, left: Expr, op: ast::BinaryOperator, right: Expr) -> Result<Expr> {
        use ast::BinaryOperator as Ast;
        let op = match op {
            Ast::Or => BinaryOp::Or,
            Ast::And => BinaryOp::And,
            Ast::Eq => BinaryOp::Eq,
            Ast::NotEq => BinaryOp::NotEq,
            Ast::Lt => BinaryOp::Lt,
            Ast::LtEq => BinaryOp::LtEq,
            Ast::Gt => BinaryOp::Gt,
            Ast::GtEq => BinaryOp::GtEq,
            Ast::Concat => BinaryOp::Concat,
            Ast::Plus => BinaryOp::Plus,
            Ast::Minus => BinaryOp::Minus,
            Ast::Multiply => BinaryOp::Multiply,
            Ast::Divide => BinaryOp::Divide,
            Ast::Modulo => BinaryOp::Modulo,
        };

        match op {
            BinaryOp::And | BinaryOp::Or => {
                let left = coerce_to_boolean(left, op_name(op))?;
                let right = coerce_to_boolean(right, op_name(op))?;
                Ok(Expr::binary(left, op, right))
            }
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq => {
                let (left, right) = align_types(left, right, op_name(op))?;
                Ok(Expr::binary(left, op, right))
            }
            BinaryOp::Concat => Ok(Expr::binary(
                cast_to(left, &DataType::Utf8, "||")?,
                op,
                cast_to(right, &DataType::Utf8, "||")?,
            )),
            _ => {
                // The result type is decided first, then both operands are cast
                // to it, so the executor's arithmetic kernels only ever see
                // matching types.
                let result = arithmetic_type(op, &left.data_type(), &right.data_type())?;
                let operand_type = if op == BinaryOp::Divide {
                    DataType::Float64
                } else {
                    result.clone()
                };
                let left = cast_to(left, &operand_type, op_name(op))?;
                let right = cast_to(right, &operand_type, op_name(op))?;
                Ok(Expr::binary(left, op, right))
            }
        }
    }

    #[allow(clippy::unused_self)] // A binder method for call-site symmetry; see the module docs.
    fn bind_in_list(&mut self, probe: Expr, list: Vec<Expr>, negated: bool) -> Result<Expr> {
        // A list of constants becomes a hash set at bind time. The threshold is
        // low because the set also removes duplicates and gives the executor a
        // single code path; below it, a chain of comparisons is cheaper than
        // hashing the probe.
        const HASH_SET_THRESHOLD: usize = 8;

        if list.is_empty() {
            // `x IN ()` is empty-set membership: always false, and always false
            // even for NULL, because no comparison is ever performed.
            return Ok(Expr::bool(negated));
        }
        let mut common = probe.data_type();
        for item in &list {
            common = coerce(&common, &item.data_type()).ok_or_else(|| {
                BasaltError::type_error(format!(
                    "IN list mixes {} with {}",
                    common,
                    item.data_type()
                ))
            })?;
        }
        let probe = cast_to(probe, &common, "IN")?;

        let all_constant = list.iter().all(Expr::is_constant);
        if all_constant && list.len() >= HASH_SET_THRESHOLD {
            let mut values = Vec::with_capacity(list.len());
            let mut has_null = false;
            for item in &list {
                let Expr::Literal(value) = item else {
                    // A constant that is not yet a literal (a cast of one, say)
                    // is left to the optimizer's folding pass.
                    return Ok(Expr::InList {
                        expr: Box::new(probe),
                        list: cast_all(list, &common, "IN")?,
                        negated,
                    });
                };
                if value.is_null() {
                    has_null = true;
                    continue;
                }
                let value = value.cast_to(&common)?;
                if !values.contains(&value) {
                    values.push(value);
                }
            }
            values.sort();
            return Ok(Expr::InSet {
                expr: Box::new(probe),
                set: Arc::new(InValueSet {
                    values,
                    has_null,
                    data_type: common,
                }),
                negated,
            });
        }

        Ok(Expr::InList {
            expr: Box::new(probe),
            list: cast_all(list, &common, "IN")?,
            negated,
        })
    }

    fn bind_case(
        &mut self,
        operand: Option<&ast::Expr>,
        arms: &[ast::CaseArm],
        else_result: Option<&ast::Expr>,
        scope: &Scope<'_>,
        sink: &mut AggregateSink,
    ) -> Result<Expr> {
        // The simple form is rewritten into the searched form so that the plan
        // has one CASE shape and the executor one implementation.
        let operand = match operand {
            Some(expr) => Some(self.bind_expr(expr, scope, sink)?),
            None => None,
        };

        let mut bound_arms = Vec::with_capacity(arms.len());
        let mut result_type: Option<DataType> = None;
        for arm in arms {
            let condition = self.bind_expr(&arm.condition, scope, sink)?;
            let condition = match &operand {
                Some(operand) => {
                    let (left, right) = align_types(operand.clone(), condition, "CASE")?;
                    Expr::binary(left, BinaryOp::Eq, right)
                }
                None => coerce_to_boolean(condition, "CASE WHEN")?,
            };
            let result = self.bind_expr(&arm.result, scope, sink)?;
            result_type = Some(match result_type {
                None => result.data_type(),
                Some(current) => coerce(&current, &result.data_type()).ok_or_else(|| {
                    BasaltError::type_error(format!(
                        "CASE branches produce incompatible types {current} and {}",
                        result.data_type()
                    ))
                })?,
            });
            bound_arms.push((condition, result));
        }

        let else_result = match else_result {
            Some(expr) => {
                let bound = self.bind_expr(expr, scope, sink)?;
                result_type = Some(match result_type {
                    None => bound.data_type(),
                    Some(current) => coerce(&current, &bound.data_type()).ok_or_else(|| {
                        BasaltError::type_error(format!(
                            "CASE ELSE produces {} which is incompatible with {current}",
                            bound.data_type()
                        ))
                    })?,
                });
                Some(bound)
            }
            None => None,
        };

        let result_type = result_type.unwrap_or(DataType::Null);
        let arms = bound_arms
            .into_iter()
            .map(|(condition, result)| Ok((condition, cast_to(result, &result_type, "CASE")?)))
            .collect::<Result<Vec<_>>>()?;
        let else_result = match else_result {
            Some(expr) => Some(Box::new(cast_to(expr, &result_type, "CASE ELSE")?)),
            None => None,
        };
        Ok(Expr::Case { arms, else_result })
    }

    fn bind_function(
        &mut self,
        function: &ast::Function,
        scope: &Scope<'_>,
        sink: &mut AggregateSink,
    ) -> Result<Expr> {
        let name = function.name.base();
        let Some(resolved) = FunctionRegistry.resolve(name) else {
            let known = FunctionRegistry.names();
            let suggestion = closest_name(name, &known);
            return Err(BasaltError::plan(match suggestion {
                Some(candidate) => format!("unknown function {name}; did you mean {candidate}?"),
                None => format!("unknown function {name}"),
            }));
        };

        // A wildcard argument is only ever `count(*)`.
        let wildcard = function
            .args
            .iter()
            .any(|arg| matches!(arg, ast::FunctionArg::Wildcard(_)));
        if wildcard {
            if function.args.len() != 1 {
                return Err(BasaltError::plan(format!(
                    "{name}(*) does not take other arguments"
                )));
            }
            if resolved != ResolvedFunction::Aggregate(AggregateFunc::Count) {
                return Err(BasaltError::plan(format!(
                    "{name}(*) is not valid; only count(*) may take a wildcard"
                )));
            }
        }

        let mut args = Vec::with_capacity(function.args.len());
        for arg in &function.args {
            if let ast::FunctionArg::Expr(expr) = arg {
                args.push(self.bind_expr(expr, scope, sink)?);
            }
        }

        if let Some(spec) = &function.over {
            return self.bind_window_function(name, resolved, args, spec, scope, sink);
        }

        match resolved {
            ResolvedFunction::Scalar(scalar) => {
                if function.distinct {
                    return Err(BasaltError::plan(format!(
                        "DISTINCT is only meaningful inside an aggregate, not in {name}"
                    )));
                }
                self.bind_scalar(scalar, args, name)
            }
            ResolvedFunction::Aggregate(aggregate) => {
                self.bind_aggregate(aggregate, args, function.distinct, wildcard, name, sink)
            }
            ResolvedFunction::Window(window) => Err(BasaltError::plan(format!(
                "{} requires an OVER clause",
                window.name()
            ))),
        }
    }

    #[allow(clippy::unused_self)] // A binder method for call-site symmetry; see the module docs.
    fn bind_scalar(&mut self, func: ScalarFunc, args: Vec<Expr>, name: &str) -> Result<Expr> {
        let arg_types: Vec<DataType> = args.iter().map(Expr::data_type).collect();
        let targets = func.coerce_args(&arg_types)?;
        if targets.len() != args.len() {
            return Err(BasaltError::internal(
                "binder::bind_scalar",
                format!(
                    "{name} coerced {} arguments into {} targets",
                    args.len(),
                    targets.len()
                ),
            ));
        }
        let return_type = func.return_type(&targets)?;
        let args = args
            .into_iter()
            .zip(&targets)
            .map(|(arg, target)| cast_to(arg, target, name))
            .collect::<Result<Vec<_>>>()?;
        let _ = return_type;
        Ok(Expr::ScalarFunction { func, args })
    }

    fn bind_aggregate(
        &mut self,
        func: AggregateFunc,
        mut args: Vec<Expr>,
        distinct: bool,
        wildcard: bool,
        name: &str,
        sink: &mut AggregateSink,
    ) -> Result<Expr> {
        if wildcard {
            let output = PlanField::new(self.ids.next(), "count(*)", DataType::Int64, false);
            return sink.push_aggregate(AggregateExpr {
                func: AggregateFunc::Count,
                arg: None,
                distinct: false,
                output,
            });
        }
        if args.len() != 1 {
            return Err(BasaltError::plan(format!(
                "{name} takes exactly one argument, found {}",
                args.len()
            )));
        }
        let arg = args.remove(0);
        if contains_aggregate(&arg) {
            return Err(BasaltError::plan(format!(
                "{name} contains another aggregate; aggregates cannot be nested"
            )));
        }
        let target = func.coerce_arg(&arg.data_type())?;
        let arg = cast_to(arg, &target, name)?;
        let return_type = func.return_type(&target)?;
        // Every aggregate except count can produce NULL: a group whose values
        // are all NULL, or -- for a global aggregate -- an empty input.
        let nullable = func != AggregateFunc::Count;
        let display = if distinct {
            format!("{name}(DISTINCT {arg})")
        } else {
            format!("{name}({arg})")
        };
        let output = PlanField::new(self.ids.next(), display, return_type, nullable);
        sink.push_aggregate(AggregateExpr {
            func,
            arg: Some(arg),
            distinct,
            output,
        })
    }

    fn bind_window_function(
        &mut self,
        name: &str,
        resolved: ResolvedFunction,
        mut args: Vec<Expr>,
        spec: &ast::WindowSpec,
        scope: &Scope<'_>,
        sink: &mut AggregateSink,
    ) -> Result<Expr> {
        let func = match resolved {
            ResolvedFunction::Window(window) => window,
            ResolvedFunction::Aggregate(aggregate) => WindowFunc::Aggregate(aggregate),
            ResolvedFunction::Scalar(_) => {
                return Err(BasaltError::plan(format!(
                    "{name} is a scalar function and cannot take an OVER clause"
                )))
            }
        };

        let arg = match func {
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank => {
                if !args.is_empty() {
                    return Err(BasaltError::plan(format!("{name} takes no arguments")));
                }
                None
            }
            WindowFunc::Aggregate(aggregate) => {
                if args.len() != 1 {
                    return Err(BasaltError::plan(format!(
                        "{name} takes exactly one argument, found {}",
                        args.len()
                    )));
                }
                let arg = args.remove(0);
                let target = aggregate.coerce_arg(&arg.data_type())?;
                Some(cast_to(arg, &target, name)?)
            }
        };

        let return_type = func.return_type(arg.as_ref().map(Expr::data_type).as_ref())?;

        let mut inner = AggregateSink::forbidden("a window function's OVER clause");
        let mut partition_by = Vec::with_capacity(spec.partition_by.len());
        for expr in &spec.partition_by {
            partition_by.push(self.bind_expr(expr, scope, &mut inner)?);
        }
        let mut order_by = Vec::with_capacity(spec.order_by.len());
        for item in &spec.order_by {
            let expr = self.bind_expr(&item.expr, scope, &mut inner)?;
            let asc = item.asc.unwrap_or(true);
            order_by.push(SortExpr::new(expr, asc, item.nulls_first.unwrap_or(!asc)));
        }

        let frame = match &spec.frame {
            Some(frame) => bind_frame(frame)?,
            None if order_by.is_empty() => WindowFrameSpec::whole_partition(),
            None => WindowFrameSpec::default_ordered(),
        };
        frame.validate()?;

        let display = format!(
            "{name}({}) OVER (...)",
            arg.as_ref().map(ToString::to_string).unwrap_or_default()
        );
        let output = PlanField::new(self.ids.next(), display, return_type, true);
        sink.push_window(WindowExpr {
            call: WindowCall { func, arg },
            partition_by,
            order_by,
            frame,
            output,
        })
    }
}

fn bind_frame(frame: &ast::WindowFrame) -> Result<WindowFrameSpec> {
    let units = match frame.units {
        ast::WindowFrameUnits::Rows => WindowFrameUnits::Rows,
        ast::WindowFrameUnits::Range => WindowFrameUnits::Range,
    };
    let start = bind_frame_bound(&frame.start)?;
    let end = match &frame.end {
        Some(bound) => bind_frame_bound(bound)?,
        None => WindowFrameBound::CurrentRow,
    };
    Ok(WindowFrameSpec { units, start, end })
}

fn bind_frame_bound(bound: &ast::WindowFrameBound) -> Result<WindowFrameBound> {
    Ok(match bound {
        ast::WindowFrameBound::UnboundedPreceding => WindowFrameBound::UnboundedPreceding,
        ast::WindowFrameBound::CurrentRow => WindowFrameBound::CurrentRow,
        ast::WindowFrameBound::UnboundedFollowing => WindowFrameBound::UnboundedFollowing,
        ast::WindowFrameBound::Preceding(expr) => {
            WindowFrameBound::Preceding(frame_offset(expr, "PRECEDING")?)
        }
        ast::WindowFrameBound::Following(expr) => {
            WindowFrameBound::Following(frame_offset(expr, "FOLLOWING")?)
        }
    })
}

fn frame_offset(expr: &ast::Expr, keyword: &str) -> Result<u64> {
    if let ast::ExprKind::Literal(ast::Literal::Number(text)) = &expr.kind {
        if let Ok(value) = text.parse::<u64>() {
            return Ok(value);
        }
    }
    Err(BasaltError::plan(format!(
        "{keyword} requires a non-negative integer literal"
    )))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
        BinaryOp::NotEq => "<>",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Plus => "+",
        BinaryOp::Minus => "-",
        BinaryOp::Multiply => "*",
        BinaryOp::Divide => "/",
        BinaryOp::Modulo => "%",
        BinaryOp::Concat => "||",
    }
}

/// Insert a cast unless the expression already has the target type.
fn cast_to(expr: Expr, target: &DataType, context: &str) -> Result<Expr> {
    let current = expr.data_type();
    if current == *target {
        return Ok(expr);
    }
    if current != DataType::Null && coerce(&current, target).as_ref() != Some(target) {
        return Err(BasaltError::type_error(format!(
            "{context} cannot convert {current} to {target}"
        )));
    }
    Ok(Expr::Cast {
        expr: Box::new(expr),
        data_type: target.clone(),
    })
}

fn cast_all(exprs: Vec<Expr>, target: &DataType, context: &str) -> Result<Vec<Expr>> {
    exprs
        .into_iter()
        .map(|expr| cast_to(expr, target, context))
        .collect()
}

/// Bring two operands to a common type, inserting casts on either side.
fn align_types(left: Expr, right: Expr, context: &str) -> Result<(Expr, Expr)> {
    let (lt, rt) = (left.data_type(), right.data_type());
    if lt == rt {
        return Ok((left, right));
    }
    let Some(common) = coerce(&lt, &rt) else {
        return Err(BasaltError::type_error(format!(
            "{context} is not defined between {lt} and {rt}"
        )));
    };
    Ok((
        cast_to(left, &common, context)?,
        cast_to(right, &common, context)?,
    ))
}

fn contains_aggregate(expr: &Expr) -> bool {
    // Aggregates are hoisted into the sink as they are bound, so by the time an
    // outer aggregate looks at its argument any nested aggregate has already
    // become a column reference to the sink's output. Detecting that is what
    // this checks: a bound argument that references a column produced by the
    // same aggregation level.
    let mut found = false;
    expr.visit(&mut |node| {
        if matches!(node, Expr::Column(reference) if reference.name.contains('(')) {
            found = true;
            return false;
        }
        true
    });
    found
}

/// A cheap edit-distance suggestion for an unknown function name.
///
/// Bounded at distance two: beyond that the "did you mean" is noise, and a
/// wrong suggestion is worse than none.
fn closest_name<'a>(name: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let lower = name.to_ascii_lowercase();
    let mut best: Option<(usize, &str)> = None;
    for candidate in candidates {
        let distance = edit_distance(&lower, candidate);
        if distance <= 2 && best.is_none_or(|(current, _)| distance < current) {
            best = Some((distance, candidate));
        }
    }
    best.map(|(_, candidate)| candidate)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

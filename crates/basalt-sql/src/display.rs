//! Printing the AST back as SQL.
//!
//! The contract is exact: for every `sql` the parser accepts,
//!
//! ```text
//! parse(print(parse(sql))) == parse(sql)
//! ```
//!
//! and the property test in `tests/roundtrip.rs` checks it against generated
//! input. That contract is what makes `EXPLAIN`, view definitions and error
//! messages that quote a subexpression trustworthy.
//!
//! # How parentheses are decided
//!
//! Parentheses are not stored in the tree, so the printer re-derives them from
//! the precedence table. The rule is one line:
//!
//! * print a child inside parentheses when its precedence is **lower** than the
//!   minimum the position requires;
//! * a left operand requires the operator's own precedence, a right operand
//!   requires one more.
//!
//! Requiring one more on the right is what preserves associativity: `a - (b - c)`
//! keeps its parentheses because the right child of `-` needs precedence 7 and
//! `-` only has 6, while `(a - b) - c` prints as `a - b - c` because the left
//! child needs only 6. The rule can add a redundant pair (`a + (b + c)` keeps
//! them even though `+` is associative on integers) — that is deliberate,
//! because `+` is *not* associative on floats, and the printer must not silently
//! reassociate.

use std::fmt;
use std::fmt::Write as _;

use basalt_types::DataType;

use crate::ast::{
    CaseArm, ColumnDef, ColumnOption, CreateTable, CreateView, Cte, DropTable, DropView, Explain,
    Expr, ExprKind, Function, FunctionArg, Ident, Insert, Join, JoinConstraint, Literal,
    ObjectName, OrderByExpr, Precedence, Query, Select, SelectItem, SetExpr, SetQuantifier,
    Statement, TableAlias, TableFactor, TableRef, UnaryOperator, Values, WindowFrame,
    WindowFrameBound, WindowFrameUnits, WindowSpec, With, PREC_COMPARISON,
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Write `items` separated by `sep`, using each item's `Display`.
fn write_list<T: fmt::Display>(f: &mut fmt::Formatter<'_>, items: &[T], sep: &str) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            f.write_str(sep)?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

/// Quote a string literal for SQL: wrap in `'` and double any embedded `'`.
pub(crate) fn quote_string(f: &mut fmt::Formatter<'_>, value: &str) -> fmt::Result {
    f.write_str("'")?;
    for ch in value.chars() {
        if ch == '\'' {
            f.write_str("''")?;
        } else {
            f.write_char(ch)?;
        }
    }
    f.write_str("'")
}

impl fmt::Display for Ident {
    /// Prints bare when it round-trips bare, quoted otherwise. Since the lexer
    /// folds bare identifiers to lower case, any name with an upper-case letter,
    /// a space, or a keyword spelling has to be quoted to come back unchanged.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.needs_quoting() {
            return f.write_str(&self.value);
        }
        f.write_str("\"")?;
        for ch in self.value.chars() {
            if ch == '"' {
                f.write_str("\"\"")?;
            } else {
                f.write_char(ch)?;
            }
        }
        f.write_str("\"")
    }
}

impl fmt::Display for ObjectName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_list(f, &self.0, ".")
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => f.write_str("NULL"),
            Literal::Boolean(true) => f.write_str("TRUE"),
            Literal::Boolean(false) => f.write_str("FALSE"),
            // Printed verbatim: a numeric literal's *spelling* determines its
            // type (see `Literal::to_value`), so reformatting it would change
            // the query's meaning.
            Literal::Number(text) => f.write_str(text),
            Literal::String(text) => quote_string(f, text),
        }
    }
}

/// Write a type the way [`Parser::parse_data_type`](crate::Parser::parse_data_type)
/// reads it back.
///
/// `DataType`'s own `Display` is already a valid Basalt type name for every
/// variant, so this is a straight delegation — kept as a named function so the
/// dependency is explicit and a future divergence has one place to be fixed.
fn write_data_type(f: &mut fmt::Formatter<'_>, data_type: &DataType) -> fmt::Result {
    write!(f, "{data_type}")
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_expr(f, self, 0)
    }
}

/// Print `expr`, parenthesising it if its precedence is below `min_precedence`.
fn write_expr(f: &mut fmt::Formatter<'_>, expr: &Expr, min_precedence: Precedence) -> fmt::Result {
    let precedence = expr.precedence();
    let parens = precedence < min_precedence;
    if parens {
        f.write_str("(")?;
    }
    write_expr_inner(f, expr)?;
    if parens {
        f.write_str(")")?;
    }
    Ok(())
}

/// One arm per expression form; splitting it would only scatter the
/// parenthesisation rules that have to be read together.
#[allow(clippy::too_many_lines)]
fn write_expr_inner(f: &mut fmt::Formatter<'_>, expr: &Expr) -> fmt::Result {
    match &expr.kind {
        ExprKind::Column(name) => write!(f, "{name}"),
        ExprKind::Literal(literal) => write!(f, "{literal}"),
        ExprKind::Parameter(param) => write!(f, "{param}"),
        ExprKind::UnaryOp { op, expr } => {
            // A word operator always needs a space. A sign operator needs one
            // only in front of another sign, because `- -x` printed as `--x`
            // would be a line comment and the round trip would silently lose
            // the rest of the statement.
            let sign_follows = matches!(
                &expr.kind,
                ExprKind::UnaryOp {
                    op: UnaryOperator::Minus | UnaryOperator::Plus,
                    ..
                }
            );
            let separator = if *op == UnaryOperator::Not || sign_follows {
                " "
            } else {
                ""
            };
            write!(f, "{op}{separator}")?;
            write_expr(f, expr, op.precedence())
        }
        ExprKind::BinaryOp { left, op, right } => {
            let precedence = op.precedence();
            write_expr(f, left, precedence)?;
            write!(f, " {op} ")?;
            write_expr(f, right, precedence + 1)
        }
        ExprKind::IsNull { expr, negated } => {
            write_expr(f, expr, PREC_COMPARISON + 1)?;
            f.write_str(if *negated { " IS NOT NULL" } else { " IS NULL" })
        }
        ExprKind::IsDistinctFrom {
            left,
            right,
            negated,
        } => {
            write_expr(f, left, PREC_COMPARISON + 1)?;
            f.write_str(if *negated {
                " IS NOT DISTINCT FROM "
            } else {
                " IS DISTINCT FROM "
            })?;
            write_expr(f, right, PREC_COMPARISON + 1)
        }
        ExprKind::InList {
            expr,
            list,
            negated,
        } => {
            write_expr(f, expr, PREC_COMPARISON + 1)?;
            f.write_str(if *negated { " NOT IN (" } else { " IN (" })?;
            write_list(f, list, ", ")?;
            f.write_str(")")
        }
        ExprKind::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            write_expr(f, expr, PREC_COMPARISON + 1)?;
            f.write_str(if *negated { " NOT IN (" } else { " IN (" })?;
            write!(f, "{subquery})")
        }
        ExprKind::Between {
            expr,
            low,
            high,
            negated,
        } => {
            write_expr(f, expr, PREC_COMPARISON + 1)?;
            f.write_str(if *negated {
                " NOT BETWEEN "
            } else {
                " BETWEEN "
            })?;
            // The bounds are parsed at comparison precedence, so anything
            // looser (`AND`, `OR`) has to be parenthesised or the `AND` of the
            // `BETWEEN` itself would be captured.
            write_expr(f, low, PREC_COMPARISON + 1)?;
            f.write_str(" AND ")?;
            write_expr(f, high, PREC_COMPARISON + 1)
        }
        ExprKind::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => {
            write_expr(f, expr, PREC_COMPARISON + 1)?;
            f.write_str(match (negated, case_insensitive) {
                (true, true) => " NOT ILIKE ",
                (true, false) => " NOT LIKE ",
                (false, true) => " ILIKE ",
                (false, false) => " LIKE ",
            })?;
            write_expr(f, pattern, PREC_COMPARISON + 1)?;
            if let Some(escape) = escape {
                f.write_str(" ESCAPE ")?;
                write_expr(f, escape, PREC_COMPARISON + 1)?;
            }
            Ok(())
        }
        ExprKind::Case {
            operand,
            arms,
            else_result,
        } => {
            f.write_str("CASE")?;
            if let Some(operand) = operand {
                write!(f, " {operand}")?;
            }
            for CaseArm { condition, result } in arms {
                write!(f, " WHEN {condition} THEN {result}")?;
            }
            if let Some(else_result) = else_result {
                write!(f, " ELSE {else_result}")?;
            }
            f.write_str(" END")
        }
        ExprKind::Cast { expr, data_type } => {
            write!(f, "CAST({expr} AS ")?;
            write_data_type(f, data_type)?;
            f.write_str(")")
        }
        ExprKind::Extract { field, expr } => write!(f, "EXTRACT({field} FROM {expr})"),
        ExprKind::Function(function) => write!(f, "{function}"),
        ExprKind::Exists { subquery, negated } => {
            if *negated {
                f.write_str("NOT ")?;
            }
            write!(f, "EXISTS ({subquery})")
        }
        ExprKind::Subquery(query) => write!(f, "({query})"),
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        if self.distinct {
            f.write_str("DISTINCT ")?;
        }
        write_list(f, &self.args, ", ")?;
        f.write_str(")")?;
        if let Some(over) = &self.over {
            write!(f, " OVER ({over})")?;
        }
        Ok(())
    }
}

impl fmt::Display for FunctionArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionArg::Expr(expr) => write!(f, "{expr}"),
            FunctionArg::Wildcard(_) => f.write_str("*"),
        }
    }
}

impl fmt::Display for WindowSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        if !self.partition_by.is_empty() {
            f.write_str("PARTITION BY ")?;
            write_list(f, &self.partition_by, ", ")?;
            first = false;
        }
        if !self.order_by.is_empty() {
            if !first {
                f.write_str(" ")?;
            }
            f.write_str("ORDER BY ")?;
            write_list(f, &self.order_by, ", ")?;
            first = false;
        }
        if let Some(frame) = &self.frame {
            if !first {
                f.write_str(" ")?;
            }
            write!(f, "{frame}")?;
        }
        Ok(())
    }
}

impl fmt::Display for WindowFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let units = match self.units {
            WindowFrameUnits::Rows => "ROWS",
            WindowFrameUnits::Range => "RANGE",
        };
        match &self.end {
            Some(end) => write!(f, "{units} BETWEEN {} AND {end}", self.start),
            None => write!(f, "{units} {}", self.start),
        }
    }
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

impl fmt::Display for OrderByExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)?;
        match self.asc {
            Some(true) => f.write_str(" ASC")?,
            Some(false) => f.write_str(" DESC")?,
            None => {}
        }
        match self.nulls_first {
            Some(true) => f.write_str(" NULLS FIRST")?,
            Some(false) => f.write_str(" NULLS LAST")?,
            None => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(with) = &self.with {
            write!(f, "{with} ")?;
        }
        write!(f, "{}", self.body)?;
        if !self.order_by.is_empty() {
            f.write_str(" ORDER BY ")?;
            write_list(f, &self.order_by, ", ")?;
        }
        if let Some(limit) = &self.limit {
            write!(f, " LIMIT {limit}")?;
        }
        if let Some(offset) = &self.offset {
            write!(f, " OFFSET {offset}")?;
        }
        Ok(())
    }
}

impl fmt::Display for With {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WITH ")?;
        if self.recursive {
            f.write_str("RECURSIVE ")?;
        }
        write_list(f, &self.ctes, ", ")
    }
}

impl fmt::Display for Cte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if !self.columns.is_empty() {
            f.write_str(" (")?;
            write_list(f, &self.columns, ", ")?;
            f.write_str(")")?;
        }
        write!(f, " AS ({})", self.query)
    }
}

impl fmt::Display for SetExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_set_expr(f, self, 0)
    }
}

fn write_set_expr(
    f: &mut fmt::Formatter<'_>,
    body: &SetExpr,
    min_precedence: Precedence,
) -> fmt::Result {
    match body {
        SetExpr::Select(select) => write!(f, "{select}"),
        SetExpr::Values(values) => write!(f, "{values}"),
        // A nested query brings its own ORDER BY / LIMIT, which only make sense
        // inside parentheses.
        SetExpr::Query(query) => write!(f, "({query})"),
        SetExpr::SetOperation {
            op,
            quantifier,
            left,
            right,
        } => {
            let precedence = op.precedence();
            let parens = precedence < min_precedence;
            if parens {
                f.write_str("(")?;
            }
            write_set_expr(f, left, precedence)?;
            write!(f, " {op}")?;
            match quantifier {
                SetQuantifier::All => f.write_str(" ALL")?,
                SetQuantifier::Distinct => f.write_str(" DISTINCT")?,
                SetQuantifier::None => {}
            }
            f.write_str(" ")?;
            write_set_expr(f, right, precedence + 1)?;
            if parens {
                f.write_str(")")?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("VALUES ")?;
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str("(")?;
            write_list(f, row, ", ")?;
            f.write_str(")")?;
        }
        Ok(())
    }
}

impl fmt::Display for Select {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SELECT ")?;
        if self.distinct {
            f.write_str("DISTINCT ")?;
        }
        write_list(f, &self.projection, ", ")?;
        if !self.from.is_empty() {
            f.write_str(" FROM ")?;
            write_list(f, &self.from, ", ")?;
        }
        if let Some(selection) = &self.selection {
            write!(f, " WHERE {selection}")?;
        }
        if !self.group_by.is_empty() {
            f.write_str(" GROUP BY ")?;
            write_list(f, &self.group_by, ", ")?;
        }
        if let Some(having) = &self.having {
            write!(f, " HAVING {having}")?;
        }
        Ok(())
    }
}

impl fmt::Display for SelectItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectItem::Wildcard(_) => f.write_str("*"),
            SelectItem::QualifiedWildcard(name) => write!(f, "{name}.*"),
            SelectItem::Expr { expr, alias } => {
                write!(f, "{expr}")?;
                if let Some(alias) = alias {
                    write!(f, " AS {alias}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for TableRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.relation)?;
        for join in &self.joins {
            write!(f, " {join}")?;
        }
        Ok(())
    }
}

impl fmt::Display for TableFactor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableFactor::Table { name, alias } => {
                write!(f, "{name}")?;
                write_alias(f, alias.as_ref())
            }
            TableFactor::Derived { subquery, alias } => {
                write!(f, "({subquery})")?;
                write_alias(f, alias.as_ref())
            }
            TableFactor::Nested(inner) => write!(f, "({inner})"),
        }
    }
}

fn write_alias(f: &mut fmt::Formatter<'_>, alias: Option<&TableAlias>) -> fmt::Result {
    let Some(alias) = alias else {
        return Ok(());
    };
    write!(f, " AS {}", alias.name)?;
    if !alias.columns.is_empty() {
        f.write_str(" (")?;
        write_list(f, &alias.columns, ", ")?;
        f.write_str(")")?;
    }
    Ok(())
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if matches!(self.constraint, JoinConstraint::Natural) {
            f.write_str("NATURAL ")?;
        }
        // `JOIN` alone means inner; the redundant `INNER` is dropped.
        write!(f, "{} {}", self.operator, self.relation)?;
        match &self.constraint {
            JoinConstraint::On(expr) => write!(f, " ON {expr}"),
            JoinConstraint::Using(columns) => {
                f.write_str(" USING (")?;
                write_list(f, columns, ", ")?;
                f.write_str(")")
            }
            JoinConstraint::Natural | JoinConstraint::None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Query(query) => write!(f, "{query}"),
            Statement::Insert(insert) => write!(f, "{insert}"),
            Statement::CreateTable(create) => write!(f, "{create}"),
            Statement::DropTable(drop) => write!(f, "{drop}"),
            Statement::CreateView(create) => write!(f, "{create}"),
            Statement::DropView(drop) => write!(f, "{drop}"),
            Statement::Explain(explain) => write!(f, "{explain}"),
        }
    }
}

impl fmt::Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "INSERT INTO {}", self.table)?;
        if !self.columns.is_empty() {
            f.write_str(" (")?;
            write_list(f, &self.columns, ", ")?;
            f.write_str(")")?;
        }
        write!(f, " {}", self.source)
    }
}

impl fmt::Display for CreateTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CREATE TABLE ")?;
        if self.if_not_exists {
            f.write_str("IF NOT EXISTS ")?;
        }
        write!(f, "{}", self.name)?;
        if let Some(query) = &self.query {
            return write!(f, " AS {query}");
        }
        f.write_str(" (")?;
        write_list(f, &self.columns, ", ")?;
        f.write_str(")")
    }
}

impl fmt::Display for ColumnDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ", self.name)?;
        write_data_type(f, &self.data_type)?;
        for option in &self.options {
            write!(f, " {option}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ColumnOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnOption::Null => f.write_str("NULL"),
            ColumnOption::NotNull => f.write_str("NOT NULL"),
            ColumnOption::PrimaryKey => f.write_str("PRIMARY KEY"),
            ColumnOption::Unique => f.write_str("UNIQUE"),
            ColumnOption::Default(expr) => write!(f, "DEFAULT {expr}"),
        }
    }
}

impl fmt::Display for DropTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DROP TABLE ")?;
        if self.if_exists {
            f.write_str("IF EXISTS ")?;
        }
        write_list(f, &self.names, ", ")
    }
}

impl fmt::Display for DropView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DROP VIEW ")?;
        if self.if_exists {
            f.write_str("IF EXISTS ")?;
        }
        write_list(f, &self.names, ", ")
    }
}

impl fmt::Display for CreateView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CREATE ")?;
        if self.or_replace {
            f.write_str("OR REPLACE ")?;
        }
        write!(f, "VIEW {}", self.name)?;
        if !self.columns.is_empty() {
            f.write_str(" (")?;
            write_list(f, &self.columns, ", ")?;
            f.write_str(")")?;
        }
        write!(f, " AS {}", self.query)
    }
}

impl fmt::Display for Explain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("EXPLAIN ")?;
        if self.analyze {
            f.write_str("ANALYZE ")?;
        }
        if self.verbose {
            f.write_str("VERBOSE ")?;
        }
        write!(f, "{}", self.statement)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use crate::{parse, parse_expression};

    /// Print, re-parse and assert the tree is unchanged, then assert the exact
    /// printed text so that a formatting regression is visible in the diff.
    fn round_trip(sql: &str, expected: &str) {
        let first = parse(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        let printed = first.to_string();
        assert_eq!(printed, expected, "printing {sql}");
        let second = parse(&printed).unwrap_or_else(|e| panic!("{printed}: {e}"));
        assert_eq!(first, second, "re-parsing {printed}");
    }

    #[test]
    fn expressions() {
        for (input, expected) in [
            ("select 1+2*3", "SELECT 1 + 2 * 3"),
            ("select (1+2)*3", "SELECT (1 + 2) * 3"),
            ("select 1-(2-3)", "SELECT 1 - (2 - 3)"),
            ("select 1-2-3", "SELECT 1 - 2 - 3"),
            ("select not a and b", "SELECT NOT a AND b"),
            ("select not (a and b)", "SELECT NOT (a AND b)"),
            ("select -x", "SELECT -x"),
            ("select - -x", "SELECT - -x"),
            ("select -(-x)", "SELECT - -x"),
            ("select a||b=c", "SELECT a || b = c"),
            ("select 'it''s'", "SELECT 'it''s'"),
            ("select \"Odd Name\"", "SELECT \"Odd Name\""),
            ("select \"select\"", "SELECT \"select\""),
            ("select 1.50, 1e9, .5", "SELECT 1.50, 1e9, .5"),
            ("select ?, ?", "SELECT $1, $2"),
            ("select $2, $1", "SELECT $2, $1"),
        ] {
            round_trip(input, expected);
        }
    }

    #[test]
    fn a_double_negative_never_prints_as_a_comment() {
        // `--x` is a line comment, so printing it would silently delete the
        // rest of the statement on the next parse.
        let printed = parse("select - -x, 1").unwrap().to_string();
        assert!(!printed.contains("--"), "{printed}");
        assert_eq!(printed, "SELECT - -x, 1");
        assert_eq!(parse(&printed).unwrap().to_string(), printed);
    }

    #[test]
    fn clauses_and_joins() {
        round_trip(
            "select a, t.*, count(*) as n from x as t inner join y on t.a=y.a left outer join z using (a) where a>1 group by a having count(*)>2 order by n desc nulls last limit 10 offset 5",
            "SELECT a, t.*, count(*) AS n FROM x AS t JOIN y ON t.a = y.a LEFT JOIN z USING (a) \
             WHERE a > 1 GROUP BY a HAVING count(*) > 2 ORDER BY n DESC NULLS LAST LIMIT 10 OFFSET 5",
        );
        round_trip(
            "select * from a natural full outer join b",
            "SELECT * FROM a NATURAL FULL JOIN b",
        );
        round_trip("select * from a, b", "SELECT * FROM a, b");
        round_trip(
            "select * from (a join b on a.x=b.x) cross join c",
            "SELECT * FROM (a JOIN b ON a.x = b.x) CROSS JOIN c",
        );
    }

    #[test]
    fn set_operations_keep_their_shape() {
        round_trip(
            "select 1 union select 2 union all select 3",
            "SELECT 1 UNION SELECT 2 UNION ALL SELECT 3",
        );
        round_trip(
            "select 1 union select 2 intersect select 3",
            "SELECT 1 UNION SELECT 2 INTERSECT SELECT 3",
        );
        round_trip(
            "(select 1 union select 2) intersect select 3",
            "(SELECT 1 UNION SELECT 2) INTERSECT SELECT 3",
        );
        round_trip(
            "select 1 except distinct select 2",
            "SELECT 1 EXCEPT DISTINCT SELECT 2",
        );
    }

    #[test]
    fn statements() {
        round_trip(
            "create table if not exists s.t (a int not null, b varchar(10) default 'x', c decimal(10,2) primary key)",
            "CREATE TABLE IF NOT EXISTS s.t (a INT32 NOT NULL, b UTF8 DEFAULT 'x', c DECIMAL(10, 2) PRIMARY KEY)",
        );
        round_trip("drop table if exists a, b", "DROP TABLE IF EXISTS a, b");
        round_trip(
            "create or replace view v (x) as select 1",
            "CREATE OR REPLACE VIEW v (x) AS SELECT 1",
        );
        round_trip("drop view v", "DROP VIEW v");
        round_trip(
            "insert into t (a,b) values (1,2),(3,4)",
            "INSERT INTO t (a, b) VALUES (1, 2), (3, 4)",
        );
        round_trip(
            "explain analyze verbose select 1",
            "EXPLAIN ANALYZE VERBOSE SELECT 1",
        );
        round_trip("create table t as select 1", "CREATE TABLE t AS SELECT 1");
    }

    #[test]
    fn windows_ctes_and_subqueries() {
        round_trip(
            "with c (x) as (select 1), d as (select 2) select * from c",
            "WITH c (x) AS (SELECT 1), d AS (SELECT 2) SELECT * FROM c",
        );
        round_trip(
            "select rank() over (partition by a order by b desc rows between unbounded preceding and current row) from t",
            "SELECT rank() OVER (PARTITION BY a ORDER BY b DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        );
        round_trip(
            "select sum(x) over (rows 3 preceding) from t",
            "SELECT sum(x) OVER (ROWS 3 PRECEDING) FROM t",
        );
        round_trip(
            "select (select max(b) from u where u.a=t.a) from t",
            "SELECT (SELECT max(b) FROM u WHERE u.a = t.a) FROM t",
        );
        round_trip(
            "select * from t where exists (select 1 from u) and a in (select b from u)",
            "SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u) AND a IN (SELECT b FROM u)",
        );
        round_trip(
            "select * from (select 1 as x) as s",
            "SELECT * FROM (SELECT 1 AS x) AS s",
        );
    }

    #[test]
    fn case_cast_and_extract() {
        round_trip(
            "select case when a then 1 when b then 2 else 3 end",
            "SELECT CASE WHEN a THEN 1 WHEN b THEN 2 ELSE 3 END",
        );
        round_trip(
            "select case a when 1 then 'x' end",
            "SELECT CASE a WHEN 1 THEN 'x' END",
        );
        round_trip(
            "select cast(a as bigint), extract(year from ts)",
            "SELECT CAST(a AS INT64), EXTRACT(YEAR FROM ts)",
        );
    }

    #[test]
    fn every_expression_form_reparses_identically() {
        for sql in [
            "a IS NULL",
            "a IS NOT NULL",
            "a IS DISTINCT FROM b",
            "a IN (1, 2)",
            "a NOT IN (1, 2)",
            "a BETWEEN 1 AND 2",
            "a NOT BETWEEN 1 AND 2",
            "a LIKE 'x%' ESCAPE '!'",
            "a NOT ILIKE 'x'",
            "(a AND b) IS NULL",
            "count(DISTINCT a)",
            "f(a, b, *)",
            "-(a + b)",
            "NOT NOT a",
        ] {
            let first = parse_expression(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
            let printed = first.to_string();
            let second = parse_expression(&printed).unwrap_or_else(|e| panic!("{printed}: {e}"));
            assert_eq!(first, second, "{sql} printed as {printed}");
        }
    }
}

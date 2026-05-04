//! The syntax tree.
//!
//! # Two decisions that shape everything else
//!
//! **Spans are provenance, not identity.** Every [`Expr`] and every [`Ident`]
//! records the byte range it came from so the binder can underline the exact
//! text it is complaining about — but `PartialEq` ignores the span. Two trees
//! are equal when they *mean* the same thing. That is what makes the round-trip
//! property `parse(print(parse(sql))) == parse(sql)` a real test of the printer
//! rather than a test of whitespace.
//!
//! **Parentheses are not in the tree.** `(a + b) * c` and `a + b * c` differ in
//! structure, so there is nothing left for a `Nested` node to record. The
//! printer re-derives the parentheses it needs from the precedence table (see
//! [`BinaryOperator::precedence`]), which means it can never emit a string that
//! parses back to a different tree — and the property test proves it.
//!
//! **Numeric literals keep their original spelling.** [`Literal::Number`] holds
//! the text, not a `Value`. Deciding that `1.50` is a `DECIMAL(3, 2)` and `1e9`
//! is a `FLOAT64` is a *semantic* judgement that depends on context, so it
//! belongs to the binder; [`Literal::to_value`] is where the rule is written
//! down. Keeping the text also makes the round trip exact.

use std::fmt;
use std::hash::{Hash, Hasher};

use basalt_common::{BasaltError, Result};
use basalt_types::{DataType, Value, DECIMAL128_MAX_PRECISION};

use crate::lexer::{Keyword, Parameter};
use crate::span::{NodeSpan, Span};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// A name: a column, a table, an alias.
///
/// The `value` is already case-folded by the lexer (bare identifiers to lower
/// case, quoted ones verbatim), so two `Ident`s are equal exactly when they name
/// the same thing. Whether the source spelled it `foo` or `"foo"` is not
/// recorded, because it makes no semantic difference and recording it would
/// break the round-trip property for no gain.
#[derive(Debug, Clone, Eq)]
pub struct Ident {
    /// The case-folded name.
    pub value: String,
    /// Where the name appeared.
    pub span: Span,
}

impl Ident {
    /// Build an identifier.
    pub fn new(value: impl Into<String>, span: Span) -> Ident {
        Ident {
            value: value.into(),
            span,
        }
    }

    /// `true` if this name must be printed inside double quotes to survive a
    /// round trip: it is empty, contains something other than `[a-z0-9_]`,
    /// starts with a digit, or collides with a keyword.
    pub fn needs_quoting(&self) -> bool {
        let mut chars = self.value.chars();
        let Some(first) = chars.next() else {
            return true;
        };
        if !(first.is_ascii_lowercase() || first == '_') {
            return true;
        }
        if !chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return true;
        }
        Keyword::from_uppercase(&self.value.to_ascii_uppercase()).is_some()
    }
}

impl PartialEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// A possibly qualified name: `t`, `s.t`, `s.t.c`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObjectName(pub Vec<Ident>);

impl ObjectName {
    /// A single-part name.
    pub fn bare(ident: Ident) -> ObjectName {
        ObjectName(vec![ident])
    }

    /// The last part, which is the object's own name.
    ///
    /// # Panics
    ///
    /// Never, in trees produced by the parser: the grammar cannot build an
    /// empty `ObjectName`. Hand-built empty names return `""`.
    pub fn base(&self) -> &str {
        self.0.last().map_or("", |i| i.value.as_str())
    }

    /// Everything before the last part.
    pub fn qualifier(&self) -> &[Ident] {
        self.0.split_last().map_or(&[], |(_, rest)| rest)
    }

    /// The span covering the whole dotted name.
    pub fn span(&self) -> Span {
        match (self.0.first(), self.0.last()) {
            (Some(f), Some(l)) => f.span.to(l.span),
            _ => Span::at(0),
        }
    }
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

/// A literal as it was written.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Literal {
    /// `NULL`.
    Null,
    /// `TRUE` / `FALSE`.
    Boolean(bool),
    /// A numeric literal in its original spelling, e.g. `1`, `1.50`, `1e-9`.
    Number(String),
    /// A string literal, already unescaped.
    String(String),
}

impl Literal {
    /// Interpret the literal as a typed [`Value`].
    ///
    /// The rules, chosen so that a literal never silently loses precision:
    ///
    /// * an integer literal is `INT64`, or `DECIMAL(p, 0)` if it needs more than
    ///   64 bits, or an error if it needs more than 38 digits;
    /// * a literal with a decimal point but no exponent is an **exact**
    ///   `DECIMAL(p, s)` preserving the digits as written — `1.50` is
    ///   `DECIMAL(3, 2)`, not `1.5` — because that is the only reading under
    ///   which `SELECT 0.1 + 0.2 = 0.3` is true;
    /// * a literal with an exponent is `FLOAT64`, since scientific notation is
    ///   how people ask for floating point.
    ///
    /// ```
    /// use basalt_sql::ast::Literal;
    /// use basalt_types::{DataType, Value};
    ///
    /// assert_eq!(Literal::Number("42".into()).to_value()?, Value::Int64(42));
    /// assert_eq!(
    ///     Literal::Number("1.50".into()).to_value()?,
    ///     Value::Decimal128(150, 3, 2),
    /// );
    /// assert_eq!(
    ///     Literal::Number("1e3".into()).to_value()?.data_type(),
    ///     DataType::Float64,
    /// );
    /// # Ok::<(), basalt_common::BasaltError>(())
    /// ```
    pub fn to_value(&self) -> Result<Value> {
        match self {
            Literal::Null => Ok(Value::Null),
            Literal::Boolean(b) => Ok(Value::Boolean(*b)),
            Literal::String(s) => Ok(Value::Utf8(s.clone())),
            Literal::Number(text) => number_to_value(text),
        }
    }
}

fn number_to_value(text: &str) -> Result<Value> {
    let has_exponent = text.bytes().any(|b| b == b'e' || b == b'E');
    if has_exponent {
        return text
            .parse::<f64>()
            .map(Value::Float64)
            .map_err(|_| BasaltError::type_error(format!("`{text}` is not a valid number")));
    }
    let Some((integral, fractional)) = text.split_once('.') else {
        return match text.parse::<i64>() {
            Ok(v) => Ok(Value::Int64(v)),
            Err(_) => exact_decimal(text, ""),
        };
    };
    exact_decimal(integral, fractional)
}

/// Build a `DECIMAL(p, s)` that keeps every digit written.
fn exact_decimal(integral: &str, fractional: &str) -> Result<Value> {
    let digits: String = format!("{integral}{fractional}");
    let scale = i8::try_from(fractional.len()).map_err(|_| too_many_digits(&digits))?;
    // Leading zeros do not count towards precision, but a value of exactly zero
    // still needs one digit.
    let significant = digits.trim_start_matches('0').len().max(1);
    let precision = u8::try_from(significant.max(fractional.len()))
        .ok()
        .filter(|p| *p <= DECIMAL128_MAX_PRECISION)
        .ok_or_else(|| too_many_digits(&digits))?;
    let unscaled: i128 = digits.parse().map_err(|_| {
        BasaltError::type_error(format!("`{integral}.{fractional}` is not a number"))
    })?;
    DataType::decimal128(precision, scale)
        .map(|_| Value::Decimal128(unscaled, precision, scale))
        .ok_or_else(|| too_many_digits(&digits))
}

fn too_many_digits(digits: &str) -> BasaltError {
    BasaltError::type_error(format!(
        "numeric literal has {} significant digits; the maximum is {DECIMAL128_MAX_PRECISION}",
        digits.len()
    ))
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    /// `OR`
    Or,
    /// `AND`
    And,
    /// `=`
    Eq,
    /// `<>`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `||`
    Concat,
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Multiply,
    /// `/`
    Divide,
    /// `%`
    Modulo,
}

/// Binding power of an operator; larger binds tighter.
///
/// # The precedence table
///
/// | Level | Operators                                              | Associativity |
/// |-------|--------------------------------------------------------|---------------|
/// | 1     | `OR`                                                   | left          |
/// | 2     | `AND`                                                  | left          |
/// | 3     | `NOT` (prefix)                                         | right         |
/// | 4     | `=` `<>` `<` `<=` `>` `>=` `IS` `IN` `BETWEEN` `LIKE`  | left          |
/// | 5     | `\|\|`                                                 | left          |
/// | 6     | `+` `-` (binary)                                       | left          |
/// | 7     | `*` `/` `%`                                            | left          |
/// | 8     | `-` `+` (unary)                                        | right         |
///
/// This is the standard SQL table. Two consequences are worth stating because
/// they are the ones people get wrong:
///
/// * `NOT` binds *looser* than comparison, so `NOT a = b` is `NOT (a = b)`.
/// * `||` binds *tighter* than comparison but *looser* than `+`, so
///   `a || b = c` is `(a || b) = c` and `a || b + c` is `a || (b + c)`.
///
/// The comparison level is left-associative here rather than non-associative as
/// in the standard: `a < b < c` parses as `(a < b) < c` and is then rejected by
/// the binder with a type error naming both sides, which is a better experience
/// than a syntax error that just says "unexpected `<`".
pub type Precedence = u8;

/// Precedence of `OR`.
pub const PREC_OR: Precedence = 1;
/// Precedence of `AND`.
pub const PREC_AND: Precedence = 2;
/// Precedence of prefix `NOT`.
pub const PREC_NOT: Precedence = 3;
/// Precedence of comparison and of the `IS` / `IN` / `BETWEEN` / `LIKE` family.
pub const PREC_COMPARISON: Precedence = 4;
/// Precedence of `||`.
pub const PREC_CONCAT: Precedence = 5;
/// Precedence of binary `+` and `-`.
pub const PREC_ADDITIVE: Precedence = 6;
/// Precedence of `*`, `/` and `%`.
pub const PREC_MULTIPLICATIVE: Precedence = 7;
/// Precedence of unary `+` and `-`.
pub const PREC_UNARY: Precedence = 8;
/// Precedence of a primary expression: a literal, a column, a call, `CASE`.
pub const PREC_PRIMARY: Precedence = 9;

impl BinaryOperator {
    /// Where this operator sits in the [precedence table](Precedence).
    pub const fn precedence(self) -> Precedence {
        match self {
            BinaryOperator::Or => PREC_OR,
            BinaryOperator::And => PREC_AND,
            BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq => PREC_COMPARISON,
            BinaryOperator::Concat => PREC_CONCAT,
            BinaryOperator::Plus | BinaryOperator::Minus => PREC_ADDITIVE,
            BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Modulo => {
                PREC_MULTIPLICATIVE
            }
        }
    }

    /// The spelling used by [`Display`](fmt::Display).
    pub const fn as_str(self) -> &'static str {
        match self {
            BinaryOperator::Or => "OR",
            BinaryOperator::And => "AND",
            BinaryOperator::Eq => "=",
            BinaryOperator::NotEq => "<>",
            BinaryOperator::Lt => "<",
            BinaryOperator::LtEq => "<=",
            BinaryOperator::Gt => ">",
            BinaryOperator::GtEq => ">=",
            BinaryOperator::Concat => "||",
            BinaryOperator::Plus => "+",
            BinaryOperator::Minus => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Modulo => "%",
        }
    }

    /// `true` for the six comparison operators.
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq
        )
    }

    /// The operator with its operands swapped, when one exists.
    ///
    /// `a < b` is `b > a`; used by the optimizer to normalise join conditions.
    pub const fn swap_operands(self) -> Option<BinaryOperator> {
        Some(match self {
            BinaryOperator::Eq => BinaryOperator::Eq,
            BinaryOperator::NotEq => BinaryOperator::NotEq,
            BinaryOperator::Lt => BinaryOperator::Gt,
            BinaryOperator::LtEq => BinaryOperator::GtEq,
            BinaryOperator::Gt => BinaryOperator::Lt,
            BinaryOperator::GtEq => BinaryOperator::LtEq,
            BinaryOperator::And | BinaryOperator::Or => self,
            _ => return None,
        })
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A prefix operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    /// `NOT`
    Not,
    /// `-`
    Minus,
    /// `+` — kept in the tree rather than discarded, so that the printer can
    /// reproduce the input and the binder can reject `+'abc'`.
    Plus,
}

impl UnaryOperator {
    /// Binding power of the operand.
    pub const fn precedence(self) -> Precedence {
        match self {
            UnaryOperator::Not => PREC_NOT,
            UnaryOperator::Minus | UnaryOperator::Plus => PREC_UNARY,
        }
    }

    /// The spelling used by [`Display`](fmt::Display).
    pub const fn as_str(self) -> &'static str {
        match self {
            UnaryOperator::Not => "NOT",
            UnaryOperator::Minus => "-",
            UnaryOperator::Plus => "+",
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A field selectable by `EXTRACT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateTimeField {
    /// Calendar year.
    Year,
    /// Quarter of the year, 1-4.
    Quarter,
    /// Calendar month, 1-12.
    Month,
    /// ISO week of the year.
    Week,
    /// Day of the month, 1-31.
    Day,
    /// Day of the week, 0 (Sunday) to 6.
    DayOfWeek,
    /// Day of the year, 1-366.
    DayOfYear,
    /// Hour, 0-23.
    Hour,
    /// Minute, 0-59.
    Minute,
    /// Second, 0-59.
    Second,
    /// Milliseconds within the second.
    Millisecond,
    /// Microseconds within the second.
    Microsecond,
    /// Whole seconds since the Unix epoch.
    Epoch,
}

impl DateTimeField {
    /// The spelling used both by the parser and by [`Display`](fmt::Display).
    pub const fn as_str(self) -> &'static str {
        match self {
            DateTimeField::Year => "YEAR",
            DateTimeField::Quarter => "QUARTER",
            DateTimeField::Month => "MONTH",
            DateTimeField::Week => "WEEK",
            DateTimeField::Day => "DAY",
            DateTimeField::DayOfWeek => "DOW",
            DateTimeField::DayOfYear => "DOY",
            DateTimeField::Hour => "HOUR",
            DateTimeField::Minute => "MINUTE",
            DateTimeField::Second => "SECOND",
            DateTimeField::Millisecond => "MILLISECOND",
            DateTimeField::Microsecond => "MICROSECOND",
            DateTimeField::Epoch => "EPOCH",
        }
    }

    /// Every field, for lookup and for "did you mean" hints.
    pub const ALL: &'static [DateTimeField] = &[
        DateTimeField::Year,
        DateTimeField::Quarter,
        DateTimeField::Month,
        DateTimeField::Week,
        DateTimeField::Day,
        DateTimeField::DayOfWeek,
        DateTimeField::DayOfYear,
        DateTimeField::Hour,
        DateTimeField::Minute,
        DateTimeField::Second,
        DateTimeField::Millisecond,
        DateTimeField::Microsecond,
        DateTimeField::Epoch,
    ];

    /// Parse a field name, which the caller has already upper-cased.
    pub fn from_uppercase(word: &str) -> Option<DateTimeField> {
        DateTimeField::ALL
            .iter()
            .copied()
            .find(|f| f.as_str() == word)
    }
}

impl fmt::Display for DateTimeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// An expression, plus the source text it came from.
///
/// `PartialEq`, `Eq` and `Hash` ignore [`Expr::span`] — see the module docs.
#[derive(Debug, Clone, Eq)]
pub struct Expr {
    /// The expression itself.
    pub kind: ExprKind,
    /// Where it appeared in the source.
    pub span: Span,
}

impl Expr {
    /// Pair a kind with a span.
    pub fn new(kind: ExprKind, span: Span) -> Expr {
        Expr { kind, span }
    }

    /// Convenience for `Expr::new(kind, span)` behind a `Box`.
    pub fn boxed(kind: ExprKind, span: Span) -> Box<Expr> {
        Box::new(Expr::new(kind, span))
    }

    /// Binding power of the outermost operator, used by the printer to decide
    /// where parentheses are needed.
    pub fn precedence(&self) -> Precedence {
        match &self.kind {
            ExprKind::BinaryOp { op, .. } => op.precedence(),
            ExprKind::UnaryOp { op, .. } => op.precedence(),
            ExprKind::IsNull { .. }
            | ExprKind::IsDistinctFrom { .. }
            | ExprKind::InList { .. }
            | ExprKind::InSubquery { .. }
            | ExprKind::Between { .. }
            | ExprKind::Like { .. } => PREC_COMPARISON,
            _ => PREC_PRIMARY,
        }
    }

    /// Walk the tree, calling `f` on every node including this one, until `f`
    /// returns `false` for a node (whose children are then skipped).
    ///
    /// The AST is walked in a handful of places (finding aggregates, finding
    /// subqueries, collecting parameters) and each of them wants a different
    /// answer, so the traversal lives here once.
    pub fn visit(&self, f: &mut impl FnMut(&Expr) -> bool) {
        if !f(self) {
            return;
        }
        self.kind.for_each_child(&mut |child| child.visit(f));
    }

    /// `true` if any node in the tree satisfies `predicate`.
    pub fn contains(&self, predicate: &mut impl FnMut(&Expr) -> bool) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if found {
                return false;
            }
            if predicate(e) {
                found = true;
                return false;
            }
            true
        });
        found
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

/// The shape of an expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExprKind {
    /// A column reference, possibly qualified: `a`, `t.a`, `s.t.a`.
    Column(ObjectName),
    /// A literal.
    Literal(Literal),
    /// A bind parameter.
    Parameter(Parameter),
    /// `op expr`.
    UnaryOp {
        /// The prefix operator.
        op: UnaryOperator,
        /// Its operand.
        expr: Box<Expr>,
    },
    /// `left op right`.
    BinaryOp {
        /// Left operand.
        left: Box<Expr>,
        /// The operator.
        op: BinaryOperator,
        /// Right operand.
        right: Box<Expr>,
    },
    /// `expr IS [NOT] NULL`.
    IsNull {
        /// The tested expression.
        expr: Box<Expr>,
        /// `true` for `IS NOT NULL`.
        negated: bool,
    },
    /// `left IS [NOT] DISTINCT FROM right` — the null-safe comparison.
    IsDistinctFrom {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
        /// `true` for `IS NOT DISTINCT FROM`.
        negated: bool,
    },
    /// `expr [NOT] IN (a, b, c)`.
    InList {
        /// The tested expression.
        expr: Box<Expr>,
        /// The candidate values.
        list: Vec<Expr>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] IN (SELECT ...)`.
    InSubquery {
        /// The tested expression.
        expr: Box<Expr>,
        /// The single-column subquery.
        subquery: Box<Query>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] BETWEEN low AND high`.
    Between {
        /// The tested expression.
        expr: Box<Expr>,
        /// Inclusive lower bound.
        low: Box<Expr>,
        /// Inclusive upper bound.
        high: Box<Expr>,
        /// `true` for `NOT BETWEEN`.
        negated: bool,
    },
    /// `expr [NOT] LIKE pattern [ESCAPE c]`.
    Like {
        /// The tested expression.
        expr: Box<Expr>,
        /// The pattern.
        pattern: Box<Expr>,
        /// The escape character, if `ESCAPE` was given.
        escape: Option<Box<Expr>>,
        /// `true` for `NOT LIKE`.
        negated: bool,
        /// `true` for `ILIKE`.
        case_insensitive: bool,
    },
    /// `CASE [operand] WHEN .. THEN .. [ELSE ..] END`.
    Case {
        /// The `CASE operand WHEN value` form, if any.
        operand: Option<Box<Expr>>,
        /// The `WHEN condition THEN result` arms, in order.
        arms: Vec<CaseArm>,
        /// The `ELSE` result, if any.
        else_result: Option<Box<Expr>>,
    },
    /// `CAST(expr AS type)`.
    Cast {
        /// The expression being converted.
        expr: Box<Expr>,
        /// The target type.
        data_type: DataType,
    },
    /// `EXTRACT(field FROM expr)`.
    Extract {
        /// Which component to take.
        field: DateTimeField,
        /// The temporal expression.
        expr: Box<Expr>,
    },
    /// A function call, which may be a scalar call, an aggregate, or a window
    /// function depending on the name and on [`Function::over`].
    Function(Function),
    /// `[NOT] EXISTS (SELECT ...)`.
    Exists {
        /// The subquery.
        subquery: Box<Query>,
        /// `true` for `NOT EXISTS`.
        negated: bool,
    },
    /// A scalar subquery: `(SELECT ...)` used where a value is expected.
    Subquery(Box<Query>),
}

impl ExprKind {
    /// Apply `f` to each direct child expression, in evaluation order.
    ///
    /// Subqueries are *not* descended into: they are separate name-resolution
    /// scopes, and every caller of this function so far wants to treat them as
    /// opaque. Callers that need to look inside a subquery walk its `Query`.
    pub fn for_each_child(&self, f: &mut impl FnMut(&Expr)) {
        match self {
            ExprKind::Column(_)
            | ExprKind::Literal(_)
            | ExprKind::Parameter(_)
            | ExprKind::Exists { .. }
            | ExprKind::Subquery(_) => {}
            ExprKind::UnaryOp { expr, .. }
            | ExprKind::IsNull { expr, .. }
            | ExprKind::Cast { expr, .. }
            | ExprKind::Extract { expr, .. }
            | ExprKind::InSubquery { expr, .. } => f(expr),
            ExprKind::BinaryOp { left, right, .. }
            | ExprKind::IsDistinctFrom { left, right, .. } => {
                f(left);
                f(right);
            }
            ExprKind::InList { expr, list, .. } => {
                f(expr);
                list.iter().for_each(&mut *f);
            }
            ExprKind::Between {
                expr, low, high, ..
            } => {
                f(expr);
                f(low);
                f(high);
            }
            ExprKind::Like {
                expr,
                pattern,
                escape,
                ..
            } => {
                f(expr);
                f(pattern);
                if let Some(e) = escape {
                    f(e);
                }
            }
            ExprKind::Case {
                operand,
                arms,
                else_result,
            } => {
                if let Some(o) = operand {
                    f(o);
                }
                for arm in arms {
                    f(&arm.condition);
                    f(&arm.result);
                }
                if let Some(e) = else_result {
                    f(e);
                }
            }
            ExprKind::Function(func) => {
                for arg in &func.args {
                    if let FunctionArg::Expr(e) = arg {
                        f(e);
                    }
                }
                if let Some(w) = &func.over {
                    w.partition_by.iter().for_each(&mut *f);
                    for o in &w.order_by {
                        f(&o.expr);
                    }
                }
            }
        }
    }
}

/// One `WHEN condition THEN result` arm of a `CASE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CaseArm {
    /// The `WHEN` expression: a predicate in the searched form, a value in the
    /// simple form.
    pub condition: Expr,
    /// The `THEN` expression.
    pub result: Expr,
}

/// A function call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    /// The function name.
    pub name: ObjectName,
    /// The arguments.
    pub args: Vec<FunctionArg>,
    /// `true` for `agg(DISTINCT x)`.
    pub distinct: bool,
    /// The `OVER (...)` clause, which turns the call into a window function.
    pub over: Option<WindowSpec>,
}

/// One argument of a call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionArg {
    /// A normal expression argument.
    Expr(Expr),
    /// The `*` of `count(*)`, which is not an expression: it denotes "the row"
    /// and is only legal for `count`.
    Wildcard(NodeSpan),
}

/// The `OVER (...)` specification of a window function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowSpec {
    /// `PARTITION BY` expressions.
    pub partition_by: Vec<Expr>,
    /// `ORDER BY` items.
    pub order_by: Vec<OrderByExpr>,
    /// The frame, if one was written.
    pub frame: Option<WindowFrame>,
}

/// `ROWS`/`RANGE BETWEEN start AND end`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowFrame {
    /// Whether offsets count rows or values.
    pub units: WindowFrameUnits,
    /// The start bound.
    pub start: WindowFrameBound,
    /// The end bound; `None` means `CURRENT ROW`, which is what the standard
    /// says a bare `ROWS n PRECEDING` means.
    pub end: Option<WindowFrameBound>,
}

/// Whether a frame offset counts rows or compares values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowFrameUnits {
    /// `ROWS`: physical offsets.
    Rows,
    /// `RANGE`: logical offsets against the `ORDER BY` value.
    Range,
}

/// One end of a window frame.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowFrameBound {
    /// `UNBOUNDED PRECEDING`
    UnboundedPreceding,
    /// `n PRECEDING`
    Preceding(Box<Expr>),
    /// `CURRENT ROW`
    CurrentRow,
    /// `n FOLLOWING`
    Following(Box<Expr>),
    /// `UNBOUNDED FOLLOWING`
    UnboundedFollowing,
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// A complete query: optional CTEs, a set-expression body, then the clauses
/// that apply to the whole thing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Query {
    /// The `WITH` clause.
    pub with: Option<With>,
    /// The body: a `SELECT`, a `VALUES`, or a set operation over them.
    pub body: SetExpr,
    /// `ORDER BY`, which applies after the set operation.
    pub order_by: Vec<OrderByExpr>,
    /// `LIMIT`.
    pub limit: Option<Expr>,
    /// `OFFSET`.
    pub offset: Option<Expr>,
}

/// A `WITH` clause.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct With {
    /// `true` if `RECURSIVE` was written. The parser accepts it so that the
    /// binder can reject it with an explanation rather than a syntax error.
    pub recursive: bool,
    /// The common table expressions, in the order written.
    pub ctes: Vec<Cte>,
}

/// One common table expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cte {
    /// The name it is bound to.
    pub name: Ident,
    /// Optional explicit column names.
    pub columns: Vec<Ident>,
    /// The query it stands for.
    pub query: Box<Query>,
}

/// The body of a [`Query`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SetExpr {
    /// A `SELECT`.
    Select(Box<Select>),
    /// A parenthesised query, which may carry its own `ORDER BY`/`LIMIT`.
    Query(Box<Query>),
    /// `left op right`.
    SetOperation {
        /// The operator.
        op: SetOperator,
        /// `ALL` or `DISTINCT`; absent means `DISTINCT`, per the standard.
        quantifier: SetQuantifier,
        /// Left input.
        left: Box<SetExpr>,
        /// Right input.
        right: Box<SetExpr>,
    },
    /// A `VALUES` list.
    Values(Values),
}

/// `UNION` / `EXCEPT` / `INTERSECT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetOperator {
    /// `UNION`
    Union,
    /// `EXCEPT`
    Except,
    /// `INTERSECT`
    Intersect,
}

impl SetOperator {
    /// The spelling used by [`Display`](fmt::Display).
    pub const fn as_str(self) -> &'static str {
        match self {
            SetOperator::Union => "UNION",
            SetOperator::Except => "EXCEPT",
            SetOperator::Intersect => "INTERSECT",
        }
    }

    /// Binding power. `INTERSECT` binds tighter than `UNION` and `EXCEPT`,
    /// which are equal and left-associative — the standard's rule.
    pub const fn precedence(self) -> Precedence {
        match self {
            SetOperator::Union | SetOperator::Except => 1,
            SetOperator::Intersect => 2,
        }
    }
}

impl fmt::Display for SetOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Duplicate handling of a set operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetQuantifier {
    /// `ALL`: keep duplicates.
    All,
    /// `DISTINCT`, written explicitly.
    Distinct,
    /// Neither written; means `DISTINCT`.
    None,
}

/// A `VALUES (..), (..)` list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Values {
    /// The rows, each of the same width (checked by the binder, not the
    /// parser, so that the error can name both row numbers).
    pub rows: Vec<Vec<Expr>>,
}

/// A `SELECT` block.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Select {
    /// `true` for `SELECT DISTINCT`.
    pub distinct: bool,
    /// The projection list.
    pub projection: Vec<SelectItem>,
    /// The `FROM` clause. More than one entry means an implicit cross join.
    pub from: Vec<TableRef>,
    /// `WHERE`.
    pub selection: Option<Expr>,
    /// `GROUP BY`.
    pub group_by: Vec<Expr>,
    /// `HAVING`.
    pub having: Option<Expr>,
}

/// One item of a projection list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SelectItem {
    /// `*`
    Wildcard(NodeSpan),
    /// `t.*`
    QualifiedWildcard(ObjectName),
    /// `expr` or `expr AS alias`.
    Expr {
        /// The expression.
        expr: Expr,
        /// The output name, if written.
        alias: Option<Ident>,
    },
}

/// An entry in the `FROM` clause: one relation plus the joins hanging off it.
///
/// The join list is flat and left-deep, which is how SQL's syntax reads
/// (`a JOIN b JOIN c` is `(a JOIN b) JOIN c`). Right-deep and bushy trees are
/// written with parentheses and land in [`TableFactor::Nested`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableRef {
    /// The leftmost relation.
    pub relation: TableFactor,
    /// Joins applied left to right.
    pub joins: Vec<Join>,
}

/// A single relation in the `FROM` clause.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TableFactor {
    /// A named table, view or CTE.
    Table {
        /// Its name.
        name: ObjectName,
        /// Its alias, if any.
        alias: Option<TableAlias>,
    },
    /// A parenthesised subquery.
    Derived {
        /// The subquery.
        subquery: Box<Query>,
        /// Its alias. Required by the binder, optional in the grammar so the
        /// error can say *why* it is required.
        alias: Option<TableAlias>,
    },
    /// A parenthesised join tree.
    Nested(Box<TableRef>),
}

/// `AS alias (col, col)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableAlias {
    /// The relation name.
    pub name: Ident,
    /// Explicit column names, which rename the relation's output positionally.
    pub columns: Vec<Ident>,
}

/// One join in a [`TableRef`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Join {
    /// The relation being joined in.
    pub relation: TableFactor,
    /// Which kind of join.
    pub operator: JoinOperator,
    /// How the rows are matched.
    pub constraint: JoinConstraint,
}

/// The kind of a join.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinOperator {
    /// `[INNER] JOIN`
    Inner,
    /// `LEFT [OUTER] JOIN`
    Left,
    /// `RIGHT [OUTER] JOIN`
    Right,
    /// `FULL [OUTER] JOIN`
    Full,
    /// `CROSS JOIN`, or a comma in the `FROM` clause.
    Cross,
}

impl JoinOperator {
    /// The spelling used by [`Display`](fmt::Display).
    pub const fn as_str(self) -> &'static str {
        match self {
            JoinOperator::Inner => "JOIN",
            JoinOperator::Left => "LEFT JOIN",
            JoinOperator::Right => "RIGHT JOIN",
            JoinOperator::Full => "FULL JOIN",
            JoinOperator::Cross => "CROSS JOIN",
        }
    }
}

impl fmt::Display for JoinOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a join matches rows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum JoinConstraint {
    /// `ON predicate`
    On(Expr),
    /// `USING (a, b)`
    Using(Vec<Ident>),
    /// `NATURAL`: join on every commonly named column.
    Natural,
    /// No constraint, as in `CROSS JOIN`.
    None,
}

/// One `ORDER BY` item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrderByExpr {
    /// The sort key: an expression, an output alias, or a 1-based ordinal.
    pub expr: Expr,
    /// `Some(true)` for `ASC`, `Some(false)` for `DESC`, `None` if unwritten.
    pub asc: Option<bool>,
    /// `Some(true)` for `NULLS FIRST`, `Some(false)` for `NULLS LAST`, `None`
    /// if unwritten.
    pub nulls_first: Option<bool>,
}

impl OrderByExpr {
    /// The effective direction: ascending unless `DESC` was written.
    pub fn is_ascending(&self) -> bool {
        self.asc.unwrap_or(true)
    }

    /// The effective null placement.
    ///
    /// Basalt follows the SQL standard's *recommended* default, which is also
    /// PostgreSQL's: nulls sort as if they were larger than every value, so they
    /// come last under `ASC` and first under `DESC`. Writing it out means an
    /// `ORDER BY` never changes meaning when the direction is edited.
    pub fn nulls_first(&self) -> bool {
        self.nulls_first.unwrap_or(!self.is_ascending())
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A top-level statement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Statement {
    /// A `SELECT`, `VALUES` or set operation.
    Query(Box<Query>),
    /// `INSERT INTO t [(cols)] <query>`.
    Insert(Insert),
    /// `CREATE TABLE`.
    CreateTable(CreateTable),
    /// `DROP TABLE`.
    DropTable(DropTable),
    /// `CREATE VIEW`.
    CreateView(CreateView),
    /// `DROP VIEW`.
    DropView(DropView),
    /// `EXPLAIN <statement>`.
    Explain(Explain),
}

/// `INSERT INTO t (a, b) SELECT ...` or `... VALUES (...)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Insert {
    /// Target table.
    pub table: ObjectName,
    /// Target columns; empty means "all columns, in schema order".
    pub columns: Vec<Ident>,
    /// The rows to insert, as a query (a `VALUES` list is a query).
    pub source: Box<Query>,
}

/// `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateTable {
    /// Table name.
    pub name: ObjectName,
    /// `IF NOT EXISTS`.
    pub if_not_exists: bool,
    /// Column definitions; empty for `CREATE TABLE ... AS SELECT`.
    pub columns: Vec<ColumnDef>,
    /// The `AS <query>` form.
    pub query: Option<Box<Query>>,
}

/// One column of a `CREATE TABLE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnDef {
    /// Column name.
    pub name: Ident,
    /// Column type.
    pub data_type: DataType,
    /// Constraints and defaults, in the order written.
    pub options: Vec<ColumnOption>,
}

/// A per-column constraint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ColumnOption {
    /// `NULL` — the default, but writable for symmetry.
    Null,
    /// `NOT NULL`.
    NotNull,
    /// `PRIMARY KEY`, which also implies `NOT NULL`.
    PrimaryKey,
    /// `UNIQUE`.
    Unique,
    /// `DEFAULT expr`.
    Default(Expr),
}

/// `DROP TABLE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropTable {
    /// The tables to drop.
    pub names: Vec<ObjectName>,
    /// `IF EXISTS`.
    pub if_exists: bool,
}

/// `CREATE [OR REPLACE] VIEW`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CreateView {
    /// View name.
    pub name: ObjectName,
    /// `OR REPLACE`.
    pub or_replace: bool,
    /// Optional explicit column names.
    pub columns: Vec<Ident>,
    /// The view body.
    pub query: Box<Query>,
}

/// `DROP VIEW`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DropView {
    /// The views to drop.
    pub names: Vec<ObjectName>,
    /// `IF EXISTS`.
    pub if_exists: bool,
}

/// `EXPLAIN [ANALYZE] [VERBOSE] <statement>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Explain {
    /// `ANALYZE`: run the statement and report actual counts.
    pub analyze: bool,
    /// `VERBOSE`: include expression detail and schemas.
    pub verbose: bool,
    /// The statement being explained.
    pub statement: Box<Statement>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn ident(name: &str) -> Ident {
        Ident::new(name, Span::new(0, name.len()))
    }

    #[test]
    fn identifier_equality_ignores_the_span() {
        let a = Ident::new("x", Span::new(0, 1));
        let b = Ident::new("x", Span::new(50, 51));
        assert_eq!(a, b);
    }

    #[test]
    fn quoting_is_needed_for_keywords_and_odd_names() {
        assert!(!ident("foo").needs_quoting());
        assert!(!ident("_a1").needs_quoting());
        assert!(ident("Foo").needs_quoting());
        assert!(ident("with space").needs_quoting());
        assert!(ident("1abc").needs_quoting());
        assert!(ident("").needs_quoting());
        assert!(ident("select").needs_quoting());
    }

    #[test]
    fn object_name_parts() {
        let name = ObjectName(vec![ident("s"), ident("t")]);
        assert_eq!(name.base(), "t");
        assert_eq!(name.qualifier().len(), 1);
        assert_eq!(ObjectName::bare(ident("t")).qualifier().len(), 0);
        assert_eq!(ObjectName(vec![]).base(), "");
        assert_eq!(ObjectName(vec![]).span(), Span::at(0));
    }

    #[test]
    fn literal_typing_rules() {
        assert_eq!(
            Literal::Number("42".into()).to_value().unwrap(),
            Value::Int64(42)
        );
        assert_eq!(
            Literal::Number("-0".into()).to_value().unwrap(),
            Value::Int64(0)
        );
        assert_eq!(
            Literal::Number("1.50".into()).to_value().unwrap(),
            Value::Decimal128(150, 3, 2)
        );
        assert_eq!(
            Literal::Number("0.5".into()).to_value().unwrap(),
            Value::Decimal128(5, 1, 1)
        );
        assert_eq!(
            Literal::Number("1e3".into()).to_value().unwrap(),
            Value::Float64(1000.0)
        );
        // Wider than i64 but within DECIMAL(38, 0).
        let big = Literal::Number("123456789012345678901234567890".into())
            .to_value()
            .unwrap();
        assert_eq!(big.data_type(), DataType::Decimal128(30, 0));
        // Wider than DECIMAL(38, 0).
        let err = Literal::Number("1".repeat(40)).to_value().unwrap_err();
        assert!(err.to_string().contains("significant digits"), "{err}");

        assert_eq!(Literal::Null.to_value().unwrap(), Value::Null);
        assert_eq!(
            Literal::Boolean(true).to_value().unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            Literal::String("hi".into()).to_value().unwrap(),
            Value::Utf8("hi".into())
        );
    }

    #[test]
    fn precedence_matches_the_documented_table() {
        use BinaryOperator as B;
        assert!(B::Or.precedence() < B::And.precedence());
        assert!(B::And.precedence() < B::Eq.precedence());
        assert!(B::Eq.precedence() < B::Concat.precedence());
        assert!(B::Concat.precedence() < B::Plus.precedence());
        assert!(B::Plus.precedence() < B::Multiply.precedence());
        assert!(B::Multiply.precedence() < UnaryOperator::Minus.precedence());
        assert_eq!(UnaryOperator::Not.precedence(), PREC_NOT);
        // The one ordering people get wrong: NOT sits *between* AND and
        // comparison, so `NOT a = b` is `NOT (a = b)`.
        assert_eq!([PREC_AND, PREC_NOT, PREC_COMPARISON], [2, 3, 4]);
    }

    #[test]
    fn operator_helpers() {
        assert!(BinaryOperator::Lt.is_comparison());
        assert!(!BinaryOperator::Plus.is_comparison());
        assert_eq!(BinaryOperator::Lt.swap_operands(), Some(BinaryOperator::Gt));
        assert_eq!(BinaryOperator::Eq.swap_operands(), Some(BinaryOperator::Eq));
        assert_eq!(BinaryOperator::Plus.swap_operands(), None);
        assert_eq!(BinaryOperator::NotEq.to_string(), "<>");
        assert_eq!(UnaryOperator::Not.to_string(), "NOT");
        assert_eq!(JoinOperator::Left.to_string(), "LEFT JOIN");
        assert_eq!(SetOperator::Union.to_string(), "UNION");
        assert!(SetOperator::Intersect.precedence() > SetOperator::Union.precedence());
    }

    #[test]
    fn datetime_fields_round_trip() {
        for f in DateTimeField::ALL {
            assert_eq!(DateTimeField::from_uppercase(f.as_str()), Some(*f));
        }
        assert_eq!(DateTimeField::from_uppercase("NOPE"), None);
    }

    #[test]
    fn order_by_defaults_put_nulls_where_the_standard_says() {
        let asc = OrderByExpr {
            expr: Expr::new(ExprKind::Literal(Literal::Null), Span::at(0)),
            asc: None,
            nulls_first: None,
        };
        assert!(asc.is_ascending());
        assert!(!asc.nulls_first(), "ASC puts nulls last");
        let desc = OrderByExpr {
            asc: Some(false),
            ..asc.clone()
        };
        assert!(desc.nulls_first(), "DESC puts nulls first");
        let forced = OrderByExpr {
            nulls_first: Some(true),
            ..asc
        };
        assert!(forced.nulls_first());
    }

    #[test]
    fn visit_reaches_every_child_but_stops_at_subqueries() {
        let sp = Span::at(0);
        let col = |n: &str| Expr::new(ExprKind::Column(ObjectName::bare(ident(n))), sp);
        let expr = Expr::new(
            ExprKind::Between {
                expr: Box::new(col("a")),
                low: Box::new(col("b")),
                high: Box::new(col("c")),
                negated: false,
            },
            sp,
        );
        let mut names = Vec::new();
        expr.visit(&mut |e| {
            if let ExprKind::Column(n) = &e.kind {
                names.push(n.base().to_string());
            }
            true
        });
        assert_eq!(names, vec!["a", "b", "c"]);
        assert!(expr.contains(&mut |e| matches!(&e.kind, ExprKind::Column(n) if n.base() == "c")));
        assert!(!expr.contains(&mut |e| matches!(e.kind, ExprKind::Literal(_))));
    }
}

//! The resolved expression tree.
//!
//! Everything a parser leaves open is closed here: names are
//! [`ColumnId`]s, every node knows its [`DataType`] and its nullability, and the
//! implicit casts the type rules call for are explicit [`Expr::Cast`] nodes. The
//! executor therefore never has to look anything up while compiling an
//! expression, and the optimizer can rewrite freely without re-deriving types.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use basalt_common::{BasaltError, Result};
use basalt_sql::ast::DateTimeField;
use basalt_types::{coerce, DataType, Value};

use crate::function::{ScalarFunc, WindowFunc};
use crate::logical::LogicalPlan;

/// A globally unique identity for one column produced somewhere in a plan.
///
/// Ids are handed out by the binder and never reused within a statement. Two
/// plan nodes referring to the same id always mean the same column, which is
/// what makes column pruning and join reordering purely structural rewrites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnId(pub u32);

impl fmt::Display for ColumnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Hands out fresh [`ColumnId`]s.
#[derive(Debug, Default)]
pub struct ColumnIdGenerator {
    next: u32,
}

impl ColumnIdGenerator {
    /// A generator starting from zero.
    pub fn new() -> ColumnIdGenerator {
        ColumnIdGenerator::default()
    }

    /// The next unused id.
    ///
    /// Named `next` despite the resemblance to [`Iterator::next`]: a generator
    /// of fresh identities is exactly what the name says, and an iterator
    /// implementation would be a worse fit because the sequence is infinite and
    /// callers never want to consume it lazily.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> ColumnId {
        let id = ColumnId(self.next);
        self.next += 1;
        id
    }
}

/// A reference to one column of an operator's input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    /// Which column, independent of position.
    pub id: ColumnId,
    /// The relation alias the column came from, for display only.
    pub qualifier: Option<String>,
    /// The user-visible name, for display only.
    pub name: String,
    /// Its type.
    pub data_type: DataType,
    /// Whether it can be null.
    pub nullable: bool,
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.qualifier {
            Some(q) => write!(f, "{q}.{}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// A binary operator in a resolved expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
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
    /// `AND`
    And,
    /// `OR`
    Or,
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
    /// `||`
    Concat,
}

impl BinaryOp {
    /// The SQL spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
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

    /// `true` for the six comparison operators.
    pub const fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::NotEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
        )
    }

    /// `true` for `AND` / `OR`.
    pub const fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// The operator that means the same thing with the operands swapped.
    ///
    /// `a < b` is `b > a`. Used by predicate pushdown to normalise
    /// `literal op column` into `column op literal`.
    pub const fn swapped(self) -> Option<BinaryOp> {
        Some(match self {
            BinaryOp::Eq => BinaryOp::Eq,
            BinaryOp::NotEq => BinaryOp::NotEq,
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            BinaryOp::And | BinaryOp::Or | BinaryOp::Plus | BinaryOp::Multiply => self,
            _ => return None,
        })
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A pre-hashed `IN` list, produced by the optimizer.
///
/// `x IN (a, b, c)` is a chain of comparisons; past a handful of elements a hash
/// set is asymptotically better, and the crossover is measured in
/// `basalt-optimizer`. The set is shared behind an [`Arc`] so that rewriting a
/// plan never re-hashes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InValueSet {
    /// The distinct non-null members, in a deterministic order.
    pub values: Vec<Value>,
    /// `true` if the original list contained a `NULL`, which changes the
    /// three-valued answer for non-members from `false` to `NULL`.
    pub has_null: bool,
    /// The type every member (and the probe) has been cast to.
    pub data_type: DataType,
}

/// A resolved expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A column of the operator's input.
    Column(ColumnRef),
    /// A column of an *enclosing* query block, in a correlated subquery.
    OuterColumn(ColumnRef),
    /// A constant.
    Literal(Value),
    /// `left op right`.
    Binary {
        /// Left operand.
        left: Box<Expr>,
        /// The operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expr>,
    },
    /// `NOT expr`.
    Not(Box<Expr>),
    /// Arithmetic negation.
    Negate(Box<Expr>),
    /// `expr IS [NOT] NULL`.
    IsNull {
        /// The tested expression.
        expr: Box<Expr>,
        /// `true` for `IS NOT NULL`.
        negated: bool,
    },
    /// `left IS [NOT] DISTINCT FROM right`: the null-safe comparison.
    IsDistinctFrom {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
        /// `true` for `IS NOT DISTINCT FROM`.
        negated: bool,
    },
    /// `expr [NOT] IN (list)`.
    InList {
        /// The probe.
        expr: Box<Expr>,
        /// The candidates, already cast to the probe's type.
        list: Vec<Expr>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] IN (set)`: the hashed form of [`Expr::InList`].
    InSet {
        /// The probe.
        expr: Box<Expr>,
        /// The pre-hashed members.
        set: Arc<InValueSet>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
    /// `expr [NOT] LIKE pattern`.
    Like {
        /// The subject.
        expr: Box<Expr>,
        /// The pattern.
        pattern: Box<Expr>,
        /// The `ESCAPE` character.
        escape: Option<char>,
        /// `true` for `NOT LIKE`.
        negated: bool,
        /// `true` for `ILIKE`.
        case_insensitive: bool,
    },
    /// `CASE WHEN .. THEN .. ELSE .. END`, already rewritten to the searched
    /// form (the simple form's operand is distributed into the conditions).
    Case {
        /// `(condition, result)` pairs, tried in order.
        arms: Vec<(Expr, Expr)>,
        /// The `ELSE` branch; `NULL` when absent.
        else_result: Option<Box<Expr>>,
    },
    /// An explicit or inserted cast.
    Cast {
        /// The operand.
        expr: Box<Expr>,
        /// The target type.
        data_type: DataType,
    },
    /// `EXTRACT(field FROM expr)`.
    Extract {
        /// Which component.
        field: DateTimeField,
        /// A `DATE32` or `TIMESTAMP` operand.
        expr: Box<Expr>,
    },
    /// A call to a built-in scalar function.
    ScalarFunction {
        /// Which function.
        func: ScalarFunc,
        /// Its arguments, already cast to the declared types.
        args: Vec<Expr>,
    },
    /// `[NOT] EXISTS (subquery)`.
    Exists {
        /// The subquery plan.
        subquery: Arc<LogicalPlan>,
        /// `true` for `NOT EXISTS`.
        negated: bool,
    },
    /// `expr [NOT] IN (subquery)`.
    InSubquery {
        /// The probe.
        expr: Box<Expr>,
        /// A single-column subquery.
        subquery: Arc<LogicalPlan>,
        /// `true` for `NOT IN`.
        negated: bool,
    },
}

impl Expr {
    /// A boolean literal.
    pub fn bool(value: bool) -> Expr {
        Expr::Literal(Value::Boolean(value))
    }

    /// Build `left op right`.
    pub fn binary(left: Expr, op: BinaryOp, right: Expr) -> Expr {
        Expr::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Conjoin two predicates, dropping literal `TRUE`s.
    pub fn and(self, other: Expr) -> Expr {
        match (&self, &other) {
            (Expr::Literal(Value::Boolean(true)), _) => other,
            (_, Expr::Literal(Value::Boolean(true))) => self,
            _ => Expr::binary(self, BinaryOp::And, other),
        }
    }

    /// The type of the value this expression produces.
    ///
    /// Never fails: the binder has already checked the types, and every node
    /// carries what it needs. Complexity is O(depth) because only the spine of
    /// the tree is walked.
    pub fn data_type(&self) -> DataType {
        match self {
            Expr::Column(c) | Expr::OuterColumn(c) => c.data_type.clone(),
            Expr::Literal(v) => v.data_type(),
            Expr::Binary { left, op, right } => binary_type(left, *op, right),
            Expr::Not(_)
            | Expr::IsNull { .. }
            | Expr::IsDistinctFrom { .. }
            | Expr::InList { .. }
            | Expr::InSet { .. }
            | Expr::Like { .. }
            | Expr::Exists { .. }
            | Expr::InSubquery { .. } => DataType::Boolean,
            Expr::Negate(e) => e.data_type(),
            Expr::Case { arms, else_result } => {
                let mut acc = else_result
                    .as_ref()
                    .map_or(DataType::Null, |e| e.data_type());
                for (_, result) in arms {
                    acc = coerce(&acc, &result.data_type()).unwrap_or(acc);
                }
                acc
            }
            Expr::Cast { data_type, .. } => data_type.clone(),
            Expr::Extract { .. } => DataType::Int64,
            Expr::ScalarFunction { func, args } => {
                let types: Vec<DataType> = args.iter().map(Expr::data_type).collect();
                func.return_type(&types).unwrap_or(DataType::Null)
            }
        }
    }

    /// Whether the expression can evaluate to `NULL`.
    ///
    /// Conservative: it may say `true` where a cleverer analysis would say
    /// `false`, never the other way round. Used to decide output nullability and
    /// to justify null-related simplifications.
    pub fn nullable(&self) -> bool {
        match self {
            Expr::Column(c) | Expr::OuterColumn(c) => c.nullable,
            Expr::Literal(v) => v.is_null(),
            Expr::IsNull { .. } | Expr::IsDistinctFrom { .. } | Expr::Exists { .. } => false,
            Expr::Binary { left, op, right } => {
                // `AND`/`OR` are null-propagating only sometimes (FALSE AND NULL
                // is FALSE), but "may be null" is the safe answer either way.
                let _ = op;
                left.nullable() || right.nullable()
            }
            Expr::Not(e) | Expr::Negate(e) | Expr::Cast { expr: e, .. } => e.nullable(),
            Expr::Extract { expr, .. } => expr.nullable(),
            Expr::InList { expr, list, .. } => expr.nullable() || list.iter().any(Expr::nullable),
            Expr::InSet { expr, set, .. } => expr.nullable() || set.has_null,
            Expr::InSubquery { .. } => true,
            Expr::Like { expr, pattern, .. } => expr.nullable() || pattern.nullable(),
            Expr::Case { arms, else_result } => {
                else_result.as_ref().is_none_or(|e| e.nullable())
                    || arms.iter().any(|(_, r)| r.nullable())
            }
            Expr::ScalarFunction { func, args } => {
                func.may_introduce_null()
                    || (func.is_null_propagating() && args.iter().any(Expr::nullable))
                    || (!func.is_null_propagating() && args.iter().all(Expr::nullable))
            }
        }
    }

    /// `true` when the expression contains no column reference and no subquery,
    /// so it can be evaluated once at planning time.
    pub fn is_constant(&self) -> bool {
        let mut constant = true;
        self.visit(&mut |e| {
            if matches!(
                e,
                Expr::Column(_)
                    | Expr::OuterColumn(_)
                    | Expr::Exists { .. }
                    | Expr::InSubquery { .. }
            ) {
                constant = false;
            }
            constant
        });
        constant
    }

    /// Every column id the expression reads from its input.
    ///
    /// Outer references are excluded: they are not satisfied by this operator's
    /// input, and including them would make column pruning ask its child for a
    /// column the child does not have.
    pub fn columns(&self) -> BTreeSet<ColumnId> {
        let mut out = BTreeSet::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns(&self, out: &mut BTreeSet<ColumnId>) {
        self.visit(&mut |e| {
            if let Expr::Column(c) = e {
                out.insert(c.id);
            }
            true
        });
    }

    /// Every *outer* column id the expression reads, i.e. its correlation.
    pub fn outer_columns(&self) -> BTreeSet<ColumnId> {
        let mut out = BTreeSet::new();
        self.visit(&mut |e| {
            if let Expr::OuterColumn(c) = e {
                out.insert(c.id);
            }
            true
        });
        out
    }

    /// `true` if the expression contains a subquery.
    pub fn has_subquery(&self) -> bool {
        let mut found = false;
        self.visit(&mut |e| {
            if matches!(e, Expr::Exists { .. } | Expr::InSubquery { .. }) {
                found = true;
            }
            !found
        });
        found
    }

    /// Pre-order walk. `f` returns `false` to stop descending into a node.
    ///
    /// Subquery plans are *not* descended into; they are separate scopes and
    /// every caller so far wants them treated as opaque.
    pub fn visit(&self, f: &mut impl FnMut(&Expr) -> bool) {
        if !f(self) {
            return;
        }
        self.for_each_child(&mut |child| child.visit(f));
    }

    /// Apply `f` to each direct child expression.
    pub fn for_each_child(&self, f: &mut impl FnMut(&Expr)) {
        match self {
            Expr::Column(_) | Expr::OuterColumn(_) | Expr::Literal(_) | Expr::Exists { .. } => {}
            Expr::Binary { left, right, .. } | Expr::IsDistinctFrom { left, right, .. } => {
                f(left);
                f(right);
            }
            Expr::Not(e)
            | Expr::Negate(e)
            | Expr::IsNull { expr: e, .. }
            | Expr::Cast { expr: e, .. }
            | Expr::Extract { expr: e, .. }
            | Expr::InSet { expr: e, .. }
            | Expr::InSubquery { expr: e, .. } => f(e),
            Expr::InList { expr, list, .. } => {
                f(expr);
                for item in list {
                    f(item);
                }
            }
            Expr::Like { expr, pattern, .. } => {
                f(expr);
                f(pattern);
            }
            Expr::Case { arms, else_result } => {
                for (condition, result) in arms {
                    f(condition);
                    f(result);
                }
                if let Some(e) = else_result {
                    f(e);
                }
            }
            Expr::ScalarFunction { args, .. } => {
                for arg in args {
                    f(arg);
                }
            }
        }
    }

    /// Bottom-up rewrite: `f` is applied to every node after its children.
    ///
    /// This is the workhorse of the optimizer. It allocates a new tree; the
    /// alternative (in-place mutation) would make rules that need the old shape
    /// for a fixpoint check impossible to write.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns.
    pub fn transform_up(self, f: &mut impl FnMut(Expr) -> Result<Expr>) -> Result<Expr> {
        let rewritten = self.map_children(&mut |child| child.transform_up(f))?;
        f(rewritten)
    }

    /// Rebuild the node with each child replaced by `f(child)`.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns.
    pub fn map_children(self, f: &mut impl FnMut(Expr) -> Result<Expr>) -> Result<Expr> {
        Ok(match self {
            Expr::Column(_) | Expr::OuterColumn(_) | Expr::Literal(_) | Expr::Exists { .. } => self,
            Expr::Binary { left, op, right } => Expr::Binary {
                left: Box::new(f(*left)?),
                op,
                right: Box::new(f(*right)?),
            },
            Expr::IsDistinctFrom {
                left,
                right,
                negated,
            } => Expr::IsDistinctFrom {
                left: Box::new(f(*left)?),
                right: Box::new(f(*right)?),
                negated,
            },
            Expr::Not(e) => Expr::Not(Box::new(f(*e)?)),
            Expr::Negate(e) => Expr::Negate(Box::new(f(*e)?)),
            Expr::IsNull { expr, negated } => Expr::IsNull {
                expr: Box::new(f(*expr)?),
                negated,
            },
            Expr::Cast { expr, data_type } => Expr::Cast {
                expr: Box::new(f(*expr)?),
                data_type,
            },
            Expr::Extract { field, expr } => Expr::Extract {
                field,
                expr: Box::new(f(*expr)?),
            },
            Expr::InSet { expr, set, negated } => Expr::InSet {
                expr: Box::new(f(*expr)?),
                set,
                negated,
            },
            Expr::InSubquery {
                expr,
                subquery,
                negated,
            } => Expr::InSubquery {
                expr: Box::new(f(*expr)?),
                subquery,
                negated,
            },
            Expr::InList {
                expr,
                list,
                negated,
            } => Expr::InList {
                expr: Box::new(f(*expr)?),
                list: list.into_iter().map(&mut *f).collect::<Result<_>>()?,
                negated,
            },
            Expr::Like {
                expr,
                pattern,
                escape,
                negated,
                case_insensitive,
            } => Expr::Like {
                expr: Box::new(f(*expr)?),
                pattern: Box::new(f(*pattern)?),
                escape,
                negated,
                case_insensitive,
            },
            Expr::Case { arms, else_result } => {
                let mut new_arms = Vec::with_capacity(arms.len());
                for (condition, result) in arms {
                    new_arms.push((f(condition)?, f(result)?));
                }
                let else_result = match else_result {
                    Some(e) => Some(Box::new(f(*e)?)),
                    None => None,
                };
                Expr::Case {
                    arms: new_arms,
                    else_result,
                }
            }
            Expr::ScalarFunction { func, args } => Expr::ScalarFunction {
                func,
                args: args.into_iter().map(&mut *f).collect::<Result<_>>()?,
            },
        })
    }

    /// Split a predicate into its top-level `AND` conjuncts.
    ///
    /// Pushdown works conjunct by conjunct, so this is the first thing every
    /// pushdown rule does.
    pub fn split_conjuncts(self) -> Vec<Expr> {
        let mut out = Vec::new();
        split_conjuncts_into(self, &mut out);
        out
    }

    /// Rebuild a conjunction from parts; an empty list is `TRUE`.
    pub fn conjunction(parts: impl IntoIterator<Item = Expr>) -> Expr {
        let mut iter = parts.into_iter();
        let Some(first) = iter.next() else {
            return Expr::bool(true);
        };
        iter.fold(first, |acc, next| Expr::binary(acc, BinaryOp::And, next))
    }

    /// A display name for this expression when used as an output column.
    pub fn output_name(&self) -> String {
        match self {
            Expr::Column(c) | Expr::OuterColumn(c) => c.name.clone(),
            Expr::Cast { expr, .. } => expr.output_name(),
            Expr::Literal(v) if v.is_null() => "NULL".to_string(),
            other => other.to_string(),
        }
    }
}

fn split_conjuncts_into(expr: Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            split_conjuncts_into(*left, out);
            split_conjuncts_into(*right, out);
        }
        Expr::Literal(Value::Boolean(true)) => {}
        other => out.push(other),
    }
}

/// The result type of an arithmetic or comparison operator.
///
/// The one deliberate deviation from PostgreSQL is integer division: `5 / 2` is
/// `2.5`, not `2`. Truncating division is a footgun that silently corrupts
/// averages, and every dialect that made the other choice has a long tail of
/// bug reports to show for it. `%` keeps the integer type, where truncation is
/// the point.
fn binary_type(left: &Expr, op: BinaryOp, right: &Expr) -> DataType {
    if op.is_comparison() || op.is_logical() {
        return DataType::Boolean;
    }
    if op == BinaryOp::Concat {
        return DataType::Utf8;
    }
    let (l, r) = (left.data_type(), right.data_type());
    if op == BinaryOp::Divide {
        // Division always widens to a float, whatever the operands are.
        // Integer division that silently truncates is the single most common
        // source of quietly wrong analytics, so Basalt refuses to offer it
        // implicitly; a caller who wants truncation writes an explicit cast.
        return DataType::Float64;
    }
    coerce(&l, &r).unwrap_or(DataType::Null)
}

/// The declared result type of an arithmetic operator, for the binder.
///
/// # Errors
///
/// If the operands have no common type.
pub fn arithmetic_type(op: BinaryOp, left: &DataType, right: &DataType) -> Result<DataType> {
    if op == BinaryOp::Concat {
        return Ok(DataType::Utf8);
    }
    if op == BinaryOp::Divide {
        if left.is_numeric() && right.is_numeric() {
            return Ok(DataType::Float64);
        }
        if *left == DataType::Null || *right == DataType::Null {
            return Ok(DataType::Float64);
        }
    }
    coerce(left, right)
        .filter(|t| t.is_numeric() || *t == DataType::Null)
        .ok_or_else(|| {
            BasaltError::type_error(format!(
                "operator {op} is not defined for {left} and {right}"
            ))
        })
}

/// A sort key: an expression plus its direction and null placement.
#[derive(Debug, Clone, PartialEq)]
pub struct SortExpr {
    /// The key.
    pub expr: Expr,
    /// Ascending when `true`.
    pub asc: bool,
    /// Nulls come first when `true`.
    pub nulls_first: bool,
}

impl SortExpr {
    /// A sort key with the standard defaults (`ASC`, nulls last).
    pub fn new(expr: Expr, asc: bool, nulls_first: bool) -> SortExpr {
        SortExpr {
            expr,
            asc,
            nulls_first,
        }
    }
}

impl fmt::Display for SortExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} NULLS {}",
            self.expr,
            if self.asc { "ASC" } else { "DESC" },
            if self.nulls_first { "FIRST" } else { "LAST" }
        )
    }
}

/// A window function call inside a [`Window`](crate::logical::LogicalPlan::Window) node.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowCall {
    /// The function.
    pub func: WindowFunc,
    /// Its argument, absent for ranking functions.
    pub arg: Option<Expr>,
}

impl fmt::Display for Expr {
    /// Fully parenthesised, so that the rendering is injective: two expressions
    /// print the same text exactly when they are structurally equal. Common
    /// subexpression elimination keys on this string rather than on a hash, so
    /// injectivity is a correctness property, not a nicety.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(c) => write!(f, "{c}"),
            Expr::OuterColumn(c) => write!(f, "outer({c})"),
            Expr::Literal(v) => match v {
                Value::Utf8(s) => write!(f, "'{}'", s.replace('\'', "''")),
                Value::Null => f.write_str("NULL"),
                other => write!(f, "{other}"),
            },
            Expr::Binary { left, op, right } => write!(f, "({left} {op} {right})"),
            Expr::Not(e) => write!(f, "(NOT {e})"),
            Expr::Negate(e) => write!(f, "(- {e})"),
            Expr::IsNull { expr, negated } => {
                write!(f, "({expr} IS {}NULL)", if *negated { "NOT " } else { "" })
            }
            Expr::IsDistinctFrom {
                left,
                right,
                negated,
            } => write!(
                f,
                "({left} IS {}DISTINCT FROM {right})",
                if *negated { "NOT " } else { "" }
            ),
            Expr::InList {
                expr,
                list,
                negated,
            } => {
                write!(f, "({expr} {}IN [", if *negated { "NOT " } else { "" })?;
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("])")
            }
            Expr::InSet { expr, set, negated } => write!(
                f,
                "({expr} {}IN <set of {}{}>)",
                if *negated { "NOT " } else { "" },
                set.values.len(),
                if set.has_null { " + NULL" } else { "" }
            ),
            Expr::Like {
                expr,
                pattern,
                escape,
                negated,
                case_insensitive,
            } => {
                write!(
                    f,
                    "({expr} {}{} {pattern}",
                    if *negated { "NOT " } else { "" },
                    if *case_insensitive { "ILIKE" } else { "LIKE" }
                )?;
                if let Some(c) = escape {
                    write!(f, " ESCAPE '{c}'")?;
                }
                f.write_str(")")
            }
            Expr::Case { arms, else_result } => {
                f.write_str("CASE")?;
                for (condition, result) in arms {
                    write!(f, " WHEN {condition} THEN {result}")?;
                }
                if let Some(e) = else_result {
                    write!(f, " ELSE {e}")?;
                }
                f.write_str(" END")
            }
            Expr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type})"),
            Expr::Extract { field, expr } => write!(f, "EXTRACT({field} FROM {expr})"),
            Expr::ScalarFunction { func, args } => {
                write!(f, "{func}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
            Expr::Exists { negated, .. } => {
                write!(
                    f,
                    "({}EXISTS <subquery>)",
                    if *negated { "NOT " } else { "" }
                )
            }
            Expr::InSubquery { expr, negated, .. } => write!(
                f,
                "({expr} {}IN <subquery>)",
                if *negated { "NOT " } else { "" }
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn col(id: u32, name: &str, ty: DataType, nullable: bool) -> Expr {
        Expr::Column(ColumnRef {
            id: ColumnId(id),
            qualifier: Some("t".into()),
            name: name.into(),
            data_type: ty,
            nullable,
        })
    }

    #[test]
    fn types_and_nullability() {
        let a = col(0, "a", DataType::Int32, false);
        let b = col(1, "b", DataType::Int64, true);
        let sum = Expr::binary(a.clone(), BinaryOp::Plus, b.clone());
        assert_eq!(sum.data_type(), DataType::Int64);
        assert!(sum.nullable());

        let div = Expr::binary(a.clone(), BinaryOp::Divide, a.clone());
        assert_eq!(div.data_type(), DataType::Float64, "no integer division");

        let cmp = Expr::binary(a.clone(), BinaryOp::Lt, b);
        assert_eq!(cmp.data_type(), DataType::Boolean);

        let is_null = Expr::IsNull {
            expr: Box::new(a.clone()),
            negated: false,
        };
        assert!(!is_null.nullable(), "IS NULL is never null");
        assert!(!a.nullable());
    }

    #[test]
    fn conjunct_splitting_round_trips() {
        let a = col(0, "a", DataType::Boolean, false);
        let b = col(1, "b", DataType::Boolean, false);
        let c = col(2, "c", DataType::Boolean, false);
        let all = Expr::binary(
            Expr::binary(a.clone(), BinaryOp::And, b.clone()),
            BinaryOp::And,
            c.clone(),
        );
        let parts = all.clone().split_conjuncts();
        assert_eq!(parts, vec![a, b, c]);
        assert_eq!(Expr::conjunction(parts).to_string(), all.to_string());
        assert_eq!(Expr::conjunction(Vec::new()), Expr::bool(true));
    }

    #[test]
    fn columns_exclude_outer_references() {
        let inner = col(3, "x", DataType::Int64, false);
        let outer = Expr::OuterColumn(ColumnRef {
            id: ColumnId(9),
            qualifier: None,
            name: "y".into(),
            data_type: DataType::Int64,
            nullable: false,
        });
        let e = Expr::binary(inner, BinaryOp::Eq, outer);
        assert_eq!(
            e.columns().into_iter().collect::<Vec<_>>(),
            vec![ColumnId(3)]
        );
        assert_eq!(
            e.outer_columns().into_iter().collect::<Vec<_>>(),
            vec![ColumnId(9)]
        );
        assert!(!e.is_constant());
    }

    #[test]
    fn transform_up_visits_children_first() {
        let e = Expr::binary(
            Expr::Literal(Value::Int64(1)),
            BinaryOp::Plus,
            Expr::Literal(Value::Int64(2)),
        );
        let mut order = Vec::new();
        let out = e
            .transform_up(&mut |node| {
                order.push(node.to_string());
                Ok(node)
            })
            .unwrap();
        assert_eq!(order, vec!["1", "2", "(1 + 2)"]);
        assert_eq!(out.to_string(), "(1 + 2)");
    }

    #[test]
    fn display_is_parenthesised_and_injective() {
        let a = col(0, "a", DataType::Int64, false);
        let b = col(1, "b", DataType::Int64, false);
        let c = col(2, "c", DataType::Int64, false);
        let left = Expr::binary(
            Expr::binary(a.clone(), BinaryOp::Plus, b.clone()),
            BinaryOp::Multiply,
            c.clone(),
        );
        let right = Expr::binary(a, BinaryOp::Plus, Expr::binary(b, BinaryOp::Multiply, c));
        assert_ne!(left.to_string(), right.to_string());
        assert_eq!(left.to_string(), "((t.a + t.b) * t.c)");
    }

    #[test]
    fn operator_helpers() {
        assert_eq!(BinaryOp::Lt.swapped(), Some(BinaryOp::Gt));
        assert_eq!(BinaryOp::Eq.swapped(), Some(BinaryOp::Eq));
        assert_eq!(BinaryOp::Minus.swapped(), None);
        assert!(BinaryOp::GtEq.is_comparison());
        assert!(BinaryOp::Or.is_logical());
    }

    #[test]
    fn constant_folding_inputs_are_recognised() {
        let lit = Expr::binary(
            Expr::Literal(Value::Int64(1)),
            BinaryOp::Plus,
            Expr::Literal(Value::Int64(2)),
        );
        assert!(lit.is_constant());
        assert!(!col(0, "a", DataType::Int64, false).is_constant());
    }

    #[test]
    fn arithmetic_type_rules() {
        assert_eq!(
            arithmetic_type(BinaryOp::Plus, &DataType::Int32, &DataType::Int64).unwrap(),
            DataType::Int64
        );
        assert_eq!(
            arithmetic_type(BinaryOp::Divide, &DataType::Int64, &DataType::Int64).unwrap(),
            DataType::Float64
        );
        assert!(arithmetic_type(BinaryOp::Plus, &DataType::Utf8, &DataType::Int64).is_err());
        assert_eq!(
            arithmetic_type(BinaryOp::Concat, &DataType::Int64, &DataType::Utf8).unwrap(),
            DataType::Utf8
        );
    }
}

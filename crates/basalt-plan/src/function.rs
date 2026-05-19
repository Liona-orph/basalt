//! The function registry: the contract between the planner and the executor.
//!
//! # The contract
//!
//! For every function the registry states three things, and the executor is
//! entitled to rely on all three:
//!
//! 1. [`ScalarFunc::coerce_args`] returns the type each argument **will** have
//!    by the time a kernel sees it. The binder inserts the casts, so a kernel
//!    never has to handle a type it did not ask for.
//! 2. [`ScalarFunc::return_type`] returns the type of the result, computed from
//!    the *coerced* argument types.
//! 3. [`ScalarFunc::is_null_propagating`] says whether a null argument forces a
//!    null result. Kernels for null-propagating functions may compute on
//!    garbage under the null mask; kernels for the others (`coalesce`, `nullif`)
//!    must look at validity themselves.
//!
//! # Why a closed enum and not trait objects
//!
//! A registry of `Box<dyn ScalarFunction>` looks more extensible, but the
//! extensibility is fake in an engine whose executor must also carry a
//! vectorised kernel for each function: registering a function without a kernel
//! would compile fine and fail at run time. With a closed enum, "every function
//! has a kernel" is `match` exhaustiveness — a compile error, in the right
//! crate, the moment a variant is added. The cost is that a user cannot add a
//! function without touching Basalt, which is the honest state of affairs
//! anyway.

use std::fmt;

use basalt_common::{BasaltError, Result};
use basalt_types::{coerce, DataType};

/// How a function's arguments are converted before a kernel sees them.
///
/// Returned by [`ScalarFunc::coerce_args`]; the binder wraps each argument in a
/// [`Cast`](crate::expr::Expr::Cast) when the actual type differs.
pub type ArgTypes = Vec<DataType>;

fn arity_error(name: &str, expected: &str, got: usize) -> BasaltError {
    BasaltError::plan(format!(
        "function {name} expects {expected} argument(s), got {got}"
    ))
}

fn type_error(name: &str, detail: &str) -> BasaltError {
    BasaltError::type_error(format!("function {name}: {detail}"))
}

/// A built-in scalar function.
///
/// Scalar means "one row in, one row out": the kernel maps whole vectors
/// element-wise and never looks across rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ScalarFunc {
    // ---- string -----------------------------------------------------------
    /// `lower(utf8) -> utf8`.
    Lower,
    /// `upper(utf8) -> utf8`.
    Upper,
    /// `length(utf8) -> int64`, counted in **characters**, not bytes.
    Length,
    /// `octet_length(utf8) -> int64`, counted in bytes.
    OctetLength,
    /// `substr(utf8, int64 [, int64]) -> utf8`, 1-based like the standard.
    Substr,
    /// `trim(utf8) -> utf8` (both ends, ASCII whitespace).
    Trim,
    /// `ltrim(utf8) -> utf8`.
    Ltrim,
    /// `rtrim(utf8) -> utf8`.
    Rtrim,
    /// `replace(utf8, utf8, utf8) -> utf8`.
    Replace,
    /// `concat(utf8...) -> utf8`. Unlike `||`, ignores nulls.
    Concat,
    /// `starts_with(utf8, utf8) -> boolean`.
    StartsWith,
    /// `ends_with(utf8, utf8) -> boolean`.
    EndsWith,
    /// `contains(utf8, utf8) -> boolean`.
    Contains,
    /// `repeat(utf8, int64) -> utf8`.
    Repeat,
    /// `reverse(utf8) -> utf8`, by character.
    Reverse,
    /// `strpos(utf8, utf8) -> int64`, 1-based, 0 when absent.
    Strpos,

    // ---- math -------------------------------------------------------------
    /// `abs(numeric) -> same type`.
    Abs,
    /// `ceil(float64) -> float64`.
    Ceil,
    /// `floor(float64) -> float64`.
    Floor,
    /// `round(float64 [, int64]) -> float64`.
    Round,
    /// `sqrt(float64) -> float64`; negative input yields `NULL`.
    Sqrt,
    /// `exp(float64) -> float64`.
    Exp,
    /// `ln(float64) -> float64`; non-positive input yields `NULL`.
    Ln,
    /// `log10(float64) -> float64`; non-positive input yields `NULL`.
    Log10,
    /// `power(float64, float64) -> float64`.
    Power,
    /// `sign(numeric) -> int64`, one of -1, 0, 1.
    Sign,
    /// `trunc(float64) -> float64`.
    Trunc,

    // ---- null handling ----------------------------------------------------
    /// `coalesce(T...) -> T`: the first non-null argument.
    Coalesce,
    /// `nullif(T, T) -> T`: `NULL` when the arguments are equal.
    NullIf,
    /// `greatest(T...) -> T`, ignoring nulls.
    Greatest,
    /// `least(T...) -> T`, ignoring nulls.
    Least,
}

impl ScalarFunc {
    /// Every function the registry knows, in a stable order.
    ///
    /// Used by tests that assert the executor has a kernel for each one.
    pub const ALL: &'static [ScalarFunc] = &[
        ScalarFunc::Lower,
        ScalarFunc::Upper,
        ScalarFunc::Length,
        ScalarFunc::OctetLength,
        ScalarFunc::Substr,
        ScalarFunc::Trim,
        ScalarFunc::Ltrim,
        ScalarFunc::Rtrim,
        ScalarFunc::Replace,
        ScalarFunc::Concat,
        ScalarFunc::StartsWith,
        ScalarFunc::EndsWith,
        ScalarFunc::Contains,
        ScalarFunc::Repeat,
        ScalarFunc::Reverse,
        ScalarFunc::Strpos,
        ScalarFunc::Abs,
        ScalarFunc::Ceil,
        ScalarFunc::Floor,
        ScalarFunc::Round,
        ScalarFunc::Sqrt,
        ScalarFunc::Exp,
        ScalarFunc::Ln,
        ScalarFunc::Log10,
        ScalarFunc::Power,
        ScalarFunc::Sign,
        ScalarFunc::Trunc,
        ScalarFunc::Coalesce,
        ScalarFunc::NullIf,
        ScalarFunc::Greatest,
        ScalarFunc::Least,
    ];

    /// The lower-case SQL name.
    pub const fn name(self) -> &'static str {
        match self {
            ScalarFunc::Lower => "lower",
            ScalarFunc::Upper => "upper",
            ScalarFunc::Length => "length",
            ScalarFunc::OctetLength => "octet_length",
            ScalarFunc::Substr => "substr",
            ScalarFunc::Trim => "trim",
            ScalarFunc::Ltrim => "ltrim",
            ScalarFunc::Rtrim => "rtrim",
            ScalarFunc::Replace => "replace",
            ScalarFunc::Concat => "concat",
            ScalarFunc::StartsWith => "starts_with",
            ScalarFunc::EndsWith => "ends_with",
            ScalarFunc::Contains => "contains",
            ScalarFunc::Repeat => "repeat",
            ScalarFunc::Reverse => "reverse",
            ScalarFunc::Strpos => "strpos",
            ScalarFunc::Abs => "abs",
            ScalarFunc::Ceil => "ceil",
            ScalarFunc::Floor => "floor",
            ScalarFunc::Round => "round",
            ScalarFunc::Sqrt => "sqrt",
            ScalarFunc::Exp => "exp",
            ScalarFunc::Ln => "ln",
            ScalarFunc::Log10 => "log10",
            ScalarFunc::Power => "power",
            ScalarFunc::Sign => "sign",
            ScalarFunc::Trunc => "trunc",
            ScalarFunc::Coalesce => "coalesce",
            ScalarFunc::NullIf => "nullif",
            ScalarFunc::Greatest => "greatest",
            ScalarFunc::Least => "least",
        }
    }

    /// Resolve a (already lower-cased) SQL name, including aliases.
    pub fn from_name(name: &str) -> Option<ScalarFunc> {
        Some(match name {
            "char_length" | "character_length" => ScalarFunc::Length,
            "position" => ScalarFunc::Strpos,
            "substring" => ScalarFunc::Substr,
            "ifnull" | "nvl" => ScalarFunc::Coalesce,
            "pow" => ScalarFunc::Power,
            "truncate" => ScalarFunc::Trunc,
            other => *ScalarFunc::ALL.iter().find(|f| f.name() == other)?,
        })
    }

    /// `true` if a `NULL` in any argument forces a `NULL` result.
    ///
    /// This is what lets a kernel compute over the whole vector and fix up the
    /// validity mask at the end instead of branching per row. `coalesce`,
    /// `nullif`, `greatest` and `least` are the exceptions: their whole purpose
    /// is to inspect nulls.
    pub const fn is_null_propagating(self) -> bool {
        !matches!(
            self,
            ScalarFunc::Coalesce | ScalarFunc::NullIf | ScalarFunc::Greatest | ScalarFunc::Least
        )
    }

    /// The type each argument must be cast to before the kernel runs.
    ///
    /// # Errors
    ///
    /// Wrong arity, or an argument whose type has no legal conversion.
    pub fn coerce_args(self, args: &[DataType]) -> Result<ArgTypes> {
        use ScalarFunc as F;
        let name = self.name();
        match self {
            F::Lower | F::Upper | F::Trim | F::Ltrim | F::Rtrim | F::Reverse => {
                exact(name, args, &[DataType::Utf8])
            }
            F::Length | F::OctetLength => exact(name, args, &[DataType::Utf8]),
            F::Replace => exact(
                name,
                args,
                &[DataType::Utf8, DataType::Utf8, DataType::Utf8],
            ),
            F::StartsWith | F::EndsWith | F::Contains | F::Strpos => {
                exact(name, args, &[DataType::Utf8, DataType::Utf8])
            }
            F::Repeat => exact(name, args, &[DataType::Utf8, DataType::Int64]),
            F::Substr => {
                if args.len() == 2 {
                    exact(name, args, &[DataType::Utf8, DataType::Int64])
                } else if args.len() == 3 {
                    exact(
                        name,
                        args,
                        &[DataType::Utf8, DataType::Int64, DataType::Int64],
                    )
                } else {
                    Err(arity_error(name, "2 or 3", args.len()))
                }
            }
            F::Concat => {
                if args.is_empty() {
                    return Err(arity_error(name, "at least 1", 0));
                }
                Ok(vec![DataType::Utf8; args.len()])
            }
            F::Abs | F::Sign => {
                if args.len() != 1 {
                    return Err(arity_error(name, "1", args.len()));
                }
                if !args[0].is_numeric() && args[0] != DataType::Null {
                    return Err(type_error(name, "argument must be numeric"));
                }
                Ok(vec![args[0].clone()])
            }
            F::Ceil | F::Floor | F::Sqrt | F::Exp | F::Ln | F::Log10 | F::Trunc => {
                exact(name, args, &[DataType::Float64])
            }
            F::Power => exact(name, args, &[DataType::Float64, DataType::Float64]),
            F::Round => {
                if args.len() == 1 {
                    exact(name, args, &[DataType::Float64])
                } else if args.len() == 2 {
                    exact(name, args, &[DataType::Float64, DataType::Int64])
                } else {
                    Err(arity_error(name, "1 or 2", args.len()))
                }
            }
            F::Coalesce | F::Greatest | F::Least => {
                if args.is_empty() {
                    return Err(arity_error(name, "at least 1", 0));
                }
                let common = common_type(name, args)?;
                Ok(vec![common; args.len()])
            }
            F::NullIf => {
                if args.len() != 2 {
                    return Err(arity_error(name, "2", args.len()));
                }
                let common = common_type(name, args)?;
                Ok(vec![common; 2])
            }
        }
    }

    /// The result type, given the types returned by [`ScalarFunc::coerce_args`].
    ///
    /// # Errors
    ///
    /// Only if called with types that `coerce_args` would have rejected.
    pub fn return_type(self, args: &[DataType]) -> Result<DataType> {
        use ScalarFunc as F;
        Ok(match self {
            F::Lower
            | F::Upper
            | F::Substr
            | F::Trim
            | F::Ltrim
            | F::Rtrim
            | F::Replace
            | F::Concat
            | F::Repeat
            | F::Reverse => DataType::Utf8,
            F::Length | F::OctetLength | F::Strpos | F::Sign => DataType::Int64,
            F::StartsWith | F::EndsWith | F::Contains => DataType::Boolean,
            F::Ceil
            | F::Floor
            | F::Round
            | F::Sqrt
            | F::Exp
            | F::Ln
            | F::Log10
            | F::Power
            | F::Trunc => DataType::Float64,
            F::Abs => args
                .first()
                .cloned()
                .ok_or_else(|| arity_error(self.name(), "1", 0))?,
            F::Coalesce | F::NullIf | F::Greatest | F::Least => args
                .first()
                .cloned()
                .ok_or_else(|| arity_error(self.name(), "at least 1", 0))?,
        })
    }

    /// `true` if the result can be null even when no argument is.
    ///
    /// `sqrt(-1)`, `ln(0)` and `nullif(x, x)` all produce `NULL` from non-null
    /// inputs. Basalt returns `NULL` rather than raising, because an analytical
    /// query that dies on one bad row out of a billion is useless; the choice is
    /// documented on each kernel.
    pub const fn may_introduce_null(self) -> bool {
        matches!(
            self,
            ScalarFunc::Sqrt | ScalarFunc::Ln | ScalarFunc::Log10 | ScalarFunc::NullIf
        )
    }
}

impl fmt::Display for ScalarFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

fn exact(name: &str, args: &[DataType], want: &[DataType]) -> Result<ArgTypes> {
    if args.len() != want.len() {
        return Err(arity_error(name, &want.len().to_string(), args.len()));
    }
    for (got, need) in args.iter().zip(want) {
        if got == need || *got == DataType::Null {
            continue;
        }
        if coerce(got, need).as_ref() != Some(need) && !castable(got, need) {
            return Err(type_error(
                name,
                &format!("argument of type {got} cannot be used where {need} is required"),
            ));
        }
    }
    Ok(want.to_vec())
}

/// Explicit-cast reachability, which is wider than implicit coercion.
///
/// `length(int_col)` is a mistake worth rejecting, but `upper(cast(x as text))`
/// spelled without the cast is what every dialect accepts, so numbers and dates
/// are allowed to reach `Utf8`, and text is not allowed to reach a number
/// (there is no locale-free reading of `'1,5'`).
fn castable(from: &DataType, to: &DataType) -> bool {
    match to {
        DataType::Utf8 => !matches!(from, DataType::Binary),
        DataType::Float64 | DataType::Int64 => from.is_numeric(),
        _ => false,
    }
}

fn common_type(name: &str, args: &[DataType]) -> Result<DataType> {
    let mut acc = DataType::Null;
    for arg in args {
        acc = coerce(&acc, arg).ok_or_else(|| {
            type_error(
                name,
                &format!("arguments have no common type ({acc} and {arg})"),
            )
        })?;
    }
    Ok(acc)
}

/// A built-in aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum AggregateFunc {
    /// `count(x)`, or `count(*)` when the argument list is empty.
    Count,
    /// `sum(x)`.
    Sum,
    /// `avg(x)`.
    Avg,
    /// `min(x)`.
    Min,
    /// `max(x)`.
    Max,
    /// `stddev(x)` / `stddev_samp(x)`: sample standard deviation.
    StddevSamp,
    /// `stddev_pop(x)`: population standard deviation.
    StddevPop,
    /// `var_samp(x)` / `variance(x)`: sample variance.
    VarSamp,
    /// `var_pop(x)`: population variance.
    VarPop,
}

impl AggregateFunc {
    /// Every aggregate, in a stable order.
    pub const ALL: &'static [AggregateFunc] = &[
        AggregateFunc::Count,
        AggregateFunc::Sum,
        AggregateFunc::Avg,
        AggregateFunc::Min,
        AggregateFunc::Max,
        AggregateFunc::StddevSamp,
        AggregateFunc::StddevPop,
        AggregateFunc::VarSamp,
        AggregateFunc::VarPop,
    ];

    /// The canonical lower-case name.
    pub const fn name(self) -> &'static str {
        match self {
            AggregateFunc::Count => "count",
            AggregateFunc::Sum => "sum",
            AggregateFunc::Avg => "avg",
            AggregateFunc::Min => "min",
            AggregateFunc::Max => "max",
            AggregateFunc::StddevSamp => "stddev",
            AggregateFunc::StddevPop => "stddev_pop",
            AggregateFunc::VarSamp => "var_samp",
            AggregateFunc::VarPop => "var_pop",
        }
    }

    /// Resolve a lower-cased SQL name, including aliases.
    pub fn from_name(name: &str) -> Option<AggregateFunc> {
        Some(match name {
            "count" => AggregateFunc::Count,
            "sum" => AggregateFunc::Sum,
            "avg" | "mean" => AggregateFunc::Avg,
            "min" => AggregateFunc::Min,
            "max" => AggregateFunc::Max,
            "stddev" | "stddev_samp" => AggregateFunc::StddevSamp,
            "stddev_pop" => AggregateFunc::StddevPop,
            "variance" | "var_samp" => AggregateFunc::VarSamp,
            "var_pop" => AggregateFunc::VarPop,
            _ => return None,
        })
    }

    /// The type the argument is cast to before accumulation.
    ///
    /// `sum` and `avg` widen to the accumulator type up front so the kernel has
    /// exactly one arm per accumulator rather than one per input type; `min`,
    /// `max` and `count` take the input unchanged.
    ///
    /// # Errors
    ///
    /// If the aggregate is not defined for the argument's type.
    pub fn coerce_arg(self, arg: &DataType) -> Result<DataType> {
        match self {
            AggregateFunc::Count | AggregateFunc::Min | AggregateFunc::Max => Ok(arg.clone()),
            AggregateFunc::Sum => {
                if arg.is_integer() {
                    Ok(DataType::Int64)
                } else if arg.is_floating() || matches!(arg, DataType::Decimal128(_, _)) {
                    Ok(DataType::Float64)
                } else if *arg == DataType::Null {
                    Ok(DataType::Int64)
                } else {
                    Err(type_error("sum", &format!("cannot sum {arg}")))
                }
            }
            AggregateFunc::Avg
            | AggregateFunc::StddevSamp
            | AggregateFunc::StddevPop
            | AggregateFunc::VarSamp
            | AggregateFunc::VarPop => {
                if arg.is_numeric() || *arg == DataType::Null {
                    Ok(DataType::Float64)
                } else {
                    Err(type_error(
                        self.name(),
                        &format!("requires a numeric argument, got {arg}"),
                    ))
                }
            }
        }
    }

    /// The output type given the *coerced* argument type.
    ///
    /// # Errors
    ///
    /// If `min`/`max` is applied to a type with no order.
    pub fn return_type(self, arg: &DataType) -> Result<DataType> {
        Ok(match self {
            AggregateFunc::Count => DataType::Int64,
            AggregateFunc::Sum => arg.clone(),
            AggregateFunc::Min | AggregateFunc::Max => {
                if !arg.is_comparable() && *arg != DataType::Null {
                    return Err(type_error(self.name(), &format!("{arg} has no order")));
                }
                arg.clone()
            }
            AggregateFunc::Avg
            | AggregateFunc::StddevSamp
            | AggregateFunc::StddevPop
            | AggregateFunc::VarSamp
            | AggregateFunc::VarPop => DataType::Float64,
        })
    }

    /// `true` if the aggregate returns `NULL` for an empty (or all-null) group.
    ///
    /// Only `count` does not: it returns 0. Getting this wrong is the classic
    /// `sum` of nothing bug, so the planner asks rather than assumes.
    pub const fn is_null_on_empty(self) -> bool {
        !matches!(self, AggregateFunc::Count)
    }
}

impl fmt::Display for AggregateFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A function that may appear with an `OVER (...)` clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum WindowFunc {
    /// `row_number()`: 1-based position within the partition.
    RowNumber,
    /// `rank()`: competition ranking; ties share a rank and leave gaps.
    Rank,
    /// `dense_rank()`: ties share a rank and leave no gaps.
    DenseRank,
    /// An aggregate used as a window function.
    Aggregate(AggregateFunc),
}

impl WindowFunc {
    /// Resolve a lower-cased SQL name.
    pub fn from_name(name: &str) -> Option<WindowFunc> {
        Some(match name {
            "row_number" => WindowFunc::RowNumber,
            "rank" => WindowFunc::Rank,
            "dense_rank" => WindowFunc::DenseRank,
            other => WindowFunc::Aggregate(AggregateFunc::from_name(other)?),
        })
    }

    /// The canonical name.
    pub fn name(self) -> &'static str {
        match self {
            WindowFunc::RowNumber => "row_number",
            WindowFunc::Rank => "rank",
            WindowFunc::DenseRank => "dense_rank",
            WindowFunc::Aggregate(a) => a.name(),
        }
    }

    /// `true` if the function takes no arguments.
    pub const fn is_ranking(self) -> bool {
        matches!(
            self,
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank
        )
    }

    /// The output type given the coerced argument type (ignored for ranking
    /// functions, which are always `INT64`).
    ///
    /// # Errors
    ///
    /// Propagates [`AggregateFunc::return_type`].
    pub fn return_type(self, arg: Option<&DataType>) -> Result<DataType> {
        match self {
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank => Ok(DataType::Int64),
            WindowFunc::Aggregate(a) => a.return_type(arg.unwrap_or(&DataType::Null)),
        }
    }
}

impl fmt::Display for WindowFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Name-based lookup over the built-ins.
///
/// A zero-sized type rather than a map: the tables are `const` and the lookup is
/// a linear scan over ~30 short strings, which is faster than hashing and needs
/// no allocation at start-up. It exists as a type so that the binder has one
/// obvious place to ask "what is this identifier".
#[derive(Debug, Clone, Copy, Default)]
pub struct FunctionRegistry;

/// What a name resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFunction {
    /// A scalar function.
    Scalar(ScalarFunc),
    /// An aggregate.
    Aggregate(AggregateFunc),
    /// A window-only function (`row_number`, `rank`, `dense_rank`).
    Window(WindowFunc),
}

impl FunctionRegistry {
    /// Look a function up by its SQL name (case-insensitive).
    ///
    /// Aggregates win over scalars when a name is both, which never happens
    /// today and is asserted by a test.
    pub fn resolve(self, name: &str) -> Option<ResolvedFunction> {
        let lower = name.to_ascii_lowercase();
        if let Some(agg) = AggregateFunc::from_name(&lower) {
            return Some(ResolvedFunction::Aggregate(agg));
        }
        if let Some(scalar) = ScalarFunc::from_name(&lower) {
            return Some(ResolvedFunction::Scalar(scalar));
        }
        match lower.as_str() {
            "row_number" => Some(ResolvedFunction::Window(WindowFunc::RowNumber)),
            "rank" => Some(ResolvedFunction::Window(WindowFunc::Rank)),
            "dense_rank" => Some(ResolvedFunction::Window(WindowFunc::DenseRank)),
            _ => None,
        }
    }

    /// Every known name, for "did you mean" diagnostics.
    pub fn names(self) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = ScalarFunc::ALL.iter().map(|f| f.name()).collect();
        out.extend(AggregateFunc::ALL.iter().map(|f| f.name()));
        out.extend(["row_number", "rank", "dense_rank"]);
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn every_scalar_resolves_by_its_own_name() {
        for f in ScalarFunc::ALL {
            assert_eq!(ScalarFunc::from_name(f.name()), Some(*f), "{f}");
            assert_eq!(
                FunctionRegistry.resolve(&f.name().to_uppercase()),
                Some(ResolvedFunction::Scalar(*f)),
                "{f}"
            );
        }
    }

    #[test]
    fn every_aggregate_resolves() {
        for f in AggregateFunc::ALL {
            assert_eq!(AggregateFunc::from_name(f.name()), Some(*f), "{f}");
        }
        assert_eq!(
            FunctionRegistry.resolve("MEAN"),
            Some(ResolvedFunction::Aggregate(AggregateFunc::Avg))
        );
    }

    #[test]
    fn no_name_is_both_scalar_and_aggregate() {
        for f in ScalarFunc::ALL {
            assert_eq!(AggregateFunc::from_name(f.name()), None, "{f}");
        }
    }

    #[test]
    fn coercion_and_return_types() {
        let args = ScalarFunc::Length.coerce_args(&[DataType::Utf8]).unwrap();
        assert_eq!(args, vec![DataType::Utf8]);
        assert_eq!(
            ScalarFunc::Length.return_type(&args).unwrap(),
            DataType::Int64
        );

        // Numbers reach Utf8 by explicit-cast reachability.
        assert!(ScalarFunc::Upper.coerce_args(&[DataType::Int64]).is_ok());
        // Text does not reach a number.
        assert!(ScalarFunc::Sqrt.coerce_args(&[DataType::Utf8]).is_err());

        let coalesce = ScalarFunc::Coalesce
            .coerce_args(&[DataType::Int32, DataType::Int64])
            .unwrap();
        assert_eq!(coalesce, vec![DataType::Int64, DataType::Int64]);
        assert_eq!(
            ScalarFunc::Coalesce.return_type(&coalesce).unwrap(),
            DataType::Int64
        );

        assert!(ScalarFunc::Substr.coerce_args(&[DataType::Utf8]).is_err());
        assert!(ScalarFunc::Substr
            .coerce_args(&[DataType::Utf8, DataType::Int64])
            .is_ok());
    }

    #[test]
    fn aggregate_types() {
        assert_eq!(
            AggregateFunc::Sum.coerce_arg(&DataType::Int32).unwrap(),
            DataType::Int64
        );
        assert_eq!(
            AggregateFunc::Sum.coerce_arg(&DataType::Float32).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            AggregateFunc::Avg.coerce_arg(&DataType::Int64).unwrap(),
            DataType::Float64
        );
        assert!(AggregateFunc::Sum.coerce_arg(&DataType::Utf8).is_err());
        assert_eq!(
            AggregateFunc::Max.return_type(&DataType::Utf8).unwrap(),
            DataType::Utf8
        );
        assert!(!AggregateFunc::Count.is_null_on_empty());
        assert!(AggregateFunc::Sum.is_null_on_empty());
    }

    #[test]
    fn null_propagation_flags() {
        assert!(ScalarFunc::Upper.is_null_propagating());
        assert!(!ScalarFunc::Coalesce.is_null_propagating());
        assert!(ScalarFunc::Sqrt.may_introduce_null());
        assert!(!ScalarFunc::Abs.may_introduce_null());
    }

    #[test]
    fn window_names() {
        assert_eq!(WindowFunc::from_name("rank"), Some(WindowFunc::Rank));
        assert_eq!(
            WindowFunc::from_name("sum"),
            Some(WindowFunc::Aggregate(AggregateFunc::Sum))
        );
        assert!(WindowFunc::RowNumber.is_ranking());
        assert!(!WindowFunc::Aggregate(AggregateFunc::Sum).is_ranking());
        assert_eq!(WindowFunc::Rank.return_type(None).unwrap(), DataType::Int64);
        assert!(FunctionRegistry.names().len() > 30);
    }
}

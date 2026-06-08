# 0007. Explicit casts inserted by the binder

## Status

Accepted.

## Context

SQL is full of implicit conversions: comparing an `INT32` column to an `INT64` literal, summing a
mixed-width expression, unioning a `FLOAT64` column with an `INT32` one, inserting an `INT64` into a
`FLOAT64` column. Something has to decide what type the operands actually have.

The conversion can be resolved in one of two places.

**At execution time.** Each kernel promotes its operands as needed. This is what "implicit
conversion" usually means in practice, and its failure mode is subtle: the join operator, the filter
kernel and the aggregate each implement promotion separately, and they eventually disagree. A join
that promotes `int64 -> float64` on one side while the filter promotes `float64 -> int64` on the
other does not error — it returns a different set of rows. The bug reports that result are the worst
kind: intermittent, data-dependent, and about *results* rather than crashes.

**At bind time.** The binder resolves each conversion once, into a node the plan carries.

There is a related question for the function registry: does a kernel handle every input type it
might be given, or is it promised a fixed set?

## Decision

The binder inserts an explicit `Expr::Cast` wherever a conversion is needed, and nothing downstream
converts anything.

The rule this leaves for a future executor is one sentence: **the operands of a binary operator have
identical types.** If they do not, that is a binder bug, and `LogicalPlan::validate` reports it
rather than letting the executor improvise.

Where it applies:

* **Binary operators.** `align_types` finds the common type from the coercion lattice and casts both
  sides. For arithmetic, the *result* type is decided first and both operands are cast to it; for
  division, both are cast to `Float64`.
* **Predicates.** `WHERE`, `HAVING`, `JOIN ON`, `NOT`, `AND`/`OR` and `CASE WHEN` conditions must be
  `Boolean`. A `Null`-typed expression is cast; anything else is a type error, because SQL has no
  truthiness and `WHERE 1` is a mistake rather than a shorthand.
* **Function arguments.** `ScalarFunc::coerce_args` declares the type each argument *will* have when
  a kernel sees it, and the binder inserts the casts to make that true.
* **Aggregates.** `coerce_arg` widens up front — `sum` to `Int64`/`Float64`, `avg` and the variance
  family to `Float64` — so a kernel has one arm per accumulator rather than one per input type.
* **Set operations.** Each `UNION` arm is projected to the column's common type, so the executor
  never reconciles types across a union. `VALUES` does the same per column.
* **`IN` lists** are cast to the probe's common type; `INSERT` sources are cast to the target column
  types.
* **`BETWEEN`** is desugared into two comparisons at the same time, so the casts are visible in the
  two range predicates a pushdown rule will see.

Casts are only inserted where the lattice permits the conversion; `cast_to` refuses anything else
with a message naming the context, both types and the direction.

## Consequences

**The plan is auditable.** Every conversion is visible in `EXPLAIN`, which means "why did this
comparison behave like that" is answerable by reading the plan:

```text
Filter: (orders.total > CAST(10 AS FLOAT64))
Filter: ((CAST(orders.qty AS INT64) >= 1) AND (CAST(orders.qty AS INT64) <= 10))
Projection: CAST(qty AS FLOAT64) AS total
```

**One implementation of the rules, in one place.** The lattice is consulted by the binder and by
nobody else. Kernels get exactly the types they asked for.

**Errors move earlier and get better.** `cannot insert INT64 into column status of type UTF8` is a
bind error naming the column, not a half-written table. `> is not defined between TIMESTAMP(us) and
INT64` names both types and the operator.

**`validate` can check it.** Join keys must have equal types; union arms must match the declared
types; projection expressions must match their declared column types. These checks are only possible
because "the types already agree" is an invariant rather than a hope, and each has a hand-built
counterexample in `crates/basalt-plan/tests/plan_display.rs`.

**Costs.**

* Plans are more verbose. `CAST(orders.qty AS INT64)` appears in the output where another engine
  would print `orders.qty`, which is noisier to read — and is the point.
* Casts of literals are not folded, because folding belongs in the optimizer that does not exist
  yet. `CAST(10 AS FLOAT64)` will be evaluated per batch until it is.
* Structural expression matching (used to match `GROUP BY` keys against select-list expressions) has
  to see the same casts on both sides. It does, because both go through the same binder path, but it
  is a coupling worth knowing about.
* A future executor cannot be lenient. If a plan ever arrives with mismatched operand types, the
  correct behaviour is to fail, not to promote.

# 0006. Column identities rather than positions in the plan

## Status

Accepted.

## Context

A resolved expression has to say which column it reads. There are two ways to say it:

* **By position** — an index into the operator's input schema. This is what an executor ultimately
  wants, because reading column 3 of a batch is an array index.
* **By identity** — a name that is unique within the plan and independent of where the column sits.

Positions are attractive because they need no lookup at execution time. But an optimizer's most
valuable rules all change the shape and width of intermediate schemas:

* **Column pruning** removes columns from a projection or a scan, shifting every position after it.
* **Join reordering** swaps sides, moving whole ranges of positions.
* **Predicate pushdown** moves a filter under a node with a different schema entirely.

Under positional references, every one of those rewrites must renumber every expression in the
subtree below it. A missed renumbering does not crash: it reads a different column of the right
type, and the query returns a **silently wrong answer**. That is the worst failure mode a database
has, and it is a bug class that positions create structurally.

## Decision

Every column produced anywhere in a plan gets a `ColumnId(u32)` from the binder's generator. Ids are
unique within a statement and never reused. `Expr::Column` and `Expr::OuterColumn` carry a
`ColumnRef { id, qualifier, name, data_type, nullable }` in which **only `id` is semantic** — the
qualifier and name are for display, and the type and nullability are cached so that `Expr::data_type`
never needs the schema.

`PlanSchema` maps identity to position with `index_of(id)`, a linear scan, for the executor's
compile step.

Consequences of the "never reused" rule that are load-bearing:

* Each reference to a CTE gets a *fresh* projection with fresh ids, so a predicate pushed into one
  reference cannot land in the other.
* A `SubqueryAlias` requalifies with fresh ids, so `s.x` and the inner `x` are distinct columns.
* `Expr::columns()` excludes `OuterColumn`, so column pruning never asks a child for a column that
  belongs to an enclosing query block.

## Consequences

**Rewrites become structural.** An expression that referred to `#7` still refers to `#7` however the
schema around it moved. A rule that reorders a join does not have to touch a single expression.

**Wrong-column bugs become impossible-by-construction rather than caught-by-review.** The remaining
failure mode is referring to an id the input does not produce, which is *not* silent:
`LogicalPlan::validate` reports "`Filter` references column #99 which its input does not produce".

**The executor pays one lookup per expression at compile time**, never per row. `index_of` is a
linear scan because schemas are tens of columns wide and the lookup happens once; a hash index would
cost more in complexity than it saves.

**Plans are more verbose.** Every field carries an id, a qualifier and a name, and `PlanField` is
therefore bigger than a `Field`. `EXPLAIN` prints names rather than ids to stay readable, which is
why validation messages — which print `#99` — look different from plan output.

**Debugging needs both views.** An id alone (`#12`) is meaningless to a reader, which is why
`ColumnRef` keeps the name for display and why validation errors quote the rendered expression as
well as the id.

**A future optimizer must respect the invariant.** Any rule that synthesises a new output column has
to allocate a fresh id rather than reuse one; the generator lives in the binder today, so a rule
framework will need access to one. That is a known piece of design debt, recorded here rather than
discovered later.

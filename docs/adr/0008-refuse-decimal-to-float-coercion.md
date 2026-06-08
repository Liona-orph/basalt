# 0008. Refuse implicit decimal-to-float coercion

## Status

Accepted.

## Context

`coerce(a, b)` must answer what happens when a `DECIMAL(p, s)` meets a `FLOAT64`. Every mainstream
SQL dialect answers "promote to double", and the operation succeeds.

That answer is convenient and wrong in a specific way. `DECIMAL` exists because someone needed exact
base-10 arithmetic — money, tax rates, quantities that must sum to a stated total. `FLOAT64` cannot
represent `0.1`. Promoting a decimal to a double converts an exact value into an approximation
*silently*, in the middle of an expression the user did not think was lossy, and the error then
propagates through a `SUM` over a billion rows.

The two type systems have no common exact supertype. There is no width of decimal that contains all
doubles and no float that contains all decimals; the promotion is a choice about which kind of
wrongness to accept, not a widening.

There is a wrinkle that makes this far more visible in Basalt than it would be elsewhere: a
fractional literal is a `DECIMAL`, not a float. `1.5` lexes as `DECIMAL(2, 1)` because the parser
keeps every digit the user wrote, so `WHERE float_col > 10.5` is decimal-meets-float.

## Decision

`coerce(Decimal128(p, s), Float64)` and `coerce(Decimal128(p, s), Float32)` return `None`. Rule 9 of
the coercion lattice, stated in the documentation of `basalt_types::coerce`:

> **Decimal with float.** `None`, deliberately. Binary floating point and decimal have no common
> exact type; the usual "promote to double" rule destroys the exactness that made someone choose
> `DECIMAL` in the first place. Make the cast explicit.

Because the binder resolves conversions through the lattice (ADR 0007), this surfaces as a type
error naming both types:

```text
SELECT total FROM orders WHERE total > 10.5
  type error: > is not defined between FLOAT64 and DECIMAL(3, 1)

SELECT amount + total FROM payments, orders
  type error: operator + is not defined for DECIMAL(10, 2) and FLOAT64
```

Decimal *does* coerce with integers (the integer becomes `DECIMAL(digits, 0)` and decimal
reconciliation applies), and decimal with decimal keeps every integral digit and the finer scale, so
`amount > 3` against a `DECIMAL(10, 2)` column widens both sides to `DECIMAL(21, 2)` and stays
exact.

## Consequences

**Exactness is preserved by default.** A `DECIMAL` column cannot become approximate without someone
writing `CAST`. That is the property the type exists to provide, and Basalt is the layer responsible
for not throwing it away.

**The surprise is real and lands on innocent-looking SQL.** `WHERE float_col > 10.5` is rejected.
Anyone arriving from PostgreSQL or DuckDB will hit it, and the honest description is "this is a
deliberate deviation that costs you a cast". The mitigations are that the message names both types
and the operator, and that the fix is mechanical: `CAST(10.5 AS FLOAT64)`, or make the column a
`DECIMAL`.

**It is asymmetric with the one lossy promotion the lattice does allow.** `Int64 -> Float64` loses
precision above `2^53` and is permitted, because refusing it would break `avg(bigint_column)` and
most ordinary SQL. The difference is one of degree and of intent: integers are routinely averaged,
and `is_lossless_coercion` reports the promotion so a planner can warn; decimals are chosen
*specifically* to avoid binary floating point, so silently undoing that choice is a different kind
of mistake.

**Related refusals follow the same principle** and are recorded together here because they come from
the same conviction — that a coercion which loses information should be written down:

* `UInt64` with any signed integer is `None` (`Int64` cannot hold `u64::MAX`, `Float64` would round
  it).
* Boolean with numeric is `None` (`WHERE flag = 1` against a boolean column is a bug).
* `Date32` with any integer is `None` (`date_col > 5` is a bug).
* Decimal with decimal is `None` when the result would need more than 38 digits, rather than
  truncating the scale — dropping fractional digits is exactly the failure decimals exist to
  prevent.

**If this is ever reversed**, the right change is not to add the coercion to the lattice but to add
an explicit opt-in (a session flag or a dialect mode) that inserts the cast for you, so that the
default stays exact and the deviation stays visible.

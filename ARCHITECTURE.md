# Architecture

How Basalt is put together, and why each piece is shaped the way it is. This document assumes you
have read the [README](README.md) — in particular the part where it says there is no optimizer and
no executor, which is why this document describes a *data model and a front end* rather than a
query engine.

Every claim here is about the code as it exists. Where a design anticipates a component that has not
been written, it says so.

---

## 1. The data model

### 1.1 Logical types

`basalt_types::DataType` is the logical type of a column. "Logical" is load-bearing: `Date32` and
`Int32` are both an `i32` in memory, but `date_col > 5` is a bug and `int_col > 5` is a query, so
the two are different types and the coercion rules treat them differently. The physical layout lives
on `ColumnVector`, and the mapping type → layout is deliberately not injective.

The set is small and closed: `Null`, `Boolean`, eight sized integers, `Float32`/`Float64`, `Utf8`,
`Binary`, `Date32`, `Timestamp(TimeUnit)` and `Decimal128(precision, scale)`. There is no nested
type. Adding lists and structs is not hard; it is *pervasive* — every kernel, every codec and every
zone map grows a case — and none of the questions this project is about need them.

`Decimal128` carries a validated `(precision, scale)` with `1 <= precision <= 38` and
`0 <= scale <= precision`. Negative scale (a multiplier) is rejected: it is rare, it doubles the
number of cases in every decimal code path, and nothing here needs it.

### 1.2 The coercion lattice

`coerce(a, b) -> Option<DataType>` is the single implicit-conversion rule for the whole engine. It
is commutative and idempotent (both property-tested), and it is written as twelve numbered rules in
the documentation of the function, so that a disagreement with it is a disagreement with a specific
sentence. The load-bearing ones:

* `Null` is the bottom of the lattice, so `SELECT NULL UNION SELECT 1` types as `Int64`.
* Mixed signedness widens to the narrowest *signed* type containing both domains: `UInt16 + Int8` is
  `Int32`. `UInt64` mixed with any signed type has **no** common type, because `Int64` cannot hold
  `u64::MAX` and `Float64` would round it.
* Integer to float: `Float32` absorbs integers up to 16 bits (a `binary32` is exact below `2^24`),
  anything wider goes to `Float64`. `Int64 -> Float64` is the *only* lossy promotion in the lattice.
  It is allowed because refusing it would break `avg(bigint_column)`, and `is_lossless_coercion`
  reports it so a future planner can warn.
* Decimal with decimal keeps every integral digit and the finer scale; if the result needs more than
  38 digits of precision the answer is `None` rather than a truncated scale.
* **Decimal with float is `None`.** See [ADR 0008](docs/adr/0008-refuse-decimal-to-float-coercion.md).
* Boolean with numeric is `None`. `WHERE flag = 1` against a boolean column is a bug worth catching.
* `Utf8` with `Binary` is `Binary`: UTF-8 is a subset of bytes and the byte order agrees with the
  code-point order, so the embedding is exact *and* order-preserving.
* Temporal: `Date32 + Timestamp(u)` is `Timestamp(u)` (a date is midnight UTC), and two timestamps
  take the finer unit so no sub-second information is dropped. Dates and integers do not coerce.

### 1.3 `Value`

`Value` is an owned scalar with a genuine total order — `Ord`, not just `PartialOrd`. That single
decision means a value can be a sort key, a hash key, a zone-map bound and a group key without three
different comparison functions that will eventually disagree. Floats compare by a total order over
their bit patterns (so `NaN` has a definite place), which is also why `NativeType::total_eq`
compares float arrays bitwise: without it, a round trip through the storage layer could not be
asserted equal.

---

## 2. The columnar representation, and why

### 2.1 Dense values plus a bitmap

A `PrimitiveArray<T>` is a `Buffer<T>` of values plus an `Option<Bitmap>` validity mask. Two choices
are worth defending.

**The values buffer is dense.** A `T` is stored for every slot, including null ones, whose contents
are unspecified. The alternative — compacting the valid values and indirecting through a rank
operation — saves memory on mostly-null columns and costs a popcount on *every* read. Dense storage
makes the array a plain `&[T]` that the compiler can auto-vectorise, and mostly-null columns are
handled far better one layer down, where the encoder turns them into almost nothing anyway.

**The mask is a bitmap with `1 = valid`, and `None` when there are no nulls.** One bit per row is
eight times smaller than a byte mask, and eight times more rows fit in cache. The polarity matters:
with `1 = valid`, "is this row usable across all its inputs" is a plain `AND` of masks. And `None`
rather than an all-ones bitmap keeps the encoding canonical — the same logical column always
produces the same bytes on disk — as well as saving the allocation.

`Bitmap` is `Arc<Vec<u64>>` plus a bit offset and a bit length, so `slice` is O(1) and copies
nothing. A slice may start mid-word, so reads go through a helper that shifts two adjacent words
together: two loads and two shifts per 64 bits, still far cheaper than materialising a copy. Words
are 64 bits so that `count_ones` is one `popcnt` and iterating set bits is one `tzcnt` per bit.

`Bitmap::set` is copy-on-write. Mutating a shared bitmap clones it; mutating an unshared one does
not. This is what makes it safe to hand bitmaps around by `Arc` without a mutability audit.

### 2.2 Variable-length data: offsets plus one buffer

`BinaryArray` (and `StringArray`, which is the same layout with a UTF-8 guarantee) is
`offsets: Buffer<i32>` of length `n + 1`, `data: Buffer<u8>`, and the usual validity mask. Value `i`
is `data[offsets[i]..offsets[i+1]]`.

The alternative is `Vec<Vec<u8>>` — one allocation per value. For 65 536 short strings that is
65 536 allocations, 65 536 pointer chases per scan, and an allocator-defined memory layout. With
offsets there is exactly one allocation for the bytes and one for the offsets, the bytes are
contiguous (so a scan is sequential and prefetchable), and slicing an array is still O(1) because a
slice is a window into the same two buffers.

The cost is that appending is the only cheap mutation and that offsets are `i32`, capping one array
at 2 GiB of data. Both are the right trade for an analytical engine whose arrays are built once and
read many times. The invariants (non-empty, non-decreasing, non-negative, within `data`) are checked
once in `try_new`, in O(n), so every read afterwards is unchecked.

`decode_column` currently materialises a `Vec<Vec<u8>>` before rebuilding a variable-length array —
exactly the pattern this layout exists to avoid. It is the clearest remaining optimisation in the
encoding crate, and it is called out in `BENCHMARKS.md` rather than quietly fixed, because
correctness came first.

### 2.3 `RecordBatch` and `Selection`

A `RecordBatch` is an `Arc<Schema>` and equal-length columns, validated once at construction so that
every operator downstream can trust it.

`Selection` is the late-materialisation vector: the set of live rows as *indices*, not data. A scan
that reads 64 Ki rows and applies three predicates should copy the survivors once, not three times,
so each predicate narrows a `Selection` and the copy happens where an operator genuinely needs
contiguous data. `Selection::All(n)` is a variant rather than a materialised `0..n`, because an
unfiltered scan is the common case and representing it as a `Vec` would cost a large allocation and
destroy the sequential access pattern for nothing. Indices are strictly increasing, which keeps
`intersect`/`union` linear and preserves source order.

---

## 3. Encoding: a size estimator and an argmin

`basalt-encoding` implements seven integer codecs (plain, varint, bit-packed, frame-of-reference,
run-length, dictionary, delta-binary-packed) and two byte codecs (plain and an in-tree LZ77), all
behind one `Codec` trait, all writing a self-describing header.

The interesting part is not the codecs, it is **how one is chosen**:

1. `analyze` makes **one linear pass** over the chunk and accumulates: min, max, first value, run
   count, sortedness, the exact total varint length, the exact total run-length body size, the
   largest zigzag magnitude, and a distinct-value set that stops growing once a dictionary is out of
   the question (above `2^14` entries).
2. `estimate_sizes` turns those statistics into a **closed-form byte count for every candidate**.
   Most of the estimates are *exact*, not approximations, because the pass accumulated the real
   sizes. Only delta is approximate, and deliberately conservative: it uses the global delta range
   where the codec uses per-block ranges, so delta only wins when it wins by more than the slack.
3. The smallest estimate wins, ties broken toward the encoding that decodes fastest.
4. The winner must beat `Plain` by a configured margin — 20 % at compression levels 1-3, 5 % at 4-6,
   1 % at 7-9 — or `Plain` is used anyway.

Step 4 is the one people skip. Compression is not free at read time: a 2 % size win that makes a
scan 3x slower is a bad trade for an engine whose queries are scan-bound. And step 2 is why the
selector does not embarrass itself on random data, where every "clever" encoding is *larger* than
plain (varint by 19 %, RLE by 31 % — see `BENCHMARKS.md`). A rule-based selector that pattern-matched
on "looks sorted" would make that data bigger; an estimator simply picks plain.

Because `estimate_sizes` is a pure function of `IntegerStats`, the whole decision procedure is unit
testable without encoding anything, and each encoder can be checked against its own estimate. That
property test — "the estimate matches reality" — is what keeps the heuristic honest as codecs change.

The byte codec does the same thing empirically rather than analytically: `encode_bytes_adaptive`
runs LZ77, compares, and keeps the plain form when compression does not pay. That is why
incompressible input costs 0.4 % instead of the token stream's worst case.

---

## 4. Storage: the segment, and pushdown

A **segment** is one immutable columnar file. The normative byte-level description is
[`FORMAT.md`](crates/basalt-storage/FORMAT.md); this is the shape of it.

```text
+---------------------------+
| file header      16 bytes |  magic "BSLT", format_version = 1
+---------------------------+
| BLOCK  rg 0, col 0 data   |  [u32 len][u32 crc32c][payload]
| BLOCK  rg 0, col 0 bloom  |  optional
| BLOCK  rg 0, col 1 data   |
| ...                       |
+---------------------------+
| BLOCK  rg 1, col 0 data   |
| ...                       |
+---------------------------+
| BLOCK  footer             |  schema + every chunk's metadata
+---------------------------+
| trailer          16 bytes |  footer offset, footer length, magic
+---------------------------+
```

Four properties fall out of that layout.

**The footer is written last**, because a chunk's offset is only known once it has been written.
That means the writer needs `Write`, not `Write + Seek`, and can therefore target a pipe or an
object-store upload. The trailer at the end is how a reader finds the footer in one 16-byte read.

**Every block is CRC32C'd** (Castagnoli, the polynomial iSCSI and modern filesystems use), over the
payload only. A mismatch is a `BasaltError::Corruption` naming the region, the absolute byte offset,
the stored checksum and the computed one — never a plausible-looking wrong answer.
See [ADR 0004](docs/adr/0004-crc32c-per-block.md). The 256 MiB cap on a block payload is part of the
format so that a damaged length field cannot make a reader allocate arbitrarily *before* the
checksum has a chance to reject it.

**Projection pushdown is addressing, not filtering.** Each column chunk has its own offset and
length in the footer, so reading one column of forty reads one fortieth of the bytes. This is
verifiable rather than asserted: wrap the source in a `CountingSource` and compare.

**Predicate pushdown is metadata first.** Every chunk carries a `ZoneMap` — min, max, null count,
estimated distinct count — and optionally a `BloomFilter`. A `Predicate` is evaluated against the
metadata *before any data byte is read*:

* Zone maps are already in memory (they are in the footer), so they are free. `min`/`max` ignore
  nulls entirely and `null_count` carries the rest of the story, which is what makes `IS NULL`
  pruning work; folding nulls into the bounds as if `NULL` were negative infinity would stop
  `col > 5` from skipping an all-null chunk.
* Bloom filters cost one small read, so they are consulted only for equality predicates that
  survived the zone-map stage. The filter's one guarantee is asymmetric: `false` is a proof of
  absence (skip the block), `true` only means "read it and check". Sizing is the classic
  `m = -n ln p / (ln 2)^2`, and the `k` probes come from `h1 + i*h2` double hashing, which Kirsch
  and Mitzenmacher showed has the same asymptotic false-positive rate for one hash instead of `k`.

The distinct-count estimate in a zone map comes from an 8192-bit linear-counting sketch (1 KiB,
transient), which keeps the relative error under about 2 % in the range where the answer changes a
decision: whether to build a bloom filter, and how big.

`SegmentSource` is the read abstraction. `FileSource` implements it with a `RefCell<File>` and
seek-then-read, which makes it `Send` but **not `Sync`** — one reader per thread. A parallel scan
wants positioned reads instead; the trait is already shaped for that change.

---

## 5. From SQL text to a logical plan

```text
text ──tokenize──> tokens+spans ──parse──> AST ──bind──> LogicalPlan ──validate──> (nothing yet)
        lexer.rs                 parser.rs        binder.rs        logical.rs
```

### 5.1 Lexer and parser

Hand-written, both of them. `tokenize` produces tokens carrying byte `Span`s; `parse` is
recursive-descent with one function per grammar production, except for expressions, which are a
Pratt (precedence-climbing) loop over a documented precedence table.

The reason is stated in [ADR 0005](docs/adr/0005-hand-written-parser.md) and is worth repeating: a
parser is judged by its errors. A generated parser can tell you a token was unexpected; this one
tells you what it wanted, what it got, where, and — when the fix is unambiguous — what to do:

```text
error: expected FROM, ... found identifier `form`
 --> 1:10
  |
1 | SELECT * FORM orders
  |          ^^^^
  = hint: did you mean `FROM`?
```

The whole input is tokenized up front rather than lexed on demand: SQL statements are small, and
having the full token vector available is what makes multi-token lookahead (is this `(` the start of
a subquery? is this `t.*` a qualified wildcard?) straightforward instead of needing a push-back
buffer.

The AST records *meaning*. Spans ride along for diagnostics but take no part in equality;
parentheses are not stored, because the tree structure already says where they were; a numeric
literal keeps its exact spelling, because the spelling is what decides its type (`1.5` is a
`DECIMAL(2, 1)`, `1e0` is a `FLOAT64`). Printing round-trips: `parse(print(parse(sql))) ==
parse(sql)` is a property test.

The parser knows nothing about tables, types or `*` expansion. Keeping that boundary sharp is what
lets it be tested exhaustively against text alone — see `crates/basalt-sql/tests/`.

### 5.2 The catalog

`Catalog` maps a case-insensitive name to a `TableDefinition`: a schema plus `TableStatistics`.
Statistics live in the catalog rather than in the (unwritten) optimizer, for two reasons. A plan is
only as good as the numbers it was costed with, so hiding them inside the optimizer would make it
impossible for the storage layer to keep them current. And the four numbers a `ColumnStatistics`
holds are deliberately the same four a `ZoneMap` holds, because that is where they come from:
merging a written segment's zone maps into the catalog is a fold, not a second scan.

Distinct counts merge with `max`, not `+`: two segments' `country` columns overlap heavily, and
summing would make every equality look far more selective than it is. `max` is the tight lower bound
and errs toward *less* selectivity, which is the safe direction for join ordering.

### 5.3 The binder

`bind_statement` answers everything the parser refused to: which table is `o`, which column is
`total`, what type is `a + b`, is `count(x)` legal *here*.

**Scopes.** Name resolution walks a chain of scopes held by reference, so binding a deeply nested
correlated subquery costs one pointer per level rather than one schema copy. Resolution reports the
depth at which it found a name: depth 0 is the current query block and becomes `Expr::Column`;
anything deeper becomes `Expr::OuterColumn`, which is exactly what marks a subquery as correlated —
the information a decorrelation rule will need. `Expr::columns()` excludes outer references on
purpose, so that column pruning never asks a child for a column the child does not have.

**Visible schema vs. output schema.** `bind_from` returns both the plan and the schema that *name
resolution* should see, and they are not always the same. `JOIN ... USING (id)` produces both `id`
columns in the join node's output — which is what `join_schema` says, and what `validate` checks —
while only one of them is visible as a bare `id`. Keeping the two apart is what lets the join node
stay consistent and `SELECT id` still resolve.

**Aggregates are hoisted.** `sum(x) + 1` is an addition over the *output* of an aggregate node that
sits below the projection, so binding hoists every aggregate into a sink and leaves a column
reference behind. Clauses where an aggregate is illegal (`WHERE`, `GROUP BY`, `JOIN ON`, `ORDER BY`)
pass a *forbidding* sink, which turns the mistake into an error naming the clause instead of a
confusing plan. An aggregate written twice is stored once: `sum(x)/count(x)` appearing in both the
select list and `HAVING` must produce identical values, and sharing the output column makes that
true by construction rather than by luck.

Group-key matching is **structural**: `GROUP BY a + b` makes `a + b` — and nothing else — legal
above the aggregate. Basalt does not try to prove that `b + a` is the same expression, because a
rule that is clever in some cases and not others is harder to explain than one that is simple.

**Casts are made explicit here and nowhere else.** See §6.

### 5.4 The logical plan

Fourteen node kinds, each carrying the schema it produces. Children are behind `Arc`, so a rewrite
that leaves a subtree alone shares it instead of cloning it — a ten-rule fixpoint over a deep plan
would otherwise copy every leaf ten times.

`with_new_children` recomputes the schema for nodes whose output depends on their input's shape (a
join) and preserves it for nodes that own it (a projection): replacing a child under a projection
must not silently change what the query returns, and a rule that does that is the thing that is
wrong. `join_schema` is the single place the outer-join nullability rule lives, because the binder
and a future join-reordering rule must arrive at the same answer, and two copies of that rule would
eventually disagree — with the symptom being a `NOT NULL` column that is sometimes null.

`validate()` checks that a tree is well-typed and internally consistent: every expression only reads
columns its input produces, predicates are boolean, projection expressions match their declared
types, join schemas match `join_schema`, join keys come from the right sides and have equal types,
union arms match the declared arity and types, and window frames are ordered. In a correct build it
never fires — the binder is supposed to produce only valid plans and every future rule is supposed
to preserve validity. It exists because "supposed to" is not a guarantee: a rule that drops a column
its parent still references produces a plan that looks fine and would execute into a panic. Running
`validate` after every rule turns that into an error naming the rule, at the moment it happened.
`crates/basalt-plan/tests/plan_display.rs` hand-builds each of those broken plans and asserts the
message.

`Display` is `EXPLAIN`: one operator per line, two spaces per level, and enough detail on the line
to answer "why is this slow" without a second command — join keys and residual filter, group keys
and aggregates, the scan's projection and pushed-down filters, the window's partition/order/frame.

---

## 6. Decisions worth defending

### 6.1 Column identities, not positions

A column reference could be a *position* into the operator's input schema (cheap to execute) or a
unique *identity* (cheap to rewrite). Basalt uses identities: `ColumnId(u32)`, handed out by the
binder, never reused within a statement.

The reason is that the optimizer's most valuable rules — column pruning, join reordering, predicate
pushdown — all change the shape and width of intermediate schemas. With positions, every one of
those rewrites has to renumber every expression in the subtree beneath it, and a single missed
renumbering is a *silently wrong answer* rather than a crash. With identities the rewrites are
structural: an expression that referred to `#7` still refers to `#7` no matter how the schema around
it moved.

The cost is one `id -> index` lookup per expression when the executor compiles a plan — once, at
compile time, never per row — and `PlanSchema::index_of` is a linear scan because schemas are tens
of columns wide and an index would cost more complexity than it saves.

It also gives a cheap correctness property for free: two references to the same CTE get *fresh*
identities, so a predicate pushed into one cannot land in the other. That is asserted in
`bind.rs::two_references_to_one_cte_do_not_share_column_identities`.

Full argument: [ADR 0006](docs/adr/0006-column-identities-not-positions.md).

### 6.2 Explicit casts, inserted at bind time

SQL's implicit conversions are a source of quiet wrongness. An engine that compares an `INT64` to a
`FLOAT64` by promoting at execution time will do it slightly differently in the join operator, the
filter kernel and the aggregate, and eventually the three disagree — for example about which rows
match, which is a wrong answer rather than an error.

Basalt resolves every conversion once, at bind time, into an explicit `Expr::Cast`. The rule the
executor then implements is exactly one sentence: **the operands of a binary operator have identical
types.** If they do not, that is a binder bug, and `validate()` reports it instead of letting the
executor improvise.

This is visible in every plan:

```text
Filter: (orders.total > CAST(10 AS FLOAT64))
Filter: ((CAST(orders.qty AS INT64) >= 1) AND (CAST(orders.qty AS INT64) <= 10))
Aggregate: groupBy=[] aggregates=[avg(CAST(orders.qty AS FLOAT64)) AS avg(CAST(orders.qty AS FLOAT64))]
Projection: CAST(qty AS FLOAT64) AS total          -- a UNION arm
```

The same rule applies to function arguments (the registry declares what each argument *will* be by
the time a kernel sees it), to aggregate accumulators (`sum` widens to `Int64`/`Float64` up front so
the kernel has one arm per accumulator instead of one per input type), to `UNION` arms (both are
projected to the common type, so the executor never reconciles types across a union) and to
`INSERT` (the cast is inserted at bind time, so an impossible conversion is a bind error naming the
column rather than a half-written table).

Full argument: [ADR 0007](docs/adr/0007-explicit-casts-at-bind-time.md).

### 6.3 Refusing lossy coercions

The lattice says `None` more often than a typical SQL dialect does, and each refusal is a decision:

* **Decimal with float.** No common exact type exists. The usual "promote to double" rule destroys
  the exactness that made someone choose `DECIMAL`. Because a fractional literal is a `DECIMAL`,
  this catches the innocent-looking `WHERE float_col > 10.5` — which is startling the first time and
  correct every time. [ADR 0008](docs/adr/0008-refuse-decimal-to-float-coercion.md)
* **`UInt64` with any signed integer.** `Int64` cannot represent `u64::MAX` and `Float64` would
  round it. A caller who wants the "just make it a double" behaviour asks for it explicitly.
* **Boolean with numeric.** Implicit `true -> 1` hides real bugs and no dialect Basalt targets
  requires it.
* **Date with integer.** `date_col > 5` is a bug, not a query.
* **Integer division.** `5 / 2` is `2.5`. Truncating division is the single most common source of
  quietly wrong analytics; `%` keeps the integer type, where truncation is the point.

The one lossy promotion that *is* allowed — `Int64 -> Float64` — is allowed because refusing it
would break ordinary SQL, and it is reported by `is_lossless_coercion` so that a planner can warn
rather than pretend.

### 6.4 A closed function registry

`ScalarFunc`, `AggregateFunc` and `WindowFunc` are enums, not a bag of trait objects. A registry of
`Box<dyn ScalarFunction>` looks more extensible, but the extensibility is fake in an engine whose
executor must also carry a vectorised kernel for each function: registering a function without a
kernel would compile fine and fail at run time. With a closed enum, "every function has a kernel" is
`match` exhaustiveness — a compile error, in the right crate, the moment a variant is added. The
cost is that a user cannot add a function without touching Basalt, which is the honest state of
affairs anyway.

The registry is a *contract* with three clauses: `coerce_args` says what type each argument will
have when a kernel sees it, `return_type` says what comes back, and `is_null_propagating` says
whether a null argument forces a null result — which is what lets a kernel compute over a whole
vector and fix the validity mask at the end instead of branching per row.

---

## 7. What is missing, structurally

The dashed boxes in the README's diagram are not decoration. Concretely:

* **No optimizer.** There is no rule framework, no cost model, no column pruning, no predicate
  pushdown *through* plan nodes (the storage layer's pushdown is a separate, working thing), no join
  reordering. The plan structures were designed for those rules — identities, `Arc` children,
  `transform_up`/`transform_down`, `validate` — but the rules do not exist.
* **No executor.** No kernels, no operators, no scheduling. `RecordBatch`, `Selection` and the
  segment reader are the pieces an executor would be built on, and they are tested on their own.
* **Nothing writes back to the catalog.** `CREATE TABLE` produces a `BoundStatement` that no one
  applies; the catalog is in-memory and populated by callers.

The honest summary is that Basalt is a complete storage and front-end stack with a hole in the
middle, and the hole is labelled.

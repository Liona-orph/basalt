# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, minor versions
may contain breaking changes.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-08-09

The first release: a columnar storage stack and a SQL front end, built from first principles. There
is **no optimizer and no executor** — statements are parsed and planned, not run. See
[Status and what is not implemented](README.md#status-and-what-is-not-implemented).

### Added

**`basalt-common`** — one `BasaltError` enum for the whole workspace, whose variants are categories
rather than call sites. Corruption reports carry the object, a structural `Region` and the absolute
byte offset. `Config` with documented defaults, `ByteSize` that parses and prints like a human wrote
it, `Id`/`IdGenerator`, and `IoResultExt` for attaching context to `std::io::Error`.

**`basalt-types`** — the logical `DataType` set (including `Decimal128` with validated precision and
scale, and `Timestamp(TimeUnit)`), a twelve-rule `coerce` lattice that is commutative and idempotent
by construction, and `is_lossless_coercion` for reporting the one lossy promotion the lattice
performs. `Value` with a genuine total order over every type, floats included. Columnar arrays —
`PrimitiveArray`, `BooleanArray`, `StringArray`, `BinaryArray`, `NullArray` behind `ColumnVector` —
with dense value buffers, bitmap validity masks (`1 = valid`, `None` for no nulls) and O(1) slicing.
`Bitmap` with copy-on-write mutation, `Buffer`, runtime-typed `ColumnBuilder`s, `Schema`/`Field`,
`RecordBatch`, and `Selection` for late materialisation.

**`basalt-encoding`** — bit packing at arbitrary widths 1..=64, zigzag LEB128 varints, and the
codecs: plain, varint, bit-packed, frame-of-reference, run-length, dictionary and
delta-binary-packed over `i64`, plus a plain and an in-tree LZ77 codec over bytes. A
statistics-driven selector (`analyze` → `estimate_sizes` → argmin, with a margin plain must lose by)
that picked the smallest candidate on all seven benchmark shapes. `encode_column`/`decode_column`
for whole columns including nulls and variable-length data. Measured numbers, and the machine that
produced them, in `BENCHMARKS.md`.

**`basalt-storage`** — the version-1 segment format, specified byte for byte in `FORMAT.md`: file
header, blocks, row groups, footer and trailer. CRC32C (Castagnoli, implemented in-tree) over every
block payload, with corruption reported as an error naming a region and a byte offset. Per-chunk
`ZoneMap`s (min, max, null count, linear-counting distinct estimate) and optional `BloomFilter`s.
`SegmentWriter` needs only `Write`, so it can target a pipe. `SegmentReader` with projection
pushdown (column chunks are addressed individually) and predicate pushdown (zone maps first, bloom
filters only for surviving equality predicates), plus `CountingSource` for verifying the bytes not
read.

**`basalt-sql`** — a hand-written lexer producing tokens with byte spans, a recursive-descent parser
with a Pratt expression loop, an AST that records meaning, and a `Display` printer for which
`parse(print(parse(sql))) == parse(sql)` is a property test. Diagnostics carry a span, a message and
often a hint, including "did you mean" suggestions for keywords, type names and functions. Recursion
is bounded, so pathological nesting is a diagnostic rather than a stack overflow.

**`basalt-plan`** — a `Catalog` of tables with statistics deliberately shaped like zone maps, a
resolved `Expr` tree in which every column reference is a `ColumnId` and every node knows its type
and nullability, a closed `FunctionRegistry` (31 scalar functions, 9 aggregates, 3 ranking window
functions) that is a contract with the executor, a binder implementing scoping, correlated
references (`Expr::OuterColumn`), aggregate placement rules, `USING`/`NATURAL` column merging,
`ORDER BY` by alias and ordinal, set-operation coercion, window specs and CTEs — and a
`LogicalPlan` with `validate()` and an `EXPLAIN`-style `Display`.

Every implicit conversion is materialised as an explicit `Expr::Cast` at bind time, so the rule
downstream is "the operands of a binary operator have identical types" and nothing else.

### Known limitations

* No optimizer and no executor: **no query returns rows**. `INSERT` binds and casts but never
  writes; `CREATE TABLE` never updates the catalog.
* Parsed and then refused as unsupported: scalar subqueries, `EXCEPT`, `INTERSECT`,
  `WITH RECURSIVE`, query parameters, `CREATE VIEW`/`DROP VIEW`, `EXPLAIN` of a non-query, and
  three-part names.
* `ORDER BY` resolves only against the query's output columns.
* `USING`/`NATURAL` keep the left copy of a merged column rather than coalescing.
* `LIMIT`/`OFFSET` require a literal; `LIMIT 1 + 1` is rejected.
* `FileSource` is `Send` but not `Sync` — one reader per thread.
* `DECIMAL` does not implicitly coerce to a float, so `WHERE float_col > 10.5` is a type error.
* Codecs are scalar; there is no SIMD.
* No nested types (list, struct, map).

### Tooling

Pinned toolchain (Rust 1.83), workspace lints including `clippy::pedantic` and
`unwrap_used = deny`, `#![forbid(unsafe_code)]` in every crate, a `Makefile` whose `ci` target is
what CI runs, GitHub Actions for fmt/clippy/test/doc/bench, a scheduled `cargo audit`, and
Dependabot for cargo and actions.

[Unreleased]: https://github.com/Liona-orph/basalt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Liona-orph/basalt/releases/tag/v0.1.0

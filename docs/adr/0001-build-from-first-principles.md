# 0001. Build from first principles rather than on Arrow

## Status

Accepted.

## Context

Basalt needs an in-memory columnar representation, a compression layer, an on-disk format and a SQL
front end. Every one of those has a mature Rust implementation available: `arrow` for the memory
format, `parquet` for storage, `datafusion` for planning and execution, `sqlparser` for the front
end, `zstd`/`lz4` for compression.

Taking them would produce a working engine faster and, on almost any measurable axis, a better one.
The Arrow memory format is a specification with a large ecosystem; Parquet is a battle-tested format;
DataFusion's optimizer has rules Basalt will never have.

The purpose of this project, though, is to understand the machinery — the bit packer, the null mask,
the block framing, the coercion lattice, the name resolver — and to be able to defend each choice.
An engine assembled from those crates would be a demonstration of gluing, and the interesting
questions ("why 1 = valid?", "why estimate sizes instead of pattern-matching?", "why identities
instead of positions?") would all have the answer "because Arrow does".

## Decision

Implement the type system, columnar arrays, compression codecs, segment format, SQL parser and
query planner **in-tree**, with no database, compression or parsing crate as a dependency.

The workspace has exactly one runtime dependency, `thiserror`, and three dev-dependencies
(`proptest`, `criterion`, `tempfile`). The dependency policy — including the crates that were
considered and declined, each with a reason — is a comment block at the top of the workspace
`Cargo.toml`, so that adding a dependency requires editing the argument against it.

Specifically written in-tree: bit packing at arbitrary widths, zigzag varints, RLE, dictionary,
frame-of-reference, delta-binary-packed, an LZ77 byte compressor, CRC32C (the Castagnoli table is 30
lines), bloom filters, zone maps, the lexer, the parser, the AST printer, and the binder.

## Consequences

**What this buys.**

* Every design choice is *ours* to explain, and the module documentation can carry the argument
  rather than a link.
* The whole workspace is readable end to end. There is no layer where understanding stops at a
  crate boundary.
* A supply chain of one runtime dependency: no transitive C libraries, no version-resolution
  archaeology, no `unsafe` we did not write (in fact none at all — every crate is
  `#![forbid(unsafe_code)]`).
* Compile times and binary size are small enough to be uninteresting.

**What it costs.**

* Basalt cannot exchange data with the Arrow ecosystem without a conversion layer that does not
  exist. That is a genuine interoperability loss and would be the first thing to reconsider if the
  project ever wanted users rather than readers.
* Its segment format is not Parquet, so no other tool can read it.
* The codecs are scalar. There is no SIMD, and a NEON/AVX2 unpacker would plausibly triple
  `BITPACKED` and `FOR` decode throughput — see `BENCHMARKS.md`. Arrow's kernels would have been
  faster for free.
* Every bug in bit packing, offset arithmetic and CRC computation is a bug we had to find. The
  mitigation is property testing: codecs are tested by round trip and against reference
  implementations, and the storage layer is tested by mutating bytes and asserting the corruption
  is caught.
* The scope is deliberately narrower than the crates it declines: no nested types, no dictionary
  *arrays* in memory, no multi-file datasets.

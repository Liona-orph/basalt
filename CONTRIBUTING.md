# Contributing to Basalt

Thank you for looking. Before anything else, read the
[Status section of the README](README.md#status-and-what-is-not-implemented): Basalt has no
optimizer and no executor, and that shapes which contributions make sense.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## What this project optimises for

Basalt is a from-first-principles engine written to be *understood*. That makes the bar different
from a normal library:

* **A change is judged by its reasoning as much as its code.** If a function chooses between two
  reasonable designs, the module documentation should say which one and why the other was rejected.
  Look at any existing module for the register.
* **Honesty beats completeness.** A refused construct with a clear message is better than an
  approximation that is wrong about `NULL`. The unsupported list in the README exists to be accurate,
  not short.
* **Dependencies must earn their place.** The workspace has one runtime dependency. The policy, and
  the list of crates deliberately declined, is a comment at the top of `Cargo.toml`. A PR that adds
  a dependency should edit that comment, because the comment is the argument against it.

## Getting set up

The toolchain is pinned by `rust-toolchain.toml` (Rust 1.83, with `rustfmt` and `clippy`), so
`rustup` will fetch the right one automatically.

```sh
make help     # list every target
make ci       # fmt --check, clippy -D warnings, tests, docs, bench compile
```

`make ci` runs exactly what the CI workflow runs, in the same order. If it passes locally it will
pass in CI.

## The bar for a change

Every one of these must hold before a PR is ready:

1. `cargo fmt --check` is clean. Formatting is `rustfmt` defaults plus `rustfmt.toml`; do not
   hand-format.
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean. The workspace lints (including
   `clippy::pedantic`, `missing_docs` and `unwrap_used = deny`) are configured in the workspace
   `Cargo.toml`, with a written reason for each one that is switched off.
3. `cargo test --workspace` passes, including doctests.
4. `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace` is clean — broken intra-doc links
   are errors.
5. Every new public item has a `///` doc comment. `missing_docs` will tell you, but write something
   worth reading rather than restating the signature.
6. `#![forbid(unsafe_code)]` stays at the top of every crate. There is currently no `unsafe` in the
   workspace and a change that needs some needs a discussion first.
7. No `unwrap()` outside tests. `expect()` is allowed but rare, and always with a comment naming the
   invariant that makes it safe. Test modules opt out with an inner
   `#![allow(clippy::unwrap_used)]`, which keeps every exception greppable.

## Tests

New behaviour comes with tests, and the style matters here more than usual.

* **Unit tests** live in a `#[cfg(test)] mod tests` next to the code, for things that need access to
  internals.
* **Integration tests** live in `crates/*/tests/` and exercise the public API. The established
  pattern is a corpus plus a helper that asserts *specifically*:
  * `basalt-sql/tests/accept.rs` — parse, print, re-parse, compare.
  * `basalt-sql/tests/reject.rs` — assert the message **and** the exact text the span underlines.
  * `basalt-plan/tests/bind.rs` — assert resolved types, nullability and the casts that were
    inserted, and validate every accepted plan.
  * `basalt-plan/tests/reject.rs` — assert the specific diagnostic, not `is_err()`.
  * `basalt-plan/tests/plan_display.rs` — `EXPLAIN` snapshots, plus hand-built invalid plans that
    `validate()` must catch.
* **Property tests** (`proptest`) are the right tool for codecs, bitmaps and anything with a
  round-trip or a reference implementation. Codec estimates are checked against reality this way.
* A test that only asserts `is_ok()` or `is_err()` is usually not pulling its weight. Assert what
  changed.

## Benchmarks

`crates/basalt-encoding/BENCHMARKS.md` contains **measured** numbers with the machine and toolchain
that produced them. If a change moves them, re-run `cargo bench -p basalt-encoding` and update the
file, including the environment table if your machine differs. Do not estimate a number into that
file.

## Commit messages and PRs

* Imperative mood, one line of subject, a body that explains *why* if it is not obvious.
* One logical change per PR. A refactor and a behaviour change in the same diff are hard to review
  and harder to revert.
* Fill in the pull request template; the "what did you consider and reject" box is the interesting
  one.

## Architecture decisions

A change that alters a documented design choice — the coercion lattice, the plan's column
identities, the segment format — should come with an ADR in [`docs/adr/`](docs/adr), in the Nygard
format the existing eight use. If it *reverses* one, add a new record that supersedes the old one
rather than editing history.

## Reporting bugs and asking for features

Use the issue templates. For a bug, the minimal SQL or the smallest input that reproduces it is
worth more than a description. For a feature, say what you are trying to do, not only what you want
built — several of the current limitations have workarounds that may be enough.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## License

Contributions are licensed under the [Apache License, Version 2.0](LICENSE), the same terms as the
project. You retain copyright in your contribution.

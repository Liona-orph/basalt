# What this changes

<!-- One paragraph. What is different after this PR that was not before? -->

## Why

<!--
The part that matters most in this project. If the change picks between two
reasonable designs, say which and why the other was rejected — and put that
reasoning in the code's documentation, not only here.
-->

## What I considered and rejected

<!--
Optional but encouraged. An empty box is fine for a typo fix; for anything that
touches the type system, the plan or the segment format it usually is not.
-->

## Checklist

- [ ] `make ci` passes locally (fmt, clippy `-D warnings`, tests, docs, bench compile).
- [ ] New public items have `///` docs that say something beyond the signature.
- [ ] New behaviour has tests that assert *what changed*, not just `is_ok()`.
- [ ] No `unwrap()` outside tests; any new `expect()` names the invariant in a comment.
- [ ] `#![forbid(unsafe_code)]` is still at the top of every crate touched.
- [ ] No new dependency — or the dependency policy comment in `Cargo.toml` is updated with the
      argument for it.
- [ ] Documentation updated: crate docs, `README.md`, `ARCHITECTURE.md`, `FORMAT.md` as applicable.
- [ ] `CHANGELOG.md` has an entry under *Unreleased* if this is user-visible.

## Affects

- [ ] The segment format (`FORMAT.md`) — this can make existing files unreadable.
- [ ] The coercion lattice or the type system — this can change what a query means.
- [ ] The logical plan or the binder's diagnostics.
- [ ] Benchmarked code paths — if so, `BENCHMARKS.md` was re-measured, not estimated.
- [ ] A documented design decision — if so, an ADR is added or superseded in `docs/adr/`.

## Related issues

<!-- Closes #... -->

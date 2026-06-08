# Architecture decision records

Each file here records one decision that is visible in the code, in
[Michael Nygard's format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):
**Title, Status, Context, Decision, Consequences**. An ADR describes the situation at the time the
decision was made and is not edited afterwards; a decision that turns out to be wrong gets a new ADR
that supersedes it, and the old one stays as the record of what was believed and why.

They exist because the *why* is the part that is expensive to reconstruct. A reader can see from the
code that `ColumnId` exists; only an ADR says what the alternative was and what it would have cost.

| # | Title | Status |
|---|---|---|
| [0001](0001-build-from-first-principles.md) | Build from first principles rather than on Arrow | Accepted |
| [0002](0002-columnar-vectors-with-bitmap-null-masks.md) | Columnar vectors with bitmap null masks | Accepted |
| [0003](0003-statistics-driven-encoding-selection.md) | Statistics-driven encoding selection | Accepted |
| [0004](0004-crc32c-per-block.md) | CRC32C per block, and failing loud on corruption | Accepted |
| [0005](0005-hand-written-parser.md) | A hand-written recursive-descent and Pratt parser | Accepted |
| [0006](0006-column-identities-not-positions.md) | Column identities rather than positions in the plan | Accepted |
| [0007](0007-explicit-casts-at-bind-time.md) | Explicit casts inserted by the binder | Accepted |
| [0008](0008-refuse-decimal-to-float-coercion.md) | Refuse implicit decimal-to-float coercion | Accepted |

## Writing a new one

Copy the structure of an existing record. Number it with the next free integer, keep the title a
sentence that states the decision, and put the honest cost in *Consequences* — an ADR whose
consequences are all positive is marketing, not a record.

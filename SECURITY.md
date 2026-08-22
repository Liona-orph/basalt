# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes — the only released line |

Basalt is pre-1.0 and has no released binaries. Fixes go to `main`.

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private reporting — *Security* → *Report a vulnerability* on
<https://github.com/Liona-orph/basalt>.

Please include:

* what the issue is and which crate it is in;
* the smallest input that reproduces it (a segment file, a SQL statement, a byte sequence);
* the toolchain and platform you saw it on;
* what you think an attacker could achieve.

You will get an acknowledgement within **3 working days** and an assessment within **10**. If the
report is accepted, we will agree a disclosure date with you, credit you in the changelog unless you
prefer otherwise, and publish the fix and an advisory together.

## Threat model

Basalt is an embedded library. It has no network surface, no server, no authentication and no
process boundary of its own: it runs inside the caller's process with the caller's privileges. That
makes the threat model narrow and specific.

**In scope — untrusted input is a real attack surface.**

* **Segment files.** A reader must survive an arbitrary byte sequence. Corruption, truncation and
  hostile length fields must produce a `BasaltError`, not a panic, an unbounded allocation or an
  infinite loop. The format caps a block payload at 256 MiB precisely so that a damaged length
  cannot make the reader allocate before the CRC has a chance to reject it, and the reader validates
  the header, the trailer and every block offset before using them.
* **SQL text.** The parser must not be made to overflow the stack; recursion is bounded by
  `MAX_RECURSION_DEPTH` and deeply nested input is a diagnostic. Pathological input should not take
  superlinear time.
* **Panics on untrusted input are vulnerabilities here**, because a panic in an embedded library
  takes the caller's process with it (or unwinds through it). Report them.

**Out of scope.**

* **A CRC is not a MAC.** CRC32C detects accidental damage. An attacker who can write the file can
  recompute it. Basalt does not authenticate segment files; if you need that, sign or encrypt them
  at a layer above.
* **Resource exhaustion from honest input.** A query planned over a very wide schema, or a segment
  legitimately containing a large column, will use memory proportional to it. `Config` exposes the
  relevant limits; setting them is the caller's job.
* **Anything about the executor**, which does not exist. There is no query execution, so there is no
  query-driven resource consumption to attack.
* **Correctness bugs with no security consequence.** A wrong answer is a bug — please report it as
  an issue, not as a vulnerability.
* **Dependencies.** There is one runtime dependency (`thiserror`); vulnerabilities in it belong
  upstream. `cargo audit` runs on a schedule in CI.

## Hardening already in place

* Every crate is `#![forbid(unsafe_code)]`. There is no `unsafe` in the workspace.
* `clippy::unwrap_used` is denied outside tests, and `expect` is used only for invariants
  established in the same module.
* All array constructors validate their invariants (offset monotonicity, buffer bounds, bitmap
  length against array length) once, at construction.
* Corruption errors carry the structural region and the absolute byte offset, so an operator can act
  on them without parsing a message.

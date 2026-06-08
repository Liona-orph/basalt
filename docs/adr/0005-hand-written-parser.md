# 0005. A hand-written recursive-descent and Pratt parser

## Status

Accepted.

## Context

Basalt needs to turn SQL text into a syntax tree. The options were a parser generator (LALRPOP,
pest, lalr tables), an existing SQL parser crate (`sqlparser`), or writing one.

SQL's grammar is not the difficulty. Expression precedence is a solved problem, and the statement
grammar is a few dozen productions. The difficulty is that **a SQL front end is judged almost
entirely by its error messages**. A user who writes valid SQL never thinks about the parser; a user
who mistypes `FORM` meets it immediately, and what it says next determines whether the tool feels
usable.

A generated parser reports failures in terms of its own machinery: "unexpected token IDENT at
position 9, expected one of {FROM, WHERE, ...}" — the expected set is the automaton's, which is
frequently either too large to read or missing the obvious suggestion. Adding a good message means
fighting the generator's error recovery, and adding a *hint* means threading context the generator
does not model.

A borrowed-AST parser crate has a second problem: the binder needs byte spans on every node, so that
`no column named total` can underline `total`, and the spans must survive into the plan-building
stage.

## Decision

Write the front end by hand:

* **`tokenize`** produces `Token`s carrying byte `Span`s. It handles identifier folding, nested
  block comments and `''` escaping, and it reports lexical problems in its own terms
  ("unterminated string literal", "`::` is not supported").
* **`parse`** is recursive descent, one function per grammar production, each named after it.
* **Expressions** are a Pratt (precedence-climbing) loop over the table documented on
  `ast::Precedence`.
* The whole input is tokenized up front rather than lexed on demand, because SQL statements are
  small and having the full token vector makes multi-token lookahead (is this `(` the start of a
  subquery? is `t.*` a qualified wildcard?) straightforward instead of requiring a push-back buffer.
* Recursion is bounded by `MAX_RECURSION_DEPTH`, so pathological nesting is a diagnostic rather than
  a stack overflow.
* Every diagnostic carries a span, a message, and — where the fix is unambiguous — a `hint`. Edit
  distance drives "did you mean" for keywords, type names and functions.
* The AST owns its data, records *meaning* (parentheses are not stored; spans are excluded from
  equality; a numeric literal keeps its exact spelling because that is what decides its type), and
  has a `Display` that round-trips.

## Consequences

**The errors are the product, and they are testable as such.** `crates/basalt-sql/tests/reject.rs`
asserts, for each rejected statement, the message *and* the exact text the span underlines — the
part that rots silently, since a parser can drift into reporting everything at end-of-input and
still pass a test that only checks `is_err()`.

```text
error: expected FROM, ... found identifier `form`
 --> 1:10
  |
1 | SELECT * FORM orders
  |          ^^^^
  = hint: did you mean `FROM`?
```

**Round-tripping is a property test, not a hope.** `parse(print(parse(sql))) == parse(sql)` holds
over the accepted corpus, which catches a printer that drops the same thing the parser dropped.

**Costs.** The parser is about 1 900 lines and the lexer about 1 000, all of which must be
maintained and tested by hand; a generator would have been a few hundred lines of grammar. Precedence
and associativity are asserted by tests rather than guaranteed by construction, so a wrong table
entry is a bug rather than a conflict report. And there is no formal grammar artifact to point at —
the grammar is the code, plus the corpus in `tests/accept.rs`.

**Scope is what we chose to write.** The supported SQL is a documented subset, not "whatever the
crate supports". Some of it parses and is then refused by the binder — scalar subqueries,
`EXCEPT`/`INTERSECT`, `WITH RECURSIVE`, parameters, views — which is a deliberate split: the parser
answers "is this valid SQL and what does it say", and the binder answers "can Basalt do it".

# 0003. Statistics-driven encoding selection

## Status

Accepted.

## Context

Basalt has seven integer codecs (plain, varint, bit-packed, frame-of-reference, run-length,
dictionary, delta-binary-packed) and two byte codecs. Something has to choose one per column chunk.

The usual approaches:

* **Fixed per column type.** Simple, and wrong often enough to be embarrassing: the right encoding
  for a monotonic key and for a random hash are not the same, and both are `INT64`.
* **User-specified.** Pushes the problem to someone with less information than the writer has.
* **Rules on shape.** "If sorted, delta; if few distinct values, dictionary; if long runs, RLE."
  Reads well, and fails on the cases where two rules are both true, on the cases where a rule is
  true but the win is small, and — worst — on high-entropy data, where every clever encoding is
  *larger* than plain (measured: varint by 19 %, RLE by 31 %).
* **Try them all and keep the smallest.** Exact, and pays the full encode cost seven times.

There is also a second-order problem the size question hides: compression is not free at read time.
Frame-of-reference and bit-packing decode at roughly half of plain's throughput because of the
shift-and-mask inner loop. A 2 % size win that makes a scan measurably slower is a bad trade for an
engine whose queries are scan-bound.

## Decision

Choose by **estimating the encoded size of every candidate in closed form, and taking the argmin**,
with a margin that plain has to lose by.

1. `analyze` makes one linear pass and collects: count, first value, min, max, run count,
   sortedness, the exact accumulated varint body size, the exact accumulated run-length body size,
   the largest zigzag magnitude, and a distinct-value set that stops growing past `2^14` entries
   (`Cardinality::AtLeast` says so).
2. `estimate_sizes` turns those statistics into a byte count per candidate. Every estimate except
   delta's is **exact**, because the pass accumulated the real sizes rather than approximating them.
   Delta's estimate is deliberately conservative: it uses the global delta range where the codec
   uses per-block ranges, so delta only wins when it wins by more than the slack.
3. Smallest wins; ties break toward the encoding that decodes fastest.
4. The winner must beat plain by `CompressionLevel::minimum_gain` — 20 % at levels 1-3, 5 % at 4-6,
   1 % at 7-9 — or plain is used anyway.

`estimate_sizes` is a pure function of `IntegerStats`, so the decision procedure is testable without
encoding anything.

For byte columns the same idea is applied empirically instead of analytically:
`encode_bytes_adaptive` runs the LZ77 compressor, compares, and keeps the plain form when
compression does not pay.

## Consequences

**It is honest, and measurably so.** On the seven benchmark shapes the selector picked the smallest
of the seven candidates in every case (`BENCHMARKS.md`). On `high_entropy` it picks plain, which is
exactly the case a rule-based selector gets wrong.

**It is testable.** The pure estimator supports a property test asserting that estimates match the
sizes the encoders actually produce. That test is what keeps the heuristic honest as codecs change,
and it is the reason the distinct-value count is exact rather than sketched.

**Analysis is the write path's bottleneck.** Measured at ~750 MiB/s, about 29x the cost of a plain
encode, dominated by the `HashSet<i64>` tracking distinct values: the two slowest shapes are exactly
the two with 65 536 distinct values. Reading is unaffected — the choice is recorded in the chunk
header — but a write-heavy workload pays for the decision procedure.

The obvious fix (sample the first N values, or swap the set for a HyperLogLog sketch) is left undone
on purpose: it trades an exact dictionary-size estimate for a guess, and the property test would
have to weaken to accept it. `CompressionLevel::wants_exact_statistics()` already exists as the
place to make that trade if write throughput ever matters more than picking the smallest encoding.

**The margin means Basalt sometimes stores more bytes than it could.** That is the intended
behaviour, and it is a *policy* exposed as a knob rather than a constant, because the right answer
depends on whether the data is going to disk or staying in cache.

**Dictionaries are capped at 2^14 entries.** 14-bit codes are 1.75 bytes per value, still a 4.5x win
over plain; above that the dictionary dominates, the gather loses cache locality, and the analysis
pass would have to hold an unbounded set.

# basalt-encoding benchmarks

All numbers below were produced by `cargo bench -p basalt-encoding` on the
machine described in *Environment*. They are **measured, not estimated**; the
size table is printed by the benchmark harness itself
(`crates/basalt-encoding/benches/codecs.rs`) so it can be reproduced with one
command.

```
cargo bench -p basalt-encoding
```

## Environment

| | |
|---|---|
| CPU | aarch64 (ARM64), 4 cores |
| OS | Linux 6.8, Ubuntu 22.04 userspace |
| Toolchain | rustc 1.83.0, `--release` (`opt-level=3`, `lto="thin"`, `codegen-units=1`) |
| Harness | criterion 0.4, 30 samples, 1.2 s measurement window |
| Workload | 65 536 `i64` values per column (512 KiB uncompressed) |

Throughputs are quoted against the **uncompressed** size, which is the number
that matters for a scan: "how fast can the engine turn bytes on disk into values
in registers".

## Compression: size in bytes, 65 536 `i64` values (512 KiB plain)

| shape | PLAIN | VARINT | BITPACKED | FOR | RLE | DICTIONARY | DELTA | selector picks |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `sorted_ids` | 524 293 | 213 573 | 180 230 | 131 086 | 279 112 | 196 619 | **1 034** | DELTA, 507x |
| `clustered_timestamps` | 524 293 | 524 293 | 425 990 | **163 854** | 589 832 | 194 537 | 174 093 | FOR, 3.2x |
| `low_cardinality` | 524 293 | 90 233 | 114 694 | 106 510 | 136 329 | **24 593** | 116 119 | DICTIONARY, 21.3x |
| `long_runs` | 524 293 | 98 309 | 65 542 | 57 358 | **455** | 57 480 | 3 064 | RLE, 1152x |
| `small_values` | 524 293 | 65 541 | **49 158** | 49 166 | 129 038 | 49 223 | 58 376 | BITPACKED, 10.7x |
| `high_entropy` | **524 293** | 622 148 | 524 294 | 524 302 | 687 687 | 597 601 | 529 929 | PLAIN, 1.0x |
| `skewed` | 524 293 | 110 685 | 524 294 | 524 302 | 175 357 | **74 529** | 133 021 | DICTIONARY, 7.0x |

Shapes: `sorted_ids` is a monotonic primary key; `clustered_timestamps` is
microsecond timestamps inside a one-second window; `low_cardinality` is an
eight-value enum; `long_runs` is a sorted status column with 512-row runs;
`small_values` is signed measurements in `-32..32`; `high_entropy` is splitmix64
output; `skewed` is mostly small values with a random 64-bit outlier every 997
rows.

**The selector picked the smallest candidate in all seven cases.** That is the
single most important line in this file: the heuristic is a size estimator plus
an argmin, and on these shapes the estimate is exact for every encoding except
delta (where it is a documented upper bound).

Note `high_entropy`, where every "clever" encoding is *larger* than plain —
varint by 19 %, RLE by 31 %. Choosing on estimated size rather than on a
pattern-matching rule is what stops the engine from making random data bigger.

## Compression: string columns, 65 536 rows

| shape | raw bytes | encoded | ratio |
|---|--:|--:|--:|
| `city_enum` (8 distinct city names) | 327 458 | 24 653 | **13.3x** |
| `unique_ids` (`order-000000000123`) | 1 179 648 | 270 026 | **4.4x** |
| `urls` (templated URLs, 5 000 distinct paths) | 3 163 418 | 132 498 | **23.9x** |

`city_enum` and `urls` take the dictionary layout; `unique_ids` is all-distinct,
so it takes the direct layout and the win comes from the LZ77 byte codec plus
bit-packed lengths.

## Compression: the LZ77 byte codec on its own

| input | raw | compressed | ratio |
|---|--:|--:|--:|
| JSON records | 138 890 | 20 079 | 6.92x |
| English text (repetitive) | 180 000 | 761 | 236x |
| Random bytes | 180 000 | 180 712 | 1.00x (0.4 % expansion) |

The 0.4 % expansion on incompressible input is the expected worst case for an
LZ77 token stream, and it is why `encode_bytes_adaptive` **measures** rather than
guesses: it runs the compressor, compares, and keeps the plain form when
compression does not pay.

## Throughput

### Integer codecs

Encode and decode, 512 KiB of `i64` per iteration. `PLAIN` is included on every
row as the baseline.

| shape | codec | encode | decode |
|---|---|--:|--:|
| `sorted_ids` | PLAIN | 23.5 µs (20.8 GiB/s) | 63.4 µs (7.70 GiB/s) |
| `sorted_ids` | **DELTA** | 30.5 µs (16.0 GiB/s) | 44.2 µs (11.1 GiB/s) |
| `clustered_timestamps` | PLAIN | 22.7 µs (21.5 GiB/s) | 61.1 µs (7.99 GiB/s) |
| `clustered_timestamps` | **FOR** | 131.9 µs (3.70 GiB/s) | 130.0 µs (3.76 GiB/s) |
| `low_cardinality` | PLAIN | 22.5 µs (21.7 GiB/s) | 62.6 µs (7.80 GiB/s) |
| `low_cardinality` | **DICTIONARY** | 227.5 µs (2.15 GiB/s) | 104.3 µs (4.68 GiB/s) |
| `long_runs` | PLAIN | 22.5 µs (21.7 GiB/s) | 61.2 µs (7.97 GiB/s) |
| `long_runs` | **RLE** | 34.0 µs (14.4 GiB/s) | **6.53 µs (74.8 GiB/s)** |
| `small_values` | PLAIN | 22.5 µs (21.7 GiB/s) | 61.2 µs (7.97 GiB/s) |
| `small_values` | **BITPACKED** | 71.8 µs (6.80 GiB/s) | 84.7 µs (5.76 GiB/s) |
| `high_entropy` | **PLAIN** | 22.7 µs (21.5 GiB/s) | 60.9 µs (8.02 GiB/s) |
| `skewed` | PLAIN | 22.8 µs (21.5 GiB/s) | 60.9 µs (8.01 GiB/s) |
| `skewed` | **DICTIONARY** | 502.1 µs (0.97 GiB/s) | 111.7 µs (4.37 GiB/s) |

Reading these:

* **RLE decode is 9x faster than plain decode** (74.8 vs 7.97 GiB/s) *and*
  1152x smaller. When a column has runs, compression is free in both currencies.
  This is the whole argument for columnar storage in one line.
* **Delta decode beats plain decode** (11.1 vs 7.7 GiB/s) because it touches 500x
  fewer bytes; the serial prefix sum costs less than the memory traffic it saves.
* **FOR and BITPACKED decode at roughly half of plain**, which is the honest
  cost of the shift-and-mask inner loop at a non-power-of-two width. They still
  win overall once the data is on a disk rather than in L2.
* **Dictionary encode is the slowest path** (0.97-2.15 GiB/s) because it sorts
  and dedups the whole chunk. Write-side cost; read side is 4.4-4.7 GiB/s.
* Plain "decode" is slower than plain "encode" because decode allocates and
  fills a `Vec<i64>` while encode appends to a pre-reserved buffer.

### The selector itself

`analyze` is one pass that computes extremes, runs, sortedness, exact varint and
run-length body sizes, and a capped distinct-value set.

| shape | time | throughput |
|---|--:|--:|
| `sorted_ids` | 659 µs | 759 MiB/s |
| `clustered_timestamps` | 664 µs | 753 MiB/s |
| `low_cardinality` | 508 µs | 984 MiB/s |
| `long_runs` | 434 µs | 1.13 GiB/s |
| `small_values` | 477 µs | 1.02 GiB/s |
| `high_entropy` | 666 µs | 750 MiB/s |
| `skewed` | 504 µs | 993 MiB/s |

**Analysis is the write path's bottleneck, not the codecs.** At ~750 MiB/s it
costs about 29x more than a plain encode. The `HashSet<i64>` that tracks distinct
values dominates: the two slowest shapes (`sorted_ids`, `high_entropy`) are
exactly the two with 65 536 distinct values, and the fastest (`long_runs`, 128
distinct) is 1.6x quicker. The obvious fix — sample the first N values, or swap
the `HashSet` for a HyperLogLog sketch — is left undone deliberately: it trades
an exact dictionary size estimate for a guess, and the current
`estimates_match_reality` property test would have to weaken. If write throughput
ever matters more than picking the smallest encoding, `CompressionLevel` is the
right place to make that trade (`wants_exact_statistics()` already exists for it).

### Byte codec

| input | compress | decompress | `memcpy` baseline |
|---|--:|--:|--:|
| JSON records (139 KiB) | 55.1 µs (2.35 GiB/s) | 49.7 µs (2.60 GiB/s) | 2.02 µs (64.1 GiB/s) |
| English text (180 KiB) | 47.9 µs (3.50 GiB/s) | 46.6 µs (3.60 GiB/s) | 2.38 µs (70.4 GiB/s) |
| Random bytes (180 KiB) | 488 µs (352 MiB/s) | 2.86 µs (58.6 GiB/s) | 2.86 µs (58.5 GiB/s) |

Compressing random bytes is 10x slower than compressing text because every
position misses in the match table and falls through to the literal path — the
match finder does maximum work for zero benefit. Decompressing it runs at
*exactly* memcpy speed (2.86 µs against a 2.86 µs baseline), because the stream
is one long literal run and the decoder does nothing but copy it.

### Whole columns

End-to-end `encode_column` / `decode_column`, including the validity bitmap and
the statistics pass.

| column | encode | decode |
|---|--:|--:|
| `int64_clustered` (65 536 rows, 512 KiB) | 1.10 ms (454 MiB/s) | 86.3 µs (5.66 GiB/s) |
| `utf8_enum` (65 536 rows, 8 distinct strings) | 3.54 ms (159 MiB/s) | 1.59 ms (353 MiB/s) |

The gap between `int64_clustered` encode (454 MiB/s) and its codec (16 GiB/s) is
the analysis pass. The string column is slower on both sides because the current
`decode_column` materialises a `Vec<Vec<u8>>` before rebuilding the array — one
allocation per value, exactly the pattern `BinaryArray` exists to avoid. That is
the clearest single optimisation left in this crate: decode straight into the
offsets + data buffers. It is called out rather than fixed because it changes the
shape of `decode_var_len`, and correctness came first.

## What these numbers do not show

* Single-threaded, single-socket, warm cache. Nothing here is I/O bound, so the
  compression ratios matter more than the throughputs for real query latency.
* No SIMD. The bit-packing loops are scalar; a NEON/AVX2 unpacker would plausibly
  triple `BITPACKED` and `FOR` decode.
* aarch64 only. The shift-heavy paths may rank differently on x86-64.

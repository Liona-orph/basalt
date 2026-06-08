# 0002. Columnar vectors with bitmap null masks

## Status

Accepted.

## Context

An analytical engine reads a few columns of many rows, so the in-memory layout is columnar: one
buffer per column rather than one struct per row. That much is settled by the workload. Three
questions are not:

1. **How are nulls represented?** A parallel `Vec<bool>` (one byte per row), a bitmap (one bit per
   row), a sentinel value in the data, or `Option<T>` in the values buffer.
2. **Are null slots stored?** Either the values buffer has a slot for every row including the nulls
   (dense), or only the valid values are stored and reads indirect through a rank operation over the
   mask (compact).
3. **How is variable-length data stored?** `Vec<Vec<u8>>`, or `offsets: [i32; n+1]` into one shared
   data buffer.

Slicing matters too: a scan hands sub-ranges of a batch to operators constantly, and if slicing
copies, everything downstream is quadratic in the number of pipeline stages.

## Decision

* **A validity bitmap, one bit per row, with `1 = valid`** (the same polarity Arrow uses), stored as
  `Option<Bitmap>`: `None` means "no nulls", not "all null".
* **Dense values.** Every row has a slot in the values buffer; the contents of a null slot are
  unspecified.
* **Offsets plus one data buffer** for `Utf8` and `Binary`: `offsets: Buffer<i32>` of length `n + 1`
  and a single `Buffer<u8>`, with the invariants checked once at construction.
* **O(1) slicing.** `Buffer<T>` is `Arc<Vec<T>>` plus an offset and a length; `Bitmap` is
  `Arc<Vec<u64>>` plus a *bit* offset and a bit length. Neither `slice` copies.
* `Bitmap::set` is copy-on-write, so a shared bitmap can be handed around without a mutability
  audit.

## Consequences

**Bitmap over byte mask.** Eight times smaller, so eight times more rows fit in cache; `count_ones`
is one `popcnt` per 64-bit word and iterating set bits is one `tzcnt` per bit. Cost: a bit read is a
shift and a mask rather than a load, and a slice that starts mid-word needs two loads and two shifts
per word. That is still far cheaper than materialising a copy.

**`1 = valid` over `1 = null`.** "Is this row usable across all its inputs" becomes a plain `AND` of
the input masks. With the opposite polarity it is an `OR` followed by a `NOT`, on every operator,
forever.

**`None` over an all-ones bitmap.** Saves the allocation, and — more importantly — keeps the
encoding canonical: the same logical column always produces the same bytes on disk. The segment
format's chunk header has an explicit "validity present" flag for this reason, and the writer omits
the bitmap even when a *slice* inherited an all-ones mask from its parent.

**Dense over compact.** A dense array is a plain `&[T]` the compiler can auto-vectorise, and a read
is one indexed load. Compact storage would save memory on mostly-null columns and cost a popcount on
every single read to do it. Mostly-null columns are handled far better one layer down, where RLE or
dictionary encoding turns them into almost nothing anyway. The visible cost is that a
99 %-null in-memory column occupies full width.

**Offsets over `Vec<Vec<u8>>`.** One allocation instead of one per value, contiguous bytes (so scans
are sequential and prefetchable), and O(1) slicing because a slice is a window into the same two
buffers. Costs: appending is the only cheap mutation; `i32` offsets cap a single array at 2 GiB of
data; and the offset invariants (non-empty, non-decreasing, non-negative, in range) must be checked
in O(n) at construction so every later read can skip the check.

**Floats compare bitwise.** `NativeType::total_eq` compares `f32`/`f64` by bit pattern so that
`NaN == NaN`. Without that, two arrays holding identical bytes would compare unequal and a
round trip through the storage layer could not be asserted. It also keeps array equality consistent
with `Value`'s total order.

**One thing this does not give.** There is no nested type. Lists and structs would need an offsets
layer over child arrays and a case in every kernel, codec and zone map, and nothing this project
studies requires them.

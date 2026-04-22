# The Basalt segment format, version 1

A **segment** is one immutable columnar file. This document is the normative
byte-level description; `crates/basalt-storage/src/format.rs`,
`metadata.rs` and `serial.rs` are its implementation, and the constants below
are the ones those files define. Every integer is **little-endian**. "uvarint"
means unsigned LEB128 and "varint" means zigzag LEB128, both defined in
`basalt-encoding`.

## 1. Overall layout

```
byte 0
+--------------------------------------------------------------+
|  FILE HEADER                                        16 bytes  |
+--------------------------------------------------------------+
|  BLOCK: row group 0, column 0 data                            |
|  BLOCK: row group 0, column 0 bloom filter        (optional)  |
|  BLOCK: row group 0, column 1 data                            |
|  ...                                                          |
+--------------------------------------------------------------+
|  BLOCK: row group 1, column 0 data                            |
|  ...                                                          |
+--------------------------------------------------------------+
|  BLOCK: footer (schema + all chunk metadata)                  |
+--------------------------------------------------------------+
|  TRAILER                                            16 bytes  |
+--------------------------------------------------------------+
                                                        byte N
```

Blocks appear in the order the writer produced them: row group by row group,
and within a row group column by column in schema order, each data block
immediately followed by its bloom filter block if it has one. Nothing depends on
that order — every block is located by an absolute offset in the footer — but it
means a sequential full scan reads the file front to back.

The footer is written **last**, because a chunk's offset is only known after the
chunk has been written. The writer therefore needs `Write`, not `Write + Seek`,
and can target a pipe or an object-store upload.

## 2. File header (16 bytes, at offset 0)

```
offset  size  field
     0     4  magic = "BSLT" (0x42 0x53 0x4C 0x54)
     4     2  format_version : u16 = 1
     6     2  flags : u16 = 0        (reserved, must be zero)
     8     8  reserved               (must be zero)
```

A reader rejects a file whose magic or version does not match, naming byte 0 or
byte 4 in the error.

## 3. Trailer (16 bytes, at the end of the file)

```
offset from EOF  size  field
            -16     8  footer_offset : u64   absolute offset of the footer BLOCK header
             -8     4  footer_length : u32   total footer block size, header included
             -4     4  magic = "BSLT"
```

The magic is repeated here so that a truncated file is detected by a single
16-byte read rather than by parsing garbage. The reader also checks that
`footer_offset >= 16` and `footer_offset + footer_length <= file_len - 16`.

## 4. Block framing

Every persisted byte outside the header and trailer lives inside a block.

```
offset  size  field
     0     4  payload_length : u32     (<= 268435456, i.e. 256 MiB)
     4     4  crc32c : u32             CRC-32/ISCSI of the payload only
     8     L  payload
```

* The checksum is CRC32C (Castagnoli, reflected polynomial `0x82F63B78`), the
  same one iSCSI and modern filesystems use. It detects every single-bit,
  double-bit and burst error up to 32 bits.
* The offsets recorded in the footer point at **byte 0 of the block header**, and
  the recorded length includes those 8 bytes.
* A mismatch produces a corruption error naming the payload's absolute offset,
  the stored checksum and the computed one.
* The 256 MiB cap is a format limit: without it a damaged `payload_length` could
  make a reader allocate arbitrarily before the checksum could reject it.

## 5. Column chunk payload

The payload of a data block is a self-describing chunk produced by
`basalt_encoding::encode_column`.

```
offset  size  field
     0     1  chunk_version : u8 = 1
     1     1  flags : u8
                bit 0  validity bitmap present
                bit 1  Decimal128 values were narrowed to 64 bits
                bits 2-7 reserved, zero
     2     v  row_count : uvarint
       (if bit 0)
           v  validity_length : uvarint
           n  validity bitmap, LSB-first, **1 = valid**, ceil(row_count/8) bytes
           .  body
```

The validity bitmap is omitted whenever the chunk has no nulls, including when a
column *slice* carries an all-ones mask inherited from its parent. That keeps
the encoding canonical: the same logical column always produces the same bytes.

### 5.1 Bodies by logical type

| Type | Body |
|---|---|
| `NULL` | empty |
| `BOOLEAN` | `uvarint length` + packed bits, LSB-first |
| `INT8`..`INT64`, `UINT8`..`UINT64`, `DATE32`, `TIMESTAMP` | one integer stream (§6) |
| `FLOAT32`, `FLOAT64` | one integer stream over the IEEE-754 **bit patterns** |
| `DECIMAL128` | if flags bit 1: one integer stream of the narrowed values; otherwise `row_count` x 16-byte little-endian `i128` |
| `UTF8`, `BINARY` | a variable-length body (§5.2) |

Floats reuse the integer codecs on their bit patterns. This is lossless, costs
nothing, and means dictionary and run-length encoding work on categorical float
data; genuinely continuous floats have high-entropy bit patterns and the size
estimator falls back to `PLAIN`.

Values under a null are **unspecified**. Writers store a type default; readers
must not interpret them.

### 5.2 Variable-length body

```
offset  size  field
     0     1  layout : u8    0 = direct, 1 = dictionary
```

**Direct** (`layout = 0`):

```
     v  lengths_stream_length : uvarint
     n  integer stream: one length per row
     v  data_stream_length : uvarint
     n  byte stream: the values, concatenated
```

Per-row *lengths* are stored rather than offsets: they are small non-negative
integers, so bit packing and run-length encoding work on them, and offsets are
recovered by a prefix sum.

**Dictionary** (`layout = 1`):

```
     v  dictionary_len : uvarint
     v  dict_lengths_stream_length : uvarint
     n  integer stream: one length per dictionary entry
     v  dict_data_stream_length : uvarint
     n  byte stream: the distinct values, sorted, concatenated
     v  codes_stream_length : uvarint
     n  integer stream: one dictionary index per row
```

The dictionary is sorted, so a future comparison kernel can evaluate range
predicates on the codes directly.

## 6. Codec streams

Both integer and byte streams begin with the same self-describing header:

```
offset  size  field
     0     1  encoding : u8      (§6.1)
     1     1  codec_version : u8 = 1
     2     v  element_count : uvarint   (<= 16777216)
     .     .  codec-specific body
```

Element counts are capped at 2^24 so a damaged header cannot request an
unbounded allocation.

### 6.1 Encoding ids

| id | name | body |
|---:|---|---|
| 1 | `PLAIN` | `count` x 8-byte little-endian `i64` |
| 2 | `VARINT` | `count` zigzag LEB128 integers |
| 3 | `BITPACKED` | `u8 width`, then `count` zigzag values packed at `width` bits |
| 4 | `FOR` | `i64 min` (8 bytes), `u8 width`, then `count` unsigned offsets from `min`, bit-packed |
| 5 | `RLE` | `uvarint run_count`, then per run: `varint value`, `uvarint run_length` |
| 6 | `DICTIONARY` | `uvarint k`, `varint dict[0]`, `k-1` uvarint gaps, `u8 code_width`, `count` bit-packed codes |
| 7 | `DELTA` | `varint first`, `uvarint block_size`, then per block: `varint min_delta`, `u8 width`, bit-packed `(delta - min_delta)` |
| 8 | `PLAIN_BYTES` | `count` raw bytes |
| 9 | `BYTE_LZ` | LZ77 stream (§6.3); the header's `count` is the **uncompressed** length |

Bit packing is LSB-first: value `i` occupies bits `i*w .. (i+1)*w` of a
little-endian bit stream. Width 0 is legal and means every value is zero; it
occupies no bytes.

### 6.2 Delta blocks

`DELTA` stores `first` verbatim and then `count - 1` deltas in blocks of
`block_size` (128 in this version). Each block carries its own minimum and bit
width, so one jump in the middle of a column does not widen the whole thing.

### 6.3 The LZ77 byte stream

A sequence of:

```
  token : u8       high nibble = literal length (15 means "read more")
                   low nibble  = match length - 4 (15 means "read more")
  [extension bytes]  255, 255, ..., r   summed with the nibble
  literals           literal_length raw bytes
  offset : u16       distance back into the *output*, 1..=65535
  [extension bytes]  for the match length
```

The stream ends either immediately after a match (when the last match reached
the end of the input) or with a literals-only sequence carrying the tail; the
decoder distinguishes them by checking whether the buffer is exhausted after
reading the literals. Encoders must not emit an empty trailing sequence, so the
stream is canonical and truncation is detectable.

Matches may overlap their own output (`offset < length`); decoders must copy
byte by byte.

## 7. Bloom filter payload

```
offset  size  field
     0     1  version : u8 = 1
     1     v  num_bits : uvarint     (multiple of 64)
     .     v  num_hashes : uvarint   (1..=16)
     .     n  num_bits/8 bytes: the bit array as little-endian u64 words
```

Bit positions come from double hashing: `h1 + i * h2 (mod num_bits)` for
`i` in `0..num_hashes`, where `h1` is the 64-bit value hash and `h2` is
`(h1 >> 32) | 1`.

The value hash is FNV-1a over the canonical value encoding (§8.2), finished with
the splitmix64 mixer. It is deliberately **stable across processes and builds** —
`std`'s `DefaultHasher` is not — and it includes the value's type tag, so a probe
must be cast to the column's type before hashing.

## 8. Footer payload

```
offset  size  field
     0     4  magic = "BFTR"
     4     1  footer_version : u8 = 1
     5     .  schema (§8.1)
     .     v  row_group_count : uvarint
     .     .  row_group_count x row group entry (§8.3)
     .     v  total_rows : uvarint
     .     .  metadata map (§8.4)
```

The reader checks that `total_rows` equals the sum of the row groups' row counts
and that each row group has exactly one chunk per schema field.

### 8.1 Schema

```
  v  field_count : uvarint
  repeated field_count times:
     v  name_length : uvarint
     n  name (UTF-8)
     .  data type (§8.2)
     1  nullable : u8   (0 or 1)
     .  field metadata map (§8.4)
  .  schema metadata map (§8.4)
```

### 8.2 Data types and values

A data type is a one-byte code plus parameters:

| code | type | parameters |
|---:|---|---|
| 0 | `NULL` | — |
| 1 | `BOOLEAN` | — |
| 2..5 | `INT8`, `INT16`, `INT32`, `INT64` | — |
| 6..9 | `UINT8`, `UINT16`, `UINT32`, `UINT64` | — |
| 10, 11 | `FLOAT32`, `FLOAT64` | — |
| 12, 13 | `UTF8`, `BINARY` | — |
| 14 | `DATE32` | — |
| 15 | `TIMESTAMP` | `u8 unit`: 0=s, 1=ms, 2=us, 3=ns |
| 16 | `DECIMAL128` | `u8 precision` (1..=38), `i8 scale` (0..=precision) |

A **value** is its data type followed by a payload:

| type | payload |
|---|---|
| `NULL` | empty |
| `BOOLEAN` | 1 byte, 0 or 1 |
| signed integers, `DATE32`, `TIMESTAMP` | zigzag varint |
| unsigned integers | uvarint |
| `FLOAT32` / `FLOAT64` | 4 / 8 bytes: `to_bits()`, little-endian |
| `UTF8` / `BINARY` | `uvarint length` + bytes |
| `DECIMAL128` | zigzag 128-bit varint of the unscaled value |

Floats are stored as bit patterns, not decimal text, so `-0.0` and `0.0` stay
distinguishable and `NaN` round-trips — matching `Value`'s total order, which is
what zone-map comparisons use.

This encoding is **canonical**: the same value always produces the same bytes.
The bloom filter's hash depends on that.

### 8.3 Row group entry

```
  v  num_rows : uvarint
  repeated once per schema field, in schema order:
     v  data_offset : uvarint          absolute offset of the data BLOCK header
     v  data_length : uvarint          block size, 8-byte header included
     1  encoding : u8                  (§6.1)
     v  uncompressed_bytes : uvarint   in-memory size, for ratio reporting
     .  zone map:
        1  has_min : u8 ; if 1: value (§8.2)
        1  has_max : u8 ; if 1: value (§8.2)
        v  null_count : uvarint
        v  distinct_estimate : uvarint
        v  row_count : uvarint
     1  bloom_present : u8
        if 1:
        v  bloom_offset : uvarint       absolute offset of the bloom BLOCK header
        v  bloom_length : uvarint       block size, 8-byte header included
```

**Zone-map semantics.** `min` and `max` are the extremes of the **non-null**
values; a chunk that is entirely null has neither, and `null_count == row_count`.
Folding nulls into the bounds would break `IS NULL` pruning and would make
`col > 5` fail to skip an all-null chunk. The reader validates
`null_count <= row_count` and `min <= max`.

`distinct_estimate` comes from a linear-counting sketch over an 8192-bit map
(`n̂ = -m ln(z/m)`), clamped to the non-null row count. It is an estimate; nothing
in the format depends on it being exact. It drives the bloom filter's sizing and
the decision whether to write one at all.

### 8.4 Metadata map

```
  v  entry_count : uvarint
  repeated entry_count times:
     v  key_length : uvarint    + key bytes (UTF-8)
     v  value_length : uvarint  + value bytes (UTF-8)
```

Entries are written in sorted key order (the in-memory type is a `BTreeMap`), so
two segments with the same metadata serialise identically.

## 9. Reading a segment

1. Read 16 bytes at offset 0; check the magic and version.
2. Read 16 bytes at `file_len - 16`; check the magic; take `footer_offset` and
   `footer_length`; check they lie inside the file.
3. Read the footer block, verify its CRC32C, parse it.
4. For each row group, evaluate the predicate against the zone maps. This costs
   no I/O.
5. For row groups that survive, and only for equality tests in a purely
   conjunctive position, read and consult the bloom filters. Under an `OR` or a
   `NOT`, "absent" does not make the predicate false, so filters must not be
   used there.
6. For row groups that still survive, read only the projected columns' data
   blocks, verify each CRC32C, and decode.

Opening a segment therefore costs three reads regardless of its size.

## 10. Compatibility rules

* `format_version` (file header) changes only if the framing changes.
* `footer_version`, `chunk_version` and the bloom filter's `version` are
  independent, so a new codec or a new statistic does not invalidate existing
  files.
* Encoding ids and data-type codes are permanent: an id is never reused for a
  different meaning.
* Reserved bytes and flag bits must be written as zero and ignored on read.

## 11. Limits

| Limit | Value | Why |
|---|---|---|
| Block payload | 256 MiB | Bounds allocation from a damaged length field |
| Elements per codec stream | 2^24 | Same, for encodings that describe many values in few bytes |
| Variable-length chunk data | 2 GiB | 32-bit offsets, half the metadata of 64-bit ones |
| Decimal precision | 38 | Fits `i128` |
| Bloom filter probes | 16 | Beyond this the probes cost more than the false positives they remove |
| LZ77 back reference | 65535 | 16-bit offset field |

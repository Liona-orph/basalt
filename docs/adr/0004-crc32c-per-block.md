# 0004. CRC32C per block, and failing loud on corruption

## Status

Accepted.

## Context

A segment file is read back long after it was written, possibly from a different machine, possibly
after being copied through object storage. Bits rot: silent disk corruption, a truncated upload, a
bug in a future writer, a file edited by something that had no business editing it.

An engine can respond to that in three ways:

1. **Trust the bytes.** Fastest, and the failure mode is the worst one available: a flipped bit in a
   bit-packed stream produces *plausible values*, and the query returns a wrong answer with no
   indication anything happened.
2. **Checksum the whole file.** Cheap to implement, but it can only be verified by reading
   everything — which defeats the projection and predicate pushdown that make the format worth
   having — and it cannot say where the damage is.
3. **Checksum every independently-addressable unit.**

There is a second, sharper problem. The block framing contains a length field. A reader that trusts
a damaged length allocates whatever the damaged value says *before* any checksum can reject it,
which turns corruption into an out-of-memory abort.

## Decision

**Every persisted block carries a CRC32C over its payload**, verified before the payload is used.

```text
offset  size  field
     0     4  payload_length : u32   (<= 256 MiB, a format limit)
     4     4  crc32c : u32           CRC-32/ISCSI of the payload only
     8     L  payload
```

* The polynomial is Castagnoli (reflected `0x82F63B78`), the CRC used by iSCSI, ext4 and btrfs. It
  detects every single-bit, double-bit and burst error up to 32 bits. The table is computed in 30
  lines in `crates/basalt-storage/src/crc.rs`, which is why no crate is needed for it.
* A block is the unit of independent addressing: a column chunk, a bloom filter, the footer. So the
  granularity of verification is exactly the granularity of reading, and a scan verifies precisely
  what it touches and nothing more.
* `payload_length` is capped at 256 MiB **by the format**, so a damaged length cannot cause an
  unbounded allocation before the checksum runs.
* The file header magic and version are checked at offset 0, and the 16-byte trailer repeats the
  magic so that a truncated file is detected by a single read rather than by parsing garbage. The
  reader also checks `footer_offset >= 16` and `footer_offset + footer_length <= file_len - 16`.
* A failure is a `BasaltError::Corruption` carrying the object name, the structural `Region`
  (`FileHeader`, `BlockHeader`, `BlockPayload`, `Footer`, `FooterTrailer`, `EncodedChunk`,
  `BloomFilter`, `ZoneMap`) and the **absolute byte offset**, plus the stored and computed
  checksums. `is_corruption()` and `corruption_offset()` let recovery tooling act on it without
  parsing the message.

## Consequences

**Corruption is detected, never returned.** This is the property the format exists to have: a
damaged segment produces an error naming a byte offset, not a plausible-looking wrong answer.
`crates/basalt-storage/tests/segment.rs` asserts it directly: a flipped bit in a data block is
reported with its offset, every byte of the footer and trailer is shown to be protected, and
truncated or bogus files are rejected. A projection that avoids the damaged column still succeeds,
because verification is per block.

**Cost.** Four bytes per block (negligible) and one CRC pass over the bytes actually read.
Hardware CRC32C instructions exist on both aarch64 and x86-64 and are not used here — the
implementation is a portable table-driven loop, because reaching for `unsafe` intrinsics in a
project that is `#![forbid(unsafe_code)]` throughout would be a bad trade for a cost that is small
next to decoding.

**A block is verified in full before it is decoded**, which means it must be buffered in full. That
is already how the reader works, and it is why the 256 MiB cap is a format rule rather than a
suggestion.

**Region plus offset is more work than a boolean.** Every failure path has to know where it is, and
the error type carries four fields instead of a string. That is the point: "byte 4096" is much
easier to act on when you also know it was the start of a block header rather than the middle of a
dictionary page.

**What this does not protect against.** A CRC detects damage, not tampering — it is not a MAC, and
an attacker who can write the file can recompute it. See `SECURITY.md` for the threat model.

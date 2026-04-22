//! Segment reader with projection and predicate pushdown.

use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use basalt_common::{BasaltError, ByteSize, IoResultExt, Region, Result};
use basalt_encoding::decode_column;
use basalt_types::{RecordBatch, Schema};

use crate::bloom::{value_hash, BloomFilter};
use crate::format::{
    parse_block_header, parse_file_header, parse_trailer, verify_block, BLOCK_HEADER_LEN,
    FILE_HEADER_LEN, TRAILER_LEN,
};
use crate::metadata::{BlockLocation, SegmentMetadata};
use crate::predicate::{Predicate, Verdict};

/// Random-access byte source for a segment.
///
/// A trait rather than a concrete `File` for three reasons: tests want an
/// in-memory segment, [`CountingSource`] wants to observe exactly how many
/// bytes a scan touches (which is how projection and predicate pushdown are
/// *proved* to work rather than asserted), and a future object-store backend
/// wants ranged HTTP GETs.
///
/// Implementations take `&self` so a reader can be shared; the file-backed one
/// uses interior mutability for the seek cursor.
pub trait SegmentSource {
    /// Total size of the segment in bytes.
    fn size(&self) -> u64;

    /// Fill `buf` with the bytes starting at `offset`.
    ///
    /// # Errors
    ///
    /// If the range extends past the end of the source, or on I/O failure.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Read `len` bytes at `offset` into a fresh `Vec`.
    fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.read_at(offset, &mut buf)?;
        Ok(buf)
    }
}

impl SegmentSource for Vec<u8> {
    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.as_slice().read_at(offset, buf)
    }
}

impl SegmentSource for &[u8] {
    fn size(&self) -> u64 {
        self.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let start =
            usize::try_from(offset).map_err(|_| out_of_range(offset, buf.len(), self.len()))?;
        let end = start
            .checked_add(buf.len())
            .filter(|end| *end <= self.len())
            .ok_or_else(|| out_of_range(offset, buf.len(), self.len()))?;
        buf.copy_from_slice(&self[start..end]);
        Ok(())
    }
}

fn out_of_range(offset: u64, len: usize, size: usize) -> BasaltError {
    BasaltError::corruption(
        "segment",
        Region::BlockPayload,
        offset,
        format!("read of {len} bytes at {offset} extends past the {size}-byte segment"),
    )
}

/// A segment backed by a file on disk.
///
/// Uses a `RefCell<File>` with seek-then-read, so a `FileSource` is `Send` but
/// not `Sync`: one reader per thread. A parallel scan would use positioned
/// reads (`pread`), which is a platform-specific API and not worth the
/// conditional compilation until there is a parallel executor to use it.
#[derive(Debug)]
pub struct FileSource {
    file: RefCell<File>,
    size: u64,
    name: String,
}

impl FileSource {
    /// Open a segment file.
    pub fn open(path: impl AsRef<Path>) -> Result<FileSource> {
        let path = path.as_ref();
        let name = path.display().to_string();
        let file = File::open(path).io_context(format!("opening {name}"))?;
        let size = file
            .metadata()
            .io_context(format!("stat-ing {name}"))?
            .len();
        Ok(FileSource {
            file: RefCell::new(file),
            size,
            name,
        })
    }

    /// The path this source was opened from.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl SegmentSource for FileSource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(offset))
            .io_context(format!("seeking to {offset} in {}", self.name))?;
        file.read_exact(buf).io_context(format!(
            "reading {} bytes at {offset} in {}",
            buf.len(),
            self.name
        ))
    }
}

/// Wraps a [`SegmentSource`] and counts the bytes it serves.
///
/// This exists so that "projection pushdown works" is a *measurement*, not a
/// claim: read one column out of ten and assert that roughly a tenth of the
/// data bytes were touched.
#[derive(Debug)]
pub struct CountingSource<S: SegmentSource> {
    inner: S,
    bytes_read: Cell<u64>,
    read_count: Cell<u64>,
}

impl<S: SegmentSource> CountingSource<S> {
    /// Wrap a source.
    pub fn new(inner: S) -> CountingSource<S> {
        CountingSource {
            inner,
            bytes_read: Cell::new(0),
            read_count: Cell::new(0),
        }
    }

    /// Bytes served so far.
    pub fn bytes_read(&self) -> ByteSize {
        ByteSize::new(self.bytes_read.get())
    }

    /// Number of `read_at` calls so far.
    pub fn read_count(&self) -> u64 {
        self.read_count.get()
    }

    /// Reset both counters, so a test can measure one scan in isolation.
    pub fn reset(&self) {
        self.bytes_read.set(0);
        self.read_count.set(0);
    }

    /// The wrapped source.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S: SegmentSource> SegmentSource for CountingSource<S> {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.bytes_read
            .set(self.bytes_read.get() + buf.len() as u64);
        self.read_count.set(self.read_count.get() + 1);
        self.inner.read_at(offset, buf)
    }
}

/// What a scan should read.
#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    projection: Option<Vec<usize>>,
    predicate: Option<Predicate>,
}

impl ScanOptions {
    /// Read every column, every row group.
    pub fn new() -> ScanOptions {
        ScanOptions::default()
    }

    /// Read only these columns, in this order.
    #[must_use]
    pub fn with_projection(mut self, columns: Vec<usize>) -> ScanOptions {
        self.projection = Some(columns);
        self
    }

    /// Skip row groups that cannot satisfy this predicate.
    #[must_use]
    pub fn with_predicate(mut self, predicate: Predicate) -> ScanOptions {
        self.predicate = Some(predicate);
        self
    }

    /// The projection, if any.
    pub fn projection(&self) -> Option<&[usize]> {
        self.projection.as_deref()
    }

    /// The predicate, if any.
    pub fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }
}

/// Reads a segment written by [`SegmentWriter`](crate::SegmentWriter).
///
/// Opening a segment reads only the file header, the trailer and the footer —
/// three small reads regardless of file size. Everything else is read on demand,
/// per column chunk, so a projection genuinely avoids the I/O rather than
/// discarding it after the fact.
///
/// ```
/// use std::sync::Arc;
/// use basalt_common::Config;
/// use basalt_storage::{ScanOptions, SegmentReader, SegmentWriter};
/// use basalt_types::{
///     ColumnVector, DataType, Field, PrimitiveArray, RecordBatch, Schema, StringArray,
/// };
///
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("id", DataType::Int64, false),
///     Field::new("city", DataType::Utf8, false),
/// ])?);
/// let batch = RecordBatch::try_new(
///     Arc::clone(&schema),
///     vec![
///         ColumnVector::Int64(PrimitiveArray::from_vec(vec![1, 2, 3])),
///         ColumnVector::Utf8(StringArray::from_strs(&["oslo", "lima", "kyoto"])),
///     ],
/// )?;
///
/// // `&mut Vec<u8>` implements `Write`, so a segment can be built in memory.
/// let mut bytes = Vec::new();
/// let mut writer = SegmentWriter::new(&mut bytes, Arc::clone(&schema), Config::default())?;
/// writer.write_batch(&batch)?;
/// writer.finish()?;
///
/// // Read back only the `city` column: the `id` chunk is never touched.
/// let reader = SegmentReader::open(bytes)?;
/// assert_eq!(reader.num_rows(), 3);
/// let options = ScanOptions::new().with_projection(vec![1]);
/// let batches: Vec<RecordBatch> = reader.scan(&options)?.collect::<Result<_, _>>()?;
/// assert_eq!(batches[0].num_columns(), 1);
/// assert_eq!(batches[0].column(0).len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SegmentReader<S: SegmentSource> {
    source: S,
    metadata: SegmentMetadata,
    object: String,
}

impl<S: SegmentSource> SegmentReader<S> {
    /// Open a segment, reading and validating only its header and footer.
    pub fn open(source: S) -> Result<SegmentReader<S>> {
        SegmentReader::open_named(source, "segment")
    }

    /// Like [`SegmentReader::open`] but names the object in error messages.
    pub fn open_named(source: S, object: impl Into<String>) -> Result<SegmentReader<S>> {
        let object = object.into();
        let size = source.size();
        if size < FILE_HEADER_LEN + TRAILER_LEN {
            return Err(BasaltError::corruption(
                object,
                Region::FileHeader,
                0,
                format!("{size} bytes is too small to be a segment"),
            ));
        }
        let header = source.read_vec(0, FILE_HEADER_LEN as usize)?;
        let format_version = parse_file_header(&header, &object)?;

        let trailer = source.read_vec(size - TRAILER_LEN, TRAILER_LEN as usize)?;
        let (footer_offset, footer_length) = parse_trailer(&trailer, size, &object)?;

        let footer_block = source.read_vec(footer_offset, footer_length as usize)?;
        let payload = unwrap_block(&footer_block, footer_offset, &object, Region::Footer)?;
        let metadata = SegmentMetadata::from_bytes(payload, format_version)?;

        for group in &metadata.row_groups {
            if group.columns.len() != metadata.schema.len() {
                return Err(BasaltError::corruption(
                    object,
                    Region::Footer,
                    footer_offset,
                    format!(
                        "row group has {} column chunks but the schema has {} fields",
                        group.columns.len(),
                        metadata.schema.len()
                    ),
                ));
            }
        }
        Ok(SegmentReader {
            source,
            metadata,
            object,
        })
    }

    /// The segment's schema.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.metadata.schema
    }

    /// The parsed footer.
    pub fn metadata(&self) -> &SegmentMetadata {
        &self.metadata
    }

    /// Total rows in the segment.
    pub fn num_rows(&self) -> u64 {
        self.metadata.total_rows
    }

    /// Number of row groups.
    pub fn num_row_groups(&self) -> usize {
        self.metadata.row_groups.len()
    }

    /// The underlying source, e.g. to read a [`CountingSource`]'s counters.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Which row groups a predicate cannot rule out.
    ///
    /// Two stages, cheapest first:
    ///
    /// 1. **Zone maps.** Free: the bounds are already in memory from the footer.
    /// 2. **Bloom filters.** Only for equality tests in a conjunctive position,
    ///    only for row groups that survived stage 1, and only for the columns
    ///    those equalities mention — each check costs one small read, so doing
    ///    it after zone maps means most of them never happen.
    pub fn prune_row_groups(&self, predicate: Option<&Predicate>) -> Result<Vec<usize>> {
        let Some(predicate) = predicate else {
            return Ok((0..self.metadata.row_groups.len()).collect());
        };
        predicate.validate(&self.metadata.schema)?;
        let equalities = predicate.conjunctive_equalities();
        let mut keep = Vec::new();
        for (index, group) in self.metadata.row_groups.iter().enumerate() {
            if predicate
                .evaluate_zone_maps(group, &self.metadata.schema)?
                .is_skip()
            {
                continue;
            }
            if self.bloom_rules_out(index, &equalities)? {
                continue;
            }
            keep.push(index);
        }
        Ok(keep)
    }

    fn bloom_rules_out(
        &self,
        group_index: usize,
        equalities: &[(usize, &basalt_types::Value)],
    ) -> Result<bool> {
        for (column, value) in equalities {
            let group = &self.metadata.row_groups[group_index];
            let Some(chunk) = group.columns.get(*column) else {
                continue;
            };
            let Some(location) = chunk.bloom else {
                continue;
            };
            let field_type = self.metadata.schema.field(*column).data_type();
            // Probe with the column's own representation, or not at all: the
            // hash includes the type tag, so an `Int64` probe would miss an
            // `Int32` column's filter and wrongly skip the group.
            let probe = if value.data_type() == *field_type {
                (*value).clone()
            } else {
                match value.cast_to(field_type) {
                    Ok(cast) => cast,
                    Err(_) => continue,
                }
            };
            let filter = self.read_bloom(location)?;
            if !filter.contains(value_hash(&probe)) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn read_bloom(&self, location: BlockLocation) -> Result<BloomFilter> {
        let block = self
            .source
            .read_vec(location.offset, location.length as usize)?;
        let payload = unwrap_block(&block, location.offset, &self.object, Region::BloomFilter)?;
        BloomFilter::from_bytes(payload)
    }

    /// Read one row group, optionally projecting columns.
    ///
    /// Only the projected columns' blocks are read; the rest of the file is
    /// never touched.
    pub fn read_row_group(
        &self,
        index: usize,
        projection: Option<&[usize]>,
    ) -> Result<RecordBatch> {
        let group = self.metadata.row_groups.get(index).ok_or_else(|| {
            BasaltError::plan(format!(
                "row group {index} does not exist ({} in this segment)",
                self.metadata.row_groups.len()
            ))
        })?;
        let indices: Vec<usize> = match projection {
            Some(columns) => columns.to_vec(),
            None => (0..self.metadata.schema.len()).collect(),
        };
        let schema = Arc::new(self.metadata.schema.project(&indices)?);
        let mut columns = Vec::with_capacity(indices.len());
        for column_index in &indices {
            let chunk = group.columns.get(*column_index).ok_or_else(|| {
                BasaltError::plan(format!(
                    "projection references column {column_index} but the row group has {}",
                    group.columns.len()
                ))
            })?;
            let block = self
                .source
                .read_vec(chunk.data.offset, chunk.data.length as usize)?;
            let payload = unwrap_block(
                &block,
                chunk.data.offset,
                &self.object,
                Region::EncodedChunk,
            )?;
            let data_type = self.metadata.schema.field(*column_index).data_type();
            let column = decode_column(payload, data_type)?;
            if column.len() as u64 != group.num_rows {
                return Err(BasaltError::corruption(
                    &self.object,
                    Region::EncodedChunk,
                    chunk.data.offset,
                    format!(
                        "chunk decoded {} rows but the row group declares {}",
                        column.len(),
                        group.num_rows
                    ),
                ));
            }
            columns.push(column);
        }
        RecordBatch::try_new(schema, columns)
    }

    /// Scan the segment, applying projection and row-group pruning.
    ///
    /// Returns an iterator that reads one row group at a time, so peak memory is
    /// one row group of the projected columns regardless of segment size.
    pub fn scan(&self, options: &ScanOptions) -> Result<SegmentScan<'_, S>> {
        if let Some(projection) = options.projection() {
            for index in projection {
                if *index >= self.metadata.schema.len() {
                    return Err(BasaltError::plan(format!(
                        "projection references column {index} but the segment has {} columns",
                        self.metadata.schema.len()
                    )));
                }
            }
        }
        let groups = self.prune_row_groups(options.predicate())?;
        Ok(SegmentScan {
            reader: self,
            groups: groups.into_iter(),
            projection: options.projection().map(<[usize]>::to_vec),
        })
    }

    /// Convenience: read the whole segment into one `Vec` of batches.
    pub fn read_all(&self) -> Result<Vec<RecordBatch>> {
        self.scan(&ScanOptions::new())?.collect()
    }
}

/// Strip and verify a block's framing, returning its payload.
fn unwrap_block<'a>(
    block: &'a [u8],
    offset: u64,
    object: &str,
    region: Region,
) -> Result<&'a [u8]> {
    let (length, checksum) = parse_block_header(block, offset, object)?;
    let start = BLOCK_HEADER_LEN as usize;
    let end = start + length as usize;
    if block.len() < end {
        return Err(BasaltError::corruption(
            object,
            region,
            offset,
            format!(
                "block claims {length} payload bytes but only {} were read",
                block.len() - start
            ),
        ));
    }
    let payload = &block[start..end];
    verify_block(payload, checksum, offset + BLOCK_HEADER_LEN, object)?;
    Ok(payload)
}

/// Iterator over the row groups a scan decided to read.
#[derive(Debug)]
pub struct SegmentScan<'a, S: SegmentSource> {
    reader: &'a SegmentReader<S>,
    groups: std::vec::IntoIter<usize>,
    projection: Option<Vec<usize>>,
}

impl<S: SegmentSource> Iterator for SegmentScan<'_, S> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.groups.next()?;
        Some(
            self.reader
                .read_row_group(index, self.projection.as_deref()),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.groups.size_hint()
    }
}

impl<S: SegmentSource> SegmentScan<'_, S> {
    /// Row groups this scan will read; the ones pruned away are already gone.
    pub fn remaining_row_groups(&self) -> usize {
        self.groups.len()
    }
}

/// Verdict for one row group, exposed for tools that want to explain a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneReport {
    /// Row group index.
    pub row_group: usize,
    /// Whether it survived pruning.
    pub verdict: Verdict,
}

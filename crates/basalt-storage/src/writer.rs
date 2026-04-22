//! Streaming segment writer.

use std::io::Write;
use std::sync::Arc;

use basalt_common::{BasaltError, ByteSize, Config, IoResultExt, Result};
use basalt_encoding::encode_column;
use basalt_types::{concat, ColumnVector, RecordBatch, Schema};

use crate::bloom::{value_hash, BloomFilter};
use crate::format::{file_header, frame_block, trailer, BLOCK_HEADER_LEN, FORMAT_VERSION};
use crate::metadata::{BlockLocation, ColumnChunkMetadata, RowGroupMetadata, SegmentMetadata};
use crate::zone_map::ZoneMap;

/// Writes a segment file, one row group at a time.
///
/// # Streaming and bounded memory
///
/// The writer never holds more than one row group. Batches handed to
/// [`SegmentWriter::write_batch`] are sliced (a zero-copy operation) and queued;
/// when the queue reaches [`Config::row_group_rows`], the row group is
/// concatenated, encoded, checksummed and written, and the queue is dropped.
/// The only thing that grows with the size of the segment is the footer
/// metadata — a few dozen bytes per column per row group — which has to be in
/// memory at the end anyway because it is written last.
///
/// A batch larger than a row group is split across several, so a caller cannot
/// blow the memory budget by handing over one enormous batch. [`SegmentWriter::buffered_rows`]
/// exposes the invariant for tests.
///
/// # Why the footer is last
///
/// The footer records each chunk's byte offset, which is only known after the
/// chunk is written. Writing it last means the writer needs `Write`, not
/// `Write + Seek`: the output can be a pipe, a network stream or an object-store
/// multipart upload. The cost is that a segment cannot be read until it is
/// complete, which is the right trade for an immutable columnar format.
///
/// ```no_run
/// use std::fs::File;
/// use std::sync::Arc;
/// use basalt_common::Config;
/// use basalt_storage::SegmentWriter;
/// use basalt_types::{ColumnVector, DataType, Field, PrimitiveArray, RecordBatch, Schema};
///
/// let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])?);
/// let file = File::create("orders.bslt")?;
/// let mut writer = SegmentWriter::new(file, Arc::clone(&schema), Config::default())?;
/// let batch = RecordBatch::try_new(
///     schema,
///     vec![ColumnVector::Int64(PrimitiveArray::from_vec((0..1000).collect()))],
/// )?;
/// writer.write_batch(&batch)?;
/// let metadata = writer.finish()?;
/// assert_eq!(metadata.total_rows, 1000);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SegmentWriter<W: Write> {
    writer: W,
    object: String,
    schema: Arc<Schema>,
    config: Config,
    position: u64,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    row_groups: Vec<RowGroupMetadata>,
    total_rows: u64,
}

impl<W: Write> SegmentWriter<W> {
    /// Start a segment, writing the file header immediately.
    ///
    /// # Errors
    ///
    /// If `config` is invalid or the header cannot be written.
    pub fn new(writer: W, schema: Arc<Schema>, config: Config) -> Result<SegmentWriter<W>> {
        SegmentWriter::with_name(writer, schema, config, "segment")
    }

    /// Like [`SegmentWriter::new`] but names the output, so corruption and I/O
    /// errors mention the file a human recognises.
    pub fn with_name(
        mut writer: W,
        schema: Arc<Schema>,
        config: Config,
        object: impl Into<String>,
    ) -> Result<SegmentWriter<W>> {
        config.validate()?;
        let object = object.into();
        let header = file_header();
        writer
            .write_all(&header)
            .io_context(format!("writing header of {object}"))?;
        Ok(SegmentWriter {
            writer,
            object,
            schema,
            config,
            position: header.len() as u64,
            pending: Vec::new(),
            pending_rows: 0,
            row_groups: Vec::new(),
            total_rows: 0,
        })
    }

    /// The schema every batch must match.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Rows currently buffered, i.e. not yet written to a row group.
    ///
    /// Always `< config.row_group_rows`. Exposed so the bounded-memory property
    /// can be asserted rather than assumed.
    pub fn buffered_rows(&self) -> usize {
        self.pending_rows
    }

    /// Bytes written to the underlying writer so far.
    pub fn bytes_written(&self) -> ByteSize {
        ByteSize::new(self.position)
    }

    /// Row groups completed so far.
    pub fn row_groups_written(&self) -> usize {
        self.row_groups.len()
    }

    /// Append a batch.
    ///
    /// The batch is split at row-group boundaries, so any batch size is legal.
    ///
    /// # Errors
    ///
    /// If the batch's schema differs from the segment's, or on I/O failure.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.schema().as_ref() != self.schema.as_ref() {
            return Err(BasaltError::schema(format!(
                "batch schema {} does not match the segment schema {}",
                batch.schema(),
                self.schema
            )));
        }
        let mut offset = 0usize;
        while offset < batch.num_rows() {
            let room = self.config.row_group_rows - self.pending_rows;
            let take = room.min(batch.num_rows() - offset);
            // `slice` is O(columns): the buffers are shared, not copied.
            self.pending.push(batch.slice(offset, take));
            self.pending_rows += take;
            offset += take;
            if self.pending_rows >= self.config.row_group_rows {
                self.flush_row_group()?;
            }
        }
        Ok(())
    }

    /// Force the buffered rows out as a row group, even if it is short.
    ///
    /// Useful when a caller wants a checkpoint; otherwise
    /// [`SegmentWriter::finish`] does it.
    pub fn flush_row_group(&mut self) -> Result<()> {
        if self.pending_rows == 0 {
            return Ok(());
        }
        let batches = std::mem::take(&mut self.pending);
        let num_rows = self.pending_rows as u64;
        self.pending_rows = 0;

        let mut columns = Vec::with_capacity(self.schema.len());
        for index in 0..self.schema.len() {
            let parts: Vec<ColumnVector> = batches
                .iter()
                .map(|batch| batch.column(index).clone())
                .collect();
            let column = concat(&parts)?;
            columns.push(self.write_column_chunk(&column)?);
        }
        drop(batches);

        self.row_groups.push(RowGroupMetadata { num_rows, columns });
        self.total_rows += num_rows;
        Ok(())
    }

    fn write_column_chunk(&mut self, column: &ColumnVector) -> Result<ColumnChunkMetadata> {
        let zone_map = ZoneMap::from_column(column);
        let encoded = encode_column(column, self.config.compression_level)?;

        let data_offset = self.position;
        let block = frame_block(&encoded.bytes);
        self.write_all(&block)?;

        let bloom = if self.config.write_bloom_filters
            && zone_map.wants_bloom_filter(&column.data_type())
        {
            let mut filter = BloomFilter::with_capacity(
                zone_map.distinct_estimate.max(1) as usize,
                self.config.bloom_filter_fpp,
            );
            for row in 0..column.len() {
                if !column.is_null(row) {
                    filter.insert(value_hash(&column.value(row)));
                }
            }
            let offset = self.position;
            let block = frame_block(&filter.to_bytes());
            let length = block.len() as u64;
            self.write_all(&block)?;
            Some(BlockLocation { offset, length })
        } else {
            None
        };

        Ok(ColumnChunkMetadata {
            data: BlockLocation {
                offset: data_offset,
                length: encoded.bytes.len() as u64 + BLOCK_HEADER_LEN,
            },
            encoding: encoded.encoding,
            uncompressed_bytes: encoded.uncompressed_bytes as u64,
            zone_map,
            bloom,
        })
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes).io_context(format!(
            "writing {} bytes to {}",
            bytes.len(),
            self.object
        ))?;
        self.position += bytes.len() as u64;
        Ok(())
    }

    /// Flush the last row group, write the footer and trailer, and return the
    /// metadata that a reader will see.
    ///
    /// **This must be called.** A writer that is simply dropped leaves a file
    /// with no footer and no trailing magic, which
    /// [`SegmentReader::open`](crate::SegmentReader::open) rejects with a
    /// corruption error. That is the intended failure mode: an incomplete
    /// segment is unreadable, never readable-but-wrong. `finish` takes `self`
    /// so the type system points this out at every call site.
    pub fn finish(mut self) -> Result<SegmentMetadata> {
        self.flush_row_group()?;
        let metadata = SegmentMetadata {
            format_version: FORMAT_VERSION,
            schema: Arc::clone(&self.schema),
            row_groups: std::mem::take(&mut self.row_groups),
            total_rows: self.total_rows,
            metadata: self.schema.metadata().clone(),
        };
        let footer_offset = self.position;
        let block = frame_block(&metadata.to_bytes());
        let footer_length = block.len() as u32;
        self.write_all(&block)?;
        self.write_all(&trailer(footer_offset, footer_length))?;
        self.writer
            .flush()
            .io_context(format!("flushing {}", self.object))?;
        Ok(metadata)
    }
}

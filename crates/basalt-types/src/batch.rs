//! `RecordBatch`: a schema plus equal-length columns.

use std::fmt;
use std::sync::Arc;

use basalt_common::{BasaltError, Result};

use crate::{ColumnVector, Schema};

/// A horizontal slice of a table: one [`Schema`] and one [`ColumnVector`] per
/// field, all of the same length.
///
/// This is the unit of work everywhere above storage. Construction is validated
/// once ([`RecordBatch::try_new`]) so that every operator downstream can assume
/// the invariants — same arity as the schema, matching types, equal lengths, and
/// no nulls in a `NOT NULL` column — without re-checking them per row.
///
/// [`RecordBatch::slice`] is O(number of columns): the columns are views, so
/// slicing a batch copies nothing but the (cheap) `ColumnVector` enums.
///
/// ```
/// use basalt_types::{ColumnVector, DataType, Field, PrimitiveArray, RecordBatch, Schema, StringArray};
/// use std::sync::Arc;
///
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("id", DataType::Int64, false),
///     Field::new("name", DataType::Utf8, true),
/// ]).unwrap());
///
/// let batch = RecordBatch::try_new(schema, vec![
///     ColumnVector::Int64(PrimitiveArray::from_vec(vec![1, 2, 3])),
///     ColumnVector::Utf8(StringArray::from_strs(&["a", "b", "c"])),
/// ]).unwrap();
///
/// assert_eq!(batch.num_rows(), 3);
/// assert_eq!(batch.slice(1, 2).num_rows(), 2);
/// assert_eq!(batch.column_by_name("name").unwrap().len(), 3);
/// ```
#[derive(Clone, PartialEq)]
pub struct RecordBatch {
    schema: Arc<Schema>,
    columns: Vec<ColumnVector>,
    num_rows: usize,
}

impl RecordBatch {
    /// Build a batch, validating it against its schema.
    ///
    /// # Errors
    ///
    /// * the number of columns does not match the schema,
    /// * a column's type does not match its field,
    /// * the columns have different lengths,
    /// * a `NOT NULL` field's column contains nulls.
    ///
    /// The null check is O(columns * rows / 64). It is worth it: a `NOT NULL`
    /// guarantee that is not enforced at the boundary is worse than no
    /// guarantee, because kernels skip the validity check on the strength of it.
    pub fn try_new(schema: Arc<Schema>, columns: Vec<ColumnVector>) -> Result<RecordBatch> {
        if schema.len() != columns.len() {
            return Err(BasaltError::schema(format!(
                "schema has {} fields but {} columns were supplied",
                schema.len(),
                columns.len()
            )));
        }
        let num_rows = columns.first().map_or(0, ColumnVector::len);
        for (i, column) in columns.iter().enumerate() {
            let field = schema.field(i);
            if column.len() != num_rows {
                return Err(BasaltError::schema(format!(
                    "column `{}` has {} rows but column `{}` has {num_rows}",
                    field.name(),
                    column.len(),
                    schema.field(0).name()
                )));
            }
            let actual = column.data_type();
            if &actual != field.data_type() {
                return Err(BasaltError::schema(format!(
                    "column `{}` is {actual} but the schema says {}",
                    field.name(),
                    field.data_type()
                )));
            }
            if !field.is_nullable() && column.null_count() > 0 {
                return Err(BasaltError::schema(format!(
                    "column `{}` is declared NOT NULL but contains {} nulls",
                    field.name(),
                    column.null_count()
                )));
            }
        }
        Ok(RecordBatch {
            schema,
            columns,
            num_rows,
        })
    }

    /// A batch with the given schema and no rows.
    pub fn empty(schema: Arc<Schema>) -> Result<RecordBatch> {
        let columns = schema
            .fields()
            .iter()
            .map(|f| crate::ColumnBuilder::new(f.data_type()).map(crate::ColumnBuilder::finish))
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(schema, columns)
    }

    /// The schema.
    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// The columns, in schema order.
    pub fn columns(&self) -> &[ColumnVector] {
        &self.columns
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// `true` if the batch has no rows.
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// The column at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= num_columns`.
    pub fn column(&self, index: usize) -> &ColumnVector {
        &self.columns[index]
    }

    /// The column called `name`.
    pub fn column_by_name(&self, name: &str) -> Result<&ColumnVector> {
        self.schema.index_of(name).map(|i| &self.columns[i])
    }

    /// A zero-copy view of `len` rows starting at `offset`.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds.
    pub fn slice(&self, offset: usize, len: usize) -> RecordBatch {
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= self.num_rows),
            "RecordBatch::slice({offset}, {len}) out of bounds for {} rows",
            self.num_rows
        );
        RecordBatch {
            schema: Arc::clone(&self.schema),
            columns: self.columns.iter().map(|c| c.slice(offset, len)).collect(),
            num_rows: len,
        }
    }

    /// Keep only the listed columns, in that order.
    ///
    /// The columns are shared, not copied, so this is the cheap half of
    /// projection pushdown; the expensive half (not reading the bytes at all)
    /// happens in `basalt-storage`.
    pub fn project(&self, indices: &[usize]) -> Result<RecordBatch> {
        let schema = Arc::new(self.schema.project(indices)?);
        let columns = indices.iter().map(|&i| self.columns[i].clone()).collect();
        Ok(RecordBatch {
            schema,
            columns,
            num_rows: self.num_rows,
        })
    }
}

impl fmt::Debug for RecordBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const PREVIEW: usize = 8;
        writeln!(f, "RecordBatch {} rows {}", self.num_rows, self.schema)?;
        for row in 0..self.num_rows.min(PREVIEW) {
            f.write_str("  ")?;
            for (i, column) in self.columns.iter().enumerate() {
                if i > 0 {
                    f.write_str(" | ")?;
                }
                write!(f, "{}", column.value(row))?;
            }
            f.write_str("\n")?;
        }
        if self.num_rows > PREVIEW {
            writeln!(f, "  ... {} more rows", self.num_rows - PREVIEW)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::{Bitmap, Buffer, DataType, Field, PrimitiveArray, StringArray, Value};

    fn schema() -> Arc<Schema> {
        Arc::new(
            Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, true),
            ])
            .unwrap(),
        )
    }

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                ColumnVector::Int64(PrimitiveArray::from_vec(vec![1, 2, 3])),
                ColumnVector::Utf8(StringArray::from_strs(&["a", "b", "c"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn construction_and_access() {
        let b = batch();
        assert_eq!(b.num_rows(), 3);
        assert_eq!(b.num_columns(), 2);
        assert!(!b.is_empty());
        assert_eq!(b.column(0).value(0), Value::Int64(1));
        assert_eq!(
            b.column_by_name("name").unwrap().value(2),
            Value::Utf8("c".into())
        );
        assert!(b.column_by_name("zzz").is_err());
        assert_eq!(b.schema().len(), 2);
        assert_eq!(b.columns().len(), 2);
    }

    #[test]
    fn slicing_is_a_view() {
        let b = batch();
        let s = b.slice(1, 2);
        assert_eq!(s.num_rows(), 2);
        assert_eq!(s.column(0).value(0), Value::Int64(2));
        assert_eq!(s.slice(1, 1).column(1).value(0), Value::Utf8("c".into()));
        assert_eq!(b.slice(0, 0).num_rows(), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_out_of_bounds_panics() {
        let _ = batch().slice(2, 2);
    }

    #[test]
    fn projection() {
        let b = batch();
        let p = b.project(&[1]).unwrap();
        assert_eq!(p.num_columns(), 1);
        assert_eq!(p.schema().field(0).name(), "name");
        assert_eq!(p.num_rows(), 3);
        assert!(b.project(&[7]).is_err());
    }

    #[test]
    fn validation_catches_every_mismatch() {
        // arity
        assert!(RecordBatch::try_new(schema(), vec![]).is_err());
        // type
        let err = RecordBatch::try_new(
            schema(),
            vec![
                ColumnVector::Int32(PrimitiveArray::from_vec(vec![1])),
                ColumnVector::Utf8(StringArray::from_strs(&["a"])),
            ],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("is INT32 but the schema says INT64"),
            "{err}"
        );
        // length
        let err = RecordBatch::try_new(
            schema(),
            vec![
                ColumnVector::Int64(PrimitiveArray::from_vec(vec![1, 2])),
                ColumnVector::Utf8(StringArray::from_strs(&["a"])),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("rows"), "{err}");
        // nullability
        let nullable_ids = PrimitiveArray::try_new(
            Buffer::from_vec(vec![1i64]),
            Some(Bitmap::from_bools(&[false])),
        )
        .unwrap();
        let err = RecordBatch::try_new(
            schema(),
            vec![
                ColumnVector::Int64(nullable_ids),
                ColumnVector::Utf8(StringArray::from_strs(&["a"])),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("NOT NULL"), "{err}");
    }

    #[test]
    fn empty_batch_matches_schema() {
        let b = RecordBatch::empty(schema()).unwrap();
        assert_eq!(b.num_rows(), 0);
        assert_eq!(b.num_columns(), 2);
        assert!(b.is_empty());
    }

    #[test]
    fn debug_shows_rows() {
        let text = format!("{:?}", batch());
        assert!(text.contains("RecordBatch 3 rows"), "{text}");
        assert!(text.contains("1 | a"), "{text}");
        let big = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]).unwrap()),
            vec![ColumnVector::Int64(PrimitiveArray::from_vec(
                (0..20).collect(),
            ))],
        )
        .unwrap();
        assert!(format!("{big:?}").contains("12 more rows"));
    }
}

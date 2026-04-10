//! Schemas: named, typed, nullable columns.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use basalt_common::{BasaltError, Result};

use crate::DataType;

/// One column of a [`Schema`].
///
/// Metadata is a sorted `BTreeMap` rather than a `HashMap` so that two schemas
/// with the same entries serialise to the same bytes — the segment footer
/// stores it, and a format whose byte layout depends on hash iteration order is
/// not reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    name: String,
    data_type: DataType,
    nullable: bool,
    metadata: BTreeMap<String, String>,
}

impl Field {
    /// A field with no metadata.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Field {
            name: name.into(),
            data_type,
            nullable,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach a metadata entry (builder style).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// The column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The column type.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Whether the column may contain nulls.
    ///
    /// A `false` here is a *guarantee* that [`RecordBatch::try_new`] and the
    /// segment writer both enforce, which lets scan kernels skip the validity
    /// check entirely.
    ///
    /// [`RecordBatch::try_new`]: crate::RecordBatch::try_new
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Free-form per-column metadata.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.data_type)?;
        if self.nullable {
            f.write_str(" NULL")
        } else {
            f.write_str(" NOT NULL")
        }
    }
}

/// An ordered list of uniquely named [`Field`]s.
///
/// Name lookup is O(1) through a side index built once at construction; the
/// alternative (a linear scan per lookup) shows up immediately in wide tables
/// during planning.
///
/// ```
/// use basalt_types::{DataType, Field, Schema};
///
/// let schema = Schema::new(vec![
///     Field::new("id", DataType::Int64, false),
///     Field::new("name", DataType::Utf8, true),
/// ]).unwrap();
///
/// assert_eq!(schema.index_of("name").unwrap(), 1);
/// assert_eq!(schema.field(0).name(), "id");
/// assert!(schema.index_of("nope").is_err());
/// ```
#[derive(Debug, Clone)]
pub struct Schema {
    fields: Vec<Field>,
    by_name: HashMap<String, usize>,
    metadata: BTreeMap<String, String>,
}

impl Schema {
    /// Build a schema, rejecting duplicate column names.
    ///
    /// Duplicates are refused rather than shadowed: `SELECT a` against a schema
    /// with two `a`s has no correct answer, and finding out at plan time beats
    /// finding out after the join.
    pub fn new(fields: Vec<Field>) -> Result<Schema> {
        let mut by_name = HashMap::with_capacity(fields.len());
        for (i, f) in fields.iter().enumerate() {
            if let Some(previous) = by_name.insert(f.name.clone(), i) {
                return Err(BasaltError::schema(format!(
                    "duplicate column name `{}` at positions {previous} and {i}",
                    f.name
                )));
            }
        }
        Ok(Schema {
            fields,
            by_name,
            metadata: BTreeMap::new(),
        })
    }

    /// The empty schema.
    pub fn empty() -> Schema {
        Schema {
            fields: Vec::new(),
            by_name: HashMap::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Attach schema-level metadata (builder style).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// The fields, in order.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// `true` if there are no columns.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= len`.
    pub fn field(&self, index: usize) -> &Field {
        &self.fields[index]
    }

    /// Position of the column called `name`.
    ///
    /// # Errors
    ///
    /// [`BasaltError::Schema`] naming the column and listing the candidates,
    /// because "column not found" without the alternatives is a bad log line.
    pub fn index_of(&self, name: &str) -> Result<usize> {
        self.by_name.get(name).copied().ok_or_else(|| {
            let mut names: Vec<&str> = self.fields.iter().map(Field::name).collect();
            names.sort_unstable();
            BasaltError::schema(format!(
                "no column named `{name}`; available columns: [{}]",
                names.join(", ")
            ))
        })
    }

    /// The field called `name`.
    pub fn field_by_name(&self, name: &str) -> Result<&Field> {
        self.index_of(name).map(|i| &self.fields[i])
    }

    /// Schema-level metadata.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// A schema containing only the listed column positions, in that order.
    ///
    /// Repeating an index is allowed only if the resulting names stay unique,
    /// which in practice means it is not; the error says so.
    pub fn project(&self, indices: &[usize]) -> Result<Schema> {
        let mut fields = Vec::with_capacity(indices.len());
        for &i in indices {
            let f = self.fields.get(i).ok_or_else(|| {
                BasaltError::schema(format!(
                    "projection index {i} out of range for a {}-column schema",
                    self.fields.len()
                ))
            })?;
            fields.push(f.clone());
        }
        let mut projected = Schema::new(fields)?;
        projected.metadata = self.metadata.clone();
        Ok(projected)
    }
}

impl PartialEq for Schema {
    /// Compares fields and metadata; the name index is derived, so it is not
    /// part of the value.
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields && self.metadata == other.metadata
    }
}

impl Eq for Schema {}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("(")?;
        for (i, field) in self.fields.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{field}")?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true).with_metadata("collation", "binary"),
        ])
        .unwrap()
    }

    #[test]
    fn lookup_and_projection() {
        let s = schema();
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
        assert_eq!(s.index_of("id").unwrap(), 0);
        assert_eq!(
            s.field_by_name("name").unwrap().data_type(),
            &DataType::Utf8
        );
        assert!(s.field(1).is_nullable());
        assert_eq!(
            s.field(1).metadata().get("collation").map(String::as_str),
            Some("binary")
        );
        let p = s.project(&[1]).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.field(0).name(), "name");
        assert!(s.project(&[5]).is_err());
        assert!(s.project(&[0, 0]).is_err());
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let err = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("a", DataType::Int64, false),
        ])
        .unwrap_err();
        assert!(
            err.to_string().contains("duplicate column name `a`"),
            "{err}"
        );
    }

    #[test]
    fn missing_column_error_lists_candidates() {
        let err = schema().index_of("nmae").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nmae"), "{msg}");
        assert!(msg.contains("id, name"), "{msg}");
    }

    #[test]
    fn display_and_metadata() {
        let s = schema().with_metadata("source", "tpch");
        assert_eq!(s.to_string(), "(id: INT64 NOT NULL, name: UTF8 NULL)");
        assert_eq!(s.metadata().get("source").map(String::as_str), Some("tpch"));
        assert_eq!(s.project(&[0]).unwrap().metadata().len(), 1);
        assert!(Schema::empty().is_empty());
        assert_eq!(schema(), schema());
        assert_ne!(schema(), Schema::empty());
    }
}

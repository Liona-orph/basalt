//! Tables, their schemas and their statistics.
//!
//! The catalog is the only thing the cost model can see, so the statistics live
//! here rather than in the optimizer: a plan is only as good as the numbers it
//! was costed with, and hiding them inside the optimizer would make it
//! impossible for the storage layer to keep them current.

use std::collections::BTreeMap;
use std::sync::Arc;

use basalt_common::{BasaltError, Result};
use basalt_types::{DataType, Schema, Value};

/// Per-column summary statistics.
///
/// Deliberately the same four numbers a `basalt_storage::ZoneMap`
/// carries, because that is where they come from: after a segment is written,
/// merging its zone maps into the catalog is a fold, not a second scan.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnStatistics {
    /// Rows where the column is null.
    pub null_count: u64,
    /// Estimated number of distinct non-null values.
    ///
    /// An *estimate*: it comes from a linear-counting sketch, and it is the
    /// single number the selectivity model leans on hardest. See
    /// `basalt_optimizer::cost` for what happens when it is wrong.
    pub distinct_count: u64,
    /// Smallest non-null value, if known.
    pub min: Option<Value>,
    /// Largest non-null value, if known.
    pub max: Option<Value>,
}

impl ColumnStatistics {
    /// Statistics that claim nothing.
    pub fn unknown() -> ColumnStatistics {
        ColumnStatistics::default()
    }

    /// `true` when nothing useful is known and the model must fall back to its
    /// default selectivities.
    pub fn is_unknown(&self) -> bool {
        self.distinct_count == 0 && self.min.is_none() && self.max.is_none()
    }

    /// Merge statistics from two disjoint row sets (e.g. two segments).
    ///
    /// Distinct counts are combined with `max`, not `+`: the two sets usually
    /// overlap heavily (think a `country` column across segments), and summing
    /// would make every equality look far more selective than it is. `max` is
    /// the tight lower bound and errs towards *less* selectivity, which is the
    /// safe direction for join ordering.
    pub fn merge(&self, other: &ColumnStatistics) -> ColumnStatistics {
        ColumnStatistics {
            null_count: self.null_count + other.null_count,
            distinct_count: self.distinct_count.max(other.distinct_count),
            min: min_option(self.min.clone(), other.min.clone()),
            max: max_option(self.max.clone(), other.max.clone()),
        }
    }
}

fn min_option(a: Option<Value>, b: Option<Value>) -> Option<Value> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a <= b { a } else { b }),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

fn max_option(a: Option<Value>, b: Option<Value>) -> Option<Value> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

/// Table-level statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableStatistics {
    /// Number of rows, or `None` when the table has never been analysed.
    pub row_count: Option<u64>,
    /// One entry per column, in schema order. May be shorter than the schema.
    pub columns: Vec<ColumnStatistics>,
}

impl TableStatistics {
    /// Statistics for an empty, freshly created table.
    pub fn empty(columns: usize) -> TableStatistics {
        TableStatistics {
            row_count: Some(0),
            columns: vec![ColumnStatistics::unknown(); columns],
        }
    }

    /// Statistics for column `index`, or `unknown` when absent.
    pub fn column(&self, index: usize) -> ColumnStatistics {
        self.columns
            .get(index)
            .cloned()
            .unwrap_or_else(ColumnStatistics::unknown)
    }
}

/// A table as the planner sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct TableDefinition {
    /// The name as written at `CREATE TABLE` time, preserved for display.
    pub name: String,
    /// Its columns.
    pub schema: Arc<Schema>,
    /// What is known about its contents.
    pub statistics: TableStatistics,
}

impl TableDefinition {
    /// A table with no statistics yet.
    pub fn new(name: impl Into<String>, schema: Schema) -> TableDefinition {
        let columns = schema.len();
        TableDefinition {
            name: name.into(),
            schema: Arc::new(schema),
            statistics: TableStatistics::empty(columns),
        }
    }

    /// Attach statistics (builder style).
    #[must_use]
    pub fn with_statistics(mut self, statistics: TableStatistics) -> TableDefinition {
        self.statistics = statistics;
        self
    }

    /// The type of column `name`.
    ///
    /// # Errors
    ///
    /// If the column does not exist.
    pub fn column_type(&self, name: &str) -> Result<DataType> {
        Ok(self.schema.field_by_name(name)?.data_type().clone())
    }
}

/// The set of tables a statement may refer to.
///
/// Lookup is case-insensitive (SQL folds unquoted identifiers), but the original
/// spelling is preserved for error messages and `EXPLAIN` output.
///
/// ```
/// use basalt_plan::{Catalog, TableDefinition};
/// use basalt_types::{DataType, Field, Schema};
///
/// let mut catalog = Catalog::new();
/// catalog.create_table(TableDefinition::new(
///     "Orders",
///     Schema::new(vec![Field::new("id", DataType::Int64, false)])?,
/// ))?;
/// assert!(catalog.table("orders").is_ok());
/// assert!(catalog.table("ORDERS").is_ok());
/// assert!(catalog.table("nope").is_err());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    tables: BTreeMap<String, Arc<TableDefinition>>,
}

impl Catalog {
    /// An empty catalog.
    pub fn new() -> Catalog {
        Catalog::default()
    }

    /// Register a table.
    ///
    /// # Errors
    ///
    /// If a table with that name (case-insensitively) already exists.
    pub fn create_table(&mut self, definition: TableDefinition) -> Result<()> {
        let key = definition.name.to_ascii_lowercase();
        if self.tables.contains_key(&key) {
            return Err(BasaltError::plan(format!(
                "table `{}` already exists",
                definition.name
            )));
        }
        self.tables.insert(key, Arc::new(definition));
        Ok(())
    }

    /// Register a table, replacing any existing definition.
    pub fn replace_table(&mut self, definition: TableDefinition) {
        self.tables
            .insert(definition.name.to_ascii_lowercase(), Arc::new(definition));
    }

    /// Remove a table, reporting whether it was there.
    pub fn drop_table(&mut self, name: &str) -> bool {
        self.tables.remove(&name.to_ascii_lowercase()).is_some()
    }

    /// Look a table up.
    ///
    /// # Errors
    ///
    /// If it does not exist. The message lists the closest known names, because
    /// "table not found" without candidates is the least useful error in a
    /// database.
    pub fn table(&self, name: &str) -> Result<Arc<TableDefinition>> {
        self.tables
            .get(&name.to_ascii_lowercase())
            .cloned()
            .ok_or_else(|| {
                let mut known: Vec<&str> = self.tables.values().map(|t| t.name.as_str()).collect();
                known.sort_unstable();
                if known.is_empty() {
                    BasaltError::plan(format!("unknown table `{name}`; the catalog is empty"))
                } else {
                    BasaltError::plan(format!(
                        "unknown table `{name}`; known tables are {}",
                        known.join(", ")
                    ))
                }
            })
    }

    /// `true` if the table exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tables.contains_key(&name.to_ascii_lowercase())
    }

    /// Every table, in name order.
    pub fn tables(&self) -> impl Iterator<Item = &Arc<TableDefinition>> {
        self.tables.values()
    }

    /// Number of tables.
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// `true` when the catalog has no tables.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// Replace a table's statistics, e.g. after an insert.
    ///
    /// # Errors
    ///
    /// If the table does not exist.
    pub fn set_statistics(&mut self, name: &str, statistics: TableStatistics) -> Result<()> {
        let key = name.to_ascii_lowercase();
        let existing = self
            .tables
            .get(&key)
            .ok_or_else(|| BasaltError::plan(format!("unknown table `{name}`")))?;
        let updated = TableDefinition {
            name: existing.name.clone(),
            schema: Arc::clone(&existing.schema),
            statistics,
        };
        self.tables.insert(key, Arc::new(updated));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use basalt_types::Field;

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ])
        .unwrap()
    }

    #[test]
    fn create_lookup_drop() {
        let mut catalog = Catalog::new();
        catalog
            .create_table(TableDefinition::new("Orders", schema()))
            .unwrap();
        assert!(catalog.contains("orders"));
        assert_eq!(catalog.table("ORDERS").unwrap().name, "Orders");
        assert!(catalog
            .create_table(TableDefinition::new("orders", schema()))
            .is_err());
        assert!(catalog.drop_table("Orders"));
        assert!(!catalog.drop_table("Orders"));
        assert!(catalog.is_empty());
    }

    #[test]
    fn unknown_table_lists_candidates() {
        let mut catalog = Catalog::new();
        catalog
            .create_table(TableDefinition::new("orders", schema()))
            .unwrap();
        let msg = catalog.table("order").unwrap_err().to_string();
        assert!(msg.contains("orders"), "{msg}");
        assert!(Catalog::new()
            .table("x")
            .unwrap_err()
            .to_string()
            .contains("empty"));
    }

    #[test]
    fn statistics_merge_conservatively() {
        let a = ColumnStatistics {
            null_count: 2,
            distinct_count: 10,
            min: Some(Value::Int64(1)),
            max: Some(Value::Int64(5)),
        };
        let b = ColumnStatistics {
            null_count: 3,
            distinct_count: 8,
            min: Some(Value::Int64(0)),
            max: Some(Value::Int64(9)),
        };
        let merged = a.merge(&b);
        assert_eq!(merged.null_count, 5);
        assert_eq!(merged.distinct_count, 10, "max, not sum");
        assert_eq!(merged.min, Some(Value::Int64(0)));
        assert_eq!(merged.max, Some(Value::Int64(9)));
        assert!(ColumnStatistics::unknown().is_unknown());
        assert!(!a.is_unknown());
    }

    #[test]
    fn statistics_are_replaceable() {
        let mut catalog = Catalog::new();
        catalog
            .create_table(TableDefinition::new("t", schema()))
            .unwrap();
        catalog
            .set_statistics(
                "t",
                TableStatistics {
                    row_count: Some(1_000),
                    columns: vec![ColumnStatistics::unknown(); 2],
                },
            )
            .unwrap();
        assert_eq!(
            catalog.table("t").unwrap().statistics.row_count,
            Some(1_000)
        );
        assert!(catalog
            .set_statistics("nope", TableStatistics::default())
            .is_err());
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog.tables().count(), 1);
    }

    #[test]
    fn column_type_lookup() {
        let table = TableDefinition::new("t", schema());
        assert_eq!(table.column_type("id").unwrap(), DataType::Int64);
        assert!(table.column_type("nope").is_err());
        assert_eq!(TableStatistics::empty(2).columns.len(), 2);
        assert!(TableStatistics::default().column(7).is_unknown());
    }
}

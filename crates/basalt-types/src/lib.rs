//! The Basalt type system and columnar in-memory representation.
//!
//! Everything in the engine is built on the four ideas in this crate:
//!
//! * **[`DataType`]** — the logical types, plus a documented [`coerce`] lattice
//!   that says exactly which implicit conversions are allowed and, more
//!   importantly, which are refused.
//! * **[`Value`]** — an owned scalar with a genuine total order, so it can be a
//!   sort key, a hash key and a zone-map bound without three different
//!   comparison functions.
//! * **[`ColumnVector`]** — one array per physical type: a contiguous values
//!   buffer plus a bitmap validity mask, with O(1) slicing.
//! * **[`RecordBatch`]** — a [`Schema`] and equal-length columns, validated once
//!   at the boundary so operators can trust it.
//!
//! Plus [`Selection`], the late-materialisation vector the executor uses to
//! chain filters without copying data at every step.
//!
//! # A worked example
//!
//! ```
//! use std::sync::Arc;
//! use basalt_types::{
//!     Bitmap, ColumnBuilder, ColumnVector, DataType, Field, RecordBatch, Schema, Selection, Value,
//! };
//!
//! // Build a two-column batch through the runtime-typed builders.
//! let schema = Arc::new(Schema::new(vec![
//!     Field::new("id", DataType::Int64, false),
//!     Field::new("city", DataType::Utf8, true),
//! ])?);
//!
//! let mut ids = ColumnBuilder::with_capacity(&DataType::Int64, 4)?;
//! let mut cities = ColumnBuilder::with_capacity(&DataType::Utf8, 4)?;
//! for (id, city) in [(1i64, Some("oslo")), (2, None), (3, Some("kyoto")), (4, Some("lima"))] {
//!     ids.append_value(&Value::Int64(id))?;
//!     match city {
//!         Some(c) => cities.append_value(&Value::Utf8(c.to_string()))?,
//!         None => cities.append_null(),
//!     }
//! }
//! let batch = RecordBatch::try_new(schema, vec![ids.finish(), cities.finish()])?;
//! assert_eq!(batch.num_rows(), 4);
//! assert_eq!(batch.column(1).null_count(), 1);
//!
//! // Filter with a selection vector: no data is copied until `apply`.
//! let keep = Bitmap::from_bools(&[true, false, true, false]);
//! let selection = Selection::from_bitmap(&keep);
//! assert_eq!(selection.len(), 2);
//! let cities = selection.apply(batch.column(1))?;
//! assert_eq!(cities.value(1), Value::Utf8("kyoto".into()));
//! # Ok::<(), basalt_common::BasaltError>(())
//! ```
#![forbid(unsafe_code)]

mod array;
mod batch;
mod bitmap;
mod buffer;
mod builder;
mod datatype;
mod schema;
mod selection;
mod value;

pub use array::{
    concat, BinaryArray, BooleanArray, ColumnVector, NativeType, NullArray, PrimitiveArray,
    StringArray,
};
pub use batch::RecordBatch;
pub use bitmap::{Bitmap, BitmapBuilder, SetIndices};
pub use buffer::Buffer;
pub use builder::{BinaryBuilder, BooleanBuilder, ColumnBuilder, PrimitiveBuilder, StringBuilder};
pub use datatype::{coerce, is_lossless_coercion, DataType, TimeUnit, DECIMAL128_MAX_PRECISION};
pub use schema::{Field, Schema};
pub use selection::Selection;
pub use value::Value;

//! Shared foundations for the Basalt columnar engine.
//!
//! `basalt-common` is deliberately tiny: it holds the things that *every* other
//! crate needs and that would otherwise be duplicated or, worse, defined twice
//! with subtly different semantics.
//!
//! * [`BasaltError`] / [`Result`] — one error type for the whole engine, with
//!   enough structure that a single log line is enough to start debugging.
//! * [`Config`] — engine-wide tuning knobs with documented defaults.
//! * [`ByteSize`] — a byte count that prints like a human wrote it.
//! * [`Id`] / [`IdGenerator`] — monotonic identifiers for segments, batches and
//!   anything else that needs a cheap unique name.
//!
//! # Why one error type?
//!
//! Layered engines are usually written with one error enum per layer and a pile
//! of `From` impls. That buys very little (the top layer still has to match on
//! everything) and costs a lot: every conversion is an opportunity to drop
//! context. Basalt instead uses a single `#[non_exhaustive]` enum whose variants
//! are *categories*, and pushes the detail into the payload.
#![forbid(unsafe_code)]

mod byte_size;
mod config;
mod error;
mod id;

pub use byte_size::{ByteSize, ParseByteSizeError};
pub use config::{CompressionLevel, Config};
pub use error::{BasaltError, IoResultExt, Region, Result};
pub use id::{Id, IdGenerator};

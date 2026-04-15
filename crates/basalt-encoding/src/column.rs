//! Encoding and decoding of whole [`ColumnVector`]s.
//!
//! This is the interface `basalt-storage` uses. It maps each logical type onto
//! one of four physical pipelines and writes a self-describing chunk:
//!
//! ```text
//! chunk := [version u8] [flags u8] [row count uvarint]
//!          [validity length uvarint] [validity bytes]?    (flags bit 0)
//!          body
//! ```
//!
//! | Logical type | Body |
//! |---|---|
//! | `Null` | empty |
//! | `Boolean` | length-prefixed packed bits |
//! | integers, `Date32`, `Timestamp` | one integer stream |
//! | `Float32` / `Float64` | one integer stream **over the IEEE bit patterns** |
//! | `Decimal128` | narrow flag, then an integer stream or plain 16-byte values |
//! | `Utf8` / `Binary` | a variable-length body (see [`crate::ByteLayout`]) |
//!
//! # Floats through the integer pipeline
//!
//! Reinterpreting a float's bits as an integer is lossless and costs nothing, so
//! floats reuse the same codecs. It looks odd but it is exactly right: bit
//! patterns of *repeated* floats are identical, so dictionary and run-length
//! encoding work perfectly on categorical float data, while genuinely
//! continuous float data has high-entropy bit patterns and the size estimator
//! correctly falls back to `Plain`. The alternative — a separate float codec —
//! would duplicate every codec to gain nothing.
//!
//! # Nulls
//!
//! The validity bitmap is stored separately from the values, and the values
//! buffer stays *dense* (a placeholder is written under each null). Compacting
//! the nulls out would shrink the body, but it would also mean the decoder
//! cannot compute a row's position without a rank query over the bitmap, and
//! the placeholder is usually zero, which the codecs squeeze out for free.

use basalt_common::{BasaltError, CompressionLevel, Region, Result};
use basalt_types::{
    BinaryArray, BinaryBuilder, Bitmap, BooleanArray, Buffer, ColumnVector, DataType,
    PrimitiveArray, StringArray, StringBuilder, TimeUnit,
};

use crate::bytes::{
    decode_bytes, encode_bytes_adaptive, read_length_prefixed, write_length_prefixed,
};
use crate::codec::{Encoding, MAX_CHUNK_ELEMENTS};
use crate::integer::{decode_integers, encode_integers};
use crate::stats::{analyze, analyze_byte_column, ByteLayout};
use crate::varint;

/// Version of the chunk framing written by [`encode_column`].
pub const CHUNK_VERSION: u8 = 1;

const FLAG_HAS_VALIDITY: u8 = 0b0000_0001;
/// Set when a `Decimal128` column was narrowed to 64-bit values.
const FLAG_DECIMAL_NARROW: u8 = 0b0000_0010;

fn corrupt(offset: usize, detail: impl Into<String>) -> BasaltError {
    BasaltError::corruption("column chunk", Region::EncodedChunk, offset as u64, detail)
}

/// An encoded column chunk plus the metadata the segment writer records.
#[derive(Debug, Clone)]
pub struct EncodedColumn {
    /// The encoded bytes.
    pub bytes: Vec<u8>,
    /// The encoding chosen for the value stream.
    ///
    /// For variable-length columns this describes the *data buffer*; the layout
    /// choice (direct or dictionary) is recorded inside the chunk.
    pub encoding: Encoding,
    /// Size the values occupy in memory, for reporting compression ratios.
    pub uncompressed_bytes: usize,
}

/// Encode a column into a self-describing chunk.
///
/// The chunk records everything needed to reconstruct the column *except* its
/// [`DataType`], which the schema in the segment footer supplies.
///
/// ```
/// use basalt_common::CompressionLevel;
/// use basalt_encoding::{decode_column, encode_column};
/// use basalt_types::{ColumnVector, DataType, PrimitiveArray};
///
/// let column = ColumnVector::Int64(PrimitiveArray::from_opt_vec(
///     (0..1000).map(|i| if i % 10 == 0 { None } else { Some(i as i64) }).collect(),
/// ));
/// let encoded = encode_column(&column, CompressionLevel::BALANCED)?;
/// assert!(encoded.bytes.len() < encoded.uncompressed_bytes / 4);
/// assert_eq!(decode_column(&encoded.bytes, &DataType::Int64)?, column);
/// # Ok::<(), basalt_common::BasaltError>(())
/// ```
pub fn encode_column(column: &ColumnVector, level: CompressionLevel) -> Result<EncodedColumn> {
    let mut out = Vec::new();
    // Drop an all-valid mask rather than storing it. Slicing a column with
    // nulls keeps the shared bitmap even when the slice itself has none, so
    // without this a slice would encode differently from an identical column
    // built from scratch -- and would pay a bitmap for nothing.
    let validity = column.validity().filter(|mask| mask.count_zeros() > 0);
    let mut flags = if validity.is_some() {
        FLAG_HAS_VALIDITY
    } else {
        0
    };

    // The flag byte is patched after the body is encoded, because the decimal
    // path only learns whether it can narrow while encoding.
    out.push(CHUNK_VERSION);
    let flags_at = out.len();
    out.push(flags);
    varint::write_uvarint(column.len() as u64, &mut out);
    if let Some(mask) = validity {
        write_length_prefixed(&mask.to_bytes(), &mut out);
    }

    let (encoding, uncompressed_bytes) = match column {
        ColumnVector::Null(_) => (Encoding::Plain, 0),
        ColumnVector::Boolean(array) => {
            write_length_prefixed(&array.values().to_bytes(), &mut out);
            (Encoding::Plain, array.len().div_ceil(8))
        }
        ColumnVector::Int8(a) => encode_ints(&widen(a.values()), level, &mut out, a.len())?,
        ColumnVector::Int16(a) => encode_ints(&widen(a.values()), level, &mut out, a.len() * 2)?,
        ColumnVector::Int32(a) | ColumnVector::Date32(a) => {
            encode_ints(&widen(a.values()), level, &mut out, a.len() * 4)?
        }
        ColumnVector::Int64(a) | ColumnVector::Timestamp(a, _) => {
            encode_ints(a.values(), level, &mut out, a.len() * 8)?
        }
        ColumnVector::UInt8(a) => {
            encode_ints(&widen_unsigned(a.values()), level, &mut out, a.len())?
        }
        ColumnVector::UInt16(a) => {
            encode_ints(&widen_unsigned(a.values()), level, &mut out, a.len() * 2)?
        }
        ColumnVector::UInt32(a) => {
            encode_ints(&widen_unsigned(a.values()), level, &mut out, a.len() * 4)?
        }
        // `as i64` is a bit cast, not a numeric conversion: it round-trips
        // exactly and the codecs never interpret the sign.
        ColumnVector::UInt64(a) => {
            let widened: Vec<i64> = a.values().iter().map(|&v| v as i64).collect();
            encode_ints(&widened, level, &mut out, a.len() * 8)?
        }
        ColumnVector::Float32(a) => {
            let bits: Vec<i64> = a.values().iter().map(|v| i64::from(v.to_bits())).collect();
            encode_ints(&bits, level, &mut out, a.len() * 4)?
        }
        ColumnVector::Float64(a) => {
            let bits: Vec<i64> = a.values().iter().map(|v| v.to_bits() as i64).collect();
            encode_ints(&bits, level, &mut out, a.len() * 8)?
        }
        ColumnVector::Decimal128(a, _, _) => {
            let narrow = a.values().iter().all(|v| i64::try_from(*v).is_ok());
            if narrow {
                flags |= FLAG_DECIMAL_NARROW;
                let narrowed: Vec<i64> = a
                    .values()
                    .iter()
                    .map(|v| i64::try_from(*v).unwrap_or(0))
                    .collect();
                encode_ints(&narrowed, level, &mut out, a.len() * 16)?
            } else {
                for value in a.values() {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                (Encoding::Plain, a.len() * 16)
            }
        }
        ColumnVector::Utf8(a) => {
            let values: Vec<&[u8]> = (0..a.len()).map(|i| a.value(i).as_bytes()).collect();
            let bytes = a.value_bytes();
            (
                encode_var_len(&values, level, &mut out)?,
                bytes + a.len() * 4,
            )
        }
        ColumnVector::Binary(a) => {
            let values: Vec<&[u8]> = (0..a.len()).map(|i| a.value(i)).collect();
            let bytes = a.value_bytes();
            (
                encode_var_len(&values, level, &mut out)?,
                bytes + a.len() * 4,
            )
        }
    };

    out[flags_at] = flags;
    Ok(EncodedColumn {
        bytes: out,
        encoding,
        uncompressed_bytes,
    })
}

fn widen<T: Copy + Into<i64>>(values: &[T]) -> Vec<i64> {
    values.iter().map(|&v| v.into()).collect()
}

fn widen_unsigned<T: Copy + Into<u64>>(values: &[T]) -> Vec<i64> {
    values.iter().map(|&v| v.into() as i64).collect()
}

fn encode_ints(
    values: &[i64],
    level: CompressionLevel,
    out: &mut Vec<u8>,
    uncompressed: usize,
) -> Result<(Encoding, usize)> {
    let plan = analyze(values, level);
    encode_integers(values, plan.encoding, out)?;
    Ok((plan.encoding, uncompressed))
}

/// Layout tag for a variable-length body.
const LAYOUT_DIRECT: u8 = 0;
const LAYOUT_DICTIONARY: u8 = 1;

fn encode_var_len(
    values: &[&[u8]],
    level: CompressionLevel,
    out: &mut Vec<u8>,
) -> Result<Encoding> {
    let plan = analyze_byte_column(values);
    match plan.layout {
        ByteLayout::Direct => {
            out.push(LAYOUT_DIRECT);
            let lengths: Vec<i64> = values.iter().map(|v| v.len() as i64).collect();
            let length_plan = analyze(&lengths, level);
            let mut length_stream = Vec::new();
            encode_integers(&lengths, length_plan.encoding, &mut length_stream)?;
            write_length_prefixed(&length_stream, out);

            let mut data = Vec::with_capacity(values.iter().map(|v| v.len()).sum());
            for value in values {
                data.extend_from_slice(value);
            }
            let mut data_stream = Vec::new();
            let encoding = encode_bytes_adaptive(
                &data,
                level.allows_byte_compression(),
                level.minimum_gain(),
                &mut data_stream,
            )?;
            write_length_prefixed(&data_stream, out);
            Ok(encoding)
        }
        ByteLayout::Dictionary => {
            out.push(LAYOUT_DICTIONARY);
            // A sorted dictionary keeps the codes meaningful for future
            // comparison pushdown, exactly as in the integer dictionary codec.
            let mut dictionary: Vec<&[u8]> = values.to_vec();
            dictionary.sort_unstable();
            dictionary.dedup();
            let codes: Vec<i64> = values
                .iter()
                .map(|v| dictionary.partition_point(|d| d < v) as i64)
                .collect();

            varint::write_uvarint(dictionary.len() as u64, out);
            let dictionary_lengths: Vec<i64> = dictionary.iter().map(|v| v.len() as i64).collect();
            let mut length_stream = Vec::new();
            let length_plan = analyze(&dictionary_lengths, level);
            encode_integers(
                &dictionary_lengths,
                length_plan.encoding,
                &mut length_stream,
            )?;
            write_length_prefixed(&length_stream, out);

            let mut data = Vec::with_capacity(dictionary.iter().map(|v| v.len()).sum());
            for value in &dictionary {
                data.extend_from_slice(value);
            }
            let mut data_stream = Vec::new();
            let encoding = encode_bytes_adaptive(
                &data,
                level.allows_byte_compression(),
                level.minimum_gain(),
                &mut data_stream,
            )?;
            write_length_prefixed(&data_stream, out);

            let code_plan = analyze(&codes, level);
            let mut code_stream = Vec::new();
            encode_integers(&codes, code_plan.encoding, &mut code_stream)?;
            write_length_prefixed(&code_stream, out);
            Ok(encoding)
        }
    }
}

/// Decode a chunk written by [`encode_column`] back into a column of
/// `data_type`.
///
/// # Errors
///
/// [`BasaltError::Corruption`] with a byte offset for any structural problem:
/// wrong version, truncated body, validity mask of the wrong length, dictionary
/// code out of range.
// The body is one exhaustive `match` over `DataType`. Splitting it would hide
// the fact that this function *is* the chunk format, and would force the
// framing state (flags, row count, validity) through an extra argument struct
// for no benefit.
#[allow(clippy::too_many_lines)]
pub fn decode_column(bytes: &[u8], data_type: &DataType) -> Result<ColumnVector> {
    let mut pos = 0usize;
    let version = *bytes
        .first()
        .ok_or_else(|| corrupt(0, "empty column chunk"))?;
    if version != CHUNK_VERSION {
        return Err(corrupt(
            0,
            format!("chunk version {version} is not supported"),
        ));
    }
    pos += 1;
    let flags = *bytes
        .get(pos)
        .ok_or_else(|| corrupt(pos, "missing flags byte"))?;
    pos += 1;
    let row_count = varint::read_uvarint(bytes, &mut pos)? as usize;
    if row_count > MAX_CHUNK_ELEMENTS {
        return Err(corrupt(
            pos,
            format!("chunk claims {row_count} rows, above the {MAX_CHUNK_ELEMENTS} format limit"),
        ));
    }

    let validity = if flags & FLAG_HAS_VALIDITY != 0 {
        let raw = read_length_prefixed(bytes, &mut pos)?;
        Some(Bitmap::from_bytes(raw, row_count)?)
    } else {
        None
    };

    let body = &bytes[pos..];
    let column = match data_type {
        DataType::Null => ColumnVector::Null(basalt_types::NullArray::new(row_count)),
        DataType::Boolean => {
            let mut cursor = 0usize;
            let raw = read_length_prefixed(body, &mut cursor)?;
            let values = Bitmap::from_bytes(raw, row_count)?;
            ColumnVector::Boolean(BooleanArray::try_new(values, validity)?)
        }
        DataType::Int8 => ColumnVector::Int8(narrow_array(
            &decode_integers(body)?,
            validity,
            i8::try_from,
        )?),
        DataType::Int16 => ColumnVector::Int16(narrow_array(
            &decode_integers(body)?,
            validity,
            i16::try_from,
        )?),
        DataType::Int32 => ColumnVector::Int32(narrow_array(
            &decode_integers(body)?,
            validity,
            i32::try_from,
        )?),
        DataType::Date32 => ColumnVector::Date32(narrow_array(
            &decode_integers(body)?,
            validity,
            i32::try_from,
        )?),
        DataType::Int64 => ColumnVector::Int64(PrimitiveArray::try_new(
            Buffer::from_vec(decode_integers(body)?),
            validity,
        )?),
        DataType::Timestamp(unit) => ColumnVector::Timestamp(
            PrimitiveArray::try_new(Buffer::from_vec(decode_integers(body)?), validity)?,
            *unit,
        ),
        DataType::UInt8 => ColumnVector::UInt8(narrow_array(
            &decode_integers(body)?,
            validity,
            u8::try_from,
        )?),
        DataType::UInt16 => ColumnVector::UInt16(narrow_array(
            &decode_integers(body)?,
            validity,
            u16::try_from,
        )?),
        DataType::UInt32 => ColumnVector::UInt32(narrow_array(
            &decode_integers(body)?,
            validity,
            u32::try_from,
        )?),
        DataType::UInt64 => {
            let values: Vec<u64> = decode_integers(body)?
                .into_iter()
                .map(|v| v as u64)
                .collect();
            ColumnVector::UInt64(PrimitiveArray::try_new(Buffer::from_vec(values), validity)?)
        }
        DataType::Float32 => {
            let values: Vec<f32> = decode_integers(body)?
                .into_iter()
                .map(|v| {
                    u32::try_from(v as u64 & 0xFFFF_FFFF)
                        .map(f32::from_bits)
                        .unwrap_or(f32::NAN)
                })
                .collect();
            ColumnVector::Float32(PrimitiveArray::try_new(Buffer::from_vec(values), validity)?)
        }
        DataType::Float64 => {
            let values: Vec<f64> = decode_integers(body)?
                .into_iter()
                .map(|v| f64::from_bits(v as u64))
                .collect();
            ColumnVector::Float64(PrimitiveArray::try_new(Buffer::from_vec(values), validity)?)
        }
        DataType::Decimal128(precision, scale) => {
            let values: Vec<i128> = if flags & FLAG_DECIMAL_NARROW != 0 {
                decode_integers(body)?.into_iter().map(i128::from).collect()
            } else {
                if body.len() < row_count * 16 {
                    return Err(corrupt(
                        pos,
                        format!(
                            "decimal body needs {} bytes, found {}",
                            row_count * 16,
                            body.len()
                        ),
                    ));
                }
                (0..row_count)
                    .map(|i| {
                        let mut buf = [0u8; 16];
                        buf.copy_from_slice(&body[i * 16..(i + 1) * 16]);
                        i128::from_le_bytes(buf)
                    })
                    .collect()
            };
            ColumnVector::Decimal128(
                PrimitiveArray::try_new(Buffer::from_vec(values), validity)?,
                *precision,
                *scale,
            )
        }
        DataType::Utf8 => {
            let values = decode_var_len(body, row_count)?;
            let mut builder =
                StringBuilder::with_capacity(row_count, values.iter().map(Vec::len).sum());
            for value in &values {
                let text = std::str::from_utf8(value).map_err(|e| {
                    corrupt(
                        pos,
                        format!(
                            "UTF8 column holds invalid UTF-8 at byte {}",
                            e.valid_up_to()
                        ),
                    )
                })?;
                builder.append_value(text)?;
            }
            let array = builder.finish()?;
            ColumnVector::Utf8(rebind_validity_utf8(array, validity)?)
        }
        DataType::Binary => {
            let values = decode_var_len(body, row_count)?;
            let mut builder =
                BinaryBuilder::with_capacity(row_count, values.iter().map(Vec::len).sum());
            for value in &values {
                builder.append_value(value)?;
            }
            let array = builder.finish()?;
            ColumnVector::Binary(rebind_validity(array, validity)?)
        }
    };

    if column.len() != row_count {
        return Err(corrupt(
            pos,
            format!(
                "chunk header declared {row_count} rows but the body decoded {}",
                column.len()
            ),
        ));
    }
    Ok(column)
}

fn narrow_array<T, F, E>(
    values: &[i64],
    validity: Option<Bitmap>,
    convert: F,
) -> Result<PrimitiveArray<T>>
where
    T: basalt_types::NativeType,
    F: Fn(i64) -> std::result::Result<T, E>,
{
    let mut narrowed = Vec::with_capacity(values.len());
    for (i, &value) in values.iter().enumerate() {
        // A null slot's payload is unspecified, so a placeholder that no longer
        // fits the narrow type is not corruption; substitute the default.
        let is_null = validity.as_ref().is_some_and(|v| !v.get(i));
        match convert(value) {
            Ok(v) => narrowed.push(v),
            Err(_) if is_null => narrowed.push(T::default()),
            Err(_) => {
                return Err(corrupt(
                    0,
                    format!("decoded value {value} does not fit the column's narrower type"),
                ))
            }
        }
    }
    PrimitiveArray::try_new(Buffer::from_vec(narrowed), validity)
}

fn rebind_validity(array: BinaryArray, validity: Option<Bitmap>) -> Result<BinaryArray> {
    match validity {
        None => Ok(array),
        Some(mask) => BinaryArray::try_new(
            Buffer::from_vec(array.offsets().to_vec()),
            Buffer::from_vec(array.data().to_vec()),
            Some(mask),
        ),
    }
}

fn rebind_validity_utf8(array: StringArray, validity: Option<Bitmap>) -> Result<StringArray> {
    match validity {
        None => Ok(array),
        Some(mask) => StringArray::try_new(
            Buffer::from_vec(array.offsets().to_vec()),
            Buffer::from_vec(array.data().to_vec()),
            Some(mask),
        ),
    }
}

fn decode_var_len(body: &[u8], row_count: usize) -> Result<Vec<Vec<u8>>> {
    let mut pos = 0usize;
    let layout = *body
        .first()
        .ok_or_else(|| corrupt(0, "empty variable-length body"))?;
    pos += 1;
    match layout {
        LAYOUT_DIRECT => {
            let length_stream = read_length_prefixed(body, &mut pos)?;
            let lengths = decode_integers(length_stream)?;
            let data_stream = read_length_prefixed(body, &mut pos)?;
            let data = decode_bytes(data_stream)?;
            split(&data, &lengths, row_count)
        }
        LAYOUT_DICTIONARY => {
            let dictionary_len = varint::read_uvarint(body, &mut pos)? as usize;
            if dictionary_len > body.len() {
                return Err(corrupt(
                    pos,
                    format!(
                        "dictionary claims {dictionary_len} entries in a {}-byte body",
                        body.len()
                    ),
                ));
            }
            let length_stream = read_length_prefixed(body, &mut pos)?;
            let dictionary_lengths = decode_integers(length_stream)?;
            let data_stream = read_length_prefixed(body, &mut pos)?;
            let dictionary_data = decode_bytes(data_stream)?;
            let dictionary = split(&dictionary_data, &dictionary_lengths, dictionary_len)?;
            let code_stream = read_length_prefixed(body, &mut pos)?;
            let codes = decode_integers(code_stream)?;
            if codes.len() != row_count {
                return Err(corrupt(
                    pos,
                    format!("expected {row_count} codes, found {}", codes.len()),
                ));
            }
            codes
                .into_iter()
                .map(|code| {
                    usize::try_from(code)
                        .ok()
                        .and_then(|i| dictionary.get(i).cloned())
                        .ok_or_else(|| {
                            corrupt(
                                pos,
                                format!("dictionary code {code} is outside a {dictionary_len}-entry dictionary"),
                            )
                        })
                })
                .collect()
        }
        other => Err(corrupt(
            0,
            format!("unknown variable-length layout tag {other}"),
        )),
    }
}

fn split(data: &[u8], lengths: &[i64], expected: usize) -> Result<Vec<Vec<u8>>> {
    if lengths.len() != expected {
        return Err(corrupt(
            0,
            format!("expected {expected} lengths, found {}", lengths.len()),
        ));
    }
    let mut out = Vec::with_capacity(expected.min(data.len() + 1));
    let mut cursor = 0usize;
    for &length in lengths {
        let length = usize::try_from(length)
            .map_err(|_| corrupt(cursor, format!("negative value length {length}")))?;
        let end = cursor
            .checked_add(length)
            .ok_or_else(|| corrupt(cursor, "value length overflows"))?;
        if end > data.len() {
            return Err(corrupt(
                cursor,
                format!(
                    "value of {length} bytes runs past the {}-byte data buffer",
                    data.len()
                ),
            ));
        }
        out.push(data[cursor..end].to_vec());
        cursor = end;
    }
    Ok(out)
}

/// In-memory footprint of a column's values, used for compression reporting.
pub fn uncompressed_size(column: &ColumnVector) -> usize {
    match column {
        ColumnVector::Null(_) => 0,
        ColumnVector::Boolean(a) => a.len().div_ceil(8),
        ColumnVector::Utf8(a) => a.value_bytes() + a.len() * 4,
        ColumnVector::Binary(a) => a.value_bytes() + a.len() * 4,
        other => other.data_type().byte_width().unwrap_or(8) * other.len(),
    }
}

/// The [`TimeUnit`] of a column, if it has one. Small helper for the writer.
pub fn time_unit_of(column: &ColumnVector) -> Option<TimeUnit> {
    match column {
        ColumnVector::Timestamp(_, unit) => Some(*unit),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use basalt_types::{ColumnBuilder, Value};

    fn round_trip(column: &ColumnVector, level: CompressionLevel) -> ColumnVector {
        let encoded = encode_column(column, level).unwrap();
        decode_column(&encoded.bytes, &column.data_type()).unwrap()
    }

    fn build(data_type: &DataType, values: &[Option<Value>]) -> ColumnVector {
        let mut builder = ColumnBuilder::with_capacity(data_type, values.len()).unwrap();
        for value in values {
            match value {
                Some(v) => builder.append_value(v).unwrap(),
                None => builder.append_null(),
            }
        }
        builder.finish()
    }

    #[test]
    // A table of one case per logical type; length is the point.
    #[allow(clippy::too_many_lines)]
    fn every_type_round_trips_with_and_without_nulls() {
        let cases: Vec<(DataType, Vec<Option<Value>>)> = vec![
            (DataType::Null, vec![None; 5]),
            (
                DataType::Boolean,
                (0..300)
                    .map(|i| {
                        if i % 7 == 0 {
                            None
                        } else {
                            Some(Value::Boolean(i % 3 == 0))
                        }
                    })
                    .collect(),
            ),
            (
                DataType::Int8,
                (0..300)
                    .map(|i| Some(Value::Int8((i % 127) as i8)))
                    .collect(),
            ),
            (
                DataType::Int16,
                (0..300)
                    .map(|i| Some(Value::Int16(i as i16 - 150)))
                    .collect(),
            ),
            (
                DataType::Int32,
                (0..300)
                    .map(|i| {
                        if i % 5 == 0 {
                            None
                        } else {
                            Some(Value::Int32(i * 3))
                        }
                    })
                    .collect(),
            ),
            (
                DataType::Int64,
                (0..300)
                    .map(|i| Some(Value::Int64(i64::from(i) * 1_000_000_007)))
                    .collect(),
            ),
            (
                DataType::UInt8,
                (0..300)
                    .map(|i| Some(Value::UInt8((i % 256) as u8)))
                    .collect(),
            ),
            (
                DataType::UInt16,
                (0..300).map(|i| Some(Value::UInt16(i as u16))).collect(),
            ),
            (
                DataType::UInt32,
                (0..300)
                    .map(|i| Some(Value::UInt32(i as u32 * 7)))
                    .collect(),
            ),
            (
                DataType::UInt64,
                vec![Some(Value::UInt64(u64::MAX)), Some(Value::UInt64(0)), None],
            ),
            (
                DataType::Float32,
                (0..300)
                    .map(|i| Some(Value::Float32(i as f32 * 0.5)))
                    .collect(),
            ),
            (
                DataType::Float64,
                vec![
                    Some(Value::Float64(1.5)),
                    Some(Value::Float64(f64::NAN)),
                    Some(Value::Float64(f64::INFINITY)),
                    Some(Value::Float64(-0.0)),
                    None,
                ],
            ),
            (
                DataType::Utf8,
                (0..300)
                    .map(|i| {
                        if i % 11 == 0 {
                            None
                        } else {
                            Some(Value::Utf8(format!("city-{}", i % 7)))
                        }
                    })
                    .collect(),
            ),
            (
                DataType::Binary,
                (0..300)
                    .map(|i| Some(Value::Binary(vec![(i % 251) as u8; i % 5])))
                    .collect(),
            ),
            (
                DataType::Date32,
                (0..300).map(|i| Some(Value::Date32(19_000 + i))).collect(),
            ),
            (
                DataType::Timestamp(TimeUnit::Microsecond),
                (0..300)
                    .map(|i| {
                        Some(Value::Timestamp(
                            1_700_000_000_000_000 + i64::from(i) * 1_000,
                            TimeUnit::Microsecond,
                        ))
                    })
                    .collect(),
            ),
            (
                DataType::Decimal128(20, 4),
                (0..300)
                    .map(|i| Some(Value::Decimal128(i128::from(i) * 137, 20, 4)))
                    .collect(),
            ),
            (
                DataType::Decimal128(38, 0),
                vec![
                    Some(Value::Decimal128(i128::MAX / 2, 38, 0)),
                    Some(Value::Decimal128(i128::MIN / 2, 38, 0)),
                    None,
                ],
            ),
        ];
        for (data_type, values) in cases {
            let column = build(&data_type, &values);
            for level in [
                CompressionLevel::NONE,
                CompressionLevel::FAST,
                CompressionLevel::MAX,
            ] {
                let decoded = round_trip(&column, level);
                assert_eq!(
                    decoded.data_type(),
                    data_type,
                    "{data_type} at level {level:?}"
                );
                assert_eq!(decoded.len(), column.len(), "{data_type}");
                assert_eq!(decoded, column, "{data_type} at level {level:?}");
            }
        }
    }

    #[test]
    fn empty_and_single_row_columns() {
        for data_type in [
            DataType::Int32,
            DataType::Utf8,
            DataType::Boolean,
            DataType::Null,
        ] {
            let empty = build(&data_type, &[]);
            assert_eq!(
                round_trip(&empty, CompressionLevel::MAX),
                empty,
                "{data_type}"
            );
            let single = build(&data_type, &[None]);
            assert_eq!(
                round_trip(&single, CompressionLevel::MAX),
                single,
                "{data_type}"
            );
        }
    }

    #[test]
    fn sliced_columns_encode_correctly() {
        let column =
            ColumnVector::Utf8(StringArray::from_strs(&["alpha", "beta", "gamma", "delta"]));
        let sliced = column.slice(1, 2);
        let decoded = round_trip(&sliced, CompressionLevel::MAX);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.value(0), Value::Utf8("beta".into()));
        assert_eq!(decoded.value(1), Value::Utf8("gamma".into()));
    }

    #[test]
    fn compression_actually_happens() {
        let ids = ColumnVector::Int64(PrimitiveArray::from_vec((0..10_000).collect()));
        let encoded = encode_column(&ids, CompressionLevel::BALANCED).unwrap();
        assert!(
            encoded.bytes.len() * 100 < encoded.uncompressed_bytes,
            "{}",
            encoded.bytes.len()
        );
        assert_eq!(encoded.encoding, Encoding::DeltaBinaryPacked);

        // 12 distinct strings over 10 000 rows: the dictionary layout reduces
        // this to 4-bit codes plus a ~90-byte dictionary.
        let cities: Vec<String> = (0..10_000).map(|i| format!("city-{}", i % 12)).collect();
        let column = ColumnVector::Utf8(StringArray::from_strs(&cities));
        let encoded = encode_column(&column, CompressionLevel::BALANCED).unwrap();
        assert!(
            encoded.bytes.len() < 6_000,
            "12 distinct strings should pack into ~4 bits each, got {}",
            encoded.bytes.len()
        );
        assert!(encoded.bytes.len() * 10 < encoded.uncompressed_bytes);
        assert_eq!(
            decode_column(&encoded.bytes, &DataType::Utf8).unwrap(),
            column
        );
    }

    #[test]
    fn corrupt_chunks_are_rejected() {
        let column = ColumnVector::Int32(PrimitiveArray::from_vec((0..100).collect()));
        let encoded = encode_column(&column, CompressionLevel::MAX).unwrap();
        assert!(decode_column(&[], &DataType::Int32).is_err());
        let mut broken = encoded.bytes.clone();
        broken[0] = 99;
        let err = decode_column(&broken, &DataType::Int32).unwrap_err();
        assert!(err.to_string().contains("chunk version 99"), "{err}");
        for cut in [1, 3, encoded.bytes.len() / 2] {
            assert!(
                decode_column(&encoded.bytes[..cut], &DataType::Int32).is_err(),
                "cut {cut}"
            );
        }
        // Decoding as the wrong (narrower) type must fail loudly.
        let wide = ColumnVector::Int64(PrimitiveArray::from_vec(vec![i64::MAX]));
        let encoded = encode_column(&wide, CompressionLevel::NONE).unwrap();
        let err = decode_column(&encoded.bytes, &DataType::Int8).unwrap_err();
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn an_all_valid_mask_is_not_stored() {
        let with_nulls = build(
            &DataType::Int32,
            &(0..64)
                .map(|i| if i < 8 { None } else { Some(Value::Int32(i)) })
                .collect::<Vec<_>>(),
        );
        // Rows 8.. contain no nulls, so the slice must encode as if it never
        // had a validity bitmap.
        let sliced = with_nulls.slice(8, 56);
        assert!(
            sliced.validity().is_some(),
            "the slice still carries the shared bitmap"
        );
        let from_slice = encode_column(&sliced, CompressionLevel::MAX).unwrap();
        let rebuilt = build(
            &DataType::Int32,
            &(8..64).map(|i| Some(Value::Int32(i))).collect::<Vec<_>>(),
        );
        let from_scratch = encode_column(&rebuilt, CompressionLevel::MAX).unwrap();
        assert_eq!(from_slice.bytes, from_scratch.bytes);
        assert_eq!(
            decode_column(&from_slice.bytes, &DataType::Int32).unwrap(),
            sliced
        );
    }

    #[test]
    fn helpers() {
        let column =
            ColumnVector::Timestamp(PrimitiveArray::from_vec(vec![1i64; 4]), TimeUnit::Second);
        assert_eq!(time_unit_of(&column), Some(TimeUnit::Second));
        assert_eq!(uncompressed_size(&column), 32);
        assert_eq!(
            time_unit_of(&ColumnVector::Null(basalt_types::NullArray::new(1))),
            None
        );
        assert_eq!(
            uncompressed_size(&ColumnVector::Boolean(BooleanArray::from_bools(&[true; 9]))),
            2
        );
        assert_eq!(
            uncompressed_size(&ColumnVector::Null(basalt_types::NullArray::new(3))),
            0
        );
    }
}

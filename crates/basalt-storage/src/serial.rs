//! Canonical binary encoding of schemas and scalar values.
//!
//! Used by the footer and, crucially, by the bloom filter: writer and reader
//! must derive the *same* bytes from a value or the filter would produce false
//! negatives. Sharing one encoder is how that is guaranteed rather than hoped
//! for.

use std::collections::BTreeMap;

use basalt_common::{BasaltError, Region, Result};
use basalt_encoding::varint;
use basalt_types::{DataType, Field, Schema, TimeUnit, Value};

/// Type discriminants. These are on-disk constants and must never be reused.
mod type_code {
    pub(crate) const NULL: u8 = 0;
    pub(crate) const BOOLEAN: u8 = 1;
    pub(crate) const INT8: u8 = 2;
    pub(crate) const INT16: u8 = 3;
    pub(crate) const INT32: u8 = 4;
    pub(crate) const INT64: u8 = 5;
    pub(crate) const UINT8: u8 = 6;
    pub(crate) const UINT16: u8 = 7;
    pub(crate) const UINT32: u8 = 8;
    pub(crate) const UINT64: u8 = 9;
    pub(crate) const FLOAT32: u8 = 10;
    pub(crate) const FLOAT64: u8 = 11;
    pub(crate) const UTF8: u8 = 12;
    pub(crate) const BINARY: u8 = 13;
    pub(crate) const DATE32: u8 = 14;
    pub(crate) const TIMESTAMP: u8 = 15;
    pub(crate) const DECIMAL128: u8 = 16;
}

fn corrupt(offset: usize, region: Region, detail: impl Into<String>) -> BasaltError {
    BasaltError::corruption("segment footer", region, offset as u64, detail)
}

const fn time_unit_code(unit: TimeUnit) -> u8 {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    }
}

const fn time_unit_from_code(code: u8) -> Option<TimeUnit> {
    Some(match code {
        0 => TimeUnit::Second,
        1 => TimeUnit::Millisecond,
        2 => TimeUnit::Microsecond,
        3 => TimeUnit::Nanosecond,
        _ => return None,
    })
}

/// Append a [`DataType`].
pub(crate) fn write_data_type(data_type: &DataType, out: &mut Vec<u8>) {
    match data_type {
        DataType::Null => out.push(type_code::NULL),
        DataType::Boolean => out.push(type_code::BOOLEAN),
        DataType::Int8 => out.push(type_code::INT8),
        DataType::Int16 => out.push(type_code::INT16),
        DataType::Int32 => out.push(type_code::INT32),
        DataType::Int64 => out.push(type_code::INT64),
        DataType::UInt8 => out.push(type_code::UINT8),
        DataType::UInt16 => out.push(type_code::UINT16),
        DataType::UInt32 => out.push(type_code::UINT32),
        DataType::UInt64 => out.push(type_code::UINT64),
        DataType::Float32 => out.push(type_code::FLOAT32),
        DataType::Float64 => out.push(type_code::FLOAT64),
        DataType::Utf8 => out.push(type_code::UTF8),
        DataType::Binary => out.push(type_code::BINARY),
        DataType::Date32 => out.push(type_code::DATE32),
        DataType::Timestamp(unit) => {
            out.push(type_code::TIMESTAMP);
            out.push(time_unit_code(*unit));
        }
        DataType::Decimal128(precision, scale) => {
            out.push(type_code::DECIMAL128);
            out.push(*precision);
            out.push(*scale as u8);
        }
    }
}

/// Read a [`DataType`].
pub(crate) fn read_data_type(bytes: &[u8], pos: &mut usize) -> Result<DataType> {
    let code = read_u8(bytes, pos, Region::Footer)?;
    Ok(match code {
        type_code::NULL => DataType::Null,
        type_code::BOOLEAN => DataType::Boolean,
        type_code::INT8 => DataType::Int8,
        type_code::INT16 => DataType::Int16,
        type_code::INT32 => DataType::Int32,
        type_code::INT64 => DataType::Int64,
        type_code::UINT8 => DataType::UInt8,
        type_code::UINT16 => DataType::UInt16,
        type_code::UINT32 => DataType::UInt32,
        type_code::UINT64 => DataType::UInt64,
        type_code::FLOAT32 => DataType::Float32,
        type_code::FLOAT64 => DataType::Float64,
        type_code::UTF8 => DataType::Utf8,
        type_code::BINARY => DataType::Binary,
        type_code::DATE32 => DataType::Date32,
        type_code::TIMESTAMP => {
            let unit = read_u8(bytes, pos, Region::Footer)?;
            DataType::Timestamp(time_unit_from_code(unit).ok_or_else(|| {
                corrupt(
                    *pos,
                    Region::Footer,
                    format!("unknown time unit code {unit}"),
                )
            })?)
        }
        type_code::DECIMAL128 => {
            let precision = read_u8(bytes, pos, Region::Footer)?;
            let scale = read_u8(bytes, pos, Region::Footer)? as i8;
            DataType::decimal128(precision, scale).ok_or_else(|| {
                corrupt(
                    *pos,
                    Region::Footer,
                    format!("invalid decimal precision {precision} / scale {scale}"),
                )
            })?
        }
        other => {
            return Err(corrupt(
                *pos,
                Region::Footer,
                format!("unknown type code {other}"),
            ))
        }
    })
}

/// Append a [`Value`], tagged with its type so a mismatch is detectable.
///
/// This encoding is *canonical*: the same value always produces the same bytes,
/// which is what lets [`crate::bloom::value_hash`] be stable across processes.
pub(crate) fn write_value(value: &Value, out: &mut Vec<u8>) {
    write_data_type(&value.data_type(), out);
    match value {
        Value::Null => {}
        Value::Boolean(v) => out.push(u8::from(*v)),
        Value::Int8(v) => varint::write_varint(i64::from(*v), out),
        Value::Int16(v) => varint::write_varint(i64::from(*v), out),
        Value::Int32(v) | Value::Date32(v) => varint::write_varint(i64::from(*v), out),
        Value::Int64(v) | Value::Timestamp(v, _) => varint::write_varint(*v, out),
        Value::UInt8(v) => varint::write_uvarint(u64::from(*v), out),
        Value::UInt16(v) => varint::write_uvarint(u64::from(*v), out),
        Value::UInt32(v) => varint::write_uvarint(u64::from(*v), out),
        Value::UInt64(v) => varint::write_uvarint(*v, out),
        // Bit patterns, not decimal text: `-0.0` and `0.0` must stay distinct,
        // matching `Value`'s total order.
        Value::Float32(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        Value::Float64(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        Value::Utf8(v) => write_bytes(v.as_bytes(), out),
        Value::Binary(v) => write_bytes(v, out),
        Value::Decimal128(v, _, _) => varint::write_varint_i128(*v, out),
    }
}

/// Read a [`Value`] written by [`write_value`].
pub(crate) fn read_value(bytes: &[u8], pos: &mut usize) -> Result<Value> {
    let data_type = read_data_type(bytes, pos)?;
    Ok(match data_type {
        DataType::Null => Value::Null,
        DataType::Boolean => Value::Boolean(read_u8(bytes, pos, Region::ZoneMap)? != 0),
        DataType::Int8 => Value::Int8(narrow(varint::read_varint(bytes, pos)?, *pos)?),
        DataType::Int16 => Value::Int16(narrow(varint::read_varint(bytes, pos)?, *pos)?),
        DataType::Int32 => Value::Int32(narrow(varint::read_varint(bytes, pos)?, *pos)?),
        DataType::Int64 => Value::Int64(varint::read_varint(bytes, pos)?),
        DataType::UInt8 => Value::UInt8(narrow_u(varint::read_uvarint(bytes, pos)?, *pos)?),
        DataType::UInt16 => Value::UInt16(narrow_u(varint::read_uvarint(bytes, pos)?, *pos)?),
        DataType::UInt32 => Value::UInt32(narrow_u(varint::read_uvarint(bytes, pos)?, *pos)?),
        DataType::UInt64 => Value::UInt64(varint::read_uvarint(bytes, pos)?),
        DataType::Float32 => {
            let mut buf = [0u8; 4];
            read_exact(bytes, pos, &mut buf, Region::ZoneMap)?;
            Value::Float32(f32::from_bits(u32::from_le_bytes(buf)))
        }
        DataType::Float64 => {
            let mut buf = [0u8; 8];
            read_exact(bytes, pos, &mut buf, Region::ZoneMap)?;
            Value::Float64(f64::from_bits(u64::from_le_bytes(buf)))
        }
        DataType::Utf8 => {
            let raw = read_bytes(bytes, pos)?;
            Value::Utf8(String::from_utf8(raw).map_err(|e| {
                corrupt(
                    *pos,
                    Region::ZoneMap,
                    format!(
                        "UTF8 zone-map bound is not valid UTF-8 at byte {}",
                        e.utf8_error().valid_up_to()
                    ),
                )
            })?)
        }
        DataType::Binary => Value::Binary(read_bytes(bytes, pos)?),
        DataType::Date32 => Value::Date32(narrow(varint::read_varint(bytes, pos)?, *pos)?),
        DataType::Timestamp(unit) => Value::Timestamp(varint::read_varint(bytes, pos)?, unit),
        DataType::Decimal128(precision, scale) => {
            Value::Decimal128(varint::read_varint_i128(bytes, pos)?, precision, scale)
        }
    })
}

/// Append a length-prefixed byte string.
pub(crate) fn write_bytes(value: &[u8], out: &mut Vec<u8>) {
    varint::write_uvarint(value.len() as u64, out);
    out.extend_from_slice(value);
}

/// Read a length-prefixed byte string.
pub(crate) fn read_bytes(bytes: &[u8], pos: &mut usize) -> Result<Vec<u8>> {
    let len = varint::read_uvarint(bytes, pos)? as usize;
    let end = pos
        .checked_add(len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| {
            corrupt(
                *pos,
                Region::Footer,
                format!(
                    "length prefix of {len} bytes runs past the {}-byte footer",
                    bytes.len()
                ),
            )
        })?;
    let out = bytes[*pos..end].to_vec();
    *pos = end;
    Ok(out)
}

/// Read a length-prefixed UTF-8 string.
pub(crate) fn read_string(bytes: &[u8], pos: &mut usize) -> Result<String> {
    let raw = read_bytes(bytes, pos)?;
    String::from_utf8(raw).map_err(|e| {
        corrupt(
            *pos,
            Region::Footer,
            format!("invalid UTF-8 in footer: {e}"),
        )
    })
}

fn read_u8(bytes: &[u8], pos: &mut usize, region: Region) -> Result<u8> {
    let value = *bytes
        .get(*pos)
        .ok_or_else(|| corrupt(*pos, region, "footer ended unexpectedly"))?;
    *pos += 1;
    Ok(value)
}

fn read_exact(bytes: &[u8], pos: &mut usize, out: &mut [u8], region: Region) -> Result<()> {
    let end = *pos + out.len();
    if end > bytes.len() {
        return Err(corrupt(
            *pos,
            region,
            format!("needed {} more bytes", out.len()),
        ));
    }
    out.copy_from_slice(&bytes[*pos..end]);
    *pos = end;
    Ok(())
}

fn narrow<T: TryFrom<i64>>(value: i64, pos: usize) -> Result<T> {
    T::try_from(value).map_err(|_| {
        corrupt(
            pos,
            Region::ZoneMap,
            format!("value {value} does not fit its type"),
        )
    })
}

fn narrow_u<T: TryFrom<u64>>(value: u64, pos: usize) -> Result<T> {
    T::try_from(value).map_err(|_| {
        corrupt(
            pos,
            Region::ZoneMap,
            format!("value {value} does not fit its type"),
        )
    })
}

/// Append a metadata map. `BTreeMap` iteration order makes this deterministic.
pub(crate) fn write_metadata(metadata: &BTreeMap<String, String>, out: &mut Vec<u8>) {
    varint::write_uvarint(metadata.len() as u64, out);
    for (key, value) in metadata {
        write_bytes(key.as_bytes(), out);
        write_bytes(value.as_bytes(), out);
    }
}

/// Read a metadata map.
pub(crate) fn read_metadata(bytes: &[u8], pos: &mut usize) -> Result<BTreeMap<String, String>> {
    let count = varint::read_uvarint(bytes, pos)? as usize;
    if count > bytes.len() {
        return Err(corrupt(
            *pos,
            Region::Footer,
            format!(
                "metadata claims {count} entries in a {}-byte footer",
                bytes.len()
            ),
        ));
    }
    let mut out = BTreeMap::new();
    for _ in 0..count {
        let key = read_string(bytes, pos)?;
        let value = read_string(bytes, pos)?;
        out.insert(key, value);
    }
    Ok(out)
}

/// Append a [`Schema`].
pub(crate) fn write_schema(schema: &Schema, out: &mut Vec<u8>) {
    varint::write_uvarint(schema.len() as u64, out);
    for field in schema.fields() {
        write_bytes(field.name().as_bytes(), out);
        write_data_type(field.data_type(), out);
        out.push(u8::from(field.is_nullable()));
        write_metadata(field.metadata(), out);
    }
    write_metadata(schema.metadata(), out);
}

/// Read a [`Schema`].
pub(crate) fn read_schema(bytes: &[u8], pos: &mut usize) -> Result<Schema> {
    let count = varint::read_uvarint(bytes, pos)? as usize;
    if count > bytes.len() {
        return Err(corrupt(
            *pos,
            Region::Footer,
            format!(
                "schema claims {count} fields in a {}-byte footer",
                bytes.len()
            ),
        ));
    }
    let mut fields = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_string(bytes, pos)?;
        let data_type = read_data_type(bytes, pos)?;
        let nullable = read_u8(bytes, pos, Region::Footer)? != 0;
        let metadata = read_metadata(bytes, pos)?;
        let mut field = Field::new(name, data_type, nullable);
        for (key, value) in metadata {
            field = field.with_metadata(key, value);
        }
        fields.push(field);
    }
    let metadata = read_metadata(bytes, pos)?;
    let mut schema = Schema::new(fields)?;
    for (key, value) in metadata {
        schema = schema.with_metadata(key, value);
    }
    Ok(schema)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn all_values() -> Vec<Value> {
        vec![
            Value::Null,
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Int8(i8::MIN),
            Value::Int16(i16::MAX),
            Value::Int32(-1),
            Value::Int64(i64::MIN),
            Value::UInt8(255),
            Value::UInt16(0),
            Value::UInt32(7),
            Value::UInt64(u64::MAX),
            Value::Float32(-0.0),
            Value::Float32(f32::NAN),
            Value::Float64(1.5),
            Value::Float64(f64::NEG_INFINITY),
            Value::Utf8("hello \u{1F600}".into()),
            Value::Binary(vec![0, 255, 128]),
            Value::Date32(-19_000),
            Value::Timestamp(1_700_000_000_000, TimeUnit::Millisecond),
            Value::Decimal128(i128::MIN, 38, 0),
            Value::Decimal128(1234, 10, 2),
        ]
    }

    #[test]
    fn values_round_trip() {
        for value in all_values() {
            let mut buf = Vec::new();
            write_value(&value, &mut buf);
            let mut pos = 0;
            let decoded = read_value(&buf, &mut pos).unwrap();
            assert_eq!(decoded, value, "{value:?}");
            assert_eq!(pos, buf.len(), "{value:?}");
        }
    }

    #[test]
    fn value_encoding_is_canonical() {
        // The bloom filter depends on this: equal values must produce equal
        // bytes, every time.
        for value in all_values() {
            let mut a = Vec::new();
            let mut b = Vec::new();
            write_value(&value, &mut a);
            write_value(&value.clone(), &mut b);
            assert_eq!(a, b);
        }
        let mut zero = Vec::new();
        let mut negative_zero = Vec::new();
        write_value(&Value::Float64(0.0), &mut zero);
        write_value(&Value::Float64(-0.0), &mut negative_zero);
        assert_ne!(
            zero, negative_zero,
            "signed zeros must stay distinguishable"
        );
    }

    #[test]
    fn schema_round_trips() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("t", DataType::Timestamp(TimeUnit::Nanosecond), true)
                .with_metadata("tz", "UTC"),
            Field::new("amount", DataType::Decimal128(18, 4), true),
        ])
        .unwrap()
        .with_metadata("source", "tpch");
        let mut buf = Vec::new();
        write_schema(&schema, &mut buf);
        let mut pos = 0;
        assert_eq!(read_schema(&buf, &mut pos).unwrap(), schema);
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn truncated_input_is_corruption_not_panic() {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]).unwrap();
        let mut buf = Vec::new();
        write_schema(&schema, &mut buf);
        for cut in 0..buf.len() {
            let mut pos = 0;
            let result = read_schema(&buf[..cut], &mut pos);
            assert!(result.is_err(), "prefix of {cut} bytes was accepted");
        }
        let mut pos = 0;
        assert!(read_data_type(&[99], &mut pos).is_err());
        let mut pos = 0;
        assert!(read_data_type(&[type_code::TIMESTAMP, 9], &mut pos).is_err());
        let mut pos = 0;
        assert!(read_data_type(&[type_code::DECIMAL128, 99, 0], &mut pos).is_err());
    }
}

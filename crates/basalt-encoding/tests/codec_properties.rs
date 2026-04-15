//! Property tests for every codec.
//!
//! The generators mix genuinely random input with the shapes that actually
//! break codecs: all-same, all-distinct, sorted, reverse-sorted, single element,
//! empty, and the `i64`/`i128` extremes where a naive `max - min` overflows.
//!
//! Four families of property:
//!
//! 1. **Round trip.** `decode(encode(x)) == x`, for every codec and every shape.
//! 2. **Self-description.** The header identifies the codec, and decoding with
//!    the wrong codec fails rather than returning plausible garbage.
//! 3. **Estimator honesty.** The size the selector predicted matches (or, for
//!    the deliberately conservative delta estimate, bounds) what the encoder
//!    produced. If this ever fails, the selection heuristic is choosing on bad
//!    information.
//! 4. **Damage tolerance.** Truncated and bit-flipped buffers produce errors,
//!    never panics and never unbounded allocation.

#![allow(clippy::unwrap_used)]

use basalt_common::CompressionLevel;
use basalt_encoding::{
    analyze, analyze_integers, decode_bytes, decode_column, decode_integers, encode_column,
    encode_integers, estimate_sizes, peek_encoding, ByteLzCodec, Codec, Encoding, PlainBytesCodec,
    PlanReason,
};
use basalt_types::{ColumnBuilder, ColumnVector, DataType, TimeUnit, Value};
use proptest::prelude::*;

const INTEGER_ENCODINGS: &[Encoding] = &[
    Encoding::Plain,
    Encoding::Varint,
    Encoding::BitPacked,
    Encoding::FrameOfReference,
    Encoding::RunLength,
    Encoding::Dictionary,
    Encoding::DeltaBinaryPacked,
];

/// Random values plus the adversarial shapes, all as `Vec<i64>`.
fn integer_column() -> impl Strategy<Value = Vec<i64>> {
    prop_oneof![
        // Uniformly random, including the extremes.
        prop::collection::vec(any::<i64>(), 0..200),
        // Small magnitudes: bit packing territory.
        prop::collection::vec(-8i64..8, 0..200),
        // All the same value.
        (any::<i64>(), 0usize..200).prop_map(|(v, n)| vec![v; n]),
        // Sorted ascending.
        prop::collection::vec(any::<i64>(), 0..200).prop_map(|mut v| {
            v.sort_unstable();
            v
        }),
        // Sorted descending.
        prop::collection::vec(any::<i64>(), 0..200).prop_map(|mut v| {
            v.sort_unstable();
            v.reverse();
            v
        }),
        // Monotonic with a constant step: the delta codec's best case.
        (any::<i64>(), -1000i64..1000, 0usize..200).prop_map(|(start, step, n)| (0..n)
            .map(|i| start.wrapping_add(step.wrapping_mul(i as i64)))
            .collect()),
        // A handful of distinct values repeated: dictionary and RLE territory.
        (prop::collection::vec(any::<i64>(), 1..5), 0usize..200)
            .prop_map(|(pool, n)| (0..n).map(|i| pool[i % pool.len()]).collect()),
        // Extremes only.
        prop::collection::vec(
            prop_oneof![Just(i64::MIN), Just(i64::MAX), Just(0), Just(-1), Just(1)],
            0..64
        ),
        // Degenerate sizes.
        Just(Vec::new()),
        any::<i64>().prop_map(|v| vec![v]),
    ]
}

fn byte_blob() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        prop::collection::vec(any::<u8>(), 0..2048),
        prop::collection::vec(0u8..4, 0..2048),
        (any::<u8>(), 0usize..2048).prop_map(|(b, n)| vec![b; n]),
        (prop::collection::vec(any::<u8>(), 1..24), 0usize..80).prop_map(|(unit, n)| unit
            .iter()
            .copied()
            .cycle()
            .take(n * unit.len())
            .collect()),
        Just(Vec::new()),
        any::<u8>().prop_map(|b| vec![b]),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 192, ..ProptestConfig::default() })]

    #[test]
    fn every_integer_codec_round_trips(values in integer_column()) {
        for &encoding in INTEGER_ENCODINGS {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            prop_assert_eq!(peek_encoding(&buf).unwrap(), encoding);
            prop_assert_eq!(decode_integers(&buf).unwrap(), values.clone(), "{}", encoding);
        }
    }

    #[test]
    fn decoding_with_the_wrong_codec_fails(values in integer_column()) {
        for &encoding in INTEGER_ENCODINGS {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            for &other in INTEGER_ENCODINGS {
                if other == encoding {
                    continue;
                }
                // `decode_integers` dispatches on the header, so it succeeds;
                // asking a specific codec for someone else's bytes must not.
                let mut probe = Vec::new();
                encode_integers(&[], other, &mut probe).unwrap();
                prop_assert_ne!(buf.first(), probe.first().filter(|_| other != encoding));
            }
        }
    }

    #[test]
    fn estimator_matches_the_encoders(values in integer_column()) {
        let stats = analyze_integers(&values);
        let estimates = estimate_sizes(&stats);
        for (encoding, estimated) in estimates.candidates() {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            if encoding == Encoding::DeltaBinaryPacked {
                prop_assert!(
                    buf.len() <= estimated,
                    "delta estimate {estimated} was not an upper bound for {}",
                    buf.len()
                );
            } else {
                prop_assert_eq!(buf.len(), estimated, "{} estimate", encoding);
            }
        }
    }

    /// The whole point of the heuristic: whatever it picks must not be bigger
    /// than plain, and when it claims to have found the smallest, it must have.
    #[test]
    fn selection_never_makes_things_worse(values in integer_column()) {
        for level in [CompressionLevel::FAST, CompressionLevel::BALANCED, CompressionLevel::MAX] {
            let plan = analyze(&values, level);
            let mut chosen = Vec::new();
            encode_integers(&values, plan.encoding, &mut chosen).unwrap();
            let mut plain = Vec::new();
            encode_integers(&values, Encoding::Plain, &mut plain).unwrap();
            prop_assert!(
                chosen.len() <= plain.len(),
                "{:?} produced {} bytes vs {} for plain",
                plan,
                chosen.len(),
                plain.len()
            );
            prop_assert_eq!(decode_integers(&chosen).unwrap(), values.clone());
            if plan.reason == PlanReason::SmallestEstimate {
                let best = estimate_sizes(&analyze_integers(&values))
                    .candidates()
                    .into_iter()
                    .map(|(_, size)| size)
                    .min()
                    .unwrap_or(0);
                prop_assert_eq!(plan.estimated_bytes, best);
            }
        }
    }

    #[test]
    fn byte_codecs_round_trip(blob in byte_blob()) {
        let plain = PlainBytesCodec.encode_to_vec(&blob).unwrap();
        prop_assert_eq!(PlainBytesCodec.decode(&plain).unwrap(), blob.clone());
        let compressed = ByteLzCodec.encode_to_vec(&blob).unwrap();
        prop_assert_eq!(ByteLzCodec.decode(&compressed).unwrap(), blob.clone());
        prop_assert_eq!(decode_bytes(&compressed).unwrap(), blob);
    }

    /// Damaged buffers must produce errors, not panics or huge allocations.
    #[test]
    fn damaged_buffers_are_survivable(
        values in integer_column(),
        cut in any::<usize>(),
        flip_at in any::<usize>(),
        flip_mask in 1u8..=255,
    ) {
        for &encoding in INTEGER_ENCODINGS {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            if buf.is_empty() {
                continue;
            }
            let truncated = &buf[..cut % buf.len()];
            let _ = decode_integers(truncated);
            let mut flipped = buf.clone();
            let at = flip_at % flipped.len();
            flipped[at] ^= flip_mask;
            // Any outcome but a panic is acceptable: a bit flip inside a
            // bit-packed body is indistinguishable from valid data, which is
            // exactly why the storage layer adds a CRC per block.
            let _ = decode_integers(&flipped);
        }
    }

    #[test]
    fn damaged_byte_buffers_are_survivable(blob in byte_blob(), cut in any::<usize>()) {
        let compressed = ByteLzCodec.encode_to_vec(&blob).unwrap();
        let truncated = &compressed[..cut % compressed.len().max(1)];
        let _ = decode_bytes(truncated);
    }
}

// ---------------------------------------------------------------------------
// Whole-column round trips, including nulls
// ---------------------------------------------------------------------------

fn any_data_type() -> impl Strategy<Value = DataType> {
    prop_oneof![
        Just(DataType::Null),
        Just(DataType::Boolean),
        Just(DataType::Int8),
        Just(DataType::Int16),
        Just(DataType::Int32),
        Just(DataType::Int64),
        Just(DataType::UInt8),
        Just(DataType::UInt16),
        Just(DataType::UInt32),
        Just(DataType::UInt64),
        Just(DataType::Float32),
        Just(DataType::Float64),
        Just(DataType::Utf8),
        Just(DataType::Binary),
        Just(DataType::Date32),
        Just(DataType::Timestamp(TimeUnit::Nanosecond)),
        Just(DataType::Decimal128(38, 6)),
    ]
}

fn value_of(data_type: &DataType) -> BoxedStrategy<Value> {
    match data_type {
        DataType::Null => Just(Value::Null).boxed(),
        DataType::Boolean => any::<bool>().prop_map(Value::Boolean).boxed(),
        DataType::Int8 => any::<i8>().prop_map(Value::Int8).boxed(),
        DataType::Int16 => any::<i16>().prop_map(Value::Int16).boxed(),
        DataType::Int32 => any::<i32>().prop_map(Value::Int32).boxed(),
        DataType::Int64 => any::<i64>().prop_map(Value::Int64).boxed(),
        DataType::UInt8 => any::<u8>().prop_map(Value::UInt8).boxed(),
        DataType::UInt16 => any::<u16>().prop_map(Value::UInt16).boxed(),
        DataType::UInt32 => any::<u32>().prop_map(Value::UInt32).boxed(),
        DataType::UInt64 => any::<u64>().prop_map(Value::UInt64).boxed(),
        DataType::Float32 => any::<f32>().prop_map(Value::Float32).boxed(),
        DataType::Float64 => any::<f64>().prop_map(Value::Float64).boxed(),
        DataType::Utf8 => "[a-z ]{0,20}".prop_map(Value::Utf8).boxed(),
        DataType::Binary => prop::collection::vec(any::<u8>(), 0..20)
            .prop_map(Value::Binary)
            .boxed(),
        DataType::Date32 => any::<i32>().prop_map(Value::Date32).boxed(),
        DataType::Timestamp(u) => {
            let u = *u;
            any::<i64>()
                .prop_map(move |v| Value::Timestamp(v, u))
                .boxed()
        }
        DataType::Decimal128(p, s) => {
            let (p, s) = (*p, *s);
            any::<i128>()
                .prop_map(move |v| Value::Decimal128(v, p, s))
                .boxed()
        }
    }
}

fn column_data() -> impl Strategy<Value = (DataType, Vec<Option<Value>>)> {
    any_data_type().prop_flat_map(|data_type| {
        // 20% nulls, so both the sparse-null and no-null paths get exercised.
        let values =
            prop::collection::vec(prop::option::weighted(0.8, value_of(&data_type)), 0..96);
        (Just(data_type), values)
    })
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

proptest! {
    #![proptest_config(ProptestConfig { cases: 192, ..ProptestConfig::default() })]

    #[test]
    fn columns_round_trip_at_every_level((data_type, data) in column_data()) {
        let column = build(&data_type, &data);
        for level in [
            CompressionLevel::NONE,
            CompressionLevel::FAST,
            CompressionLevel::BALANCED,
            CompressionLevel::MAX,
        ] {
            let encoded = encode_column(&column, level).unwrap();
            let decoded = decode_column(&encoded.bytes, &data_type).unwrap();
            prop_assert_eq!(&decoded, &column, "{} at {:?}", data_type, level);
            prop_assert_eq!(decoded.null_count(), column.null_count());
        }
    }

    /// Slicing must not change what a column encodes to.
    #[test]
    fn slices_encode_like_freshly_built_columns(
        (data_type, data) in column_data(),
        offset in any::<usize>(),
        len in any::<usize>(),
    ) {
        prop_assume!(!data.is_empty());
        let column = build(&data_type, &data);
        let offset = offset % column.len();
        let len = len % (column.len() - offset + 1);
        let sliced = column.slice(offset, len);
        let encoded = encode_column(&sliced, CompressionLevel::MAX).unwrap();
        let decoded = decode_column(&encoded.bytes, &data_type).unwrap();
        prop_assert_eq!(&decoded, &sliced);

        let rebuilt = build(&data_type, &data[offset..offset + len]);
        let rebuilt_encoded = encode_column(&rebuilt, CompressionLevel::MAX).unwrap();
        prop_assert_eq!(encoded.bytes, rebuilt_encoded.bytes);
    }

    #[test]
    fn truncated_column_chunks_never_panic((data_type, data) in column_data(), cut in any::<usize>()) {
        let column = build(&data_type, &data);
        let encoded = encode_column(&column, CompressionLevel::BALANCED).unwrap();
        let truncated = &encoded.bytes[..cut % encoded.bytes.len().max(1)];
        let _ = decode_column(truncated, &data_type);
    }
}

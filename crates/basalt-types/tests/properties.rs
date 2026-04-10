//! Property tests for the columnar representation.
//!
//! Three families:
//!
//! 1. **Builder round trips.** For every [`DataType`], a randomly generated
//!    sequence of values-and-nulls must survive builder -> array -> read-back
//!    unchanged. This is the invariant everything else depends on.
//! 2. **Bitmap against a reference implementation.** Every `Bitmap` operation is
//!    checked against the obvious `Vec<bool>` version, which is easy to read and
//!    obviously correct, at random lengths chosen to straddle the 64-bit word
//!    boundaries where packed bitmaps go wrong.
//! 3. **Slicing invariants.** Slices are views, so the interesting properties
//!    are compositional: slicing twice equals slicing once, and a slice reads
//!    the same values as the original at shifted indices.

#![allow(clippy::unwrap_used)]

use basalt_types::{
    Bitmap, BitmapBuilder, ColumnBuilder, ColumnVector, DataType, Field, RecordBatch, Schema,
    Selection, TimeUnit, Value,
};
use proptest::prelude::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Every type Basalt can store, including two decimal shapes and two timestamp
/// resolutions so the parameterised variants are exercised too.
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
        Just(DataType::Timestamp(TimeUnit::Second)),
        Just(DataType::Timestamp(TimeUnit::Nanosecond)),
        Just(DataType::Decimal128(10, 2)),
        Just(DataType::Decimal128(38, 0)),
    ]
}

/// A value of exactly `dt`, or `Value::Null`. Extreme values are included
/// explicitly because that is where width handling breaks.
fn value_of(dt: &DataType) -> BoxedStrategy<Value> {
    match dt {
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
        // `any::<f32>()` already generates NaN and the infinities, which is the
        // point: `Value`'s total order has to survive them.
        DataType::Float32 => any::<f32>().prop_map(Value::Float32).boxed(),
        DataType::Float64 => any::<f64>().prop_map(Value::Float64).boxed(),
        DataType::Utf8 => ".{0,24}".prop_map(Value::Utf8).boxed(),
        DataType::Binary => prop::collection::vec(any::<u8>(), 0..24)
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
            // Stay inside the declared precision so the value is representable.
            let bound = 10i128.pow(u32::from(p).min(38)) - 1;
            (-bound..=bound)
                .prop_map(move |v| Value::Decimal128(v, p, s))
                .boxed()
        }
    }
}

/// A column's worth of optional values plus the type they belong to.
fn column_data() -> impl Strategy<Value = (DataType, Vec<Option<Value>>)> {
    any_data_type().prop_flat_map(|dt| {
        let values = prop::collection::vec(prop::option::weighted(0.75, value_of(&dt)), 0..64);
        (Just(dt), values)
    })
}

fn build_column(dt: &DataType, data: &[Option<Value>]) -> ColumnVector {
    let mut builder = ColumnBuilder::with_capacity(dt, data.len()).unwrap();
    for slot in data {
        match slot {
            Some(v) => builder.append_value(v).unwrap(),
            None => builder.append_null(),
        }
    }
    builder.finish()
}

// ---------------------------------------------------------------------------
// 1. Builder round trips
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// builder -> array -> read-back is the identity, for every type and every
    /// null pattern.
    #[test]
    fn builder_round_trip((dt, data) in column_data()) {
        let column = build_column(&dt, &data);
        prop_assert_eq!(column.data_type(), dt.clone());
        prop_assert_eq!(column.len(), data.len());

        let expected_nulls = data.iter().filter(|v| v.is_none() || dt == DataType::Null).count();
        prop_assert_eq!(column.null_count(), expected_nulls);

        for (i, slot) in data.iter().enumerate() {
            let read = column.value(i);
            match slot {
                // A `Null`-typed column reads back as NULL whatever went in.
                _ if dt == DataType::Null => prop_assert_eq!(read, Value::Null),
                Some(v) => prop_assert_eq!(&read, v),
                None => prop_assert_eq!(read, Value::Null),
            }
            prop_assert_eq!(column.is_null(i), slot.is_none() || dt == DataType::Null);
            prop_assert_eq!(column.is_valid(i), !column.is_null(i));
        }
    }

    /// A column with no nulls must not allocate a validity bitmap, because the
    /// null-free fast path in every kernel keys off `validity().is_none()`.
    #[test]
    fn null_free_columns_have_no_validity_mask((dt, data) in column_data()) {
        let dense: Vec<Option<Value>> = data
            .into_iter()
            .map(|v| v.or_else(|| Some(default_value(&dt))))
            .collect();
        let column = build_column(&dt, &dense);
        if dt != DataType::Null {
            prop_assert!(column.validity().is_none());
            prop_assert_eq!(column.null_count(), 0);
        }
    }

    /// `take` is a gather: the result reads the source's values at the given
    /// indices, in the given order, nulls included.
    #[test]
    fn take_gathers((dt, data) in column_data(), seed in any::<u64>()) {
        prop_assume!(!data.is_empty());
        let column = build_column(&dt, &data);
        // A deterministic pseudo-random index list derived from `seed`.
        let mut state = seed | 1;
        let indices: Vec<u32> = (0..data.len().min(16))
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((state >> 33) as usize % data.len()) as u32
            })
            .collect();
        let taken = column.take(&indices).unwrap();
        prop_assert_eq!(taken.len(), indices.len());
        for (out, &src) in indices.iter().enumerate() {
            prop_assert_eq!(taken.value(out), column.value(src as usize));
            prop_assert_eq!(taken.is_null(out), column.is_null(src as usize));
        }
    }

    /// `filter` keeps exactly the rows whose bit is set, in order, and agrees
    /// with the `Selection` path that the executor actually uses.
    #[test]
    fn filter_matches_selection(
        (dt, data) in column_data(),
        pattern in prop::collection::vec(any::<bool>(), 1..17),
    ) {
        // Tile the generated pattern to the column's length: generating two
        // independent vectors and rejecting the mismatches wastes almost every
        // case, and the tiled mask still covers every interesting shape.
        let bits: Vec<bool> = (0..data.len()).map(|i| pattern[i % pattern.len()]).collect();
        let column = build_column(&dt, &data);
        let mask = Bitmap::from_bools(&bits);
        let filtered = column.filter(&mask).unwrap();
        let via_selection = Selection::from_bitmap(&mask).apply(&column).unwrap();
        prop_assert_eq!(filtered.len(), bits.iter().filter(|b| **b).count());
        prop_assert_eq!(&filtered, &via_selection);
        let mut out = 0;
        for (i, keep) in bits.iter().enumerate() {
            if *keep {
                prop_assert_eq!(filtered.value(out), column.value(i));
                out += 1;
            }
        }
    }
}

fn default_value(dt: &DataType) -> Value {
    match dt {
        DataType::Null => Value::Null,
        DataType::Boolean => Value::Boolean(false),
        DataType::Int8 => Value::Int8(0),
        DataType::Int16 => Value::Int16(0),
        DataType::Int32 => Value::Int32(0),
        DataType::Int64 => Value::Int64(0),
        DataType::UInt8 => Value::UInt8(0),
        DataType::UInt16 => Value::UInt16(0),
        DataType::UInt32 => Value::UInt32(0),
        DataType::UInt64 => Value::UInt64(0),
        DataType::Float32 => Value::Float32(0.0),
        DataType::Float64 => Value::Float64(0.0),
        DataType::Utf8 => Value::Utf8(String::new()),
        DataType::Binary => Value::Binary(Vec::new()),
        DataType::Date32 => Value::Date32(0),
        DataType::Timestamp(u) => Value::Timestamp(0, *u),
        DataType::Decimal128(p, s) => Value::Decimal128(0, *p, *s),
    }
}

// ---------------------------------------------------------------------------
// 2. Bitmap versus a naive reference implementation
// ---------------------------------------------------------------------------

/// Lengths that straddle word boundaries are where packed bitmaps break, so the
/// generator is biased towards them.
fn bool_vec() -> impl Strategy<Value = Vec<bool>> {
    prop_oneof![
        prop::collection::vec(any::<bool>(), 0..8),
        prop::collection::vec(any::<bool>(), 60..70),
        prop::collection::vec(any::<bool>(), 120..140),
        prop::collection::vec(any::<bool>(), 0..300),
    ]
}

fn same_length_pair() -> impl Strategy<Value = (Vec<bool>, Vec<bool>)> {
    bool_vec().prop_flat_map(|a| {
        let n = a.len();
        (Just(a), prop::collection::vec(any::<bool>(), n..=n))
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn bitmap_matches_vec_bool(bits in bool_vec()) {
        let bm = Bitmap::from_bools(&bits);
        prop_assert_eq!(bm.len(), bits.len());
        prop_assert_eq!(bm.count_ones(), bits.iter().filter(|b| **b).count());
        prop_assert_eq!(bm.count_zeros(), bits.iter().filter(|b| !**b).count());
        prop_assert_eq!(bm.is_all_set(), bits.iter().all(|b| *b));
        prop_assert_eq!(bm.is_all_unset(), bits.iter().all(|b| !*b));
        prop_assert_eq!(bm.to_bools(), bits.clone());
        for (i, expected) in bits.iter().enumerate() {
            prop_assert_eq!(bm.get(i), *expected);
            prop_assert_eq!(bm.try_get(i), Some(*expected));
        }
        prop_assert_eq!(bm.try_get(bits.len()), None);

        let expected_indices: Vec<usize> =
            bits.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i).collect();
        prop_assert_eq!(bm.iter_set_indices().collect::<Vec<_>>(), expected_indices);

        // Serialisation round trip.
        prop_assert_eq!(Bitmap::from_bytes(&bm.to_bytes(), bm.len()).unwrap(), bm.clone());
        // Builder path must agree with the `from_bools` path.
        let mut builder = BitmapBuilder::with_capacity(bits.len());
        for b in &bits {
            builder.push(*b);
        }
        prop_assert_eq!(builder.finish(), bm);
    }

    #[test]
    fn bitmap_logic_matches_reference((a, b) in same_length_pair()) {
        let (ba, bb) = (Bitmap::from_bools(&a), Bitmap::from_bools(&b));
        let and_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x && *y).collect();
        let or_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x || *y).collect();
        let xor_ref: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x ^ *y).collect();
        let not_ref: Vec<bool> = a.iter().map(|x| !*x).collect();
        prop_assert_eq!(ba.and(&bb).unwrap().to_bools(), and_ref);
        prop_assert_eq!(ba.or(&bb).unwrap().to_bools(), or_ref);
        prop_assert_eq!(ba.xor(&bb).unwrap().to_bools(), xor_ref);
        prop_assert_eq!(ba.not().to_bools(), not_ref);
        // De Morgan, as a cheap algebraic cross-check of the tail masking.
        prop_assert_eq!(
            ba.and(&bb).unwrap().not(),
            ba.not().or(&bb.not()).unwrap()
        );
    }

    #[test]
    fn bitmap_set_matches_reference(bits in bool_vec(), writes in prop::collection::vec((any::<usize>(), any::<bool>()), 0..32)) {
        prop_assume!(!bits.is_empty());
        let mut reference = bits.clone();
        let mut bm = Bitmap::from_bools(&bits);
        for (raw_index, value) in writes {
            let i = raw_index % reference.len();
            reference[i] = value;
            bm.set(i, value);
        }
        prop_assert_eq!(bm.to_bools(), reference);
    }

    /// Sliced bitmaps must behave exactly like a freshly built bitmap of the
    /// same bits, including through the offset-aware word extraction.
    #[test]
    fn bitmap_slices_are_transparent(bits in bool_vec(), offset in any::<usize>(), len in any::<usize>()) {
        prop_assume!(!bits.is_empty());
        let offset = offset % bits.len();
        let len = len % (bits.len() - offset + 1);
        let bm = Bitmap::from_bools(&bits);
        let sliced = bm.slice(offset, len);
        let reference = Bitmap::from_bools(&bits[offset..offset + len]);
        prop_assert_eq!(&sliced, &reference);
        prop_assert_eq!(sliced.count_ones(), reference.count_ones());
        prop_assert_eq!(sliced.to_bytes(), reference.to_bytes());
        prop_assert_eq!(sliced.not(), reference.not());
        prop_assert_eq!(
            sliced.iter_set_indices().collect::<Vec<_>>(),
            reference.iter_set_indices().collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Slicing invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn column_slice_is_a_shifted_view((dt, data) in column_data(), o in any::<usize>(), l in any::<usize>()) {
        prop_assume!(!data.is_empty());
        let column = build_column(&dt, &data);
        let offset = o % column.len();
        let len = l % (column.len() - offset + 1);
        let sliced = column.slice(offset, len);

        prop_assert_eq!(sliced.len(), len);
        prop_assert_eq!(sliced.data_type(), dt.clone());
        for i in 0..len {
            prop_assert_eq!(sliced.value(i), column.value(offset + i));
            prop_assert_eq!(sliced.is_null(i), column.is_null(offset + i));
        }
        prop_assert_eq!(
            sliced.null_count(),
            (offset..offset + len).filter(|i| column.is_null(*i)).count()
        );
        // Identity slice.
        prop_assert_eq!(&column.slice(0, column.len()), &column);
        // Slicing a slice is slicing once.
        if len > 0 {
            let inner = sliced.slice(len / 2, len - len / 2);
            let direct = column.slice(offset + len / 2, len - len / 2);
            prop_assert_eq!(&inner, &direct);
        }
        // Slicing then rebuilding gives the same column, which is what makes it
        // safe to hand a slice to the writer.
        let rebuilt = build_column(
            &dt,
            &(0..len)
                .map(|i| {
                    if sliced.is_null(i) {
                        None
                    } else {
                        Some(sliced.value(i))
                    }
                })
                .collect::<Vec<_>>(),
        );
        prop_assert_eq!(&rebuilt, &sliced);
    }

    #[test]
    fn record_batch_slice_agrees_per_column(
        (dt_a, data_a) in column_data(),
        (dt_b, _) in column_data(),
        o in any::<usize>(),
        l in any::<usize>(),
    ) {
        prop_assume!(!data_a.is_empty());
        // Give the second column the same number of rows as the first.
        let data_b: Vec<Option<Value>> = data_a.iter().map(|_| None).collect();
        let col_a = build_column(&dt_a, &data_a);
        let col_b = build_column(&dt_b, &data_b);
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("a", dt_a.clone(), true),
                Field::new("b", dt_b.clone(), true),
            ])
            .unwrap(),
        );
        let batch = RecordBatch::try_new(schema, vec![col_a, col_b]).unwrap();
        let offset = o % batch.num_rows();
        let len = l % (batch.num_rows() - offset + 1);
        let sliced = batch.slice(offset, len);
        prop_assert_eq!(sliced.num_rows(), len);
        for c in 0..batch.num_columns() {
            prop_assert_eq!(sliced.column(c).len(), len);
            for i in 0..len {
                prop_assert_eq!(sliced.column(c).value(i), batch.column(c).value(offset + i));
            }
        }
        // Projection commutes with slicing.
        let project_then_slice = batch.project(&[1]).unwrap().slice(offset, len);
        let slice_then_project = sliced.project(&[1]).unwrap();
        prop_assert_eq!(slice_then_project.column(0), project_then_slice.column(0));
    }

    /// Chained selections must equal the single equivalent selection: this is
    /// the correctness condition for filter pushdown.
    #[test]
    fn selection_refinement_composes(
        first in prop::collection::vec(any::<bool>(), 1..80),
        seed in any::<u64>(),
    ) {
        let base = Selection::all(first.len());
        let mask1 = Bitmap::from_bools(&first);
        let step1 = base.refine(&mask1).unwrap();

        let mut state = seed | 1;
        let second: Vec<bool> = (0..step1.len())
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (state >> 60) & 1 == 1
            })
            .collect();
        let step2 = step1.refine(&Bitmap::from_bools(&second)).unwrap();

        // Reference: apply both masks by hand.
        let mut expected = Vec::new();
        let mut k = 0;
        for (i, keep) in first.iter().enumerate() {
            if *keep {
                if second[k] {
                    expected.push(i as u32);
                }
                k += 1;
            }
        }
        prop_assert_eq!(step2.to_indices(), expected);
        prop_assert_eq!(step2.source_len(), first.len());
        // Intersecting with the equivalent bitmap-derived selection agrees.
        let direct = Selection::from_bitmap(&step2.to_bitmap());
        prop_assert_eq!(step2.intersect(&direct).unwrap(), direct);
    }
}

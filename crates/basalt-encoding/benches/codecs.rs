//! Codec benchmarks over synthetic column shapes.
//!
//! Run with `cargo bench -p basalt-encoding`. The harness first prints a size
//! table (every codec against every shape, so the compression ratios in
//! `BENCHMARKS.md` are reproducible), then measures encode and decode
//! throughput for the encoding the selector would actually choose.

#![allow(clippy::unwrap_used)]

use std::time::Duration;

use basalt_common::CompressionLevel;
use basalt_encoding::{
    analyze, decode_bytes, decode_column, decode_integers, encode_bytes_adaptive, encode_column,
    encode_integers, Codec, Encoding, PlainBytesCodec,
};
use basalt_types::{ColumnVector, PrimitiveArray, StringArray};
use criterion::{black_box, BenchmarkId, Criterion, Throughput};

const N: usize = 65_536;

/// Deterministic splitmix64, so the numbers are reproducible run to run.
///
/// A plain LCG will not do here: its low bits have tiny periods, so
/// `lcg.next() as u8` produces a *highly compressible* stream and the
/// "incompressible data" benchmark would measure the wrong thing entirely.
/// splitmix64's finaliser mixes every bit position.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The column shapes an analytical engine actually meets.
fn shapes() -> Vec<(&'static str, Vec<i64>)> {
    let mut rng = Rng::new(0x5EED);
    vec![
        // A synthetic primary key.
        ("sorted_ids", (0..N as i64).map(|i| 1_000_000 + i).collect()),
        // Event timestamps within a one-second window, in microseconds.
        (
            "clustered_timestamps",
            (0..N)
                .map(|_| 1_700_000_000_000_000 + (rng.next() % 1_000_000) as i64)
                .collect(),
        ),
        // An enum-like column: 8 distinct values, no ordering.
        (
            "low_cardinality",
            (0..N)
                .map(|_| [3i64, 17, 42, 99, 512, 4096, -7, 0][(rng.next() % 8) as usize])
                .collect(),
        ),
        // A sorted status column: long runs.
        ("long_runs", (0..N as i64).map(|i| i / 512).collect()),
        // Small signed measurements.
        (
            "small_values",
            (0..N).map(|_| (rng.next() % 64) as i64 - 32).collect(),
        ),
        // Random 64-bit values: the incompressible case.
        ("high_entropy", (0..N).map(|_| rng.next() as i64).collect()),
        // Mostly small with rare huge outliers: varint's case.
        (
            "skewed",
            (0..N)
                .map(|i| {
                    if i % 997 == 0 {
                        rng.next() as i64
                    } else {
                        (rng.next() % 200) as i64
                    }
                })
                .collect(),
        ),
    ]
}

fn integer_encodings() -> Vec<Encoding> {
    vec![
        Encoding::Plain,
        Encoding::Varint,
        Encoding::BitPacked,
        Encoding::FrameOfReference,
        Encoding::RunLength,
        Encoding::Dictionary,
        Encoding::DeltaBinaryPacked,
    ]
}

fn print_size_table() {
    println!(
        "\n== compressed size, bytes ({N} i64 values = {} KiB plain) ==",
        N * 8 / 1024
    );
    print!("{:<22}", "shape");
    for encoding in integer_encodings() {
        print!("{:>12}", encoding.to_string());
    }
    println!("{:>14}", "chosen");
    for (name, values) in shapes() {
        print!("{name:<22}");
        for encoding in integer_encodings() {
            let mut buf = Vec::new();
            encode_integers(&values, encoding, &mut buf).unwrap();
            print!("{:>12}", buf.len());
        }
        let plan = analyze(&values, CompressionLevel::BALANCED);
        println!(
            "{:>14}",
            format!("{} {:.1}x", plan.encoding, plan.estimated_ratio())
        );
    }

    println!("\n== string column, 65536 rows ==");
    for (name, rows) in string_shapes() {
        let raw: usize = rows.iter().map(String::len).sum();
        let column = ColumnVector::Utf8(StringArray::from_strs(&rows));
        let encoded = encode_column(&column, CompressionLevel::BALANCED).unwrap();
        println!(
            "{name:<22} raw {raw:>9}  encoded {:>9}  {:.1}x",
            encoded.bytes.len(),
            raw as f64 / encoded.bytes.len() as f64
        );
    }

    println!("\n== byte codec (LZ77) ==");
    for (name, blob) in byte_shapes() {
        let compressed = basalt_encoding::ByteLzCodec.encode_to_vec(&blob).unwrap();
        println!(
            "{name:<22} raw {:>9}  encoded {:>9}  {:.2}x",
            blob.len(),
            compressed.len(),
            blob.len() as f64 / compressed.len() as f64
        );
    }
    println!();
}

fn string_shapes() -> Vec<(&'static str, Vec<String>)> {
    let mut rng = Rng::new(7);
    let cities = [
        "oslo", "kyoto", "lima", "accra", "hobart", "tromso", "quito", "dakar",
    ];
    vec![
        (
            "city_enum",
            (0..N)
                .map(|_| cities[(rng.next() % 8) as usize].to_string())
                .collect(),
        ),
        (
            "unique_ids",
            (0..N).map(|i| format!("order-{i:012}")).collect(),
        ),
        (
            "urls",
            (0..N)
                .map(|i| {
                    format!(
                        "https://example.com/products/{}/reviews?page={}",
                        i % 5000,
                        i % 20
                    )
                })
                .collect(),
        ),
    ]
}

fn byte_shapes() -> Vec<(&'static str, Vec<u8>)> {
    let mut rng = Rng::new(99);
    vec![
        ("json_records", {
            let mut text = String::new();
            for i in 0..4000 {
                use std::fmt::Write;
                let _ = write!(text, r#"{{"id":{i},"ok":true,"tag":"alpha"}}"#);
            }
            text.into_bytes()
        }),
        (
            "english_text",
            "the quick brown fox jumps over the lazy dog. "
                .repeat(4000)
                .into_bytes(),
        ),
        (
            "random_bytes",
            (0..180_000).map(|_| rng.next() as u8).collect(),
        ),
    ]
}

fn bench_integer_codecs(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_codec");
    group.throughput(Throughput::Bytes((N * 8) as u64));
    for (name, values) in shapes() {
        let chosen = analyze(&values, CompressionLevel::BALANCED).encoding;
        for encoding in [Encoding::Plain, chosen] {
            let mut encoded = Vec::new();
            encode_integers(&values, encoding, &mut encoded).unwrap();
            group.bench_function(BenchmarkId::new(format!("encode/{encoding}"), name), |b| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(encoded.len());
                    encode_integers(black_box(&values), encoding, &mut out).unwrap();
                    out
                });
            });
            group.bench_function(BenchmarkId::new(format!("decode/{encoding}"), name), |b| {
                b.iter(|| decode_integers(black_box(&encoded)).unwrap());
            });
            if encoding == chosen {
                break;
            }
        }
    }
    group.finish();
}

fn bench_analyze(c: &mut Criterion) {
    let mut group = c.benchmark_group("analyze");
    group.throughput(Throughput::Bytes((N * 8) as u64));
    for (name, values) in shapes() {
        group.bench_function(name, |b| {
            b.iter(|| analyze(black_box(&values), CompressionLevel::BALANCED));
        });
    }
    group.finish();
}

fn bench_byte_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_codec");
    for (name, blob) in byte_shapes() {
        group.throughput(Throughput::Bytes(blob.len() as u64));
        let mut compressed = Vec::new();
        encode_bytes_adaptive(&blob, true, 0.0, &mut compressed).unwrap();
        group.bench_function(BenchmarkId::new("compress", name), |b| {
            b.iter(|| {
                let mut out = Vec::new();
                encode_bytes_adaptive(black_box(&blob), true, 0.0, &mut out).unwrap();
                out
            });
        });
        group.bench_function(BenchmarkId::new("decompress", name), |b| {
            b.iter(|| decode_bytes(black_box(&compressed)).unwrap());
        });
        group.bench_function(BenchmarkId::new("memcpy_baseline", name), |b| {
            b.iter(|| PlainBytesCodec.encode_to_vec(black_box(&blob)).unwrap());
        });
    }
    group.finish();
}

fn bench_column_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("column");
    let integers = ColumnVector::Int64(PrimitiveArray::from_vec(
        (0..N as i64).map(|i| 1_700_000_000 + i / 4).collect(),
    ));
    let strings = ColumnVector::Utf8(StringArray::from_strs(&string_shapes()[0].1));
    for (name, column) in [("int64_clustered", &integers), ("utf8_enum", &strings)] {
        let encoded = encode_column(column, CompressionLevel::BALANCED).unwrap();
        group.throughput(Throughput::Bytes(encoded.uncompressed_bytes as u64));
        group.bench_function(BenchmarkId::new("encode", name), |b| {
            b.iter(|| encode_column(black_box(column), CompressionLevel::BALANCED).unwrap());
        });
        let data_type = column.data_type();
        group.bench_function(BenchmarkId::new("decode", name), |b| {
            b.iter(|| decode_column(black_box(&encoded.bytes), &data_type).unwrap());
        });
    }
    group.finish();
}

fn main() {
    print_size_table();
    let mut criterion = Criterion::default()
        .sample_size(30)
        .measurement_time(Duration::from_millis(1200))
        .warm_up_time(Duration::from_millis(400))
        .configure_from_args();
    bench_integer_codecs(&mut criterion);
    bench_analyze(&mut criterion);
    bench_byte_codec(&mut criterion);
    bench_column_round_trip(&mut criterion);
    criterion.final_summary();
}

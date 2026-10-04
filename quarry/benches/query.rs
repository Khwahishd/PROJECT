//! Benchmarks for the pieces that dominate an analytical query.
//!
//! Two comparisons matter here and are measured explicitly rather than
//! asserted in prose:
//!
//! * **Vectorized vs row-at-a-time**: the same filter expressed as a batch
//!   operation and as a per-row interpreter loop, to show what batch-at-a-time
//!   evaluation is actually worth.
//! * **Optimized vs unoptimized**: the same query with the rule set on and off,
//!   to show what predicate and projection pushdown are worth.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quarry::array::{Array, ArrayData};
use quarry::batch::RecordBatch;
use quarry::types::{DataType, Field, Schema, Value};
use quarry::Context;
use std::hint::black_box;
use std::sync::Arc;

/// Builds a CSV fact table with `n` rows and several unread columns, so that
/// projection pushdown has something to prune.
fn make_csv(n: usize) -> String {
    let cities = ["Austin", "Dallas", "Houston", "Seattle", "Portland", "Denver"];
    let mut s = String::from("id,city,rider,fare,tip,minutes,note\n");
    // A simple deterministic PRNG keeps the benchmark reproducible without a
    // dependency, and without the generator dominating the setup cost.
    let mut state = 0x2545F491_4F6CDD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..n {
        let r = next();
        let city = cities[(r % 6) as usize];
        let fare = (r >> 8) % 10_000;
        s.push_str(&format!(
            "{i},{city},rider_{},{}.{:02},{}.{:02},{},note-{}\n",
            r % 500,
            fare / 100,
            fare % 100,
            (fare / 500) % 20,
            r % 97,
            (r >> 16) % 60,
            i
        ));
    }
    s
}

fn ctx_with(n: usize, optimize: bool) -> Context {
    let mut c = Context::new();
    c.register_csv_str("trips", &make_csv(n)).unwrap();
    c.set_optimize(optimize);
    c
}

/// Vectorized filter: one type dispatch, then a tight loop over `&[f64]`.
fn vectorized_filter(batch: &RecordBatch, threshold: f64) -> usize {
    let col = batch.column(0);
    let values = col.as_f64().expect("a float column");
    let mut n = 0;
    for (i, &v) in values.iter().enumerate() {
        if col.is_valid(i) && v > threshold {
            n += 1;
        }
    }
    n
}

/// Row-at-a-time filter: the same predicate, but paying the enum dispatch and
/// the `Value` materialization once per row -- what a tree-walking interpreter
/// over rows does.
fn row_at_a_time_filter(batch: &RecordBatch, threshold: f64) -> usize {
    let mut n = 0;
    for i in 0..batch.num_rows() {
        match batch.value(i, 0) {
            Value::Float64(v) if v > threshold => n += 1,
            Value::Int64(v) if (v as f64) > threshold => n += 1,
            _ => {}
        }
    }
    n
}

fn bench_evaluation_model(c: &mut Criterion) {
    let n = 1 << 20;
    let schema = Arc::new(Schema::new(vec![Field::new("fare", DataType::Float64)]));
    let data: Vec<f64> = (0..n).map(|i| (i % 10_000) as f64 / 100.0).collect();
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Array::new(ArrayData::Float64(data)))]).unwrap();

    let mut g = c.benchmark_group("filter_1M_rows");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("vectorized", |b| {
        b.iter(|| black_box(vectorized_filter(black_box(&batch), 50.0)))
    });
    g.bench_function("row_at_a_time", |b| {
        b.iter(|| black_box(row_at_a_time_filter(black_box(&batch), 50.0)))
    });
    g.finish();
}

fn bench_optimizer(c: &mut Criterion) {
    // Selectivity is measured at both extremes on purpose. Projection pushdown
    // saves work in `take`, which only matters when many rows survive the
    // filter; with a highly selective predicate almost nothing is gathered and
    // the rules have little left to save.
    let cases = [
        ("selective", "SELECT id FROM trips WHERE fare > 90 AND city = 'Austin'"),
        ("permissive", "SELECT id FROM trips WHERE fare > 5"),
    ];

    let n = 100_000usize;
    let opt = ctx_with(n, true);
    let plain = ctx_with(n, false);

    let mut g = c.benchmark_group("optimizer_100k_rows");
    g.throughput(Throughput::Elements(n as u64));
    for (label, sql) in cases {
        g.bench_with_input(BenchmarkId::new("optimized", label), &sql, |b, sql| {
            b.iter(|| black_box(opt.sql(black_box(sql)).unwrap().num_rows()))
        });
        g.bench_with_input(BenchmarkId::new("unoptimized", label), &sql, |b, sql| {
            b.iter(|| black_box(plain.sql(black_box(sql)).unwrap().num_rows()))
        });
    }
    g.finish();
}

fn bench_operators(c: &mut Criterion) {
    let n = 100_000;
    let ctx = ctx_with(n, true);

    let queries = [
        ("scan_and_filter", "SELECT id FROM trips WHERE fare > 50"),
        ("projection", "SELECT fare * 1.2 + tip AS total FROM trips"),
        ("group_by_low_cardinality", "SELECT city, COUNT(*), AVG(fare) FROM trips GROUP BY city"),
        ("group_by_high_cardinality", "SELECT rider, COUNT(*) FROM trips GROUP BY rider"),
        ("sort", "SELECT id FROM trips ORDER BY fare DESC"),
        ("top_n", "SELECT id FROM trips ORDER BY fare DESC LIMIT 10"),
        ("distinct", "SELECT DISTINCT city, minutes FROM trips"),
        ("case_expression",
         "SELECT CASE WHEN fare > 70 THEN 'high' WHEN fare > 30 THEN 'mid' ELSE 'low' END AS band FROM trips"),
        ("like", "SELECT id FROM trips WHERE note LIKE '%99%'"),
    ];

    let mut g = c.benchmark_group("operators_100k_rows");
    g.throughput(Throughput::Elements(n as u64));
    for (name, sql) in queries {
        g.bench_function(name, |b| {
            b.iter(|| black_box(ctx.sql(black_box(sql)).unwrap().num_rows()))
        });
    }
    g.finish();
}

fn bench_join(c: &mut Criterion) {
    let mut ctx = Context::new();
    ctx.register_csv_str("trips", &make_csv(100_000)).unwrap();
    ctx.register_csv_str(
        "cities",
        "name,state\nAustin,TX\nDallas,TX\nHouston,TX\nSeattle,WA\nPortland,OR\nDenver,CO\n",
    )
    .unwrap();

    let mut g = c.benchmark_group("hash_join_100k_x_6");
    g.throughput(Throughput::Elements(100_000));
    g.bench_function("inner_join", |b| {
        b.iter(|| {
            black_box(
                ctx.sql("SELECT t.id, c.state FROM trips t JOIN cities c ON t.city = c.name")
                    .unwrap()
                    .num_rows(),
            )
        })
    });
    g.bench_function("join_then_aggregate", |b| {
        b.iter(|| {
            black_box(
                ctx.sql(
                    "SELECT c.state, COUNT(*), AVG(t.fare) FROM trips t
                     JOIN cities c ON t.city = c.name GROUP BY c.state",
                )
                .unwrap()
                .num_rows(),
            )
        })
    });
    g.finish();
}

fn bench_parse_and_plan(c: &mut Criterion) {
    let ctx = ctx_with(1000, true);
    let sql = "SELECT city, COUNT(*) AS n, AVG(fare) AS avg_fare FROM trips
               WHERE fare > 10 AND city <> 'Denver' GROUP BY city
               HAVING COUNT(*) > 5 ORDER BY n DESC LIMIT 10";

    let mut g = c.benchmark_group("front_end");
    g.bench_function("parse", |b| {
        b.iter(|| black_box(quarry::sql::parse(black_box(sql)).unwrap()))
    });
    g.bench_function("parse_plan_optimize", |b| {
        b.iter(|| black_box(ctx.plan(black_box(sql)).unwrap()))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_evaluation_model,
    bench_optimizer,
    bench_operators,
    bench_join,
    bench_parse_and_plan
);
criterion_main!(benches);

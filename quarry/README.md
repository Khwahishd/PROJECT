# quarry

A vectorized analytical SQL query engine, written from scratch in Rust with **zero third-party
dependencies**: a hand-written lexer and Pratt parser, a typed logical planner, a rule-based
optimizer, and a columnar execution engine that works a batch at a time.

```sql
quarry> SELECT c.state, COUNT(*) AS n, AVG(t.fare) AS avg_fare
   ...> FROM trips t JOIN cities c ON t.city = c.name
   ...> WHERE t.fare > 10 GROUP BY c.state ORDER BY n DESC;
+-------+------+--------------------+
| state | n    | avg_fare           |
+-------+------+--------------------+
| TX    | 1818 | 18.288910891089106 |
| WA    | 639  | 18.524694835680734 |
| OR    | 604  | 18.495364238410563 |
| CO    | 577  | 18.336395147313702 |
+-------+------+--------------------+
(4 rows) in 18.412ms
```

[![CI](https://github.com/Khwahishd/quarry/actions/workflows/ci.yml/badge.svg)](https://github.com/Khwahishd/quarry/actions/workflows/ci.yml)
![Rust](https://img.shields.io/badge/rust-1.75%2B-orange)
![Dependencies](https://img.shields.io/badge/dependencies-0-blue)
![Tests](https://img.shields.io/badge/tests-96-brightgreen)

---

## The pipeline

```
   SQL text
     │   sql::parse            hand-written lexer → Pratt parser → AST
     ▼
   AST                         a faithful record of what was written
     │   logical::planner      name resolution, type checking, aggregate extraction
     ▼
   LogicalPlan                 WHAT to compute
     │   logical::optimizer    predicate pushdown · projection pushdown · constant folding
     ▼
   LogicalPlan'                the same answer, less work
     │   physical::planner     operator selection (hash join vs cross join, …)
     ▼
   PhysicalPlan                HOW to compute it — vectorized, batch at a time
```

`EXPLAIN` prints all three stages, so the optimizer's work is visible rather than asserted:

```
quarry> EXPLAIN SELECT id FROM trips WHERE fare > 90 AND city = 'Austin';

== Logical plan ==
Projection: id
  Filter: ((fare > 90) AND (city = 'Austin'))
    Scan: trips [*]

== Optimized plan ==
Projection: id
  Scan: trips [id, city, fare] filters=[(fare > 90) AND (city = 'Austin')]

== Physical plan ==
ProjectionExec: id
  ScanExec: trips [id, city, fare] filters=[(fare > 90) AND (city = 'Austin')]

rules applied: constant_folding, predicate_pushdown, projection_pushdown
```

Four of seven columns are never touched, and the predicate has moved into the scan.

---

## Why vectorized

Every expression evaluates over a whole batch and returns a whole column. That single decision is
what separates this from a tree-walking interpreter: **the type dispatch happens once per batch
instead of once per row**, and the inner loops run over primitive slices that LLVM can unroll.

The difference is measured, not assumed — the same filter over 1M rows, written both ways:

| | time | per row |
|---|---|---|
| Vectorized (`&[f64]` loop) | **431 µs** | 0.41 ns |
| Row-at-a-time (`Value` per row) | 3.11 ms | 2.97 ns |

**7.2× faster**, from the same algorithm, purely by changing where the dispatch happens.

### Columnar arrays

An `Array` is a contiguous typed buffer plus a *separate* validity bitmap. Nulls are tracked
out-of-band rather than as sentinel values, so the value buffer stays densely packed — and a column
with no nulls carries no bitmap at all, making the common case free.

Filtering, sorting and the probe side of a hash join are all expressed as "compute positions, then
`take`". Keeping the data movement in one type-specialized primitive means the expensive part is
written once.

---

## What it supports

**Queries** — `SELECT` with projections and aliases · `WHERE` · `GROUP BY` / `HAVING` ·
`ORDER BY` (multiple keys, `ASC`/`DESC`, ordinals, aliases) · `LIMIT`/`OFFSET` · `DISTINCT` ·
`INNER`/`LEFT`/`RIGHT`/`CROSS JOIN` · `EXPLAIN`.

**Expressions** — arithmetic and comparison · `AND`/`OR`/`NOT` · `IS [NOT] NULL` ·
`[NOT] IN` · `[NOT] BETWEEN` · `[NOT] LIKE` (with `%` and `_`) · `CASE WHEN` · `CAST`.

**Aggregates** — `COUNT` (incl. `COUNT(*)` and `COUNT(DISTINCT …)`) · `SUM` · `AVG` · `MIN` · `MAX`.

**Types** — `INT64`, `FLOAT64`, `UTF8`, `BOOLEAN`, `DATE32`, with CSV schema inference.

### SQL semantics taken seriously

The fiddly cases are where a toy engine and a real one diverge, so they are implemented
deliberately and each has a test:

- **Three-valued logic.** `FALSE AND NULL` is `FALSE` and `TRUE OR NULL` is `TRUE` — a known
  result short-circuits even when the other operand is unknown. Treating NULL as "just propagate"
  gets both wrong.
- **`NULL` never equals `NULL`.** A row whose join key is NULL is excluded from the hash table
  rather than being allowed to collide with other NULLs.
- **`SUM` over no rows is `NULL`, not `0`.** `COUNT` is `0`. SQL distinguishes "nothing to add up"
  from "adds up to nothing".
- **`AVG` and `/` always produce floats.** Integer division that silently truncates is a classic
  source of quietly wrong analytics.
- **Integer overflow produces `NULL`**, in both execution and constant folding. Wrapping would
  silently corrupt an aggregate; panicking would kill a query over one bad row.
- **NULLs sort last ascending, first descending** — and the direction applies to the null handling
  too, not just to the value comparison.
- **`ORDER BY` a column you didn't select** works: the key is added as a hidden column and trimmed
  off after the sort.
- **`GROUP BY` accepts a `SELECT` alias or an ordinal.** Not in the standard, but every engine people
  actually use supports it, and the alternative is writing a long `CASE` expression twice.

---

## The optimizer

Three rules, applied to a fixed point (bounded, so a future rule that oscillates degrades to
"not fully optimized" rather than hanging):

| Rule | What it does | Why it matters |
|---|---|---|
| **Predicate pushdown** | Splits conjunctions and moves each conjunct toward the scans | Rows are discarded before they are joined or aggregated |
| **Projection pushdown** | Prunes columns no operator reads | In a columnar engine an unread column is never touched at all |
| **Constant folding** | Evaluates literal-only subexpressions once | `b > 2 * 5 + 5` becomes `b > 15` at plan time, not per row |

Correctness is where the interesting constraints are, and each is enforced:

- A predicate **cannot** pass a projection unless every column it references is a plain passthrough
  — pushing a predicate on a *computed* column would mean evaluating an expression that does not
  exist below the projection.
- A predicate on a **grouping key** can go below the aggregation (filtering rows before grouping
  cannot change which groups survive). A predicate on an **aggregate** (i.e. `HAVING`) cannot.
- For an **outer join**, a predicate must not be pushed to the null-extended side: it would discard
  rows the join is obliged to emit with NULLs.
- Nothing is pushed through a **`LIMIT`** — filter-then-limit is not limit-then-filter.

And the property that makes all of it safe is tested directly: a battery of 17 queries covering
joins, aggregates, `HAVING`, `DISTINCT`, `CASE` and sorting is run **with the optimizer on and off**
and required to produce identical results.

### A bug this caught

Projection pushdown originally treated every column index as scan-relative. But a `Projection` above
an `Aggregate` addresses the *aggregate's* output columns:

```
Projection: c, SUM(b)          ← `c` here is aggregate output column 0
  Aggregate: groupBy=[c]
    Scan: wide [a, b, c]       ← ...but scan column 0 is `a`
```

So the rule kept column `a` alive for no reason — and, worse, was prepared to renumber an index that
was never in scan space at all. The fix tracks where the index space is redefined and stops
rewriting there ([`optimizer.rs`](src/logical/optimizer.rs)).

---

## Performance

100k rows, 7 columns, release build (Xeon @ 2.10GHz):

| Operation | Time |
|---|---|
| Scan + filter | 0.83 ms |
| Projection (arithmetic) | 0.62 ms |
| `GROUP BY` (6 groups) | 7.8 ms |
| `GROUP BY` (500 groups) | 9.5 ms |
| `LIKE '%99%'` | 4.1 ms |
| `DISTINCT` (2 columns) | 9.7 ms |
| `CASE WHEN` (3 branches) | 14.0 ms |
| Sort | 22.9 ms |
| Hash join (100k × 6) | 82 ms |
| Parse | 3.6 µs |
| Parse + plan + optimize | 20.1 µs |

Front-end cost is ~20 µs — four orders of magnitude below execution. Planning is effectively free at
this scale, which is exactly why spending it on optimization pays.

### What the optimizer is worth

Measured at both ends of the selectivity range, because a single number here would be misleading:

| Query | Unoptimized | Optimized | |
|---|---|---|---|
| `WHERE fare > 90 AND city = 'Austin'` (1.7% pass) | 4.45 ms | 4.08 ms | 1.1× |
| `WHERE fare > 5` (most rows pass) | 8.65 ms | **0.50 ms** | **17×** |

The selective query barely improves, and that is the honest expected result: almost nothing survives
the filter, so there is almost nothing for projection pushdown to save. The permissive query is
17× faster because pruning four unread columns — three of them strings — means `take` never copies
them for the 100k rows that *do* survive.

### Two performance bugs the benchmarks caught

Both were invisible in the tests and obvious the moment they were measured.

**Literal broadcast — `LIKE` was 26× slower than it needed to be.** Evaluating `city = 'Austin'`
broadcast the literal into a full column, heap-allocating 100,000 identical `String` clones just to
compare against the same text every row. Adding a scalar path for string comparison and `LIKE`
against a literal took that benchmark from 104 ms to **4.1 ms**.

**Per-row allocation in the join.** `encode_join_key` built a fresh one-element `Array` *per key
column, per row* purely to reuse the group-key encoder. Reusing a single buffer across rows cut the
join by 24%.

### A limitation this benchmark exposes

82 ms for a 100k × 6 hash join is slower than it should be, and the cause is specific: **projection
pushdown does not descend through joins.** The join therefore materializes all nine columns —
including `note`, which is unique per row — and `Array::take` clones every `String` it copies.

The fix is two things that reinforce each other: pushing projections into each join input so unread
columns are never gathered, and replacing `Vec<String>` with an Arrow-style offsets-plus-bytes
buffer so that gathering a string column copies indices rather than allocating. Both are listed
below rather than quietly omitted.

## Quick start

```bash
git clone https://github.com/Khwahishd/quarry && cd quarry
cargo build --release

# One-shot
./target/release/quarry -t trips=data/trips.csv -t cities=data/cities.csv \
  -c "SELECT city, COUNT(*) AS n, AVG(fare) AS avg FROM trips GROUP BY city ORDER BY n DESC"

# Interactive
./target/release/quarry -t trips=data/trips.csv -t cities=data/cities.csv
```

Shell commands: `.tables`, `.schema [table]`, `.timing on`, `.quit`.

### As a library

```rust
use quarry::Context;

let mut ctx = Context::new();
ctx.register_csv("trips", "data/trips.csv")?;

let result = ctx.sql("
    SELECT city, COUNT(*) AS n, AVG(fare) AS avg_fare
    FROM trips WHERE fare > 10
    GROUP BY city ORDER BY n DESC LIMIT 5
")?;

println!("{}", result.to_table());
for row in result.rows() { /* … */ }
# Ok::<(), quarry::Error>(())
```

---

## Testing

```bash
cargo test     # 96 tests
cargo clippy --all-targets -- -D warnings
cargo bench
```

| Suite | Covers |
|---|---|
| `src/sql/parser.rs` | Operator precedence and associativity, `BETWEEN` not swallowing its `AND`, postfix `NOT`, quoted identifiers, `''` escapes, exponent-vs-identifier lexing (`1east`), 16 malformed inputs |
| `src/array.rs` | Bitmap arithmetic, the no-null fast path, `take`/`slice`/`concat` preserving validity |
| `src/csv.rs` | Type inference and widening, quoted fields with embedded delimiters, ragged rows, outliers past the inference window, date round-trips across leap years and century boundaries |
| `tests/sql.rs` | 42 end-to-end semantics tests — NULL handling, joins, aggregates, `CASE`, `CAST`, errors |
| `tests/optimizer.rs` | Plan-shape assertions per rule, **plus optimized-vs-unoptimized result equivalence** |

Error messages are tested too, because they are part of the interface:

```
quarry> SELECT nope FROM trips;
planning error: no column named "nope"; available columns are
  [trips.trip_id, trips.city, trips.rider, trips.fare, trips.tip, trips.minutes, trips.day]

quarry> SELECT city, rider, COUNT(*) FROM trips GROUP BY city;
planning error: column "rider" must appear in GROUP BY or be used inside an aggregate function

quarry> SELECT a FROM t WHERE;
parse error: expected an expression at line 1, column 23 (found end of input)
```

---

## What I'd do next

Honest limitations, in rough priority order:

- **Projection pushdown through joins**, and **Arrow-style string buffers** (offsets + bytes
  instead of `Vec<String>`). Together these are worth the largest single speedup available — see the
  join measurement above.
- **Parallelism.** Execution is single-threaded. Scan, filter and the build side of a hash join are
  all embarrassingly parallel over batches; partitioned aggregation would follow.
- **A real storage format.** CSV is parsed fully into memory up front. Column chunks with min/max
  statistics would let the scan skip whole row groups — and would finally make projection pushdown
  pay what it should.
- **Subqueries and CTEs.** The parser and planner handle a single `SELECT` block; correlated
  subqueries would need a decorrelation pass.
- **Cost-based join ordering.** The physical planner always builds the hash table from the left
  input. With table cardinalities it should build from the smaller side and reorder multi-way joins.
- **SIMD and null-aware fast paths.** The comparison kernels are scalar loops. Explicit SIMD, and a
  null-free specialization that skips the validity check entirely, are both available.
- **Spilling.** Aggregation, sort and the join build side are all fully in-memory, so a result
  larger than RAM fails rather than spilling to disk.

## References

- Boncz, Zukowski & Nes, [*MonetDB/X100: Hyper-Pipelining Query Execution*](https://www.cidrdb.org/cidr2005/papers/P19.pdf) (2005) — the case for vectorized execution
- Graefe, [*Volcano — An Extensible and Parallel Query Evaluation System*](https://dl.acm.org/doi/10.1109/69.273032) (1994)
- Pratt, [*Top Down Operator Precedence*](https://tdop.github.io/) (1973)
- [Apache Arrow columnar format](https://arrow.apache.org/docs/format/Columnar.html) — the inspiration for the validity-bitmap layout

## License

MIT — see [LICENSE](LICENSE).

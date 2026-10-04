//! Physical operators.
//!
//! Operators form a pull-based (Volcano) pipeline, but each `next()` returns a
//! whole [`RecordBatch`] rather than a single row. This keeps Volcano's
//! composability -- any operator can feed any other -- while amortizing its
//! famous per-row virtual-call overhead across thousands of rows.
//!
//! Operators divide into two kinds, and the distinction decides the memory
//! profile of a query:
//!
//! * **Streaming** (scan, filter, projection, limit) process one batch at a
//!   time and hold nothing, so they run in constant memory.
//! * **Blocking** (aggregate, sort, hash join build side, distinct) must
//!   consume their entire input before producing anything.

use super::expr::{evaluate, selection_indices};
use crate::array::Array;
use crate::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use crate::error::{Error, Result};
use crate::logical::expr::LogicalExpr;
use crate::logical::optimizer::compare_values;
use crate::types::{SchemaRef, Value};
use std::sync::Arc;

/// A node in the physical plan.
pub trait Executor: Send {
    /// The schema of the batches this operator produces.
    fn schema(&self) -> SchemaRef;

    /// Produces the next batch, or `None` when exhausted.
    ///
    /// An operator may return an empty batch (for instance a filter that
    /// rejected everything in the current input batch); callers must treat
    /// `Some(empty)` as "keep going", not as end of stream.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    /// A one-line description for `EXPLAIN`.
    fn describe(&self) -> String;

    /// This operator's children, for `EXPLAIN`.
    fn children(&self) -> Vec<&dyn Executor>;

    /// Drains the operator into a vector of batches.
    fn collect(&mut self) -> Result<Vec<RecordBatch>>
    where
        Self: Sized,
    {
        let mut out = Vec::new();
        while let Some(b) = self.next_batch()? {
            if b.num_rows() > 0 {
                out.push(b);
            }
        }
        Ok(out)
    }
}

/// Drains any boxed executor.
pub fn collect(exec: &mut Box<dyn Executor>) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::new();
    while let Some(b) = exec.next_batch()? {
        if b.num_rows() > 0 {
            out.push(b);
        }
    }
    Ok(out)
}

/// Renders a physical plan as an indented tree.
pub fn explain(exec: &dyn Executor) -> String {
    fn go(e: &dyn Executor, depth: usize, out: &mut String) {
        out.push_str(&"  ".repeat(depth));
        out.push_str(&e.describe());
        out.push('\n');
        for c in e.children() {
            go(c, depth + 1, out);
        }
    }
    let mut s = String::new();
    go(exec, 0, &mut s);
    s
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// Reads batches from an in-memory table, applying projection and any
/// predicates that were pushed into the scan.
pub struct ScanExec {
    table: String,
    batches: Vec<RecordBatch>,
    projection: Option<Vec<usize>>,
    filters: Vec<LogicalExpr>,
    schema: SchemaRef,
    pos: usize,
}

impl ScanExec {
    /// Builds a scan over pre-materialized batches.
    pub fn new(
        table: String,
        batches: Vec<RecordBatch>,
        projection: Option<Vec<usize>>,
        filters: Vec<LogicalExpr>,
        schema: SchemaRef,
    ) -> Self {
        ScanExec { table, batches, projection, filters, schema, pos: 0 }
    }
}

impl Executor for ScanExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.batches.get(self.pos) else {
            return Ok(None);
        };
        self.pos += 1;

        // Project first: the pushed-down filters were rewritten to index into
        // the projected schema, and narrowing before filtering means the
        // filter's `take` copies fewer columns.
        let batch = match &self.projection {
            None => batch.clone(),
            Some(idx) => {
                let cols = idx.iter().map(|&i| Arc::clone(batch.column(i))).collect();
                RecordBatch::try_new(Arc::clone(&self.schema), cols)?
            }
        };

        if self.filters.is_empty() {
            return Ok(Some(batch));
        }

        // Conjoin the pushed-down predicates into one mask, then gather once.
        // Filtering per predicate would copy every column once per conjunct.
        let mut mask: Option<Array> = None;
        for f in &self.filters {
            let m = evaluate(f, &batch)?;
            mask = Some(match mask {
                None => m,
                Some(prev) => and_masks(&prev, &m)?,
            });
        }
        let indices = selection_indices(mask.as_ref().expect("filters is non-empty"))?;
        if indices.len() == batch.num_rows() {
            return Ok(Some(batch));
        }
        Ok(Some(batch.take(&indices)?))
    }

    fn describe(&self) -> String {
        let cols = match &self.projection {
            None => "*".to_string(),
            Some(_) => {
                self.schema.fields().iter().map(|f| f.name.clone()).collect::<Vec<_>>().join(", ")
            }
        };
        let mut s = format!("ScanExec: {} [{}]", self.table, cols);
        if !self.filters.is_empty() {
            let f: Vec<String> = self.filters.iter().map(|e| e.to_string()).collect();
            s.push_str(&format!(" filters=[{}]", f.join(" AND ")));
        }
        s
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![]
    }
}

/// Combines two boolean masks with AND semantics, treating NULL as false.
fn and_masks(a: &Array, b: &Array) -> Result<Array> {
    let (av, bv) = (
        a.as_bool().ok_or_else(|| Error::typ("filter mask must be boolean"))?,
        b.as_bool().ok_or_else(|| Error::typ("filter mask must be boolean"))?,
    );
    let out: Vec<bool> =
        (0..a.len()).map(|i| a.is_valid(i) && av[i] && b.is_valid(i) && bv[i]).collect();
    Ok(Array::new(crate::array::ArrayData::Boolean(out)))
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

/// Evaluates a predicate and keeps the rows where it is true.
pub struct FilterExec {
    input: Box<dyn Executor>,
    predicate: LogicalExpr,
}

impl FilterExec {
    /// Wraps an input with a predicate.
    pub fn new(input: Box<dyn Executor>, predicate: LogicalExpr) -> Self {
        FilterExec { input, predicate }
    }
}

impl Executor for FilterExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.input.next_batch()? else {
            return Ok(None);
        };
        let mask = evaluate(&self.predicate, &batch)?;
        let indices = selection_indices(&mask)?;
        // Skip the gather entirely when nothing was filtered out -- a very
        // common case for a selective-looking predicate that happens to pass.
        if indices.len() == batch.num_rows() {
            return Ok(Some(batch));
        }
        Ok(Some(batch.take(&indices)?))
    }

    fn describe(&self) -> String {
        format!("FilterExec: {}", self.predicate)
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// Evaluates a list of expressions, producing a new batch.
pub struct ProjectionExec {
    input: Box<dyn Executor>,
    exprs: Vec<LogicalExpr>,
    schema: SchemaRef,
}

impl ProjectionExec {
    /// Builds a projection.
    pub fn new(input: Box<dyn Executor>, exprs: Vec<LogicalExpr>, schema: SchemaRef) -> Self {
        ProjectionExec { input, exprs, schema }
    }
}

impl Executor for ProjectionExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let Some(batch) = self.input.next_batch()? else {
            return Ok(None);
        };
        let mut cols = Vec::with_capacity(self.exprs.len());
        for e in &self.exprs {
            cols.push(Arc::new(evaluate(e, &batch)?));
        }
        Ok(Some(RecordBatch::try_new(Arc::clone(&self.schema), cols)?))
    }

    fn describe(&self) -> String {
        let e: Vec<String> = self.exprs.iter().map(|x| x.to_string()).collect();
        format!("ProjectionExec: {}", e.join(", "))
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// Limit
// ---------------------------------------------------------------------------

/// Skips `skip` rows and emits at most `fetch`.
pub struct LimitExec {
    input: Box<dyn Executor>,
    skip: usize,
    fetch: Option<usize>,
    skipped: usize,
    emitted: usize,
}

impl LimitExec {
    /// Builds a limit operator.
    pub fn new(input: Box<dyn Executor>, skip: usize, fetch: Option<usize>) -> Self {
        LimitExec { input, skip, fetch, skipped: 0, emitted: 0 }
    }
}

impl Executor for LimitExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        loop {
            // Stop pulling as soon as the quota is met: this is what makes
            // `LIMIT 10` over a huge table cheap, since the scan below is
            // never asked for another batch.
            if let Some(f) = self.fetch {
                if self.emitted >= f {
                    return Ok(None);
                }
            }
            let Some(batch) = self.input.next_batch()? else {
                return Ok(None);
            };
            if batch.num_rows() == 0 {
                continue;
            }

            let mut batch = batch;
            if self.skipped < self.skip {
                let to_skip = (self.skip - self.skipped).min(batch.num_rows());
                self.skipped += to_skip;
                if to_skip == batch.num_rows() {
                    continue;
                }
                batch = batch.slice(to_skip, batch.num_rows() - to_skip)?;
            }

            if let Some(f) = self.fetch {
                let remaining = f - self.emitted;
                if batch.num_rows() > remaining {
                    batch = batch.slice(0, remaining)?;
                }
            }
            self.emitted += batch.num_rows();
            return Ok(Some(batch));
        }
    }

    fn describe(&self) -> String {
        match self.fetch {
            Some(f) => format!("LimitExec: skip={} fetch={}", self.skip, f),
            None => format!("LimitExec: skip={}", self.skip),
        }
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

/// Sorts the entire input.
///
/// Sorting is blocking by nature. The implementation sorts an index vector and
/// gathers once at the end, so the comparator moves `usize`s rather than whole
/// rows -- the strings and floats are copied exactly once.
pub struct SortExec {
    input: Box<dyn Executor>,
    exprs: Vec<(LogicalExpr, bool)>,
    output: Option<std::vec::IntoIter<RecordBatch>>,
}

impl SortExec {
    /// Builds a sort operator.
    pub fn new(input: Box<dyn Executor>, exprs: Vec<(LogicalExpr, bool)>) -> Self {
        SortExec { input, exprs, output: None }
    }

    fn build(&mut self) -> Result<()> {
        let schema = self.input.schema();
        let batches = collect(&mut self.input)?;
        let combined = RecordBatch::concat(Arc::clone(&schema), &batches)?;
        let n = combined.num_rows();

        // Materialize the sort keys once. Evaluating them inside the
        // comparator would re-run the expression O(n log n) times.
        let mut keys = Vec::with_capacity(self.exprs.len());
        for (e, asc) in &self.exprs {
            let arr = evaluate(e, &combined)?;
            let values: Vec<Value> = (0..n).map(|i| arr.value(i)).collect();
            keys.push((values, *asc));
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            for (values, asc) in &keys {
                let (x, y) = (&values[a], &values[b]);
                // NULLs sort last ascending and first descending. The
                // direction must be applied to the null handling as well as to
                // the value comparison: reversing only the latter would pin
                // NULLs to the end in both directions.
                let base = match (x.is_null(), y.is_null()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => compare_values(x, y).unwrap_or(std::cmp::Ordering::Equal),
                };
                let ord = if *asc { base } else { base.reverse() };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            // Fall back to input order, making the sort stable and its output
            // deterministic for equal keys.
            a.cmp(&b)
        });

        let sorted = combined.take(&indices)?;
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < sorted.num_rows() {
            let len = DEFAULT_BATCH_SIZE.min(sorted.num_rows() - offset);
            out.push(sorted.slice(offset, len)?);
            offset += len;
        }
        self.output = Some(out.into_iter());
        Ok(())
    }
}

impl Executor for SortExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.output.is_none() {
            self.build()?;
        }
        Ok(self.output.as_mut().expect("built above").next())
    }

    fn describe(&self) -> String {
        let e: Vec<String> = self
            .exprs
            .iter()
            .map(|(x, asc)| format!("{x} {}", if *asc { "ASC" } else { "DESC" }))
            .collect();
        format!("SortExec: {}", e.join(", "))
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

// ---------------------------------------------------------------------------
// Distinct
// ---------------------------------------------------------------------------

/// Removes duplicate rows using a hash set over row keys.
pub struct DistinctExec {
    input: Box<dyn Executor>,
    seen: std::collections::HashSet<Vec<u8>>,
}

impl DistinctExec {
    /// Builds a distinct operator.
    pub fn new(input: Box<dyn Executor>) -> Self {
        DistinctExec { input, seen: std::collections::HashSet::new() }
    }
}

impl Executor for DistinctExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        // Streaming rather than blocking: a batch's unique rows can be emitted
        // immediately, and only the key set grows.
        loop {
            let Some(batch) = self.input.next_batch()? else {
                return Ok(None);
            };
            if batch.num_rows() == 0 {
                continue;
            }
            let mut keep = Vec::new();
            for r in 0..batch.num_rows() {
                let key = super::aggregate::encode_row_key(&batch, r);
                if self.seen.insert(key) {
                    keep.push(r);
                }
            }
            if keep.is_empty() {
                continue;
            }
            return Ok(Some(batch.take(&keep)?));
        }
    }

    fn describe(&self) -> String {
        "DistinctExec".to_string()
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

pub use super::aggregate::AggregateExec;
pub use super::join::HashJoinExec;

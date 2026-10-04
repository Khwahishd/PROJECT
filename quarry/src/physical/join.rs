//! Hash joins.
//!
//! The smaller side is buffered into a hash table keyed by the join columns,
//! then the other side streams through probing it. This turns an O(n·m)
//! nested-loop join into O(n+m), and is the single most important physical
//! operator choice an analytical engine makes.

use super::plan::{collect, Executor};
use crate::array::Array;
use crate::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use crate::error::Result;
use crate::logical::expr::LogicalExpr;
use crate::physical::expr::{evaluate, selection_indices};
use crate::sql::ast::JoinType;
use crate::types::SchemaRef;
use std::collections::HashMap;
use std::sync::Arc;

/// Encodes the join key for one row of a batch into `buf`.
///
/// `buf` is reused across rows by the caller. Allocating a fresh key per row
/// is the obvious implementation and costs an allocation for every row on both
/// sides of the join -- which, on a 100k-row probe, dominates everything else
/// the operator does.
///
/// Returns false when any key column is NULL. SQL's NULL never equals
/// anything, including another NULL, so such a row can never match and is
/// excluded from the hash table rather than being allowed to collide with
/// other NULLs.
fn encode_join_key(batch: &RecordBatch, row: usize, cols: &[usize], buf: &mut Vec<u8>) -> bool {
    buf.clear();
    for &c in cols {
        let col = batch.column(c);
        if !col.is_valid(row) {
            return false;
        }
        super::aggregate::encode_value(&col.value(row), buf);
    }
    true
}

/// The build-side hash table.
pub struct HashJoin {
    /// Maps a key to the build-side row positions carrying it.
    table: HashMap<Vec<u8>, Vec<usize>>,
    /// The fully materialized build side.
    build: RecordBatch,
    /// Which build rows found at least one match, for outer joins.
    matched: Vec<bool>,
}

/// The hash join operator.
pub struct HashJoinExec {
    left: Option<Box<dyn Executor>>,
    right: Box<dyn Executor>,
    on: Vec<(usize, usize)>,
    filter: Option<LogicalExpr>,
    join_type: JoinType,
    schema: SchemaRef,

    state: Option<HashJoin>,
    /// Buffered output batches, produced lazily as the probe side streams.
    pending: std::collections::VecDeque<RecordBatch>,
    probe_done: bool,
    emitted_unmatched: bool,
}

impl HashJoinExec {
    /// Builds a hash join.
    pub fn new(
        left: Box<dyn Executor>,
        right: Box<dyn Executor>,
        on: Vec<(usize, usize)>,
        filter: Option<LogicalExpr>,
        join_type: JoinType,
        schema: SchemaRef,
    ) -> Self {
        HashJoinExec {
            left: Some(left),
            right,
            on,
            filter,
            join_type,
            schema,
            state: None,
            pending: std::collections::VecDeque::new(),
            probe_done: false,
            emitted_unmatched: false,
        }
    }

    /// Materializes the left side into a hash table.
    fn build(&mut self) -> Result<()> {
        let mut left = self.left.take().expect("build runs exactly once");
        let schema = left.schema();
        let batches = collect(&mut left)?;
        let build = RecordBatch::concat(schema, &batches)?;

        let left_cols: Vec<usize> = self.on.iter().map(|(l, _)| *l).collect();
        let mut table: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
        if !left_cols.is_empty() {
            let mut key = Vec::with_capacity(left_cols.len() * 9);
            for row in 0..build.num_rows() {
                if encode_join_key(&build, row, &left_cols, &mut key) {
                    // Only the keys that are actually inserted are cloned, so a
                    // duplicate key costs a lookup rather than an allocation.
                    match table.get_mut(&key) {
                        Some(rows) => rows.push(row),
                        None => {
                            table.insert(key.clone(), vec![row]);
                        }
                    }
                }
            }
        }
        let matched = vec![false; build.num_rows()];
        self.state = Some(HashJoin { table, build, matched });
        Ok(())
    }

    /// Joins one probe batch against the hash table.
    fn probe(&mut self, probe: &RecordBatch) -> Result<()> {
        let state = self.state.as_mut().expect("built before probing");
        let right_cols: Vec<usize> = self.on.iter().map(|(_, r)| *r).collect();

        let mut left_idx = Vec::new();
        let mut right_idx = Vec::new();
        // Probe rows with no match, needed for a RIGHT join.
        let mut right_unmatched = Vec::new();
        // Reused across rows; see encode_join_key.
        let mut key = Vec::with_capacity(right_cols.len() * 9);

        for row in 0..probe.num_rows() {
            // A cross join pairs every row with every row.
            if self.on.is_empty() {
                for l in 0..state.build.num_rows() {
                    left_idx.push(l);
                    right_idx.push(row);
                }
                continue;
            }

            let hit = encode_join_key(probe, row, &right_cols, &mut key)
                .then(|| state.table.get(&key))
                .flatten();
            match hit {
                Some(rows) => {
                    for &l in rows {
                        state.matched[l] = true;
                        left_idx.push(l);
                        right_idx.push(row);
                    }
                }
                None => right_unmatched.push(row),
            }
        }

        if !left_idx.is_empty() {
            let l = state.build.take(&left_idx)?;
            let r = probe.take(&right_idx)?;
            let mut joined = concat_columns(&self.schema, &l, &r)?;

            // Apply any non-equi residual predicate after the join, since it
            // cannot be used as a hash key.
            if let Some(f) = &self.filter {
                let mask = evaluate(f, &joined)?;
                let keep = selection_indices(&mask)?;
                joined = joined.take(&keep)?;
            }
            if joined.num_rows() > 0 {
                self.pending.push_back(joined);
            }
        }

        // RIGHT join: emit unmatched probe rows with NULLs on the left.
        if self.join_type == JoinType::Right && !right_unmatched.is_empty() {
            let r = probe.take(&right_unmatched)?;
            let l = null_batch(&state.build.schema(), r.num_rows());
            self.pending.push_back(concat_columns(&self.schema, &l, &r)?);
        }
        Ok(())
    }

    /// Emits build-side rows that never matched, for a LEFT join.
    fn emit_unmatched_left(&mut self) -> Result<()> {
        let state = self.state.as_ref().expect("built");
        let rows: Vec<usize> = (0..state.build.num_rows()).filter(|&i| !state.matched[i]).collect();
        if rows.is_empty() {
            return Ok(());
        }
        let l = state.build.take(&rows)?;
        let r = null_batch(&self.right.schema(), l.num_rows());
        let joined = concat_columns(&self.schema, &l, &r)?;
        let mut offset = 0;
        while offset < joined.num_rows() {
            let len = DEFAULT_BATCH_SIZE.min(joined.num_rows() - offset);
            self.pending.push_back(joined.slice(offset, len)?);
            offset += len;
        }
        Ok(())
    }
}

impl Executor for HashJoinExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.state.is_none() {
            self.build()?;
        }

        loop {
            if let Some(b) = self.pending.pop_front() {
                return Ok(Some(b));
            }
            if self.probe_done {
                if self.join_type == JoinType::Left && !self.emitted_unmatched {
                    self.emitted_unmatched = true;
                    self.emit_unmatched_left()?;
                    continue;
                }
                return Ok(None);
            }
            match self.right.next_batch()? {
                Some(b) => {
                    if b.num_rows() > 0 {
                        self.probe(&b)?;
                    }
                }
                None => self.probe_done = true,
            }
        }
    }

    fn describe(&self) -> String {
        format!(
            "HashJoinExec: {} on {} key(s){}",
            self.join_type,
            self.on.len(),
            match &self.filter {
                Some(f) => format!(" filter={f}"),
                None => String::new(),
            }
        )
    }

    fn children(&self) -> Vec<&dyn Executor> {
        match &self.left {
            Some(l) => vec![l.as_ref(), self.right.as_ref()],
            None => vec![self.right.as_ref()],
        }
    }
}

/// Side-by-side concatenation of two batches with the same row count.
fn concat_columns(schema: &SchemaRef, l: &RecordBatch, r: &RecordBatch) -> Result<RecordBatch> {
    let mut cols: Vec<Arc<Array>> = l.columns().to_vec();
    cols.extend(r.columns().iter().cloned());
    RecordBatch::try_new(Arc::clone(schema), cols)
}

/// An all-NULL batch matching a schema, used to pad an outer join.
fn null_batch(schema: &SchemaRef, rows: usize) -> RecordBatch {
    let cols: Vec<Arc<Array>> =
        schema.fields().iter().map(|f| Arc::new(Array::new_null(f.data_type, rows))).collect();
    RecordBatch::try_new(Arc::clone(schema), cols)
        .expect("a batch of equal-length null columns is always valid")
}

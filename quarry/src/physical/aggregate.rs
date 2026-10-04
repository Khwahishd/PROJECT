//! Hash aggregation.
//!
//! Rows are grouped by a serialized key and each group holds one accumulator
//! per aggregate. Aggregation is blocking: nothing can be emitted until the
//! last input row has been seen, because any row might belong to any group.

use super::plan::Executor;
use crate::array::Array;
use crate::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use crate::error::Result;
use crate::logical::expr::{AggregateExpr, AggregateFunction, LogicalExpr};
use crate::logical::optimizer::compare_values;
use crate::physical::expr::evaluate;
use crate::types::{DataType, SchemaRef, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Serializes one row into a byte key for hashing.
///
/// A byte key is used instead of `Vec<Value>` because `f64` is not `Hash` or
/// `Eq`, and because comparing a packed byte string is far cheaper than
/// comparing a vector of enums. The type tag prefix keeps `1` (int) and `1.0`
/// (float) in distinct groups, and the length prefix on strings prevents
/// `("a","bc")` and `("ab","c")` from colliding.
pub fn encode_row_key(batch: &RecordBatch, row: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(batch.num_columns() * 9);
    for c in 0..batch.num_columns() {
        encode_value(&batch.column(c).value(row), &mut key);
    }
    key
}

/// Serializes a group key built from evaluated key columns.
pub fn encode_key_from_arrays(keys: &[Array], row: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(keys.len() * 9);
    for k in keys {
        encode_value(&k.value(row), &mut key);
    }
    key
}

/// Appends one value's canonical byte encoding to `out`.
pub(crate) fn encode_value(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(0),
        Value::Int64(x) => {
            out.push(1);
            out.extend_from_slice(&x.to_le_bytes());
        }
        Value::Float64(x) => {
            out.push(2);
            // Canonicalize NaN so that all NaNs land in one group rather than
            // each forming its own, which bit patterns alone would allow.
            let bits = if x.is_nan() { f64::NAN.to_bits() } else { x.to_bits() };
            out.extend_from_slice(&bits.to_le_bytes());
        }
        Value::Utf8(s) => {
            out.push(3);
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Value::Boolean(b) => {
            out.push(4);
            out.push(*b as u8);
        }
        Value::Date32(d) => {
            out.push(5);
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
}

/// Running state for one aggregate within one group.
#[derive(Debug, Clone)]
enum Accumulator {
    /// COUNT: a running total.
    Count {
        n: i64,
        /// Distinct inputs seen so far, when COUNT(DISTINCT ...) was requested.
        seen: Option<std::collections::HashSet<Vec<u8>>>,
    },
    /// SUM over integers, switching to float on overflow.
    SumInt { total: Option<i64>, overflowed: f64, any: bool },
    /// SUM over floats.
    SumFloat { total: f64, any: bool },
    /// AVG: sum and count, divided at the end.
    Avg { sum: f64, n: i64 },
    /// MIN / MAX over any ordered type.
    MinMax { current: Option<Value>, is_min: bool },
}

impl Accumulator {
    fn new(func: AggregateFunction, input: DataType, distinct: bool) -> Accumulator {
        match func {
            AggregateFunction::Count => {
                Accumulator::Count { n: 0, seen: distinct.then(std::collections::HashSet::new) }
            }
            AggregateFunction::Sum => match input {
                DataType::Float64 => Accumulator::SumFloat { total: 0.0, any: false },
                _ => Accumulator::SumInt { total: Some(0), overflowed: 0.0, any: false },
            },
            AggregateFunction::Avg => Accumulator::Avg { sum: 0.0, n: 0 },
            AggregateFunction::Min => Accumulator::MinMax { current: None, is_min: true },
            AggregateFunction::Max => Accumulator::MinMax { current: None, is_min: false },
        }
    }

    fn update(&mut self, v: &Value) {
        match self {
            Accumulator::Count { n, seen } => {
                // COUNT(expr) ignores NULLs; COUNT(*) is handled by the caller,
                // which passes a non-null placeholder.
                if v.is_null() {
                    return;
                }
                if let Some(set) = seen {
                    let mut key = Vec::new();
                    encode_value(v, &mut key);
                    if !set.insert(key) {
                        return;
                    }
                }
                *n += 1;
            }
            Accumulator::SumInt { total, overflowed, any } => {
                let Some(x) = (match v {
                    Value::Int64(x) => Some(*x),
                    Value::Float64(x) => Some(*x as i64),
                    _ => None,
                }) else {
                    return;
                };
                *any = true;
                // Fall back to float accumulation on overflow rather than
                // wrapping: a wrong total is worse than a slightly imprecise
                // one, and the column type widens to match.
                match total.and_then(|t| t.checked_add(x)) {
                    Some(t) => *total = Some(t),
                    None => {
                        if let Some(t) = total.take() {
                            *overflowed = t as f64;
                        }
                        *overflowed += x as f64;
                    }
                }
            }
            Accumulator::SumFloat { total, any } => {
                if let Some(x) = v.as_f64() {
                    *total += x;
                    *any = true;
                }
            }
            Accumulator::Avg { sum, n } => {
                if let Some(x) = v.as_f64() {
                    *sum += x;
                    *n += 1;
                }
            }
            Accumulator::MinMax { current, is_min } => {
                if v.is_null() {
                    return;
                }
                match current {
                    None => *current = Some(v.clone()),
                    Some(c) => {
                        if let Some(ord) = compare_values(v, c) {
                            let better = if *is_min {
                                ord == std::cmp::Ordering::Less
                            } else {
                                ord == std::cmp::Ordering::Greater
                            };
                            if better {
                                *current = Some(v.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    fn finish(&self) -> Value {
        match self {
            Accumulator::Count { n, .. } => Value::Int64(*n),
            // SUM over an empty set is NULL, not zero -- SQL distinguishes
            // "nothing to add up" from "adds up to nothing".
            Accumulator::SumInt { total, overflowed, any } => {
                if !*any {
                    Value::Null
                } else {
                    match total {
                        Some(t) => Value::Int64(*t),
                        None => Value::Float64(*overflowed),
                    }
                }
            }
            Accumulator::SumFloat { total, any } => {
                if *any {
                    Value::Float64(*total)
                } else {
                    Value::Null
                }
            }
            Accumulator::Avg { sum, n } => {
                if *n == 0 {
                    Value::Null
                } else {
                    Value::Float64(sum / *n as f64)
                }
            }
            Accumulator::MinMax { current, .. } => current.clone().unwrap_or(Value::Null),
        }
    }
}

/// Builds and holds the hash table for a grouped aggregation.
pub struct GroupedAggregator {
    group_exprs: Vec<LogicalExpr>,
    aggregates: Vec<AggregateExpr>,
    schema: SchemaRef,

    /// Maps a serialized key to its position in `group_values` / `accumulators`.
    table: HashMap<Vec<u8>, usize>,
    /// The group key values, in insertion order.
    group_values: Vec<Vec<Value>>,
    /// One accumulator row per group.
    accumulators: Vec<Vec<Accumulator>>,
}

impl GroupedAggregator {
    /// Creates an aggregator.
    pub fn new(
        group_exprs: Vec<LogicalExpr>,
        aggregates: Vec<AggregateExpr>,
        schema: SchemaRef,
    ) -> Self {
        GroupedAggregator {
            group_exprs,
            aggregates,
            schema,
            table: HashMap::new(),
            group_values: Vec::new(),
            accumulators: Vec::new(),
        }
    }

    /// Folds one input batch into the hash table.
    pub fn update(&mut self, batch: &RecordBatch) -> Result<()> {
        let n = batch.num_rows();
        if n == 0 {
            return Ok(());
        }

        // Evaluate the key and argument columns once per batch, not per row.
        let keys: Vec<Array> =
            self.group_exprs.iter().map(|e| evaluate(e, batch)).collect::<Result<_>>()?;

        let args: Vec<Option<Array>> = self
            .aggregates
            .iter()
            .map(|a| match &a.arg {
                Some(e) => evaluate(e, batch).map(Some),
                None => Ok(None),
            })
            .collect::<Result<_>>()?;

        for row in 0..n {
            let key = if keys.is_empty() {
                // No GROUP BY: a single global group.
                Vec::new()
            } else {
                encode_key_from_arrays(&keys, row)
            };

            let idx = match self.table.get(&key) {
                Some(&i) => i,
                None => {
                    let i = self.group_values.len();
                    let values: Vec<Value> = keys.iter().map(|k| k.value(row)).collect();
                    self.group_values.push(values);
                    let accs = self
                        .aggregates
                        .iter()
                        .map(|a| {
                            let t = a
                                .arg
                                .as_ref()
                                .map(|e| e.data_type().unwrap_or(DataType::Null))
                                .unwrap_or(DataType::Int64);
                            Accumulator::new(a.func, t, a.distinct)
                        })
                        .collect();
                    self.accumulators.push(accs);
                    self.table.insert(key, i);
                    i
                }
            };

            for (ai, arg) in args.iter().enumerate() {
                let v = match arg {
                    // COUNT(*) counts rows, so feed a non-null placeholder.
                    None => Value::Int64(1),
                    Some(a) => a.value(row),
                };
                self.accumulators[idx][ai].update(&v);
            }
        }
        Ok(())
    }

    /// Finalizes the aggregation into output batches.
    pub fn finish(&mut self) -> Result<Vec<RecordBatch>> {
        // A global aggregate over zero rows still produces one row: SELECT
        // COUNT(*) FROM empty is 0, not an empty result.
        if self.group_values.is_empty() && self.group_exprs.is_empty() {
            let accs: Vec<Accumulator> = self
                .aggregates
                .iter()
                .map(|a| Accumulator::new(a.func, DataType::Int64, a.distinct))
                .collect();
            self.group_values.push(Vec::new());
            self.accumulators.push(accs);
        }

        let n_groups = self.group_values.len();
        let n_group_cols = self.group_exprs.len();

        let mut columns: Vec<Vec<Value>> = vec![Vec::with_capacity(n_groups); self.schema.len()];
        for (g, keys) in self.group_values.iter().enumerate() {
            for (c, col) in columns.iter_mut().take(n_group_cols).enumerate() {
                col.push(keys[c].clone());
            }
            for (a, acc) in self.accumulators[g].iter().enumerate() {
                columns[n_group_cols + a].push(acc.finish());
            }
        }

        let arrays: Vec<Arc<Array>> = columns
            .into_iter()
            .enumerate()
            .map(|(i, vals)| {
                let t = self.schema.field(i).data_type;
                Array::from_values(coerce(vals, t), t).map(Arc::new)
            })
            .collect::<Result<_>>()?;

        let full = RecordBatch::try_new(Arc::clone(&self.schema), arrays)?;
        let mut out = Vec::new();
        let mut offset = 0;
        while offset < full.num_rows() {
            let len = DEFAULT_BATCH_SIZE.min(full.num_rows() - offset);
            out.push(full.slice(offset, len)?);
            offset += len;
        }
        if out.is_empty() {
            out.push(RecordBatch::empty(Arc::clone(&self.schema)));
        }
        Ok(out)
    }
}

/// Widens integer values where the output column is float-typed.
fn coerce(values: Vec<Value>, target: DataType) -> Vec<Value> {
    values
        .into_iter()
        .map(|v| match (&v, target) {
            (Value::Int64(x), DataType::Float64) => Value::Float64(*x as f64),
            (Value::Float64(x), DataType::Int64) => Value::Int64(*x as i64),
            _ => v,
        })
        .collect()
}

/// The aggregation operator.
pub struct AggregateExec {
    input: Box<dyn Executor>,
    agg: GroupedAggregator,
    schema: SchemaRef,
    output: Option<std::vec::IntoIter<RecordBatch>>,
    group_count: usize,
    agg_count: usize,
}

impl AggregateExec {
    /// Builds an aggregation operator.
    pub fn new(
        input: Box<dyn Executor>,
        group_exprs: Vec<LogicalExpr>,
        aggregates: Vec<AggregateExpr>,
        schema: SchemaRef,
    ) -> Self {
        let group_count = group_exprs.len();
        let agg_count = aggregates.len();
        AggregateExec {
            input,
            agg: GroupedAggregator::new(group_exprs, aggregates, Arc::clone(&schema)),
            schema,
            output: None,
            group_count,
            agg_count,
        }
    }
}

impl Executor for AggregateExec {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.output.is_none() {
            while let Some(b) = self.input.next_batch()? {
                self.agg.update(&b)?;
            }
            self.output = Some(self.agg.finish()?.into_iter());
        }
        Ok(self.output.as_mut().expect("built above").next())
    }

    fn describe(&self) -> String {
        format!("AggregateExec: groups={} aggregates={}", self.group_count, self.agg_count)
    }

    fn children(&self) -> Vec<&dyn Executor> {
        vec![self.input.as_ref()]
    }
}

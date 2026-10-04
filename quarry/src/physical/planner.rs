//! Physical planning: choosing operators for a logical plan.
//!
//! The mapping is mostly one-to-one; the decisions worth making are:
//!
//! * which join algorithm to use (hash join when equi-keys exist, otherwise a
//!   cross join plus a filter), and
//! * which side of a join to build the hash table from.

use super::aggregate::AggregateExec;
use super::join::HashJoinExec;
use super::plan::{
    DistinctExec, Executor, FilterExec, LimitExec, ProjectionExec, ScanExec, SortExec,
};
use crate::batch::RecordBatch;
use crate::error::{Error, Result};
use crate::logical::plan::LogicalPlan;
use crate::types::SchemaRef;
use std::sync::Arc;

/// Supplies the materialized data behind a table name.
pub trait BatchSource {
    /// Returns the batches making up `table`.
    fn batches(&self, table: &str) -> Option<Vec<RecordBatch>>;
}

/// Converts a logical plan into an executable physical plan.
pub fn create_physical_plan(
    plan: &LogicalPlan,
    source: &dyn BatchSource,
) -> Result<Box<dyn Executor>> {
    Ok(match plan {
        LogicalPlan::Scan { table, schema, projection, filters } => {
            let batches = source
                .batches(table)
                .ok_or_else(|| Error::exec(format!("table {table:?} has no data registered")))?;
            // The scan's output schema reflects the pushed-down projection.
            let out_schema: SchemaRef = match projection {
                None => Arc::clone(schema),
                Some(idx) => Arc::new(schema.project(idx)?),
            };
            Box::new(ScanExec::new(
                table.clone(),
                batches,
                projection.clone(),
                filters.clone(),
                out_schema,
            ))
        }

        LogicalPlan::Filter { predicate, input } => {
            Box::new(FilterExec::new(create_physical_plan(input, source)?, predicate.clone()))
        }

        LogicalPlan::Projection { exprs, schema, input } => Box::new(ProjectionExec::new(
            create_physical_plan(input, source)?,
            exprs.clone(),
            Arc::clone(schema),
        )),

        LogicalPlan::Aggregate { group_by, aggregates, schema, input } => {
            Box::new(AggregateExec::new(
                create_physical_plan(input, source)?,
                group_by.clone(),
                aggregates.clone(),
                Arc::clone(schema),
            ))
        }

        LogicalPlan::Sort { exprs, input } => {
            Box::new(SortExec::new(create_physical_plan(input, source)?, exprs.clone()))
        }

        LogicalPlan::Limit { skip, fetch, input } => {
            Box::new(LimitExec::new(create_physical_plan(input, source)?, *skip, *fetch))
        }

        LogicalPlan::Distinct { input } => {
            Box::new(DistinctExec::new(create_physical_plan(input, source)?))
        }

        LogicalPlan::Join { left, right, on, filter, join_type, schema } => {
            // The left input becomes the build side. A cost-based planner would
            // pick the smaller side here; with no statistics available, keeping
            // the written order makes plans predictable and EXPLAIN output
            // match what the user wrote.
            Box::new(HashJoinExec::new(
                create_physical_plan(left, source)?,
                create_physical_plan(right, source)?,
                on.clone(),
                filter.clone(),
                *join_type,
                Arc::clone(schema),
            ))
        }

        LogicalPlan::EmptyRelation { produce_one_row, schema } => {
            // A one-row, zero-column batch is what lets `SELECT 1` work: the
            // projection above it evaluates its literals once.
            let batches = if *produce_one_row {
                vec![RecordBatch::try_new(Arc::clone(schema), vec![])
                    .map(|b| one_row(b, schema))??]
            } else {
                vec![]
            };
            Box::new(ScanExec::new(
                "(empty)".to_string(),
                batches,
                None,
                vec![],
                Arc::clone(schema),
            ))
        }
    })
}

/// Produces a batch with one row and no columns.
///
/// `RecordBatch` derives its row count from column 0, so a zero-column batch
/// reports zero rows. A synthetic single-column batch gives the projection
/// above it something to iterate.
fn one_row(_empty: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    use crate::array::{Array, ArrayData};
    use crate::types::{Field, Schema};
    if !schema.is_empty() {
        return Err(Error::exec("EmptyRelation must have an empty schema"));
    }
    let s = Arc::new(Schema::new(vec![Field::new("__one_row", crate::types::DataType::Int64)]));
    RecordBatch::try_new(s, vec![Arc::new(Array::new(ArrayData::Int64(vec![0])))])
}

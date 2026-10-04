//! The logical plan: a tree describing *what* to compute, not how.

use super::expr::{AggregateExpr, LogicalExpr};
use crate::error::Result;
use crate::sql::ast::JoinType;
use crate::types::{Field, Schema, SchemaRef};
use std::fmt;
use std::sync::Arc;

/// A node in the logical plan tree.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Reads a registered table.
    Scan {
        /// Table name.
        table: String,
        /// The table's full schema.
        schema: SchemaRef,
        /// Columns to read; `None` means all of them.
        ///
        /// Projection pushdown fills this in, which is what turns a wide table
        /// into a narrow scan.
        projection: Option<Vec<usize>>,
        /// Predicates the scan can evaluate itself, filtering rows before they
        /// are ever handed upward.
        filters: Vec<LogicalExpr>,
    },
    /// Filters rows by a predicate.
    Filter {
        /// The predicate.
        predicate: LogicalExpr,
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Computes a list of expressions.
    Projection {
        /// The expressions.
        exprs: Vec<LogicalExpr>,
        /// The resulting schema.
        schema: SchemaRef,
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Groups rows and computes aggregates.
    Aggregate {
        /// Grouping expressions; empty means a single global group.
        group_by: Vec<LogicalExpr>,
        /// Aggregates to compute.
        aggregates: Vec<AggregateExpr>,
        /// The output schema: group columns followed by aggregate columns.
        schema: SchemaRef,
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Sorts rows.
    Sort {
        /// Sort keys with their directions.
        exprs: Vec<(LogicalExpr, bool)>,
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Skips and/or truncates rows.
    Limit {
        /// Rows to skip.
        skip: usize,
        /// Rows to emit, or `None` for all remaining.
        fetch: Option<usize>,
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Removes duplicate rows.
    Distinct {
        /// The input.
        input: Box<LogicalPlan>,
    },
    /// Joins two inputs.
    Join {
        /// Left input.
        left: Box<LogicalPlan>,
        /// Right input.
        right: Box<LogicalPlan>,
        /// Equi-join key pairs, as (left index, right index).
        ///
        /// Splitting equality conjuncts out of the ON clause is what lets the
        /// physical planner choose a hash join instead of a nested loop.
        on: Vec<(usize, usize)>,
        /// Any remaining non-equi predicate.
        filter: Option<LogicalExpr>,
        /// The join kind.
        join_type: JoinType,
        /// The combined output schema.
        schema: SchemaRef,
    },
    /// A single row with no columns, the input for `SELECT 1`.
    EmptyRelation {
        /// Whether to produce one row (true) or none (false).
        produce_one_row: bool,
        /// The (empty) schema.
        schema: SchemaRef,
    },
}

impl LogicalPlan {
    /// The schema this plan produces.
    pub fn schema(&self) -> SchemaRef {
        match self {
            LogicalPlan::Scan { schema, projection, .. } => match projection {
                None => Arc::clone(schema),
                Some(idx) => Arc::new(
                    schema
                        .project(idx)
                        .expect("scan projection indices are validated when they are set"),
                ),
            },
            LogicalPlan::Filter { input, .. } => input.schema(),
            LogicalPlan::Projection { schema, .. } => Arc::clone(schema),
            LogicalPlan::Aggregate { schema, .. } => Arc::clone(schema),
            LogicalPlan::Sort { input, .. } => input.schema(),
            LogicalPlan::Limit { input, .. } => input.schema(),
            LogicalPlan::Distinct { input } => input.schema(),
            LogicalPlan::Join { schema, .. } => Arc::clone(schema),
            LogicalPlan::EmptyRelation { schema, .. } => Arc::clone(schema),
        }
    }

    /// The plan's children, left to right.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::EmptyRelation { .. } => vec![],
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
        }
    }

    /// Builds a schema for a projection's output expressions.
    pub fn projection_schema(exprs: &[LogicalExpr]) -> Result<Schema> {
        let mut fields = Vec::with_capacity(exprs.len());
        for e in exprs {
            fields.push(Field::new(e.display_name(), e.data_type()?));
        }
        Ok(Schema::new(fields))
    }

    /// Renders the plan as an indented tree, for `EXPLAIN`.
    pub fn display_indent(&self) -> String {
        let mut s = String::new();
        self.fmt_indent(0, &mut s);
        s
    }

    fn fmt_indent(&self, depth: usize, out: &mut String) {
        let pad = "  ".repeat(depth);
        out.push_str(&pad);
        out.push_str(&self.one_line());
        out.push('\n');
        for c in self.children() {
            c.fmt_indent(depth + 1, out);
        }
    }

    /// A single-line description of this node, without its children.
    pub fn one_line(&self) -> String {
        match self {
            LogicalPlan::Scan { table, schema, projection, filters } => {
                let cols = match projection {
                    None => "*".to_string(),
                    Some(idx) => idx
                        .iter()
                        .map(|&i| schema.field(i).name.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                };
                let mut s = format!("Scan: {table} [{cols}]");
                if !filters.is_empty() {
                    let f: Vec<String> = filters.iter().map(|e| e.to_string()).collect();
                    s.push_str(&format!(" filters=[{}]", f.join(" AND ")));
                }
                s
            }
            LogicalPlan::Filter { predicate, .. } => format!("Filter: {predicate}"),
            LogicalPlan::Projection { exprs, .. } => {
                let e: Vec<String> = exprs.iter().map(|x| x.to_string()).collect();
                format!("Projection: {}", e.join(", "))
            }
            LogicalPlan::Aggregate { group_by, aggregates, .. } => {
                let g: Vec<String> = group_by.iter().map(|x| x.to_string()).collect();
                let a: Vec<String> = aggregates.iter().map(|x| x.to_string()).collect();
                format!("Aggregate: groupBy=[{}] aggs=[{}]", g.join(", "), a.join(", "))
            }
            LogicalPlan::Sort { exprs, .. } => {
                let e: Vec<String> = exprs
                    .iter()
                    .map(|(x, asc)| format!("{x} {}", if *asc { "ASC" } else { "DESC" }))
                    .collect();
                format!("Sort: {}", e.join(", "))
            }
            LogicalPlan::Limit { skip, fetch, .. } => match fetch {
                Some(n) => format!("Limit: skip={skip} fetch={n}"),
                None => format!("Limit: skip={skip}"),
            },
            LogicalPlan::Distinct { .. } => "Distinct".to_string(),
            LogicalPlan::Join { on, filter, join_type, left, right, .. } => {
                let lschema = left.schema();
                let rschema = right.schema();
                let keys: Vec<String> = on
                    .iter()
                    .map(|(l, r)| {
                        format!("{} = {}", lschema.field(*l).name, rschema.field(*r).name)
                    })
                    .collect();
                let mut s = format!("{join_type}Join: on=[{}]", keys.join(", "));
                if let Some(f) = filter {
                    s.push_str(&format!(" filter={f}"));
                }
                s
            }
            LogicalPlan::EmptyRelation { produce_one_row, .. } => {
                format!("EmptyRelation: rows={}", if *produce_one_row { 1 } else { 0 })
            }
        }
    }
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display_indent())
    }
}

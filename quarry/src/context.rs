//! The user-facing entry point: register tables, run SQL.

use crate::batch::{format_batches, RecordBatch};
use crate::csv::{self, ReadOptions};
use crate::error::{Error, Result};
use crate::logical::planner::{Planner, TableProvider};
use crate::logical::{optimize, LogicalPlan, Optimizer};
use crate::physical::planner::{create_physical_plan, BatchSource};
use crate::physical::{collect, explain};
use crate::sql::ast::Statement;
use crate::types::{SchemaRef, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A registered table: its schema and its materialized data.
struct Table {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

/// Holds registered tables and executes queries against them.
///
/// # Example
///
/// ```
/// use quarry::Context;
///
/// let mut ctx = Context::new();
/// ctx.register_csv_str("t", "a,b\n1,x\n2,y\n").unwrap();
/// let r = ctx.sql("SELECT a FROM t WHERE b = 'y'").unwrap();
/// assert_eq!(r.num_rows(), 1);
/// ```
pub struct Context {
    tables: HashMap<String, Table>,
    optimizer: Optimizer,
    /// When false, plans run unoptimized -- used by tests and benchmarks to
    /// measure what the optimizer is actually worth.
    optimize_enabled: bool,
}

impl Context {
    /// Creates an empty context.
    pub fn new() -> Self {
        Context { tables: HashMap::new(), optimizer: Optimizer::new(), optimize_enabled: true }
    }

    /// Enables or disables optimization.
    pub fn set_optimize(&mut self, enabled: bool) {
        self.optimize_enabled = enabled;
    }

    /// Registers a CSV file as a table.
    pub fn register_csv(&mut self, name: &str, path: impl AsRef<Path>) -> Result<()> {
        self.register_csv_with(name, path, &ReadOptions::default())
    }

    /// Registers a CSV file with explicit options.
    pub fn register_csv_with(
        &mut self,
        name: &str,
        path: impl AsRef<Path>,
        opts: &ReadOptions,
    ) -> Result<()> {
        let t = csv::read_file(path, opts)?;
        self.register_batches(name, t.schema, t.batches);
        Ok(())
    }

    /// Registers CSV text as a table.
    pub fn register_csv_str(&mut self, name: &str, text: &str) -> Result<()> {
        let t = csv::read_str(text, &ReadOptions::default())?;
        self.register_batches(name, t.schema, t.batches);
        Ok(())
    }

    /// Registers pre-built batches as a table.
    pub fn register_batches(&mut self, name: &str, schema: SchemaRef, batches: Vec<RecordBatch>) {
        self.tables.insert(name.to_ascii_lowercase(), Table { schema, batches });
    }

    /// The names of registered tables, sorted.
    pub fn table_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tables.keys().cloned().collect();
        v.sort();
        v
    }

    /// The schema of a registered table.
    pub fn schema_of(&self, name: &str) -> Option<SchemaRef> {
        self.tables.get(&name.to_ascii_lowercase()).map(|t| Arc::clone(&t.schema))
    }

    /// Parses, plans, optimizes and executes a SQL statement.
    pub fn sql(&self, sql: &str) -> Result<QueryResult> {
        let stmt = crate::sql::parse(sql)?;
        match stmt {
            Statement::Query(q) => {
                let logical = Planner::new(self).plan_query(&q)?;
                let optimized = if self.optimize_enabled {
                    self.optimizer.optimize(logical.clone())?
                } else {
                    logical.clone()
                };
                let mut physical = create_physical_plan(&optimized, self)?;
                let schema = physical.schema();
                let batches = collect(&mut physical)?;
                Ok(QueryResult { schema, batches, explain: None })
            }
            Statement::Explain(q) => {
                let logical = Planner::new(self).plan_query(&q)?;
                let optimized = self.optimizer.optimize(logical.clone())?;
                let physical = create_physical_plan(&optimized, self)?;

                let text = format!(
                    "== Logical plan ==\n{}\n== Optimized plan ==\n{}\n== Physical plan ==\n{}\n\
                     rules applied: {}",
                    logical.display_indent(),
                    optimized.display_indent(),
                    explain(physical.as_ref()),
                    self.optimizer.rule_names().join(", "),
                );
                Ok(QueryResult {
                    schema: Arc::new(crate::types::Schema::new(vec![crate::types::Field::new(
                        "plan",
                        crate::types::DataType::Utf8,
                    )])),
                    batches: vec![],
                    explain: Some(text),
                })
            }
        }
    }

    /// Produces the optimized logical plan for a query, without running it.
    pub fn plan(&self, sql: &str) -> Result<LogicalPlan> {
        let stmt = crate::sql::parse(sql)?;
        let q = match stmt {
            Statement::Query(q) | Statement::Explain(q) => q,
        };
        let logical = Planner::new(self).plan_query(&q)?;
        if self.optimize_enabled {
            optimize(logical)
        } else {
            Ok(logical)
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl TableProvider for Context {
    fn schema_of(&self, name: &str) -> Option<SchemaRef> {
        Context::schema_of(self, name)
    }
    fn table_names(&self) -> Vec<String> {
        Context::table_names(self)
    }
}

impl BatchSource for Context {
    fn batches(&self, table: &str) -> Option<Vec<RecordBatch>> {
        self.tables.get(&table.to_ascii_lowercase()).map(|t| t.batches.clone())
    }
}

/// The result of a query.
///
/// `Debug` prints the rendered table rather than the internal buffers, which
/// is what a failing `assert_eq!` in a test actually wants to show.
pub struct QueryResult {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    explain: Option<String>,
}

impl std::fmt::Debug for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QueryResult({} rows)\n{}", self.num_rows(), self.to_table())
    }
}

impl QueryResult {
    /// The result schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The result batches.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Total number of result rows.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// Reads a single cell, with `row` counted across all batches.
    pub fn value(&self, row: usize, col: usize) -> Result<Value> {
        let mut remaining = row;
        for b in &self.batches {
            if remaining < b.num_rows() {
                if col >= b.num_columns() {
                    return Err(Error::exec(format!(
                        "column {col} is out of range for a result with {} columns",
                        b.num_columns()
                    )));
                }
                return Ok(b.value(remaining, col));
            }
            remaining -= b.num_rows();
        }
        Err(Error::exec(format!(
            "row {row} is out of range for a result with {} rows",
            self.num_rows()
        )))
    }

    /// Materializes the result as rows of strings, for assertions and display.
    pub fn rows(&self) -> Vec<Vec<String>> {
        let mut out = Vec::new();
        for b in &self.batches {
            for r in 0..b.num_rows() {
                out.push((0..b.num_columns()).map(|c| b.value(r, c).to_string()).collect());
            }
        }
        out
    }

    /// Renders the result as an aligned text table, or the EXPLAIN output.
    pub fn to_table(&self) -> String {
        match &self.explain {
            Some(text) => text.clone(),
            None => format_batches(&self.schema, &self.batches),
        }
    }

    /// The EXPLAIN text, if this was an EXPLAIN statement.
    pub fn explain_text(&self) -> Option<&str> {
        self.explain.as_deref()
    }
}

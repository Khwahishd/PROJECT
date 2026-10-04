//! Record batches: the unit of work passed between operators.

use crate::array::{Array, ArrayRef};
use crate::error::{Error, Result};
use crate::types::{Schema, SchemaRef, Value};
use std::fmt;
use std::sync::Arc;

/// The number of rows an operator aims to produce per batch.
///
/// The value is a compromise, and the reason vectorized engines pick something
/// in this range: large enough that per-batch overhead (virtual dispatch,
/// bounds checks, allocation) is amortized across thousands of rows, small
/// enough that a batch's working set still fits comfortably in L2 cache.
pub const DEFAULT_BATCH_SIZE: usize = 8192;

/// A horizontal slice of a table: some rows, all columns.
#[derive(Debug, Clone)]
pub struct RecordBatch {
    schema: SchemaRef,
    columns: Vec<ArrayRef>,
    num_rows: usize,
}

impl RecordBatch {
    /// Builds a batch, checking that the columns match the schema.
    pub fn try_new(schema: SchemaRef, columns: Vec<ArrayRef>) -> Result<Self> {
        if schema.len() != columns.len() {
            return Err(Error::exec(format!(
                "schema has {} fields but {} columns were supplied",
                schema.len(),
                columns.len()
            )));
        }
        let num_rows = columns.first().map_or(0, |c| c.len());
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(Error::exec(format!(
                    "column {} has {} rows but column 0 has {}",
                    i,
                    col.len(),
                    num_rows
                )));
            }
        }
        Ok(RecordBatch { schema, columns, num_rows })
    }

    /// Builds an empty batch with the given schema.
    pub fn empty(schema: SchemaRef) -> Self {
        let columns =
            schema.fields().iter().map(|f| Arc::new(Array::new_null(f.data_type, 0))).collect();
        RecordBatch { schema, columns, num_rows: 0 }
    }

    /// The batch's schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// The columns, in schema order.
    pub fn columns(&self) -> &[ArrayRef] {
        &self.columns
    }

    /// The column at `i`.
    pub fn column(&self, i: usize) -> &ArrayRef {
        &self.columns[i]
    }

    /// Looks up a column by name.
    pub fn column_by_name(&self, name: &str) -> Result<&ArrayRef> {
        Ok(&self.columns[self.schema.index_of(name)?])
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Whether the batch has no rows.
    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Gathers the rows at `indices`.
    pub fn take(&self, indices: &[usize]) -> Result<RecordBatch> {
        let columns = self.columns.iter().map(|c| Arc::new(c.take(indices))).collect();
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
    }

    /// Returns rows `[offset, offset + length)`.
    pub fn slice(&self, offset: usize, length: usize) -> Result<RecordBatch> {
        let columns = self.columns.iter().map(|c| Arc::new(c.slice(offset, length))).collect();
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
    }

    /// Reads a single cell, for result materialization.
    pub fn value(&self, row: usize, col: usize) -> Value {
        self.columns[col].value(row)
    }

    /// Concatenates batches that share a schema.
    pub fn concat(schema: SchemaRef, batches: &[RecordBatch]) -> Result<RecordBatch> {
        if batches.is_empty() {
            return Ok(RecordBatch::empty(schema));
        }
        let mut columns: Vec<Array> = batches[0].columns.iter().map(|c| (**c).clone()).collect();
        for b in &batches[1..] {
            if b.num_columns() != columns.len() {
                return Err(Error::exec("cannot concatenate batches with different widths"));
            }
            for (i, col) in b.columns.iter().enumerate() {
                columns[i] = columns[i].concat(col)?;
            }
        }
        RecordBatch::try_new(schema, columns.into_iter().map(Arc::new).collect())
    }
}

/// Renders a set of batches as an aligned text table.
///
/// Output formatting lives here rather than in the CLI so that tests can assert
/// on exactly what a user would see.
pub fn format_batches(schema: &Schema, batches: &[RecordBatch]) -> String {
    let headers: Vec<String> = schema.fields().iter().map(|f| f.name.clone()).collect();
    let mut rows: Vec<Vec<String>> = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            rows.push((0..b.num_columns()).map(|c| b.value(r, c).to_string()).collect());
        }
    }

    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.chars().count());
            }
        }
    }

    let sep: String =
        widths.iter().map(|w| format!("+{}", "-".repeat(w + 2))).collect::<String>() + "+";

    let mut out = String::new();
    out.push_str(&sep);
    out.push('\n');
    out.push('|');
    for (h, w) in headers.iter().zip(&widths) {
        out.push_str(&format!(" {:w$} |", h, w = w));
    }
    out.push('\n');
    out.push_str(&sep);
    out.push('\n');
    for row in &rows {
        out.push('|');
        for (cell, w) in row.iter().zip(&widths) {
            out.push_str(&format!(" {:w$} |", cell, w = w));
        }
        out.push('\n');
    }
    out.push_str(&sep);
    out
}

impl fmt::Display for RecordBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_batches(&self.schema, std::slice::from_ref(self)))
    }
}

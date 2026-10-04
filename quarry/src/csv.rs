//! CSV ingestion with schema inference.
//!
//! The reader makes two passes over the data conceptually: it infers a type per
//! column by widening as it reads, then builds columnar arrays. Inference uses
//! a lattice -- Boolean ⊏ Int64 ⊏ Float64 ⊏ Utf8, with Date32 beside Int64 --
//! where a column's type only ever widens, so a column of `1, 2, 3.5` becomes
//! Float64 and `1, 2, hello` becomes Utf8 rather than failing.

use crate::array::Array;
use crate::batch::{RecordBatch, DEFAULT_BATCH_SIZE};
use crate::error::{Error, Result};
use crate::types::{parse_date32, DataType, Field, Schema, SchemaRef, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;

/// Options controlling CSV parsing.
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Whether the first row holds column names.
    pub has_header: bool,
    /// The field separator.
    pub delimiter: char,
    /// How many rows to examine when inferring types. `None` reads all of them.
    pub infer_rows: Option<usize>,
    /// Rows per output batch.
    pub batch_size: usize,
    /// Values treated as NULL, in addition to the empty string.
    pub null_values: Vec<String>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        ReadOptions {
            has_header: true,
            delimiter: ',',
            // Sampling the whole file is correct but slow for large inputs;
            // 1000 rows is enough to classify almost any real column, and a
            // value that does not fit the inferred type later becomes NULL
            // rather than an error.
            infer_rows: Some(1000),
            batch_size: DEFAULT_BATCH_SIZE,
            null_values: vec!["NULL".into(), "null".into(), "NA".into(), "\\N".into()],
        }
    }
}

/// A CSV file parsed into columnar batches.
pub struct CsvTable {
    /// The inferred schema.
    pub schema: SchemaRef,
    /// The data, in batches.
    pub batches: Vec<RecordBatch>,
}

/// Reads a CSV file from disk.
pub fn read_file(path: impl AsRef<Path>, opts: &ReadOptions) -> Result<CsvTable> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::Io(format!("cannot open {}: {e}", path.as_ref().display())))?;
    read(BufReader::new(file), opts)
}

/// Reads CSV from any reader.
pub fn read(mut input: impl Read, opts: &ReadOptions) -> Result<CsvTable> {
    let mut text = String::new();
    input.read_to_string(&mut text)?;
    read_str(&text, opts)
}

/// Reads CSV from a string.
pub fn read_str(text: &str, opts: &ReadOptions) -> Result<CsvTable> {
    let mut rows = Vec::new();
    let mut reader = BufReader::new(text.as_bytes());
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if !trimmed.is_empty() {
            rows.push(split_row(trimmed, opts.delimiter));
        }
        line.clear();
    }

    if rows.is_empty() {
        return Err(Error::Io("CSV input is empty".into()));
    }

    let (header, data) = if opts.has_header {
        let h = rows[0].clone();
        (h, &rows[1..])
    } else {
        let n = rows[0].len();
        let h: Vec<String> = (0..n).map(|i| format!("column_{}", i + 1)).collect();
        (h, &rows[..])
    };

    let ncols = header.len();
    for (i, row) in data.iter().enumerate() {
        if row.len() != ncols {
            return Err(Error::Io(format!(
                "row {} has {} fields but the header has {ncols}",
                i + 1 + opts.has_header as usize,
                row.len()
            )));
        }
    }

    // --- type inference ---
    let sample = match opts.infer_rows {
        Some(n) => &data[..n.min(data.len())],
        None => data,
    };
    let mut types = vec![InferredType::Unknown; ncols];
    for row in sample {
        for (c, cell) in row.iter().enumerate() {
            if is_null(cell, &opts.null_values) {
                continue;
            }
            types[c] = types[c].widen(classify(cell));
        }
    }

    let fields: Vec<Field> =
        header.iter().zip(&types).map(|(name, t)| Field::new(name.trim(), t.data_type())).collect();
    let schema: SchemaRef = Arc::new(Schema::new(fields));

    // --- materialization ---
    let mut batches = Vec::new();
    let batch_size = opts.batch_size.max(1);
    for chunk in data.chunks(batch_size) {
        let mut columns = Vec::with_capacity(ncols);
        for c in 0..ncols {
            let dt = schema.field(c).data_type;
            let values: Vec<Value> =
                chunk.iter().map(|row| parse_cell(&row[c], dt, &opts.null_values)).collect();
            columns.push(Arc::new(Array::from_values(values, dt)?));
        }
        batches.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
    }
    if batches.is_empty() {
        batches.push(RecordBatch::empty(Arc::clone(&schema)));
    }

    Ok(CsvTable { schema, batches })
}

/// Splits one CSV row, honouring double-quoted fields and `""` escapes.
fn split_row(line: &str, delimiter: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            c if c == delimiter && !in_quotes => {
                out.push(std::mem::take(&mut field));
            }
            c => field.push(c),
        }
    }
    out.push(field);
    out
}

fn is_null(cell: &str, null_values: &[String]) -> bool {
    let t = cell.trim();
    t.is_empty() || null_values.iter().any(|n| n == t)
}

/// The inference lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferredType {
    Unknown,
    Boolean,
    Int64,
    Float64,
    Date32,
    Utf8,
}

impl InferredType {
    fn data_type(self) -> DataType {
        match self {
            // A column that was entirely NULL in the sample has no evidence
            // for any type; Utf8 holds anything, so it is the safe default.
            InferredType::Unknown | InferredType::Utf8 => DataType::Utf8,
            InferredType::Boolean => DataType::Boolean,
            InferredType::Int64 => DataType::Int64,
            InferredType::Float64 => DataType::Float64,
            InferredType::Date32 => DataType::Date32,
        }
    }

    /// Returns the narrowest type that accommodates both `self` and `other`.
    fn widen(self, other: InferredType) -> InferredType {
        use InferredType::*;
        match (self, other) {
            (Unknown, t) | (t, Unknown) => t,
            (a, b) if a == b => a,
            // Integers and floats unify as floats.
            (Int64, Float64) | (Float64, Int64) => Float64,
            // Anything else mixed is text.
            _ => Utf8,
        }
    }
}

fn classify(cell: &str) -> InferredType {
    let t = cell.trim();
    if t.parse::<i64>().is_ok() {
        return InferredType::Int64;
    }
    if t.parse::<f64>().is_ok() {
        return InferredType::Float64;
    }
    if matches!(t.to_ascii_lowercase().as_str(), "true" | "false") {
        return InferredType::Boolean;
    }
    if parse_date32(t).is_some() {
        return InferredType::Date32;
    }
    InferredType::Utf8
}

/// Parses one cell into the column's inferred type.
///
/// A value that does not fit becomes NULL rather than failing the load: in a
/// file of a million rows, one malformed cell should not cost the other
/// 999,999.
fn parse_cell(cell: &str, dt: DataType, null_values: &[String]) -> Value {
    if is_null(cell, null_values) {
        return Value::Null;
    }
    let t = cell.trim();
    match dt {
        DataType::Int64 => t.parse::<i64>().map(Value::Int64).unwrap_or(Value::Null),
        DataType::Float64 => t.parse::<f64>().map(Value::Float64).unwrap_or(Value::Null),
        DataType::Boolean => match t.to_ascii_lowercase().as_str() {
            "true" | "t" | "1" | "yes" => Value::Boolean(true),
            "false" | "f" | "0" | "no" => Value::Boolean(false),
            _ => Value::Null,
        },
        DataType::Date32 => parse_date32(t).map(Value::Date32).unwrap_or(Value::Null),
        DataType::Utf8 => Value::Utf8(cell.trim().to_string()),
        DataType::Null => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> CsvTable {
        read_str(text, &ReadOptions::default()).unwrap()
    }

    #[test]
    fn infers_column_types() {
        let t = load("i,f,s,b,d\n1,1.5,hello,true,2024-01-05\n2,2.5,world,false,2024-02-06\n");
        let types: Vec<DataType> = t.schema.fields().iter().map(|f| f.data_type).collect();
        assert_eq!(
            types,
            vec![
                DataType::Int64,
                DataType::Float64,
                DataType::Utf8,
                DataType::Boolean,
                DataType::Date32
            ]
        );
    }

    #[test]
    fn mixed_integers_and_floats_widen_to_float() {
        let t = load("x\n1\n2\n3.5\n");
        assert_eq!(t.schema.field(0).data_type, DataType::Float64);
        assert_eq!(t.batches[0].value(0, 0), Value::Float64(1.0));
    }

    #[test]
    fn anything_unclassifiable_falls_back_to_text() {
        let t = load("x\n1\n2\nbanana\n");
        assert_eq!(t.schema.field(0).data_type, DataType::Utf8);
    }

    #[test]
    fn an_all_empty_column_becomes_text() {
        // With no evidence for any type, the widest type is the safe choice.
        let t = load("a,b\n1,\n2,\n");
        assert_eq!(t.schema.field(1).data_type, DataType::Utf8);
        assert_eq!(t.batches[0].value(0, 1), Value::Null);
    }

    #[test]
    fn empty_fields_and_null_markers_become_null() {
        let t = load("x,y\n1,a\n,b\nNULL,c\nNA,d\n");
        let b = &t.batches[0];
        assert_eq!(b.value(0, 0), Value::Int64(1));
        for row in 1..4 {
            assert_eq!(b.value(row, 0), Value::Null, "row {row}");
        }
    }

    #[test]
    fn quoted_fields_may_contain_delimiters_and_quotes() {
        let t = load("a,b\n\"x,y\",1\n\"he said \"\"hi\"\"\",2\n");
        let b = &t.batches[0];
        assert_eq!(b.value(0, 0), Value::Utf8("x,y".into()));
        assert_eq!(b.value(1, 0), Value::Utf8("he said \"hi\"".into()));
        assert_eq!(b.value(0, 1), Value::Int64(1));
    }

    #[test]
    fn a_ragged_row_is_an_error_not_silent_corruption() {
        let err = read_str("a,b\n1,2\n3\n", &ReadOptions::default());
        assert!(err.is_err(), "a row with the wrong field count must be reported");
    }

    #[test]
    fn a_value_that_does_not_fit_the_inferred_type_becomes_null() {
        // Inference samples the first rows; a later outlier must not fail the
        // whole load.
        let opts = ReadOptions { infer_rows: Some(2), ..Default::default() };
        let t = read_str("x\n1\n2\nbanana\n", &opts).unwrap();
        assert_eq!(t.schema.field(0).data_type, DataType::Int64);
        assert_eq!(t.batches[0].value(2, 0), Value::Null);
    }

    #[test]
    fn headerless_files_get_generated_names() {
        let opts = ReadOptions { has_header: false, ..Default::default() };
        let t = read_str("1,2\n3,4\n", &opts).unwrap();
        assert_eq!(t.schema.field(0).name, "column_1");
        assert_eq!(t.batches[0].num_rows(), 2);
    }

    #[test]
    fn rows_are_split_across_batches() {
        let opts = ReadOptions { batch_size: 3, ..Default::default() };
        let rows: String = (0..10).map(|i| format!("{i}\n")).collect();
        let t = read_str(&format!("x\n{rows}"), &opts).unwrap();
        assert_eq!(t.batches.len(), 4, "10 rows at 3 per batch");
        assert_eq!(t.batches.iter().map(|b| b.num_rows()).sum::<usize>(), 10);
    }

    #[test]
    fn an_empty_input_is_an_error() {
        assert!(read_str("", &ReadOptions::default()).is_err());
    }

    #[test]
    fn date_round_trips_through_the_civil_calendar() {
        // Epoch, a leap day, and a century boundary that is not a leap year.
        for (days, iso) in [
            (0, "1970-01-01"),
            (19723, "2024-01-01"),
            (19782, "2024-02-29"),
            (11016, "2000-02-29"),
            (-1, "1969-12-31"),
        ] {
            assert_eq!(crate::types::format_date32(days), iso);
            assert_eq!(crate::types::parse_date32(iso), Some(days));
        }
        assert_eq!(crate::types::parse_date32("not-a-date"), None);
        assert_eq!(crate::types::parse_date32("2024-13-01"), None);
    }
}

//! The type system: logical data types, scalar values, and schemas.

use crate::error::{Error, Result};
use std::fmt;
use std::sync::Arc;

/// A logical column type.
///
/// The set is deliberately small. Every type here has a dense, fixed-width or
/// easily-vectorized representation, which is what lets the execution engine
/// work on whole batches rather than individual values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 64-bit signed integer.
    Int64,
    /// 64-bit IEEE-754 float.
    Float64,
    /// UTF-8 string.
    Utf8,
    /// Boolean.
    Boolean,
    /// Days since the Unix epoch, stored as an i32.
    Date32,
    /// The type of an untyped NULL literal, resolved during planning.
    Null,
}

impl DataType {
    /// Whether this type participates in arithmetic.
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::Int64 | DataType::Float64)
    }

    /// The type that results from combining `self` and `other` in a binary
    /// operation, following SQL's usual widening rules.
    ///
    /// `Null` is absorbing in the sense that it takes the other side's type:
    /// a literal NULL has no type of its own until it meets one.
    pub fn unify(&self, other: &DataType) -> Result<DataType> {
        use DataType::*;
        Ok(match (self, other) {
            (Null, t) | (t, Null) => *t,
            (a, b) if a == b => *a,
            (Int64, Float64) | (Float64, Int64) => Float64,
            (a, b) => {
                return Err(Error::typ(format!("cannot combine incompatible types {a} and {b}")))
            }
        })
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Int64 => "INT64",
            DataType::Float64 => "FLOAT64",
            DataType::Utf8 => "UTF8",
            DataType::Boolean => "BOOLEAN",
            DataType::Date32 => "DATE32",
            DataType::Null => "NULL",
        };
        f.write_str(s)
    }
}

/// A single scalar value, used for literals, aggregate state and result rows.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// SQL NULL.
    Null,
    /// Integer value.
    Int64(i64),
    /// Floating point value.
    Float64(f64),
    /// String value.
    Utf8(String),
    /// Boolean value.
    Boolean(bool),
    /// Date, as days since the Unix epoch.
    Date32(i32),
}

impl Value {
    /// The logical type of this value.
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null => DataType::Null,
            Value::Int64(_) => DataType::Int64,
            Value::Float64(_) => DataType::Float64,
            Value::Utf8(_) => DataType::Utf8,
            Value::Boolean(_) => DataType::Boolean,
            Value::Date32(_) => DataType::Date32,
        }
    }

    /// Whether this value is NULL.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Interpret the value as a float, for numeric contexts.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int64(v) => Some(*v as f64),
            Value::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Interpret the value as a boolean, treating NULL as "not true".
    ///
    /// SQL's three-valued logic means NULL is neither true nor false; for
    /// filtering purposes a row whose predicate is NULL is excluded, which is
    /// what this models.
    pub fn as_bool(&self) -> bool {
        matches!(self, Value::Boolean(true))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => f.write_str("NULL"),
            Value::Int64(v) => write!(f, "{v}"),
            Value::Float64(v) => {
                // Render floats so that whole numbers still read as floats,
                // which makes result comparison in tests unambiguous.
                if v.fract() == 0.0 && v.abs() < 1e15 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            Value::Utf8(v) => f.write_str(v),
            Value::Boolean(v) => write!(f, "{v}"),
            Value::Date32(v) => write!(f, "{}", format_date32(*v)),
        }
    }
}

/// Formats days-since-epoch as an ISO date.
///
/// Implemented directly rather than pulling in a date library: the civil-from-
/// days algorithm is short, exact for the whole i32 range, and keeps the
/// dependency count at zero.
pub fn format_date32(days: i32) -> String {
    // Howard Hinnant's civil_from_days.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Parses an ISO `YYYY-MM-DD` date into days since the Unix epoch.
pub fn parse_date32(s: &str) -> Option<i32> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y: i64 = s[0..4].parse().ok()?;
    let m: i64 = s[5..7].parse().ok()?;
    let d: i64 = s[8..10].parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // days_from_civil, the inverse of the above.
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64;
    let doy = (153 * mp + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe as i64 - 719_468) as i32)
}

/// A named, typed column in a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Column name, always unqualified.
    pub name: String,
    /// Column type.
    pub data_type: DataType,
    /// Whether the column may contain NULLs.
    pub nullable: bool,
    /// The table or alias this column came from, if any.
    ///
    /// The qualifier is metadata rather than part of `name` deliberately. If
    /// `SELECT t.city` produced a column literally called `t.city`, the output
    /// header would leak the query's internal aliasing, and `SELECT city`
    /// against the same table would produce a differently-named column for the
    /// identical value.
    pub qualifier: Option<String>,
}

impl Field {
    /// Creates a nullable, unqualified field.
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Field { name: name.into(), data_type, nullable: true, qualifier: None }
    }

    /// Creates a non-nullable, unqualified field.
    pub fn not_null(name: impl Into<String>, data_type: DataType) -> Self {
        Field { name: name.into(), data_type, nullable: false, qualifier: None }
    }

    /// Returns a copy of this field attributed to `qualifier`.
    pub fn with_qualifier(&self, qualifier: impl Into<String>) -> Self {
        Field { qualifier: Some(qualifier.into()), ..self.clone() }
    }

    /// The fully qualified name, for display and error messages.
    pub fn qualified_name(&self) -> String {
        match &self.qualifier {
            Some(q) => format!("{q}.{}", self.name),
            None => self.name.clone(),
        }
    }
}

/// An ordered list of fields describing a table or an intermediate result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
}

/// A reference-counted schema, cheap to clone between plan nodes and batches.
pub type SchemaRef = Arc<Schema>;

impl Schema {
    /// Creates a schema from fields.
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    /// Creates an empty schema.
    pub fn empty() -> Self {
        Schema { fields: Vec::new() }
    }

    /// The fields, in order.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the schema has no columns.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The field at `i`.
    pub fn field(&self, i: usize) -> &Field {
        &self.fields[i]
    }

    /// Finds a column by name, case-insensitively.
    ///
    /// SQL identifiers are conventionally case-insensitive when unquoted, and
    /// resolving them that way here means `SELECT Name` finds a column stored
    /// as `name` -- which is what a user typing a query expects.
    ///
    /// A dotted name (`t.city`) matches only fields carrying that qualifier; a
    /// bare name matches on the column name alone and is an error if more than
    /// one field answers to it, so `SELECT id FROM a JOIN b` is rejected rather
    /// than silently resolving to whichever side came first.
    pub fn index_of(&self, name: &str) -> Result<usize> {
        let (qualifier, column) = match name.split_once('.') {
            Some((q, c)) => (Some(q), c),
            None => (None, name),
        };
        self.index_of_qualified(qualifier, column)
    }

    /// Finds a column by an optional qualifier and a column name.
    pub fn index_of_qualified(&self, qualifier: Option<&str>, column: &str) -> Result<usize> {
        let mut found = None;
        for (i, f) in self.fields.iter().enumerate() {
            if !f.name.eq_ignore_ascii_case(column) {
                continue;
            }
            if let Some(q) = qualifier {
                match &f.qualifier {
                    Some(fq) if fq.eq_ignore_ascii_case(q) => {}
                    _ => continue,
                }
            }
            if found.is_some() {
                return Err(Error::plan(format!(
                    "column reference {:?} is ambiguous; qualify it with a table name",
                    display_ref(qualifier, column)
                )));
            }
            found = Some(i);
        }
        found.ok_or_else(|| {
            let names: Vec<String> = self.fields.iter().map(|f| f.qualified_name()).collect();
            Error::plan(format!(
                "no column named {:?}; available columns are [{}]",
                display_ref(qualifier, column),
                names.join(", ")
            ))
        })
    }

    /// Returns a copy of this schema with every field attributed to `alias`.
    pub fn qualified(&self, alias: &str) -> Schema {
        Schema::new(self.fields.iter().map(|f| f.with_qualifier(alias)).collect())
    }

    /// Concatenates two schemas, as produced by a join.
    pub fn join(&self, other: &Schema) -> Schema {
        let mut fields = self.fields.clone();
        fields.extend(other.fields.iter().cloned());
        Schema { fields }
    }

    /// Returns a schema containing only the columns at the given indices.
    pub fn project(&self, indices: &[usize]) -> Result<Schema> {
        let mut fields = Vec::with_capacity(indices.len());
        for &i in indices {
            if i >= self.fields.len() {
                return Err(Error::plan(format!(
                    "projection index {i} is out of range for a schema with {} columns",
                    self.fields.len()
                )));
            }
            fields.push(self.fields[i].clone());
        }
        Ok(Schema { fields })
    }
}

fn display_ref(qualifier: Option<&str>, column: &str) -> String {
    match qualifier {
        Some(q) => format!("{q}.{column}"),
        None => column.to_string(),
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .fields
            .iter()
            .map(|fd| format!("{}: {}", fd.qualified_name(), fd.data_type))
            .collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

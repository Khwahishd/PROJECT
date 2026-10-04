//! Resolved expressions.
//!
//! A [`LogicalExpr`] is what an AST [`Expr`](crate::sql::ast::Expr) becomes
//! once names have been resolved to column positions and types have been
//! checked. Resolving column references to indices here, rather than carrying
//! names into execution, means the hot path never does a string lookup.

use crate::error::{Error, Result};
use crate::types::{DataType, Schema, Value};
use std::fmt;

/// An expression over a known schema.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalExpr {
    /// A constant.
    Literal(Value),
    /// A column, resolved to its position in the input schema.
    Column {
        /// Position in the input schema.
        index: usize,
        /// The original name, retained for display and for schema construction.
        name: String,
        /// The column's type.
        data_type: DataType,
    },
    /// A binary operation.
    Binary {
        /// Left operand.
        left: Box<LogicalExpr>,
        /// The operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<LogicalExpr>,
    },
    /// A unary operation.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// The operand.
        expr: Box<LogicalExpr>,
    },
    /// `IS NULL` / `IS NOT NULL`.
    IsNull {
        /// The operand.
        expr: Box<LogicalExpr>,
        /// True for `IS NOT NULL`.
        negated: bool,
    },
    /// `LIKE` with SQL wildcards.
    Like {
        /// The operand.
        expr: Box<LogicalExpr>,
        /// The pattern.
        pattern: Box<LogicalExpr>,
        /// True for `NOT LIKE`.
        negated: bool,
    },
    /// `CASE WHEN`.
    Case {
        /// WHEN/THEN pairs.
        when_then: Vec<(LogicalExpr, LogicalExpr)>,
        /// The ELSE branch.
        else_expr: Option<Box<LogicalExpr>>,
    },
    /// A type conversion.
    Cast {
        /// The operand.
        expr: Box<LogicalExpr>,
        /// The target type.
        data_type: DataType,
    },
    /// A reference to an aggregate computed by a lower Aggregate node.
    ///
    /// The planner rewrites `SUM(x) + 1` into `AggregateRef(0) + 1`, so the
    /// projection above an aggregation is an ordinary scalar expression.
    AggregateRef {
        /// Position among the aggregate node's output columns.
        index: usize,
        /// Display name, e.g. `SUM(x)`.
        name: String,
        /// The aggregate's output type.
        data_type: DataType,
    },
    /// An alias, which renames without changing the value.
    Alias {
        /// The underlying expression.
        expr: Box<LogicalExpr>,
        /// The new name.
        name: String,
    },
}

pub use crate::sql::ast::{BinaryOp, UnaryOp};

impl LogicalExpr {
    /// The type this expression produces.
    pub fn data_type(&self) -> Result<DataType> {
        Ok(match self {
            LogicalExpr::Literal(v) => v.data_type(),
            LogicalExpr::Column { data_type, .. } => *data_type,
            LogicalExpr::AggregateRef { data_type, .. } => *data_type,
            LogicalExpr::Alias { expr, .. } => expr.data_type()?,
            LogicalExpr::IsNull { .. } => DataType::Boolean,
            LogicalExpr::Like { .. } => DataType::Boolean,
            LogicalExpr::Cast { data_type, .. } => *data_type,
            LogicalExpr::Unary { op, expr } => match op {
                UnaryOp::Not => DataType::Boolean,
                UnaryOp::Negate => expr.data_type()?,
            },
            LogicalExpr::Binary { left, op, right } => {
                if op.is_comparison() || op.is_logical() {
                    DataType::Boolean
                } else {
                    let lt = left.data_type()?;
                    let rt = right.data_type()?;
                    // Division always produces a float: integer division that
                    // silently truncates is a classic source of wrong answers
                    // in analytics queries.
                    if *op == BinaryOp::Divide {
                        DataType::Float64
                    } else {
                        lt.unify(&rt)?
                    }
                }
            }
            LogicalExpr::Case { when_then, else_expr } => {
                let mut t = when_then
                    .first()
                    .ok_or_else(|| Error::plan("CASE with no branches"))?
                    .1
                    .data_type()?;
                for (_, then) in when_then.iter().skip(1) {
                    t = t.unify(&then.data_type()?)?;
                }
                if let Some(e) = else_expr {
                    t = t.unify(&e.data_type()?)?;
                }
                t
            }
        })
    }

    /// The output column name this expression would produce.
    pub fn display_name(&self) -> String {
        match self {
            LogicalExpr::Alias { name, .. } => name.clone(),
            LogicalExpr::Column { name, .. } => name.clone(),
            LogicalExpr::AggregateRef { name, .. } => name.clone(),
            other => other.to_string(),
        }
    }

    /// Collects the input column indices this expression reads.
    ///
    /// Used by projection pushdown to work out the minimum set of columns a
    /// scan must produce.
    pub fn referenced_columns(&self, out: &mut Vec<usize>) {
        match self {
            LogicalExpr::Column { index, .. } => {
                if !out.contains(index) {
                    out.push(*index);
                }
            }
            LogicalExpr::Binary { left, right, .. } => {
                left.referenced_columns(out);
                right.referenced_columns(out);
            }
            LogicalExpr::Unary { expr, .. }
            | LogicalExpr::IsNull { expr, .. }
            | LogicalExpr::Cast { expr, .. }
            | LogicalExpr::Alias { expr, .. } => expr.referenced_columns(out),
            LogicalExpr::Like { expr, pattern, .. } => {
                expr.referenced_columns(out);
                pattern.referenced_columns(out);
            }
            LogicalExpr::Case { when_then, else_expr } => {
                for (w, t) in when_then {
                    w.referenced_columns(out);
                    t.referenced_columns(out);
                }
                if let Some(e) = else_expr {
                    e.referenced_columns(out);
                }
            }
            LogicalExpr::Literal(_) | LogicalExpr::AggregateRef { .. } => {}
        }
    }

    /// Rewrites column indices through a mapping, used after a projection is
    /// pushed down and the input schema narrows.
    pub fn remap_columns(&self, map: &dyn Fn(usize) -> Option<usize>) -> Result<LogicalExpr> {
        Ok(match self {
            LogicalExpr::Column { index, name, data_type } => {
                let new = map(*index).ok_or_else(|| {
                    Error::plan(format!("column {name:?} was pruned but is still referenced"))
                })?;
                LogicalExpr::Column { index: new, name: name.clone(), data_type: *data_type }
            }
            LogicalExpr::Binary { left, op, right } => LogicalExpr::Binary {
                left: Box::new(left.remap_columns(map)?),
                op: *op,
                right: Box::new(right.remap_columns(map)?),
            },
            LogicalExpr::Unary { op, expr } => {
                LogicalExpr::Unary { op: *op, expr: Box::new(expr.remap_columns(map)?) }
            }
            LogicalExpr::IsNull { expr, negated } => {
                LogicalExpr::IsNull { expr: Box::new(expr.remap_columns(map)?), negated: *negated }
            }
            LogicalExpr::Like { expr, pattern, negated } => LogicalExpr::Like {
                expr: Box::new(expr.remap_columns(map)?),
                pattern: Box::new(pattern.remap_columns(map)?),
                negated: *negated,
            },
            LogicalExpr::Cast { expr, data_type } => LogicalExpr::Cast {
                expr: Box::new(expr.remap_columns(map)?),
                data_type: *data_type,
            },
            LogicalExpr::Alias { expr, name } => {
                LogicalExpr::Alias { expr: Box::new(expr.remap_columns(map)?), name: name.clone() }
            }
            LogicalExpr::Case { when_then, else_expr } => {
                let mut wt = Vec::with_capacity(when_then.len());
                for (w, t) in when_then {
                    wt.push((w.remap_columns(map)?, t.remap_columns(map)?));
                }
                LogicalExpr::Case {
                    when_then: wt,
                    else_expr: match else_expr {
                        Some(e) => Some(Box::new(e.remap_columns(map)?)),
                        None => None,
                    },
                }
            }
            other => other.clone(),
        })
    }

    /// Splits a conjunction into its individual conjuncts.
    ///
    /// Predicate pushdown operates on one conjunct at a time: `a.x = 1 AND
    /// b.y = 2` can push each side to a different join input, which is
    /// impossible if the predicate is treated as a single opaque expression.
    pub fn split_conjunction(self, out: &mut Vec<LogicalExpr>) {
        match self {
            LogicalExpr::Binary { left, op: BinaryOp::And, right } => {
                left.split_conjunction(out);
                right.split_conjunction(out);
            }
            other => out.push(other),
        }
    }

    /// Recombines conjuncts into a single predicate.
    pub fn conjunction(exprs: Vec<LogicalExpr>) -> Option<LogicalExpr> {
        exprs.into_iter().reduce(|a, b| LogicalExpr::Binary {
            left: Box::new(a),
            op: BinaryOp::And,
            right: Box::new(b),
        })
    }
}

impl fmt::Display for LogicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalExpr::Literal(Value::Utf8(s)) => write!(f, "'{s}'"),
            LogicalExpr::Literal(v) => write!(f, "{v}"),
            LogicalExpr::Column { name, .. } => f.write_str(name),
            LogicalExpr::AggregateRef { name, .. } => f.write_str(name),
            LogicalExpr::Binary { left, op, right } => write!(f, "({left} {op} {right})"),
            LogicalExpr::Unary { op, expr } => write!(f, "{op}{expr}"),
            LogicalExpr::IsNull { expr, negated } => {
                write!(f, "{expr} IS {}NULL", if *negated { "NOT " } else { "" })
            }
            LogicalExpr::Like { expr, pattern, negated } => {
                write!(f, "{expr} {}LIKE {pattern}", if *negated { "NOT " } else { "" })
            }
            LogicalExpr::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type})"),
            LogicalExpr::Alias { expr, name } => write!(f, "{expr} AS {name}"),
            LogicalExpr::Case { when_then, else_expr } => {
                write!(f, "CASE")?;
                for (w, t) in when_then {
                    write!(f, " WHEN {w} THEN {t}")?;
                }
                if let Some(e) = else_expr {
                    write!(f, " ELSE {e}")?;
                }
                write!(f, " END")
            }
        }
    }
}

/// An aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Row or value count.
    Count,
    /// Sum.
    Sum,
    /// Arithmetic mean.
    Avg,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
}

impl AggregateFunction {
    /// Parses a function name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_uppercase().as_str() {
            "COUNT" => AggregateFunction::Count,
            "SUM" => AggregateFunction::Sum,
            "AVG" | "MEAN" => AggregateFunction::Avg,
            "MIN" => AggregateFunction::Min,
            "MAX" => AggregateFunction::Max,
            _ => return None,
        })
    }

    /// The output type given the input type.
    pub fn return_type(&self, input: DataType) -> Result<DataType> {
        Ok(match self {
            // COUNT is always an integer, whatever it counts.
            AggregateFunction::Count => DataType::Int64,
            // AVG is always a float: averaging integers and truncating would
            // silently give wrong answers.
            AggregateFunction::Avg => DataType::Float64,
            // SUM widens to avoid overflow surprises on large integer columns.
            AggregateFunction::Sum => match input {
                DataType::Int64 => DataType::Int64,
                DataType::Float64 => DataType::Float64,
                DataType::Null => DataType::Null,
                other => return Err(Error::typ(format!("SUM does not accept {other}"))),
            },
            // MIN/MAX preserve their input type, and work on any ordered type.
            AggregateFunction::Min | AggregateFunction::Max => input,
        })
    }
}

impl fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AggregateFunction::Count => "COUNT",
            AggregateFunction::Sum => "SUM",
            AggregateFunction::Avg => "AVG",
            AggregateFunction::Min => "MIN",
            AggregateFunction::Max => "MAX",
        })
    }
}

/// An aggregate to compute: a function applied to an argument expression.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    /// Which aggregate.
    pub func: AggregateFunction,
    /// The argument; `None` for `COUNT(*)`.
    pub arg: Option<LogicalExpr>,
    /// Whether to deduplicate inputs first.
    pub distinct: bool,
    /// Display name, e.g. `SUM(price)`.
    pub name: String,
}

impl AggregateExpr {
    /// The output type of this aggregate.
    pub fn data_type(&self) -> Result<DataType> {
        let input = match &self.arg {
            Some(e) => e.data_type()?,
            None => DataType::Int64,
        };
        self.func.return_type(input)
    }
}

impl fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// Resolves a column name against a schema, producing a `Column` expression.
pub fn resolve_column(schema: &Schema, name: &str) -> Result<LogicalExpr> {
    let index = schema.index_of(name)?;
    let field = schema.field(index);
    Ok(LogicalExpr::Column { index, name: field.name.clone(), data_type: field.data_type })
}

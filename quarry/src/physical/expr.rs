//! Vectorized expression evaluation.
//!
//! Every expression evaluates over a whole [`RecordBatch`] and returns a whole
//! [`Array`]. That is the single decision that separates this engine from a
//! row-at-a-time interpreter: the type dispatch happens *once per batch*
//! instead of once per row, and the inner loops run over primitive slices that
//! LLVM can unroll and vectorize.
//!
//! The cost model is worth stating plainly. A row-at-a-time engine evaluating
//! `price * quantity` over a million rows performs a million enum matches, a
//! million bounds checks and a million virtual calls. The same expression here
//! performs three matches and one tight loop over `&[f64]`.

use crate::array::{Array, ArrayData, Bitmap};
use crate::batch::RecordBatch;
use crate::error::{Error, Result};
use crate::logical::expr::{BinaryOp, LogicalExpr, UnaryOp};
use crate::types::{DataType, Value};

/// Evaluates `expr` over `batch`, producing one column.
pub fn evaluate(expr: &LogicalExpr, batch: &RecordBatch) -> Result<Array> {
    let n = batch.num_rows();
    match expr {
        LogicalExpr::Literal(v) => Ok(literal_array(v, n)),

        LogicalExpr::Column { index, name, .. } => {
            batch.columns().get(*index).map(|c| (**c).clone()).ok_or_else(|| {
                Error::exec(format!(
                    "column {name:?} resolved to index {index} but the batch has {} columns",
                    batch.num_columns()
                ))
            })
        }

        LogicalExpr::Alias { expr, .. } => evaluate(expr, batch),

        LogicalExpr::Binary { left, op, right } => {
            // Comparing a string column against a literal is the single most
            // common predicate shape there is; the same broadcast-allocation
            // problem described under Like applies, so take a scalar path.
            if op.is_comparison() {
                if let LogicalExpr::Literal(Value::Utf8(lit)) = right.as_ref() {
                    let l = evaluate(left, batch)?;
                    if let Some(out) = compare_str_scalar(&l, *op, lit, false) {
                        return Ok(out);
                    }
                }
                if let LogicalExpr::Literal(Value::Utf8(lit)) = left.as_ref() {
                    let r = evaluate(right, batch)?;
                    if let Some(out) = compare_str_scalar(&r, *op, lit, true) {
                        return Ok(out);
                    }
                }
            }
            let l = evaluate(left, batch)?;
            let r = evaluate(right, batch)?;
            binary(&l, *op, &r)
        }

        LogicalExpr::Unary { op, expr } => {
            let v = evaluate(expr, batch)?;
            unary(*op, &v)
        }

        LogicalExpr::IsNull { expr, negated } => {
            let v = evaluate(expr, batch)?;
            let flags: Vec<bool> = (0..v.len()).map(|i| v.is_valid(i) == *negated).collect();
            // IS NULL is never itself null -- it is exactly the predicate that
            // gives three-valued logic an escape hatch.
            Ok(Array::new(ArrayData::Boolean(flags)))
        }

        LogicalExpr::Like { expr, pattern, negated } => {
            let v = evaluate(expr, batch)?;
            // A literal pattern is the overwhelmingly common case, and
            // broadcasting it through literal_array would heap-allocate one
            // String *per row* just to compare against the same text every
            // time. On a 100k-row scan that allocation dominates the match.
            if let LogicalExpr::Literal(Value::Utf8(p)) = pattern.as_ref() {
                return Ok(like_const(&v, p, *negated));
            }
            let p = evaluate(pattern, batch)?;
            like(&v, &p, *negated)
        }

        LogicalExpr::Cast { expr, data_type } => {
            let v = evaluate(expr, batch)?;
            cast(&v, *data_type)
        }

        LogicalExpr::Case { when_then, else_expr } => {
            let out_type = expr.data_type()?;
            let mut result = match else_expr {
                Some(e) => evaluate(e, batch)?,
                None => Array::new_null(out_type, n),
            };
            // Evaluate branches in reverse so earlier branches overwrite later
            // ones, giving CASE its first-match-wins semantics in a single
            // pass with no per-row branching.
            for (when, then) in when_then.iter().rev() {
                let cond = evaluate(when, batch)?;
                let value = evaluate(then, batch)?;
                result = select(&cond, &value, &result)?;
            }
            Ok(result)
        }

        LogicalExpr::AggregateRef { index, name, .. } => {
            batch.columns().get(*index).map(|c| (**c).clone()).ok_or_else(|| {
                Error::exec(format!("aggregate {name:?} resolved to a missing column {index}"))
            })
        }
    }
}

/// Builds a constant column of `len` copies of `v`.
fn literal_array(v: &Value, len: usize) -> Array {
    match v {
        Value::Null => Array::new_null(DataType::Null, len),
        Value::Int64(x) => Array::new(ArrayData::Int64(vec![*x; len])),
        Value::Float64(x) => Array::new(ArrayData::Float64(vec![*x; len])),
        Value::Utf8(x) => Array::new(ArrayData::Utf8(vec![x.clone(); len])),
        Value::Boolean(x) => Array::new(ArrayData::Boolean(vec![*x; len])),
        Value::Date32(x) => Array::new(ArrayData::Date32(vec![*x; len])),
    }
}

/// Combines two validity bitmaps: a result is valid only where both inputs are.
///
/// This is SQL's NULL propagation rule, applied once per batch over packed
/// bytes rather than once per row.
fn combine_validity(a: &Array, b: &Array) -> Option<Bitmap> {
    match (a.validity(), b.validity()) {
        (None, None) => None,
        (Some(x), None) => Some(x.clone()),
        (None, Some(y)) => Some(y.clone()),
        (Some(x), Some(y)) => Some(x.and(y)),
    }
}

/// Evaluates a binary operation over two columns.
fn binary(l: &Array, op: BinaryOp, r: &Array) -> Result<Array> {
    if l.len() != r.len() {
        return Err(Error::exec(format!(
            "binary operands have different lengths: {} and {}",
            l.len(),
            r.len()
        )));
    }
    let n = l.len();

    // A NULL-typed operand makes the whole result NULL.
    if l.data_type() == DataType::Null || r.data_type() == DataType::Null {
        let out =
            if op.is_comparison() || op.is_logical() { DataType::Boolean } else { DataType::Null };
        return Ok(Array::new_null(out, n));
    }

    if op.is_logical() {
        return logical(l, op, r);
    }
    if op.is_comparison() {
        return compare(l, op, r);
    }
    arithmetic(l, op, r)
}

/// AND / OR with SQL's three-valued logic.
fn logical(l: &Array, op: BinaryOp, r: &Array) -> Result<Array> {
    let (lb, rb) = match (l.as_bool(), r.as_bool()) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err(Error::typ(format!(
                "{op} requires boolean operands, found {} and {}",
                l.data_type(),
                r.data_type()
            )))
        }
    };
    let n = l.len();
    let mut values = Vec::with_capacity(n);
    let mut validity = Bitmap::all_valid(n);
    let mut any_null = false;

    for i in 0..n {
        let lv = l.is_valid(i).then(|| lb[i]);
        let rv = r.is_valid(i).then(|| rb[i]);
        // FALSE AND NULL is FALSE, and TRUE OR NULL is TRUE: a known result
        // short-circuits even when the other operand is unknown. Treating NULL
        // as simply "propagate" would get both of these wrong.
        let out = match op {
            BinaryOp::And => match (lv, rv) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            BinaryOp::Or => match (lv, rv) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            _ => unreachable!("logical is only called for And and Or"),
        };
        match out {
            Some(b) => values.push(b),
            None => {
                values.push(false);
                validity.set_null(i);
                any_null = true;
            }
        }
    }
    Ok(Array::with_validity(ArrayData::Boolean(values), any_null.then_some(validity)))
}

/// Comparison operators, specialized per type.
fn compare(l: &Array, op: BinaryOp, r: &Array) -> Result<Array> {
    use BinaryOp::*;
    let n = l.len();
    let validity = combine_validity(l, r);

    macro_rules! cmp_slices {
        ($a:expr, $b:expr) => {{
            let a = $a;
            let b = $b;
            // One match on the operator, then a branch-free loop per row.
            let out: Vec<bool> = match op {
                Eq => (0..n).map(|i| a[i] == b[i]).collect(),
                NotEq => (0..n).map(|i| a[i] != b[i]).collect(),
                Lt => (0..n).map(|i| a[i] < b[i]).collect(),
                LtEq => (0..n).map(|i| a[i] <= b[i]).collect(),
                Gt => (0..n).map(|i| a[i] > b[i]).collect(),
                GtEq => (0..n).map(|i| a[i] >= b[i]).collect(),
                _ => unreachable!("compare is only called for comparison operators"),
            };
            out
        }};
    }

    let values = match (l.data(), r.data()) {
        (ArrayData::Int64(a), ArrayData::Int64(b)) => cmp_slices!(a, b),
        (ArrayData::Float64(a), ArrayData::Float64(b)) => cmp_slices!(a, b),
        (ArrayData::Utf8(a), ArrayData::Utf8(b)) => cmp_slices!(a, b),
        (ArrayData::Boolean(a), ArrayData::Boolean(b)) => cmp_slices!(a, b),
        (ArrayData::Date32(a), ArrayData::Date32(b)) => cmp_slices!(a, b),
        // Mixed int/float: widen to float. Done as a separate arm so the
        // same-type cases above stay allocation-free.
        (ArrayData::Int64(a), ArrayData::Float64(b)) => {
            let a: Vec<f64> = a.iter().map(|&v| v as f64).collect();
            cmp_slices!(&a, b)
        }
        (ArrayData::Float64(a), ArrayData::Int64(b)) => {
            let b: Vec<f64> = b.iter().map(|&v| v as f64).collect();
            cmp_slices!(a, &b)
        }
        (a, b) => {
            return Err(Error::typ(format!(
                "cannot compare {} with {}",
                type_name(a),
                type_name(b)
            )))
        }
    };

    Ok(Array::with_validity(ArrayData::Boolean(values), validity))
}

/// Arithmetic operators.
fn arithmetic(l: &Array, op: BinaryOp, r: &Array) -> Result<Array> {
    use BinaryOp::*;
    let n = l.len();
    let mut validity = combine_validity(l, r).unwrap_or_else(|| Bitmap::all_valid(n));
    let mut any_null = validity.null_count() > 0;

    // Integer arithmetic stays integral unless the operator is division.
    if let (ArrayData::Int64(a), ArrayData::Int64(b)) = (l.data(), r.data()) {
        if op != Divide {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                // Overflow produces NULL rather than a wrapped value or a
                // panic: silently wrapping would corrupt an aggregate, and
                // panicking would kill a query over one bad row.
                let v = match op {
                    Plus => a[i].checked_add(b[i]),
                    Minus => a[i].checked_sub(b[i]),
                    Multiply => a[i].checked_mul(b[i]),
                    Modulo => a[i].checked_rem(b[i]),
                    _ => unreachable!("arithmetic handles only these operators"),
                };
                match v {
                    Some(x) => out.push(x),
                    None => {
                        out.push(0);
                        validity.set_null(i);
                        any_null = true;
                    }
                }
            }
            return Ok(Array::with_validity(ArrayData::Int64(out), any_null.then_some(validity)));
        }
    }

    // Everything else is float arithmetic.
    let a = to_f64(l)?;
    let b = to_f64(r)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = match op {
            Plus => a[i] + b[i],
            Minus => a[i] - b[i],
            Multiply => a[i] * b[i],
            Divide | Modulo => {
                // SQL defines division by zero as an error, but returning NULL
                // keeps a single bad row from failing an entire analytical
                // query -- the same choice most warehouses make.
                if b[i] == 0.0 {
                    validity.set_null(i);
                    any_null = true;
                    out.push(0.0);
                    continue;
                }
                if op == Divide {
                    a[i] / b[i]
                } else {
                    a[i] % b[i]
                }
            }
            _ => unreachable!("arithmetic handles only these operators"),
        };
        out.push(v);
    }
    Ok(Array::with_validity(ArrayData::Float64(out), any_null.then_some(validity)))
}

fn to_f64(a: &Array) -> Result<Vec<f64>> {
    match a.data() {
        ArrayData::Float64(v) => Ok(v.clone()),
        ArrayData::Int64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        other => Err(Error::typ(format!("expected a numeric column, found {}", type_name(other)))),
    }
}

fn unary(op: UnaryOp, v: &Array) -> Result<Array> {
    match op {
        UnaryOp::Not => {
            let b = v.as_bool().ok_or_else(|| {
                Error::typ(format!("NOT requires a boolean, found {}", v.data_type()))
            })?;
            // NOT NULL is still NULL, so validity passes straight through.
            Ok(Array::with_validity(
                ArrayData::Boolean(b.iter().map(|&x| !x).collect()),
                v.validity().cloned(),
            ))
        }
        UnaryOp::Negate => match v.data() {
            ArrayData::Int64(a) => Ok(Array::with_validity(
                ArrayData::Int64(a.iter().map(|&x| x.wrapping_neg()).collect()),
                v.validity().cloned(),
            )),
            ArrayData::Float64(a) => Ok(Array::with_validity(
                ArrayData::Float64(a.iter().map(|&x| -x).collect()),
                v.validity().cloned(),
            )),
            other => {
                Err(Error::typ(format!("cannot negate a column of type {}", type_name(other))))
            }
        },
    }
}

/// `column LIKE 'literal'` -- the pattern is matched without being broadcast.
fn like_const(v: &Array, pattern: &str, negated: bool) -> Array {
    let Some(values) = v.as_str() else {
        // A non-string column can never match; preserve NULLs so that
        // `NULL LIKE 'x'` stays NULL rather than becoming false.
        return Array::new_null(DataType::Boolean, v.len());
    };
    let out: Vec<bool> = values.iter().map(|s| like_match(s, pattern) != negated).collect();
    Array::with_validity(ArrayData::Boolean(out), v.validity().cloned())
}

/// Compares a string column against a literal without broadcasting it.
///
/// `flipped` is set when the literal was written on the left, so that
/// `'b' < col` is evaluated as `col > 'b'` rather than silently reversing the
/// comparison's meaning.
fn compare_str_scalar(v: &Array, op: BinaryOp, lit: &str, flipped: bool) -> Option<Array> {
    use BinaryOp::*;
    let values = v.as_str()?;
    let op = if flipped {
        match op {
            Lt => Gt,
            LtEq => GtEq,
            Gt => Lt,
            GtEq => LtEq,
            other => other,
        }
    } else {
        op
    };
    let out: Vec<bool> = match op {
        Eq => values.iter().map(|s| s.as_str() == lit).collect(),
        NotEq => values.iter().map(|s| s.as_str() != lit).collect(),
        Lt => values.iter().map(|s| s.as_str() < lit).collect(),
        LtEq => values.iter().map(|s| s.as_str() <= lit).collect(),
        Gt => values.iter().map(|s| s.as_str() > lit).collect(),
        GtEq => values.iter().map(|s| s.as_str() >= lit).collect(),
        _ => return None,
    };
    Some(Array::with_validity(ArrayData::Boolean(out), v.validity().cloned()))
}

/// `expr LIKE pattern`, supporting `%` and `_`.
fn like(v: &Array, p: &Array, negated: bool) -> Result<Array> {
    let values = v
        .as_str()
        .ok_or_else(|| Error::typ(format!("LIKE requires a string, found {}", v.data_type())))?;
    let patterns = p.as_str().ok_or_else(|| {
        Error::typ(format!("LIKE pattern must be a string, found {}", p.data_type()))
    })?;

    let n = v.len();
    let validity = combine_validity(v, p);
    let out: Vec<bool> = (0..n).map(|i| like_match(&values[i], &patterns[i]) != negated).collect();
    Ok(Array::with_validity(ArrayData::Boolean(out), validity))
}

/// Matches a string against a SQL LIKE pattern.
///
/// Implemented as an iterative backtracking matcher rather than by translating
/// to a regex: it needs no allocation, handles the only two wildcards SQL has,
/// and runs in O(n·m) worst case with O(1) space.
///
/// The ASCII path operates directly on bytes. That matters more than it looks:
/// the obvious implementation collects both sides into `Vec<char>`, which costs
/// two heap allocations *per row*, and in a 100k-row scan the allocator ends up
/// dominating a predicate that should be a tight loop. Non-ASCII input still
/// needs character-wise comparison, so it falls back to the collecting version.
fn like_match(text: &str, pattern: &str) -> bool {
    if text.is_ascii() && pattern.is_ascii() {
        return like_match_slice(text.as_bytes(), pattern.as_bytes(), b'_', b'%');
    }
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    like_match_slice(&t, &p, '_', '%')
}

/// The matcher proper, generic over the element type so bytes and chars share
/// one implementation.
fn like_match_slice<T: Copy + PartialEq>(t: &[T], p: &[T], single: T, many: T) -> bool {
    let (mut ti, mut pi) = (0usize, 0usize);
    // Position to resume from if the current `%` guess turns out to be wrong.
    let (mut star, mut resume) = (usize::MAX, 0usize);

    while ti < t.len() {
        if pi < p.len() && (p[pi] == single || p[pi] == t[ti]) {
            ti += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == many {
            star = pi;
            resume = ti;
            pi += 1;
        } else if star != usize::MAX {
            // Backtrack: let the last `%` absorb one more character.
            pi = star + 1;
            resume += 1;
            ti = resume;
        } else {
            return false;
        }
    }
    // Trailing `%`s can match the empty remainder.
    while pi < p.len() && p[pi] == many {
        pi += 1;
    }
    pi == p.len()
}

/// Type conversion.
fn cast(v: &Array, target: DataType) -> Result<Array> {
    if v.data_type() == target {
        return Ok(v.clone());
    }
    let n = v.len();
    let mut validity = Bitmap::all_valid(n);
    let mut any_null = false;

    macro_rules! convert {
        ($variant:ident, $default:expr, $f:expr) => {{
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                if !v.is_valid(i) {
                    out.push($default);
                    validity.set_null(i);
                    any_null = true;
                    continue;
                }
                // A failed conversion yields NULL rather than failing the
                // query, matching how CAST behaves in most warehouses.
                match $f(v.value(i)) {
                    Some(x) => out.push(x),
                    None => {
                        out.push($default);
                        validity.set_null(i);
                        any_null = true;
                    }
                }
            }
            ArrayData::$variant(out)
        }};
    }

    let data = match target {
        DataType::Int64 => convert!(Int64, 0i64, |val: Value| match val {
            Value::Int64(x) => Some(x),
            Value::Float64(x) => Some(x as i64),
            Value::Boolean(b) => Some(b as i64),
            Value::Utf8(s) => s.trim().parse::<i64>().ok(),
            _ => None,
        }),
        DataType::Float64 => convert!(Float64, 0f64, |val: Value| match val {
            Value::Int64(x) => Some(x as f64),
            Value::Float64(x) => Some(x),
            Value::Boolean(b) => Some(b as i64 as f64),
            Value::Utf8(s) => s.trim().parse::<f64>().ok(),
            _ => None,
        }),
        DataType::Utf8 => convert!(Utf8, String::new(), |val: Value| Some(val.to_string())),
        DataType::Boolean => convert!(Boolean, false, |val: Value| match val {
            Value::Boolean(b) => Some(b),
            Value::Int64(x) => Some(x != 0),
            Value::Utf8(s) => match s.trim().to_ascii_lowercase().as_str() {
                "true" | "t" | "1" | "yes" => Some(true),
                "false" | "f" | "0" | "no" => Some(false),
                _ => None,
            },
            _ => None,
        }),
        DataType::Date32 => convert!(Date32, 0i32, |val: Value| match val {
            Value::Date32(d) => Some(d),
            Value::Utf8(s) => crate::types::parse_date32(s.trim()),
            _ => None,
        }),
        DataType::Null => return Ok(Array::new_null(DataType::Null, n)),
    };

    Ok(Array::with_validity(data, any_null.then_some(validity)))
}

/// Chooses between two columns row-wise, based on a boolean mask.
///
/// A NULL or FALSE condition selects `b`, which is what gives `CASE WHEN`
/// its semantics without a per-row branch in the caller.
fn select(cond: &Array, a: &Array, b: &Array) -> Result<Array> {
    let n = cond.len();
    let flags = cond.as_bool().ok_or_else(|| Error::typ("CASE WHEN condition must be boolean"))?;

    let mut values = Vec::with_capacity(n);
    for (i, &flag) in flags.iter().enumerate().take(n) {
        let take_a = cond.is_valid(i) && flag;
        values.push(if take_a { a.value(i) } else { b.value(i) });
    }

    // The two arms may have different types (INT and FLOAT); unify them.
    let out_type = a.data_type().unify(&b.data_type())?;
    Array::from_values(coerce_values(values, out_type), out_type)
}

fn coerce_values(values: Vec<Value>, target: DataType) -> Vec<Value> {
    values
        .into_iter()
        .map(|v| match (&v, target) {
            (Value::Int64(x), DataType::Float64) => Value::Float64(*x as f64),
            _ => v,
        })
        .collect()
}

fn type_name(d: &ArrayData) -> &'static str {
    match d {
        ArrayData::Int64(_) => "INT64",
        ArrayData::Float64(_) => "FLOAT64",
        ArrayData::Utf8(_) => "UTF8",
        ArrayData::Boolean(_) => "BOOLEAN",
        ArrayData::Date32(_) => "DATE32",
        ArrayData::Null(_) => "NULL",
    }
}

/// Computes the row positions where a boolean mask is true.
///
/// Filtering is expressed as "build positions, then `take`" rather than as a
/// fused copy, because the same positions are then applied to every column
/// without re-evaluating the predicate.
pub fn selection_indices(mask: &Array) -> Result<Vec<usize>> {
    let flags = mask.as_bool().ok_or_else(|| {
        Error::typ(format!("filter requires a boolean, found {}", mask.data_type()))
    })?;
    // A NULL predicate excludes the row: in SQL, only TRUE passes a WHERE.
    Ok((0..mask.len()).filter(|&i| mask.is_valid(i) && flags[i]).collect())
}

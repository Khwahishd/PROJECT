//! Rule-based plan rewriting.
//!
//! Each rule is a total function from plan to plan that must preserve the
//! query's result. Rules are applied repeatedly until the plan stops changing
//! (to a fixed bound, so a pair of rules that undo each other cannot loop
//! forever), because rules enable one another: constant folding can make a
//! predicate trivially true, which lets predicate pushdown remove a Filter
//! entirely, which can then let projection pushdown prune another column.
//!
//! The three rules implemented here are the ones with the largest effect on an
//! analytical workload, in order:
//!
//! 1. **Predicate pushdown** moves filters toward the scans, so rows are
//!    discarded before they are joined or aggregated.
//! 2. **Projection pushdown** prunes columns a query never reads, which in a
//!    columnar engine means they are never touched at all.
//! 3. **Constant folding** evaluates literal-only subexpressions once at plan
//!    time rather than once per row.

use super::expr::{BinaryOp, LogicalExpr, UnaryOp};
use super::plan::LogicalPlan;
use crate::error::Result;
use crate::sql::ast::JoinType;
use crate::types::{DataType, Value};

/// A single rewrite rule.
pub trait OptimizerRule {
    /// The rule's name, used in `EXPLAIN` output.
    fn name(&self) -> &str;
    /// Rewrites a plan.
    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan>;
}

/// Applies a set of rules to a fixed point.
pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
    max_passes: usize,
}

impl Optimizer {
    /// Builds an optimizer with the default rule set.
    pub fn new() -> Self {
        Optimizer {
            rules: vec![
                Box::new(ConstantFolding),
                Box::new(PredicatePushdown),
                Box::new(ProjectionPushdown),
            ],
            // Rules are confluent in practice, but bounding the loop means a
            // future rule that oscillates degrades to "not fully optimized"
            // rather than hanging the engine.
            max_passes: 5,
        }
    }

    /// The names of the configured rules.
    pub fn rule_names(&self) -> Vec<&str> {
        self.rules.iter().map(|r| r.name()).collect()
    }

    /// Optimizes a plan.
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let mut current = plan;
        for _ in 0..self.max_passes {
            let before = current.display_indent();
            for rule in &self.rules {
                current = rule.rewrite(current)?;
            }
            if current.display_indent() == before {
                break;
            }
        }
        Ok(current)
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimizes a plan with the default rules.
pub fn optimize(plan: LogicalPlan) -> Result<LogicalPlan> {
    Optimizer::new().optimize(plan)
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

/// Evaluates subexpressions whose operands are all literals.
pub struct ConstantFolding;

impl OptimizerRule for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        map_plan(plan, &|p| {
            Ok(match p {
                LogicalPlan::Filter { predicate, input } => {
                    let folded = fold(predicate)?;
                    // A predicate that folded to TRUE filters nothing.
                    if matches!(folded, LogicalExpr::Literal(Value::Boolean(true))) {
                        *input
                    } else {
                        LogicalPlan::Filter { predicate: folded, input }
                    }
                }
                LogicalPlan::Projection { exprs, schema, input } => {
                    let exprs = exprs.into_iter().map(fold).collect::<Result<Vec<_>>>()?;
                    LogicalPlan::Projection { exprs, schema, input }
                }
                LogicalPlan::Scan { table, schema, projection, filters } => {
                    let filters = filters.into_iter().map(fold).collect::<Result<Vec<_>>>()?;
                    LogicalPlan::Scan { table, schema, projection, filters }
                }
                other => other,
            })
        })
    }
}

/// Recursively folds constant subexpressions.
fn fold(expr: LogicalExpr) -> Result<LogicalExpr> {
    Ok(match expr {
        LogicalExpr::Binary { left, op, right } => {
            let l = fold(*left)?;
            let r = fold(*right)?;

            // Short-circuit the boolean identities first: `x AND true` is `x`
            // even when `x` is not a constant, which is what makes pushing a
            // folded conjunct worthwhile.
            if op == BinaryOp::And {
                match (&l, &r) {
                    (LogicalExpr::Literal(Value::Boolean(false)), _)
                    | (_, LogicalExpr::Literal(Value::Boolean(false))) => {
                        return Ok(LogicalExpr::Literal(Value::Boolean(false)))
                    }
                    (LogicalExpr::Literal(Value::Boolean(true)), other)
                    | (other, LogicalExpr::Literal(Value::Boolean(true))) => {
                        return Ok(other.clone())
                    }
                    _ => {}
                }
            }
            if op == BinaryOp::Or {
                match (&l, &r) {
                    (LogicalExpr::Literal(Value::Boolean(true)), _)
                    | (_, LogicalExpr::Literal(Value::Boolean(true))) => {
                        return Ok(LogicalExpr::Literal(Value::Boolean(true)))
                    }
                    (LogicalExpr::Literal(Value::Boolean(false)), other)
                    | (other, LogicalExpr::Literal(Value::Boolean(false))) => {
                        return Ok(other.clone())
                    }
                    _ => {}
                }
            }

            if let (LogicalExpr::Literal(a), LogicalExpr::Literal(b)) = (&l, &r) {
                if let Some(v) = eval_binary_literal(a, op, b) {
                    return Ok(LogicalExpr::Literal(v));
                }
            }
            LogicalExpr::Binary { left: Box::new(l), op, right: Box::new(r) }
        }

        LogicalExpr::Unary { op, expr } => {
            let e = fold(*expr)?;
            match (&op, &e) {
                (UnaryOp::Not, LogicalExpr::Literal(Value::Boolean(b))) => {
                    LogicalExpr::Literal(Value::Boolean(!b))
                }
                (UnaryOp::Negate, LogicalExpr::Literal(Value::Int64(v))) => {
                    LogicalExpr::Literal(Value::Int64(-v))
                }
                (UnaryOp::Negate, LogicalExpr::Literal(Value::Float64(v))) => {
                    LogicalExpr::Literal(Value::Float64(-v))
                }
                // Double negation cancels.
                (UnaryOp::Not, LogicalExpr::Unary { op: UnaryOp::Not, expr: inner }) => {
                    (**inner).clone()
                }
                _ => LogicalExpr::Unary { op, expr: Box::new(e) },
            }
        }

        LogicalExpr::IsNull { expr, negated } => {
            let e = fold(*expr)?;
            if let LogicalExpr::Literal(v) = &e {
                return Ok(LogicalExpr::Literal(Value::Boolean(v.is_null() != negated)));
            }
            LogicalExpr::IsNull { expr: Box::new(e), negated }
        }

        LogicalExpr::Alias { expr, name } => {
            LogicalExpr::Alias { expr: Box::new(fold(*expr)?), name }
        }

        LogicalExpr::Cast { expr, data_type } => {
            LogicalExpr::Cast { expr: Box::new(fold(*expr)?), data_type }
        }

        LogicalExpr::Like { expr, pattern, negated } => LogicalExpr::Like {
            expr: Box::new(fold(*expr)?),
            pattern: Box::new(fold(*pattern)?),
            negated,
        },

        LogicalExpr::Case { when_then, else_expr } => {
            let mut wt = Vec::with_capacity(when_then.len());
            for (w, t) in when_then {
                let w = fold(w)?;
                let t = fold(t)?;
                // A branch whose condition is constantly false can never fire.
                if matches!(w, LogicalExpr::Literal(Value::Boolean(false))) {
                    continue;
                }
                // A branch whose condition is constantly true makes every
                // later branch, and the ELSE, unreachable.
                if matches!(w, LogicalExpr::Literal(Value::Boolean(true))) {
                    if wt.is_empty() {
                        return Ok(t);
                    }
                    wt.push((w, t));
                    return Ok(LogicalExpr::Case { when_then: wt, else_expr: None });
                }
                wt.push((w, t));
            }
            let else_expr = match else_expr {
                Some(e) => Some(Box::new(fold(*e)?)),
                None => None,
            };
            if wt.is_empty() {
                return Ok(match else_expr {
                    Some(e) => *e,
                    None => LogicalExpr::Literal(Value::Null),
                });
            }
            LogicalExpr::Case { when_then: wt, else_expr }
        }

        other => other,
    })
}

/// Evaluates a binary operation over two literals.
///
/// Returns `None` where the result is not a simple constant -- notably any
/// operation involving NULL, and division by zero, both of which are left to
/// the execution engine so that their SQL semantics live in exactly one place.
fn eval_binary_literal(a: &Value, op: BinaryOp, b: &Value) -> Option<Value> {
    use BinaryOp::*;
    if a.is_null() || b.is_null() {
        return None;
    }

    if op.is_logical() {
        let (x, y) = match (a, b) {
            (Value::Boolean(x), Value::Boolean(y)) => (*x, *y),
            _ => return None,
        };
        return Some(Value::Boolean(match op {
            And => x && y,
            Or => x || y,
            _ => unreachable!("is_logical covers exactly And and Or"),
        }));
    }

    if op.is_comparison() {
        let ord = compare_values(a, b)?;
        return Some(Value::Boolean(match op {
            Eq => ord == std::cmp::Ordering::Equal,
            NotEq => ord != std::cmp::Ordering::Equal,
            Lt => ord == std::cmp::Ordering::Less,
            LtEq => ord != std::cmp::Ordering::Greater,
            Gt => ord == std::cmp::Ordering::Greater,
            GtEq => ord != std::cmp::Ordering::Less,
            _ => unreachable!("is_comparison covers exactly these six"),
        }));
    }

    // Integer arithmetic stays integral; mixed or float operands widen.
    if let (Value::Int64(x), Value::Int64(y)) = (a, b) {
        return Some(match op {
            // Use checked arithmetic: folding an overflow into a wrapped
            // constant would silently change the query's meaning.
            Plus => Value::Int64(x.checked_add(*y)?),
            Minus => Value::Int64(x.checked_sub(*y)?),
            Multiply => Value::Int64(x.checked_mul(*y)?),
            Modulo if *y != 0 => Value::Int64(x.checked_rem(*y)?),
            Divide if *y != 0 => Value::Float64(*x as f64 / *y as f64),
            _ => return None,
        });
    }

    let (x, y) = (a.as_f64()?, b.as_f64()?);
    Some(match op {
        Plus => Value::Float64(x + y),
        Minus => Value::Float64(x - y),
        Multiply => Value::Float64(x * y),
        Divide if y != 0.0 => Value::Float64(x / y),
        Modulo if y != 0.0 => Value::Float64(x % y),
        _ => return None,
    })
}

/// Total ordering over comparable values, used by folding and by sorting.
pub fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    Some(match (a, b) {
        (Value::Int64(x), Value::Int64(y)) => x.cmp(y),
        (Value::Utf8(x), Value::Utf8(y)) => x.cmp(y),
        (Value::Boolean(x), Value::Boolean(y)) => x.cmp(y),
        (Value::Date32(x), Value::Date32(y)) => x.cmp(y),
        _ => {
            let (x, y) = (a.as_f64()?, b.as_f64()?);
            // NaN is unordered; treating it as equal would make sorts
            // inconsistent, so report no ordering and let callers decide.
            x.partial_cmp(&y).unwrap_or(Ordering::Equal)
        }
    })
}

// ---------------------------------------------------------------------------
// Predicate pushdown
// ---------------------------------------------------------------------------

/// Moves filters as close to the scans as correctness allows.
pub struct PredicatePushdown;

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate_pushdown"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        push(plan, Vec::new())
    }
}

/// Pushes `predicates` down through `plan`, returning the rewritten plan.
///
/// Any predicate that cannot be pushed further is re-applied as a Filter
/// immediately above the node that blocked it.
fn push(plan: LogicalPlan, mut predicates: Vec<LogicalExpr>) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { predicate, input } => {
            predicate.split_conjunction(&mut predicates);
            push(*input, predicates)
        }

        LogicalPlan::Scan { table, schema, projection, mut filters } => {
            // The scan is the bottom: everything that reached it stays here.
            filters.extend(predicates);
            Ok(LogicalPlan::Scan { table, schema, projection, filters })
        }

        LogicalPlan::Projection { exprs, schema, input } => {
            // A predicate can pass through a projection only if every column it
            // references is a plain passthrough of an input column. Pushing a
            // predicate on a *computed* column would mean evaluating an
            // expression that does not exist below the projection.
            let (pushable, blocked) = split_through_projection(&exprs, predicates)?;
            let new_input = push(*input, pushable)?;
            let plan = LogicalPlan::Projection { exprs, schema, input: Box::new(new_input) };
            Ok(wrap_filter(plan, blocked))
        }

        LogicalPlan::Aggregate { group_by, aggregates, schema, input } => {
            // A predicate over a grouping key may be pushed below the
            // aggregation: filtering rows out before grouping cannot change
            // which groups survive, because the key value is unchanged.
            // Predicates over aggregate outputs (i.e. HAVING) cannot move.
            let group_count = group_by.len();
            let mut pushable = Vec::new();
            let mut blocked = Vec::new();
            for p in predicates {
                let mut cols = Vec::new();
                p.referenced_columns(&mut cols);
                let only_groups = !cols.is_empty() && cols.iter().all(|&c| c < group_count);
                let touches_agg = mentions_aggregate(&p);
                if only_groups && !touches_agg {
                    // Rewrite indices from the aggregate's output space into
                    // the input space by substituting the grouping expression.
                    match substitute_group_exprs(&p, &group_by) {
                        Some(rewritten) => pushable.push(rewritten),
                        None => blocked.push(p),
                    }
                } else {
                    blocked.push(p);
                }
            }
            let new_input = push(*input, pushable)?;
            let plan =
                LogicalPlan::Aggregate { group_by, aggregates, schema, input: Box::new(new_input) };
            Ok(wrap_filter(plan, blocked))
        }

        LogicalPlan::Join { left, right, on, filter, join_type, schema } => {
            let left_width = left.schema().len();
            let mut to_left = Vec::new();
            let mut to_right = Vec::new();
            let mut stay = Vec::new();

            // Combine the join's own residual filter with anything pushed from
            // above, so both get the same treatment.
            let mut all = predicates;
            if let Some(f) = filter.clone() {
                f.split_conjunction(&mut all);
            }

            for p in all {
                let mut cols = Vec::new();
                p.referenced_columns(&mut cols);
                let all_left = cols.iter().all(|&c| c < left_width);
                let all_right = cols.iter().all(|&c| c >= left_width);

                // For an outer join, pushing a predicate to the null-extended
                // side is NOT safe: it would discard rows that the join is
                // obliged to emit (with NULLs) rather than filter out.
                let left_ok =
                    matches!(join_type, JoinType::Inner | JoinType::Cross | JoinType::Left);
                let right_ok =
                    matches!(join_type, JoinType::Inner | JoinType::Cross | JoinType::Right);

                match (all_left, all_right, cols.is_empty()) {
                    (_, _, true) => stay.push(p), // constant: cheap, leave it
                    (true, _, _) if left_ok => to_left.push(p),
                    (_, true, _) if right_ok => {
                        let shifted = p.remap_columns(&|i| i.checked_sub(left_width))?;
                        to_right.push(shifted);
                    }
                    _ => stay.push(p),
                }
            }

            let new_left = push(*left, to_left)?;
            let new_right = push(*right, to_right)?;
            let plan = LogicalPlan::Join {
                left: Box::new(new_left),
                right: Box::new(new_right),
                on,
                filter: None,
                join_type,
                schema,
            };
            Ok(wrap_filter(plan, stay))
        }

        // Sorting, limiting and deduplication do not change column values, but
        // pushing a filter below a Limit WOULD change which rows survive --
        // `LIMIT 10` then filter is not the same as filter then `LIMIT 10`.
        LogicalPlan::Sort { exprs, input } => {
            let new_input = push(*input, predicates)?;
            Ok(LogicalPlan::Sort { exprs, input: Box::new(new_input) })
        }
        LogicalPlan::Distinct { input } => {
            let new_input = push(*input, predicates)?;
            Ok(LogicalPlan::Distinct { input: Box::new(new_input) })
        }
        LogicalPlan::Limit { skip, fetch, input } => {
            let new_input = push(*input, Vec::new())?;
            let plan = LogicalPlan::Limit { skip, fetch, input: Box::new(new_input) };
            Ok(wrap_filter(plan, predicates))
        }

        other => Ok(wrap_filter(other, predicates)),
    }
}

/// Rebuilds a Filter node above `plan` for predicates that could not be pushed.
fn wrap_filter(plan: LogicalPlan, predicates: Vec<LogicalExpr>) -> LogicalPlan {
    match LogicalExpr::conjunction(predicates) {
        None => plan,
        Some(predicate) => LogicalPlan::Filter { predicate, input: Box::new(plan) },
    }
}

/// Splits predicates into those that can pass through a projection and those
/// that cannot.
fn split_through_projection(
    exprs: &[LogicalExpr],
    predicates: Vec<LogicalExpr>,
) -> Result<(Vec<LogicalExpr>, Vec<LogicalExpr>)> {
    // Map each projection output position back to an input column, where the
    // output is a plain column reference.
    let passthrough: Vec<Option<(usize, String, DataType)>> = exprs
        .iter()
        .map(|e| {
            let base = match e {
                LogicalExpr::Alias { expr, .. } => expr.as_ref(),
                other => other,
            };
            match base {
                LogicalExpr::Column { index, name, data_type } => {
                    Some((*index, name.clone(), *data_type))
                }
                _ => None,
            }
        })
        .collect();

    let mut pushable = Vec::new();
    let mut blocked = Vec::new();
    for p in predicates {
        let mut cols = Vec::new();
        p.referenced_columns(&mut cols);
        if cols.iter().all(|&c| c < passthrough.len() && passthrough[c].is_some()) {
            let rewritten = p.remap_columns(&|i| {
                passthrough.get(i).and_then(|o| o.as_ref()).map(|(idx, _, _)| *idx)
            })?;
            pushable.push(rewritten);
        } else {
            blocked.push(p);
        }
    }
    Ok((pushable, blocked))
}

/// Replaces aggregate-output column references with the grouping expressions
/// they came from, so a predicate can be evaluated below the aggregation.
fn substitute_group_exprs(
    predicate: &LogicalExpr,
    group_by: &[LogicalExpr],
) -> Option<LogicalExpr> {
    match predicate {
        LogicalExpr::Column { index, .. } => group_by.get(*index).cloned(),
        LogicalExpr::Literal(_) => Some(predicate.clone()),
        LogicalExpr::Binary { left, op, right } => Some(LogicalExpr::Binary {
            left: Box::new(substitute_group_exprs(left, group_by)?),
            op: *op,
            right: Box::new(substitute_group_exprs(right, group_by)?),
        }),
        LogicalExpr::Unary { op, expr } => Some(LogicalExpr::Unary {
            op: *op,
            expr: Box::new(substitute_group_exprs(expr, group_by)?),
        }),
        LogicalExpr::IsNull { expr, negated } => Some(LogicalExpr::IsNull {
            expr: Box::new(substitute_group_exprs(expr, group_by)?),
            negated: *negated,
        }),
        LogicalExpr::Like { expr, pattern, negated } => Some(LogicalExpr::Like {
            expr: Box::new(substitute_group_exprs(expr, group_by)?),
            pattern: Box::new(substitute_group_exprs(pattern, group_by)?),
            negated: *negated,
        }),
        LogicalExpr::Cast { expr, data_type } => Some(LogicalExpr::Cast {
            expr: Box::new(substitute_group_exprs(expr, group_by)?),
            data_type: *data_type,
        }),
        // Anything else (aggregate refs, CASE over aggregates) cannot move.
        _ => None,
    }
}

fn mentions_aggregate(e: &LogicalExpr) -> bool {
    match e {
        LogicalExpr::AggregateRef { .. } => true,
        LogicalExpr::Binary { left, right, .. } => {
            mentions_aggregate(left) || mentions_aggregate(right)
        }
        LogicalExpr::Unary { expr, .. }
        | LogicalExpr::IsNull { expr, .. }
        | LogicalExpr::Cast { expr, .. }
        | LogicalExpr::Alias { expr, .. } => mentions_aggregate(expr),
        LogicalExpr::Like { expr, pattern, .. } => {
            mentions_aggregate(expr) || mentions_aggregate(pattern)
        }
        LogicalExpr::Case { when_then, else_expr } => {
            when_then.iter().any(|(w, t)| mentions_aggregate(w) || mentions_aggregate(t))
                || else_expr.as_ref().is_some_and(|e| mentions_aggregate(e))
        }
        LogicalExpr::Literal(_) | LogicalExpr::Column { .. } => false,
    }
}

// ---------------------------------------------------------------------------
// Projection pushdown
// ---------------------------------------------------------------------------

/// Prunes columns that no operator above a scan actually reads.
///
/// In a columnar engine this is the single highest-leverage rule: an unread
/// column is never decoded, never copied by `take`, and never materialized.
pub struct ProjectionPushdown;

impl OptimizerRule for ProjectionPushdown {
    fn name(&self) -> &str {
        "projection_pushdown"
    }

    fn rewrite(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        prune(plan)
    }
}

/// Rewrites every Scan under `plan` to read only the columns reachable from
/// above it.
fn prune(plan: LogicalPlan) -> Result<LogicalPlan> {
    // The rule is implemented on the single-scan shape, which covers the
    // queries this engine plans today. A join has two scans whose index spaces
    // are concatenated, so pruning them independently would require rewriting
    // the join's key indices too; rather than do that incorrectly, joins are
    // left alone and only their inputs are visited.
    match plan {
        LogicalPlan::Join { left, right, on, filter, join_type, schema } => Ok(LogicalPlan::Join {
            left: Box::new(prune(*left)?),
            right: Box::new(prune(*right)?),
            on,
            filter,
            join_type,
            schema,
        }),
        other => prune_linear(other),
    }
}

fn prune_linear(plan: LogicalPlan) -> Result<LogicalPlan> {
    let Some(scan_width) = scan_width(&plan) else {
        return Ok(plan);
    };

    let mut needed = Vec::new();
    if collect_scan_space(&plan, &mut needed).is_none() {
        // The plan contains a shape this rule does not model; leave it alone.
        return Ok(plan);
    }
    needed.retain(|&c| c < scan_width);
    needed.sort_unstable();
    needed.dedup();

    if needed.len() == scan_width {
        return Ok(plan); // nothing to prune
    }
    if needed.is_empty() {
        // The query reads no columns at all (`SELECT 1 FROM t`, `COUNT(*)`).
        // One column still has to be read, because a batch's row count comes
        // from its columns -- but one is enough.
        needed.push(0);
    }

    // Map an old scan column index to its position in the narrowed scan.
    let mut mapping: Vec<Option<usize>> = vec![None; scan_width];
    for (new, &old) in needed.iter().enumerate() {
        mapping[old] = Some(new);
    }

    let (rewritten, _) = rewrite_scan_space(plan, &needed, &mapping)?;
    Ok(rewritten)
}

/// Walks down to the scan, returning its column count.
fn scan_width(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::Scan { schema, projection, .. } => match projection {
            None => Some(schema.len()),
            // Already pruned once; do not prune again, so the index mapping
            // stays a single step and remains easy to reason about.
            Some(_) => None,
        },
        LogicalPlan::Join { .. } | LogicalPlan::EmptyRelation { .. } => None,
        other => other.children().first().and_then(|c| scan_width(c)),
    }
}

/// Collects the scan-relative column indices a plan reads.
///
/// Only part of a plan addresses the scan's columns. Walking up from the scan,
/// indices stay scan-relative until the first Projection or Aggregate, which
/// *redefines* the index space for everything above it: in
/// `Projection[c, SUM(b)] → Aggregate[groupBy=c] → Scan[a,b,c]`, the
/// projection's `c` is output column 0 of the aggregate, not column 0 of the
/// scan. Treating it as scan-relative keeps column `a` alive and, worse, lets
/// the rewrite below renumber an index that was never in scan space.
///
/// Returns `None` for a shape this rule does not model, and otherwise whether
/// that boundary has already been crossed.
fn collect_scan_space(plan: &LogicalPlan, out: &mut Vec<usize>) -> Option<bool> {
    match plan {
        LogicalPlan::Scan { filters, .. } => {
            // Predicates pushed into the scan read columns too. Omitting them
            // here prunes a column the scan still needs, which surfaces as
            // "column was pruned but is still referenced" during the rewrite.
            for f in filters {
                f.referenced_columns(out);
            }
            Some(false)
        }

        LogicalPlan::Filter { predicate, input } => {
            let crossed = collect_scan_space(input, out)?;
            if !crossed {
                predicate.referenced_columns(out);
            }
            Some(crossed)
        }

        LogicalPlan::Projection { exprs, input, .. } => {
            let crossed = collect_scan_space(input, out)?;
            if !crossed {
                for e in exprs {
                    e.referenced_columns(out);
                }
            }
            Some(true)
        }

        LogicalPlan::Aggregate { group_by, aggregates, input, .. } => {
            let crossed = collect_scan_space(input, out)?;
            if !crossed {
                for g in group_by {
                    g.referenced_columns(out);
                }
                for a in aggregates {
                    if let Some(arg) = &a.arg {
                        arg.referenced_columns(out);
                    }
                }
            }
            Some(true)
        }

        LogicalPlan::Sort { exprs, input } => {
            let crossed = collect_scan_space(input, out)?;
            if !crossed {
                for (e, _) in exprs {
                    e.referenced_columns(out);
                }
            }
            Some(crossed)
        }

        LogicalPlan::Limit { input, .. } | LogicalPlan::Distinct { input } => {
            collect_scan_space(input, out)
        }

        LogicalPlan::Join { .. } | LogicalPlan::EmptyRelation { .. } => None,
    }
}

/// Applies the narrowed projection to the scan and renumbers every index that
/// is genuinely scan-relative, mirroring [`collect_scan_space`] exactly.
///
/// Returns the rewritten plan and whether the index-space boundary has been
/// crossed; nodes above it are returned untouched.
fn rewrite_scan_space(
    plan: LogicalPlan,
    projection: &[usize],
    mapping: &[Option<usize>],
) -> Result<(LogicalPlan, bool)> {
    let remap = |i: usize| mapping.get(i).copied().flatten();

    Ok(match plan {
        LogicalPlan::Scan { table, schema, filters, .. } => {
            let filters =
                filters.into_iter().map(|f| f.remap_columns(&remap)).collect::<Result<Vec<_>>>()?;
            (
                LogicalPlan::Scan { table, schema, projection: Some(projection.to_vec()), filters },
                false,
            )
        }

        LogicalPlan::Filter { predicate, input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            let predicate = if crossed { predicate } else { predicate.remap_columns(&remap)? };
            (LogicalPlan::Filter { predicate, input: Box::new(input) }, crossed)
        }

        LogicalPlan::Projection { exprs, schema, input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            let exprs = if crossed {
                exprs
            } else {
                exprs.into_iter().map(|e| e.remap_columns(&remap)).collect::<Result<Vec<_>>>()?
            };
            (LogicalPlan::Projection { exprs, schema, input: Box::new(input) }, true)
        }

        LogicalPlan::Aggregate { group_by, aggregates, schema, input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            let (group_by, aggregates) = if crossed {
                (group_by, aggregates)
            } else {
                let group_by = group_by
                    .into_iter()
                    .map(|e| e.remap_columns(&remap))
                    .collect::<Result<Vec<_>>>()?;
                let mut aggs = Vec::with_capacity(aggregates.len());
                for a in aggregates {
                    let arg = match a.arg {
                        Some(e) => Some(e.remap_columns(&remap)?),
                        None => None,
                    };
                    aggs.push(super::expr::AggregateExpr { arg, ..a });
                }
                (group_by, aggs)
            };
            (LogicalPlan::Aggregate { group_by, aggregates, schema, input: Box::new(input) }, true)
        }

        LogicalPlan::Sort { exprs, input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            let exprs = if crossed {
                exprs
            } else {
                let mut out = Vec::with_capacity(exprs.len());
                for (e, asc) in exprs {
                    out.push((e.remap_columns(&remap)?, asc));
                }
                out
            };
            (LogicalPlan::Sort { exprs, input: Box::new(input) }, crossed)
        }

        LogicalPlan::Limit { skip, fetch, input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            (LogicalPlan::Limit { skip, fetch, input: Box::new(input) }, crossed)
        }

        LogicalPlan::Distinct { input } => {
            let (input, crossed) = rewrite_scan_space(*input, projection, mapping)?;
            (LogicalPlan::Distinct { input: Box::new(input) }, crossed)
        }

        other => (other, true),
    })
}

// ---------------------------------------------------------------------------
// Plan traversal helper
// ---------------------------------------------------------------------------

/// Applies `f` bottom-up to every node in the plan.
fn map_plan(
    plan: LogicalPlan,
    f: &dyn Fn(LogicalPlan) -> Result<LogicalPlan>,
) -> Result<LogicalPlan> {
    let rebuilt = match plan {
        LogicalPlan::Filter { predicate, input } => {
            LogicalPlan::Filter { predicate, input: Box::new(map_plan(*input, f)?) }
        }
        LogicalPlan::Projection { exprs, schema, input } => {
            LogicalPlan::Projection { exprs, schema, input: Box::new(map_plan(*input, f)?) }
        }
        LogicalPlan::Aggregate { group_by, aggregates, schema, input } => LogicalPlan::Aggregate {
            group_by,
            aggregates,
            schema,
            input: Box::new(map_plan(*input, f)?),
        },
        LogicalPlan::Sort { exprs, input } => {
            LogicalPlan::Sort { exprs, input: Box::new(map_plan(*input, f)?) }
        }
        LogicalPlan::Limit { skip, fetch, input } => {
            LogicalPlan::Limit { skip, fetch, input: Box::new(map_plan(*input, f)?) }
        }
        LogicalPlan::Distinct { input } => {
            LogicalPlan::Distinct { input: Box::new(map_plan(*input, f)?) }
        }
        LogicalPlan::Join { left, right, on, filter, join_type, schema } => LogicalPlan::Join {
            left: Box::new(map_plan(*left, f)?),
            right: Box::new(map_plan(*right, f)?),
            on,
            filter,
            join_type,
            schema,
        },
        leaf => leaf,
    };
    f(rebuilt)
}

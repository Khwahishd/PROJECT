//! Turns a parsed AST into a resolved, type-checked [`LogicalPlan`].
//!
//! This is where a query stops being text and starts having meaning: column
//! names become indices, types are checked, and aggregate calls are lifted out
//! of the projection into a dedicated Aggregate node.

use super::expr::{AggregateExpr, AggregateFunction, LogicalExpr};
use super::plan::LogicalPlan;
use crate::error::{Error, Result};
use crate::sql::ast::{self, BinaryOp, Expr, JoinType, Query, SelectItem, TableRef, UnaryOp};
use crate::types::{DataType, Field, Schema, SchemaRef, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Resolves table names to schemas.
pub trait TableProvider {
    /// Returns the schema of `name`, or `None` if it is not registered.
    fn schema_of(&self, name: &str) -> Option<SchemaRef>;
    /// Lists registered table names, for error messages.
    fn table_names(&self) -> Vec<String>;
}

/// Builds logical plans from parsed queries.
pub struct Planner<'a> {
    catalog: &'a dyn TableProvider,
}

impl<'a> Planner<'a> {
    /// Creates a planner over a catalog.
    pub fn new(catalog: &'a dyn TableProvider) -> Self {
        Planner { catalog }
    }

    /// Plans a complete query.
    pub fn plan_query(&self, q: &Query) -> Result<LogicalPlan> {
        // 1. FROM: build the input and its schema.
        let mut plan = match &q.from {
            Some(tref) => self.plan_table_ref(tref)?,
            None => LogicalPlan::EmptyRelation {
                produce_one_row: true,
                schema: Arc::new(Schema::empty()),
            },
        };
        let input_schema = plan.schema();

        // 2. WHERE. Note this is resolved against the *input* schema: a WHERE
        //    clause cannot see projection aliases, because filtering logically
        //    happens before the SELECT list is computed.
        if let Some(sel) = &q.selection {
            let predicate = self.resolve_expr(sel, &input_schema, None)?;
            let t = predicate.data_type()?;
            if t != DataType::Boolean && t != DataType::Null {
                return Err(Error::plan(format!("WHERE requires a boolean predicate, found {t}")));
            }
            plan = LogicalPlan::Filter { predicate, input: Box::new(plan) };
        }

        // 3. Expand `*` into explicit columns, so everything downstream deals
        //    only with concrete expressions.
        let projection_items = self.expand_wildcards(&q.projection, &input_schema)?;

        // 4. Collect every aggregate appearing in SELECT, HAVING and ORDER BY.
        let mut aggregates: Vec<AggregateExpr> = Vec::new();
        for item in &projection_items {
            collect_aggregates(&item.0, &input_schema, self, &mut aggregates)?;
        }
        if let Some(h) = &q.having {
            collect_aggregates(h, &input_schema, self, &mut aggregates)?;
        }
        for ob in &q.order_by {
            collect_aggregates(&ob.expr, &input_schema, self, &mut aggregates)?;
        }

        let has_grouping = !q.group_by.is_empty() || !aggregates.is_empty();

        // 5. GROUP BY / aggregation.
        let (plan, agg_context) = if has_grouping {
            let mut group_exprs = Vec::with_capacity(q.group_by.len());
            for g in &q.group_by {
                // `GROUP BY band` where `band` is a SELECT alias, and
                // `GROUP BY 1` meaning the first select item, are both outside
                // the SQL standard but supported by every engine people
                // actually use. Rejecting them would force the user to write
                // a long CASE expression twice.
                let target = resolve_group_reference(g, &projection_items)?;
                group_exprs.push(self.resolve_expr(target, &input_schema, None)?);
            }

            let mut fields = Vec::new();
            for g in &group_exprs {
                fields.push(Field::new(g.display_name(), g.data_type()?));
            }
            for a in &aggregates {
                fields.push(Field::new(a.name.clone(), a.data_type()?));
            }
            let schema: SchemaRef = Arc::new(Schema::new(fields));

            let ctx = AggContext {
                group_exprs: group_exprs.clone(),
                aggregates: aggregates.clone(),
                schema: Arc::clone(&schema),
            };
            let plan = LogicalPlan::Aggregate {
                group_by: group_exprs,
                aggregates,
                schema,
                input: Box::new(plan),
            };
            (plan, Some(ctx))
        } else {
            (plan, None)
        };

        // 6. HAVING, resolved against the aggregate output.
        let mut plan = plan;
        if let Some(h) = &q.having {
            if agg_context.is_none() {
                return Err(Error::plan("HAVING requires GROUP BY or an aggregate function"));
            }
            let predicate = self.resolve_expr(h, &input_schema, agg_context.as_ref())?;
            plan = LogicalPlan::Filter { predicate, input: Box::new(plan) };
        }

        // 7. SELECT list.
        let mut proj_exprs = Vec::with_capacity(projection_items.len());
        for (expr, alias) in &projection_items {
            let mut resolved = self.resolve_expr(expr, &input_schema, agg_context.as_ref())?;
            if let Some(a) = alias {
                resolved = LogicalExpr::Alias { expr: Box::new(resolved), name: a.clone() };
            }
            proj_exprs.push(resolved);
        }

        // 8. ORDER BY.
        //
        // The sort runs above the projection, so a sort key must be one of the
        // projection's output columns. `ORDER BY fare` on a query that does not
        // select `fare` is nonetheless legal SQL, so any such key is appended
        // to the projection as a *hidden* column and trimmed off again after
        // the sort.
        let visible_width = proj_exprs.len();
        let mut sort_exprs = Vec::new();
        let mut hidden = Vec::new();

        for ob in &q.order_by {
            if let Some(key) = self.resolve_order_key_in_projection(&ob.expr, &proj_exprs)? {
                sort_exprs.push((key, ob.asc));
                continue;
            }
            let resolved = self.resolve_expr(&ob.expr, &input_schema, agg_context.as_ref())?;

            // Reuse a hidden column if the same key was requested twice.
            let position = match hidden.iter().position(|h| *h == resolved) {
                Some(i) => visible_width + i,
                None => {
                    if q.distinct {
                        // Adding a hidden column would change which rows are
                        // distinct, so SQL requires the key to be selected.
                        return Err(Error::plan(format!(
                            "ORDER BY expression {} must appear in the SELECT list \
                             when DISTINCT is used",
                            ob.expr
                        )));
                    }
                    hidden.push(resolved.clone());
                    visible_width + hidden.len() - 1
                }
            };
            sort_exprs.push((
                LogicalExpr::Column {
                    index: position,
                    name: resolved.display_name(),
                    data_type: resolved.data_type()?,
                },
                ob.asc,
            ));
        }

        let mut all_exprs = proj_exprs.clone();
        all_exprs.extend(hidden.iter().cloned());

        let schema = Arc::new(LogicalPlan::projection_schema(&all_exprs)?);
        plan = LogicalPlan::Projection { exprs: all_exprs, schema, input: Box::new(plan) };

        if q.distinct {
            plan = LogicalPlan::Distinct { input: Box::new(plan) };
        }

        if !sort_exprs.is_empty() {
            plan = LogicalPlan::Sort { exprs: sort_exprs, input: Box::new(plan) };
        }

        if q.limit.is_some() || q.offset.is_some() {
            plan = LogicalPlan::Limit {
                skip: q.offset.unwrap_or(0),
                fetch: q.limit,
                input: Box::new(plan),
            };
        }

        // Trim the hidden sort columns. Doing this after LIMIT means the
        // projection only runs over the rows that survived.
        if !hidden.is_empty() {
            let trim: Vec<LogicalExpr> = (0..visible_width)
                .map(|i| {
                    let f = plan.schema().field(i).clone();
                    LogicalExpr::Column { index: i, name: f.name, data_type: f.data_type }
                })
                .collect();
            let schema = Arc::new(LogicalPlan::projection_schema(&trim)?);
            plan = LogicalPlan::Projection { exprs: trim, schema, input: Box::new(plan) };
        }

        Ok(plan)
    }

    /// Resolves an ORDER BY key against the projection list.
    ///
    /// Returns `Ok(None)` when the key names something the projection does not
    /// produce; the caller then adds it as a hidden column.
    fn resolve_order_key_in_projection(
        &self,
        expr: &Expr,
        proj: &[LogicalExpr],
    ) -> Result<Option<LogicalExpr>> {
        // An ordinal: ORDER BY 2.
        if let Expr::Literal(Value::Int64(n)) = expr {
            let idx = *n as usize;
            if idx == 0 || idx > proj.len() {
                return Err(Error::plan(format!(
                    "ORDER BY position {n} is out of range for a select list with {} items",
                    proj.len()
                )));
            }
            let e = &proj[idx - 1];
            return Ok(Some(LogicalExpr::Column {
                index: idx - 1,
                name: e.display_name(),
                data_type: e.data_type()?,
            }));
        }

        // An alias or an expression matching a projection item.
        if let Expr::Column { qualifier: None, name } = expr {
            for (i, p) in proj.iter().enumerate() {
                if p.display_name().eq_ignore_ascii_case(name) {
                    return Ok(Some(LogicalExpr::Column {
                        index: i,
                        name: p.display_name(),
                        data_type: p.data_type()?,
                    }));
                }
            }
        }

        // An expression written out in full that the projection already
        // computes: sort on that column rather than recomputing it.
        for (i, p) in proj.iter().enumerate() {
            let unaliased = match p {
                LogicalExpr::Alias { expr, .. } => expr.as_ref(),
                other => other,
            };
            if unaliased.to_string() == expr.to_string() {
                return Ok(Some(LogicalExpr::Column {
                    index: i,
                    name: p.display_name(),
                    data_type: p.data_type()?,
                }));
            }
        }
        Ok(None)
    }

    fn expand_wildcards(
        &self,
        items: &[SelectItem],
        schema: &SchemaRef,
    ) -> Result<Vec<(Expr, Option<String>)>> {
        let mut out = Vec::new();
        for item in items {
            match item {
                SelectItem::Wildcard => {
                    for f in schema.fields() {
                        out.push((
                            Expr::Column { qualifier: f.qualifier.clone(), name: f.name.clone() },
                            None,
                        ));
                    }
                }
                SelectItem::QualifiedWildcard(q) => {
                    let mut found = false;
                    for f in schema.fields() {
                        if f.qualifier.as_deref().is_some_and(|fq| fq.eq_ignore_ascii_case(q)) {
                            out.push((
                                Expr::Column {
                                    qualifier: f.qualifier.clone(),
                                    name: f.name.clone(),
                                },
                                None,
                            ));
                            found = true;
                        }
                    }
                    if !found {
                        return Err(Error::plan(format!("no table or alias named {q:?} in FROM")));
                    }
                }
                SelectItem::Expr { expr, alias } => out.push((expr.clone(), alias.clone())),
            }
        }
        if out.is_empty() {
            return Err(Error::plan("SELECT list is empty"));
        }
        Ok(out)
    }

    fn plan_table_ref(&self, tref: &TableRef) -> Result<LogicalPlan> {
        match tref {
            TableRef::Table { name, alias } => {
                let schema = self.catalog.schema_of(name).ok_or_else(|| {
                    let known = self.catalog.table_names();
                    Error::plan(format!(
                        "no table named {name:?}; registered tables are [{}]",
                        known.join(", ")
                    ))
                })?;
                // Every scanned table is qualified, by its alias when one was
                // given and otherwise by its own name. Qualifying only aliased
                // tables would make `FROM a JOIN b ON a.k = b.k` fail purely
                // because the user did not bother to write `AS a`.
                let qualifier = alias.clone().unwrap_or_else(|| name.clone());
                let schema = Arc::new(schema.qualified(&qualifier));
                Ok(LogicalPlan::Scan {
                    table: name.clone(),
                    schema,
                    projection: None,
                    filters: Vec::new(),
                })
            }
            TableRef::Join { left, right, join_type, on } => {
                let lplan = self.plan_table_ref(left)?;
                let rplan = self.plan_table_ref(right)?;
                let lschema = lplan.schema();
                let rschema = rplan.schema();
                let combined: SchemaRef = Arc::new(lschema.join(&rschema));

                let (keys, residual) = match on {
                    None => (Vec::new(), None),
                    Some(expr) => {
                        let resolved = self.resolve_expr(expr, &combined, None)?;
                        split_join_keys(resolved, lschema.len())?
                    }
                };

                if *join_type != JoinType::Cross && keys.is_empty() && residual.is_none() {
                    return Err(Error::plan("join ON clause has no conditions"));
                }

                Ok(LogicalPlan::Join {
                    left: Box::new(lplan),
                    right: Box::new(rplan),
                    on: keys,
                    filter: residual,
                    join_type: *join_type,
                    schema: combined,
                })
            }
        }
    }

    /// Resolves an AST expression against a schema.
    ///
    /// `agg` is present when the expression sits above an Aggregate node, in
    /// which case aggregate calls and group-by keys resolve to that node's
    /// output columns rather than to raw input columns.
    pub fn resolve_expr(
        &self,
        expr: &Expr,
        schema: &SchemaRef,
        agg: Option<&AggContext>,
    ) -> Result<LogicalExpr> {
        // Above an aggregation, anything that matches a grouping key or an
        // aggregate resolves to that output column.
        if let Some(ctx) = agg {
            if let Some(e) = ctx.try_resolve(expr, schema, self)? {
                return Ok(e);
            }
        }

        Ok(match expr {
            Expr::Literal(v) => LogicalExpr::Literal(v.clone()),
            Expr::Nested(e) => self.resolve_expr(e, schema, agg)?,

            Expr::Column { qualifier, name } => {
                // Above an aggregation, AggContext::try_resolve has already had
                // its chance to match this column against a grouping key. If it
                // did not, the column is neither grouped nor inside an
                // aggregate and the query is invalid -- resolving it against the
                // *input* schema here would let it through planning and then
                // fail at execution with an incomprehensible index error.
                if agg.is_some() {
                    return Err(Error::plan(format!(
                        "column {name:?} must appear in GROUP BY or be used inside \
                         an aggregate function"
                    )));
                }
                let index = schema.index_of_qualified(qualifier.as_deref(), name)?;
                let f = schema.field(index);
                LogicalExpr::Column { index, name: f.name.clone(), data_type: f.data_type }
            }

            Expr::Binary { left, op, right } => {
                let l = self.resolve_expr(left, schema, agg)?;
                let r = self.resolve_expr(right, schema, agg)?;
                check_binary_types(&l, *op, &r)?;
                LogicalExpr::Binary { left: Box::new(l), op: *op, right: Box::new(r) }
            }

            Expr::Unary { op, expr } => {
                let e = self.resolve_expr(expr, schema, agg)?;
                if *op == UnaryOp::Negate && !e.data_type()?.is_numeric() {
                    return Err(Error::typ(format!(
                        "cannot negate a value of type {}",
                        e.data_type()?
                    )));
                }
                LogicalExpr::Unary { op: *op, expr: Box::new(e) }
            }

            Expr::IsNull { expr, negated } => LogicalExpr::IsNull {
                expr: Box::new(self.resolve_expr(expr, schema, agg)?),
                negated: *negated,
            },

            Expr::Like { expr, pattern, negated } => {
                let e = self.resolve_expr(expr, schema, agg)?;
                let p = self.resolve_expr(pattern, schema, agg)?;
                for (what, t) in
                    [("LIKE operand", e.data_type()?), ("LIKE pattern", p.data_type()?)]
                {
                    if t != DataType::Utf8 && t != DataType::Null {
                        return Err(Error::typ(format!("{what} must be a string, found {t}")));
                    }
                }
                LogicalExpr::Like { expr: Box::new(e), pattern: Box::new(p), negated: *negated }
            }

            Expr::Cast { expr, data_type } => {
                let target = parse_type_name(data_type)?;
                LogicalExpr::Cast {
                    expr: Box::new(self.resolve_expr(expr, schema, agg)?),
                    data_type: target,
                }
            }

            Expr::Case { when_then, else_expr } => {
                let mut wt = Vec::with_capacity(when_then.len());
                for (w, t) in when_then {
                    let wr = self.resolve_expr(w, schema, agg)?;
                    let wt_type = wr.data_type()?;
                    if wt_type != DataType::Boolean && wt_type != DataType::Null {
                        return Err(Error::typ(format!(
                            "CASE WHEN requires a boolean condition, found {wt_type}"
                        )));
                    }
                    wt.push((wr, self.resolve_expr(t, schema, agg)?));
                }
                let e = match else_expr {
                    Some(e) => Some(Box::new(self.resolve_expr(e, schema, agg)?)),
                    None => None,
                };
                let case = LogicalExpr::Case { when_then: wt, else_expr: e };
                // Force the unification check now, so a type error is reported
                // against the query rather than surfacing during execution.
                case.data_type()?;
                case
            }

            // Desugared forms. Rewriting them into primitives here means the
            // optimizer and the execution engine each have one fewer case.
            Expr::InList { expr, list, negated } => {
                let target = self.resolve_expr(expr, schema, agg)?;
                if list.is_empty() {
                    // `x IN ()` is always false; `x NOT IN ()` always true.
                    return Ok(LogicalExpr::Literal(Value::Boolean(*negated)));
                }
                let mut disjuncts = Vec::with_capacity(list.len());
                for item in list {
                    let v = self.resolve_expr(item, schema, agg)?;
                    check_binary_types(&target, BinaryOp::Eq, &v)?;
                    disjuncts.push(LogicalExpr::Binary {
                        left: Box::new(target.clone()),
                        op: BinaryOp::Eq,
                        right: Box::new(v),
                    });
                }
                let any = disjuncts
                    .into_iter()
                    .reduce(|a, b| LogicalExpr::Binary {
                        left: Box::new(a),
                        op: BinaryOp::Or,
                        right: Box::new(b),
                    })
                    .expect("list is non-empty");
                if *negated {
                    LogicalExpr::Unary { op: UnaryOp::Not, expr: Box::new(any) }
                } else {
                    any
                }
            }

            Expr::Between { expr, low, high, negated } => {
                let target = self.resolve_expr(expr, schema, agg)?;
                let lo = self.resolve_expr(low, schema, agg)?;
                let hi = self.resolve_expr(high, schema, agg)?;
                check_binary_types(&target, BinaryOp::GtEq, &lo)?;
                check_binary_types(&target, BinaryOp::LtEq, &hi)?;
                let range = LogicalExpr::Binary {
                    left: Box::new(LogicalExpr::Binary {
                        left: Box::new(target.clone()),
                        op: BinaryOp::GtEq,
                        right: Box::new(lo),
                    }),
                    op: BinaryOp::And,
                    right: Box::new(LogicalExpr::Binary {
                        left: Box::new(target),
                        op: BinaryOp::LtEq,
                        right: Box::new(hi),
                    }),
                };
                if *negated {
                    LogicalExpr::Unary { op: UnaryOp::Not, expr: Box::new(range) }
                } else {
                    range
                }
            }

            Expr::Function { name, .. } => {
                if AggregateFunction::from_name(name).is_some() {
                    return Err(Error::plan(format!(
                        "aggregate function {name} is not allowed here"
                    )));
                }
                return Err(Error::plan(format!("unknown function {name}")));
            }
        })
    }
}

/// The output of an Aggregate node, used to resolve expressions above it.
pub struct AggContext {
    /// The grouping expressions, in output order.
    pub group_exprs: Vec<LogicalExpr>,
    /// The aggregates, following the group columns.
    pub aggregates: Vec<AggregateExpr>,
    /// The aggregate node's output schema.
    pub schema: SchemaRef,
}

impl AggContext {
    /// Resolves `expr` to one of the aggregate node's output columns, if it
    /// matches a grouping key or an aggregate call.
    fn try_resolve(
        &self,
        expr: &Expr,
        input_schema: &SchemaRef,
        planner: &Planner,
    ) -> Result<Option<LogicalExpr>> {
        // An aggregate call maps to its output column.
        if let Expr::Function { name, .. } = expr {
            if AggregateFunction::from_name(name).is_some() {
                let target = aggregate_display_name(expr);
                for (i, a) in self.aggregates.iter().enumerate() {
                    if a.name == target {
                        let index = self.group_exprs.len() + i;
                        return Ok(Some(LogicalExpr::AggregateRef {
                            index,
                            name: a.name.clone(),
                            data_type: a.data_type()?,
                        }));
                    }
                }
                return Err(Error::plan(format!("internal: aggregate {target} was not collected")));
            }
        }

        // A grouping key maps to its output column. Comparison is done on the
        // *resolved* form so that `GROUP BY city` matches `SELECT city` even
        // when one is written with a qualifier and the other is not.
        if let Ok(resolved) = planner.resolve_expr(expr, input_schema, None) {
            for (i, g) in self.group_exprs.iter().enumerate() {
                if *g == resolved {
                    return Ok(Some(LogicalExpr::Column {
                        index: i,
                        name: self.schema.field(i).name.clone(),
                        data_type: self.schema.field(i).data_type,
                    }));
                }
            }
        }
        Ok(None)
    }
}

/// Rewrites a GROUP BY term that names a SELECT alias or an ordinal position
/// into the expression it stands for.
fn resolve_group_reference<'e>(
    expr: &'e Expr,
    projection: &'e [(Expr, Option<String>)],
) -> Result<&'e Expr> {
    // An ordinal: GROUP BY 2.
    if let Expr::Literal(Value::Int64(n)) = expr {
        let idx = *n as usize;
        if idx == 0 || idx > projection.len() {
            return Err(Error::plan(format!(
                "GROUP BY position {n} is out of range for a select list with {} items",
                projection.len()
            )));
        }
        return Ok(&projection[idx - 1].0);
    }

    // An alias defined in the select list.
    if let Expr::Column { qualifier: None, name } = expr {
        for (e, alias) in projection {
            if alias.as_deref().is_some_and(|a| a.eq_ignore_ascii_case(name)) {
                return Ok(e);
            }
        }
    }

    Ok(expr)
}

/// Walks an expression collecting aggregate calls, deduplicating by name.
fn collect_aggregates(
    expr: &Expr,
    schema: &SchemaRef,
    planner: &Planner,
    out: &mut Vec<AggregateExpr>,
) -> Result<()> {
    match expr {
        Expr::Function { name, args, star, distinct } => {
            let Some(func) = AggregateFunction::from_name(name) else {
                return Err(Error::plan(format!("unknown function {name}")));
            };

            if *star && func != AggregateFunction::Count {
                return Err(Error::plan(format!(
                    "{name}(*) is not supported; {name} needs an argument"
                )));
            }
            if !*star && args.len() != 1 {
                return Err(Error::plan(format!(
                    "{name} takes exactly one argument, {} given",
                    args.len()
                )));
            }

            let arg = if *star {
                None
            } else {
                let resolved = planner.resolve_expr(&args[0], schema, None)?;
                // SUM and AVG are only meaningful over numbers; catching it
                // here gives a clear message instead of an execution failure.
                if matches!(func, AggregateFunction::Sum | AggregateFunction::Avg) {
                    let t = resolved.data_type()?;
                    if !t.is_numeric() && t != DataType::Null {
                        return Err(Error::typ(format!(
                            "{name} requires a numeric argument, found {t}"
                        )));
                    }
                }
                Some(resolved)
            };

            let display = aggregate_display_name(expr);
            if !out.iter().any(|a| a.name == display) {
                out.push(AggregateExpr { func, arg, distinct: *distinct, name: display });
            }
            Ok(())
        }
        Expr::Binary { left, right, .. } => {
            collect_aggregates(left, schema, planner, out)?;
            collect_aggregates(right, schema, planner, out)
        }
        Expr::Unary { expr, .. }
        | Expr::IsNull { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::Nested(expr) => collect_aggregates(expr, schema, planner, out),
        Expr::Like { expr, pattern, .. } => {
            collect_aggregates(expr, schema, planner, out)?;
            collect_aggregates(pattern, schema, planner, out)
        }
        Expr::InList { expr, list, .. } => {
            collect_aggregates(expr, schema, planner, out)?;
            for e in list {
                collect_aggregates(e, schema, planner, out)?;
            }
            Ok(())
        }
        Expr::Between { expr, low, high, .. } => {
            collect_aggregates(expr, schema, planner, out)?;
            collect_aggregates(low, schema, planner, out)?;
            collect_aggregates(high, schema, planner, out)
        }
        Expr::Case { when_then, else_expr } => {
            for (w, t) in when_then {
                collect_aggregates(w, schema, planner, out)?;
                collect_aggregates(t, schema, planner, out)?;
            }
            if let Some(e) = else_expr {
                collect_aggregates(e, schema, planner, out)?;
            }
            Ok(())
        }
        Expr::Literal(_) | Expr::Column { .. } => Ok(()),
    }
}

/// The canonical display name of an aggregate call, used both as the output
/// column name and as the key for deduplication.
fn aggregate_display_name(expr: &Expr) -> String {
    match expr {
        Expr::Function { name, args, star, distinct } => {
            let inner = if *star {
                "*".to_string()
            } else {
                args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
            };
            let d = if *distinct { "DISTINCT " } else { "" };
            format!("{name}({d}{inner})")
        }
        other => other.to_string(),
    }
}

/// Splits a join predicate into equi-join key pairs and a residual predicate.
///
/// Only conjuncts of the form `left_col = right_col` become keys; anything
/// else stays in the residual and is evaluated after the join. This is what
/// decides whether the physical planner can use a hash join.
///
/// The result is the equi-key pairs, as (left column, right column), together
/// with whatever predicate could not be turned into a key.
fn split_join_keys(predicate: LogicalExpr, left_width: usize) -> Result<SplitJoinPredicate> {
    let mut conjuncts = Vec::new();
    predicate.split_conjunction(&mut conjuncts);

    let mut keys = Vec::new();
    let mut residual = Vec::new();

    for c in conjuncts {
        if let LogicalExpr::Binary { left, op: BinaryOp::Eq, right } = &c {
            if let (LogicalExpr::Column { index: li, .. }, LogicalExpr::Column { index: ri, .. }) =
                (left.as_ref(), right.as_ref())
            {
                // The sides may be written in either order.
                if *li < left_width && *ri >= left_width {
                    keys.push((*li, *ri - left_width));
                    continue;
                }
                if *ri < left_width && *li >= left_width {
                    keys.push((*ri, *li - left_width));
                    continue;
                }
            }
        }
        residual.push(c);
    }

    Ok((keys, LogicalExpr::conjunction(residual)))
}

/// Equi-join keys plus the predicate that could not be expressed as keys.
type SplitJoinPredicate = (Vec<(usize, usize)>, Option<LogicalExpr>);

/// Type-checks a binary operation.
fn check_binary_types(left: &LogicalExpr, op: BinaryOp, right: &LogicalExpr) -> Result<()> {
    let lt = left.data_type()?;
    let rt = right.data_type()?;
    if lt == DataType::Null || rt == DataType::Null {
        return Ok(());
    }

    if op.is_logical() {
        if lt != DataType::Boolean || rt != DataType::Boolean {
            return Err(Error::typ(format!("{op} requires boolean operands, found {lt} and {rt}")));
        }
        return Ok(());
    }

    if op.is_comparison() {
        // Comparison is allowed between identical types, and between the two
        // numeric types. Comparing a string to a number is almost always a
        // mistake, so it is rejected rather than coerced.
        if lt == rt || (lt.is_numeric() && rt.is_numeric()) {
            return Ok(());
        }
        return Err(Error::typ(format!("cannot compare {lt} with {rt}")));
    }

    // Arithmetic.
    if !lt.is_numeric() || !rt.is_numeric() {
        return Err(Error::typ(format!("{op} requires numeric operands, found {lt} and {rt}")));
    }
    Ok(())
}

/// Maps a SQL type name onto a [`DataType`].
fn parse_type_name(name: &str) -> Result<DataType> {
    Ok(match name.to_ascii_uppercase().as_str() {
        "INT" | "INTEGER" | "BIGINT" | "INT64" => DataType::Int64,
        "FLOAT" | "DOUBLE" | "REAL" | "FLOAT64" => DataType::Float64,
        "TEXT" | "STRING" | "VARCHAR" | "UTF8" => DataType::Utf8,
        "BOOL" | "BOOLEAN" => DataType::Boolean,
        "DATE" | "DATE32" => DataType::Date32,
        other => return Err(Error::plan(format!("unknown type name {other:?} in CAST"))),
    })
}

/// A convenience catalog backed by a map, used by tests and the CLI.
pub struct MapCatalog {
    tables: HashMap<String, SchemaRef>,
}

impl MapCatalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        MapCatalog { tables: HashMap::new() }
    }

    /// Registers a table schema.
    pub fn insert(&mut self, name: impl Into<String>, schema: SchemaRef) {
        self.tables.insert(name.into().to_ascii_lowercase(), schema);
    }
}

impl Default for MapCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl TableProvider for MapCatalog {
    fn schema_of(&self, name: &str) -> Option<SchemaRef> {
        self.tables.get(&name.to_ascii_lowercase()).map(Arc::clone)
    }
    fn table_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tables.keys().cloned().collect();
        v.sort();
        v
    }
}

// Re-exported so downstream modules can name the AST types they handle.
pub use ast::Expr as AstExpr;

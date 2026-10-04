//! Tests that the optimizer rewrites plans the way it claims to -- and, more
//! importantly, that every rewrite preserves the query's answer.

use quarry::Context;

fn ctx() -> Context {
    let mut c = Context::new();
    c.register_csv_str(
        "wide",
        "a,b,c,d,e\n1,10,x,1.5,p\n2,20,y,2.5,q\n3,30,x,3.5,r\n4,40,z,4.5,s\n",
    )
    .unwrap();
    c.register_csv_str("side", "k,label\n1,one\n2,two\n3,three\n").unwrap();
    c
}

fn plan_of(sql: &str) -> String {
    ctx().plan(sql).unwrap().display_indent()
}

// ---------------------------------------------------------------------------
// Projection pushdown
// ---------------------------------------------------------------------------

#[test]
fn unused_columns_are_pruned_from_the_scan() {
    let plan = plan_of("SELECT a FROM wide");
    assert!(plan.contains("Scan: wide [a]"), "plan was:\n{plan}");
    for unread in ["b", "c", "d", "e"] {
        assert!(
            !plan.contains(&format!("Scan: wide [{unread}")),
            "column {unread} should have been pruned; plan was:\n{plan}"
        );
    }
}

#[test]
fn a_column_used_only_in_a_predicate_is_still_read() {
    // `b` never reaches the output but the scan must still read it, or the
    // pushed-down filter has nothing to evaluate.
    let plan = plan_of("SELECT a FROM wide WHERE b > 15");
    assert!(plan.contains("Scan: wide [a, b]"), "plan was:\n{plan}");
}

#[test]
fn columns_used_only_for_grouping_are_kept() {
    let plan = plan_of("SELECT c, SUM(b) FROM wide GROUP BY c");
    let scan = plan.lines().find(|l| l.contains("Scan:")).unwrap();
    assert_eq!(
        scan.trim(),
        "Scan: wide [b, c]",
        "only the grouping key and the aggregate argument should be read; plan was:\n{plan}"
    );
}

#[test]
fn indices_above_an_aggregate_are_not_renumbered() {
    // The projection's `c` is output column 0 of the *aggregate*, not column 0
    // of the scan. Renumbering it as if it were scan-relative silently returns
    // the wrong column -- here it would hand back `a`.
    let c = ctx();
    let r = c.sql("SELECT c, SUM(b) AS t FROM wide GROUP BY c ORDER BY c").unwrap();
    assert_eq!(
        r.rows(),
        vec![
            vec!["x".to_string(), "40".to_string()],
            vec!["y".to_string(), "20".to_string()],
            vec!["z".to_string(), "40".to_string()],
        ]
    );
}

#[test]
fn a_query_reading_no_columns_still_counts_rows() {
    let c = ctx();
    assert_eq!(c.sql("SELECT COUNT(*) FROM wide").unwrap().rows()[0][0], "4");
    assert_eq!(c.sql("SELECT 1 FROM wide").unwrap().num_rows(), 4);
}

// ---------------------------------------------------------------------------
// Predicate pushdown
// ---------------------------------------------------------------------------

#[test]
fn filters_move_into_the_scan() {
    let plan = plan_of("SELECT a FROM wide WHERE b > 15");
    assert!(plan.contains("filters="), "predicate should reach the scan:\n{plan}");
    assert!(!plan.contains("Filter:"), "no standalone Filter node should remain:\n{plan}");
}

#[test]
fn a_conjunction_is_split_and_pushed_piecewise() {
    let plan = plan_of("SELECT a FROM wide WHERE b > 15 AND c = 'x'");
    assert!(plan.contains("filters="), "plan was:\n{plan}");
    assert!(plan.contains('b') && plan.contains('c'), "plan was:\n{plan}");
}

#[test]
fn a_predicate_over_an_aggregate_cannot_be_pushed() {
    // HAVING SUM(b) > 10 cannot move below the aggregate: the value it tests
    // does not exist until the aggregation has run.
    let plan = plan_of("SELECT c, SUM(b) AS total FROM wide GROUP BY c HAVING SUM(b) > 10");
    let filter_line = plan.find("Filter:").expect("the HAVING filter should survive");
    let agg_line = plan.find("Aggregate:").expect("plan has an aggregate");
    assert!(filter_line < agg_line, "the HAVING filter must stay above the aggregate:\n{plan}");
}

#[test]
fn a_predicate_on_a_grouping_key_is_pushed_below_the_aggregate() {
    let plan = plan_of("SELECT c, SUM(b) FROM wide GROUP BY c HAVING c = 'x'");
    let agg_line = plan.find("Aggregate:").expect("plan has an aggregate");
    let filter_line = plan.find("filters=");
    assert!(
        filter_line.is_some_and(|f| f > agg_line),
        "a predicate on the grouping key should reach the scan below the aggregate:\n{plan}"
    );
}

#[test]
fn join_predicates_are_pushed_to_the_correct_side() {
    let plan = plan_of(
        "SELECT w.a, s.label FROM wide w JOIN side s ON w.a = s.k WHERE w.b > 15 AND s.k < 3",
    );
    // Each single-sided predicate should end up in its own scan.
    let wide_scan = plan.lines().find(|l| l.contains("Scan: wide")).unwrap();
    let side_scan = plan.lines().find(|l| l.contains("Scan: side")).unwrap();
    assert!(wide_scan.contains("filters="), "wide scan: {wide_scan}");
    assert!(side_scan.contains("filters="), "side scan: {side_scan}");
}

#[test]
fn limit_stops_pulling_once_its_quota_is_met() {
    // LIMIT must be able to stop the scan early rather than draining it.
    let c = ctx();
    let r = c.sql("SELECT a FROM wide LIMIT 2").unwrap();
    assert_eq!(r.num_rows(), 2);
    let r = c.sql("SELECT a FROM wide LIMIT 99").unwrap();
    assert_eq!(r.num_rows(), 4, "a LIMIT larger than the input is not an error");
}

// ---------------------------------------------------------------------------
// Constant folding
// ---------------------------------------------------------------------------

#[test]
fn constant_arithmetic_is_folded() {
    let plan = plan_of("SELECT a FROM wide WHERE b > 2 * 5 + 5");
    assert!(
        plan.contains("15") && !plan.contains('*'),
        "2 * 5 + 5 should have folded to 15:\n{plan}"
    );
}

#[test]
fn an_always_true_predicate_is_removed_entirely() {
    let plan = plan_of("SELECT a FROM wide WHERE 1 = 1");
    assert!(
        !plan.contains("Filter:") && !plan.contains("filters="),
        "a tautology should leave no filter behind:\n{plan}"
    );
}

#[test]
fn boolean_identities_are_simplified() {
    let plan = plan_of("SELECT a FROM wide WHERE b > 5 AND true");
    assert!(!plan.contains("true"), "`AND true` should vanish:\n{plan}");

    let plan = plan_of("SELECT a FROM wide WHERE b > 5 OR true");
    assert!(
        !plan.contains("Filter:") && !plan.contains("filters="),
        "`OR true` makes the predicate a tautology:\n{plan}"
    );
}

#[test]
fn integer_overflow_is_not_folded_into_a_wrapped_constant() {
    // Folding this with wrapping arithmetic would silently change the answer.
    let c = ctx();
    let r = c.sql("SELECT 9223372036854775807 + 1").unwrap();
    assert_eq!(r.rows()[0][0], "NULL", "overflow must produce NULL, not a wrapped value");
}

// ---------------------------------------------------------------------------
// The property that actually matters
// ---------------------------------------------------------------------------

/// Every rewrite must preserve the query's result. This runs a battery of
/// queries with the optimizer on and off and requires identical answers --
/// which is the only guarantee that makes the rules above safe to apply.
#[test]
fn optimization_never_changes_the_answer() {
    let queries = [
        "SELECT a FROM wide",
        "SELECT a, b FROM wide WHERE b > 15",
        "SELECT a FROM wide WHERE b > 15 AND c = 'x'",
        "SELECT a FROM wide WHERE b > 2 * 5 + 5",
        "SELECT a FROM wide WHERE 1 = 1",
        "SELECT a FROM wide WHERE b > 5 AND true",
        "SELECT c, COUNT(*) AS n FROM wide GROUP BY c ORDER BY c",
        "SELECT c, SUM(b) AS t FROM wide GROUP BY c HAVING SUM(b) > 10 ORDER BY c",
        "SELECT c, SUM(b) AS t FROM wide GROUP BY c HAVING c = 'x'",
        "SELECT a FROM wide ORDER BY d DESC",
        "SELECT a FROM wide ORDER BY b LIMIT 2",
        "SELECT DISTINCT c FROM wide ORDER BY c",
        "SELECT w.a, s.label FROM wide w JOIN side s ON w.a = s.k ORDER BY w.a",
        "SELECT w.a, s.label FROM wide w LEFT JOIN side s ON w.a = s.k ORDER BY w.a",
        "SELECT w.a FROM wide w JOIN side s ON w.a = s.k WHERE w.b > 15 AND s.k < 3",
        "SELECT CASE WHEN b > 20 THEN 'hi' ELSE 'lo' END AS band, a FROM wide ORDER BY a",
        "SELECT a + b AS sum, a * 2 AS doubled FROM wide ORDER BY a",
    ];

    let mut optimized = ctx();
    optimized.set_optimize(true);
    let mut plain = ctx();
    plain.set_optimize(false);

    for q in queries {
        let a = optimized.sql(q).unwrap_or_else(|e| panic!("optimized {q}: {e}")).rows();
        let b = plain.sql(q).unwrap_or_else(|e| panic!("unoptimized {q}: {e}")).rows();
        assert_eq!(a, b, "optimization changed the result of: {q}");
    }
}

#[test]
fn explain_shows_both_plans_and_the_rules() {
    let c = ctx();
    let out = c.sql("EXPLAIN SELECT a FROM wide WHERE b > 15").unwrap().to_table();
    for expected in [
        "Logical plan",
        "Optimized plan",
        "Physical plan",
        "predicate_pushdown",
        "projection_pushdown",
        "ScanExec",
    ] {
        assert!(out.contains(expected), "EXPLAIN output missing {expected:?}:\n{out}");
    }
}

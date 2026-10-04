//! End-to-end SQL semantics: parse, plan, optimize, execute, check answers.

use quarry::{Context, Value};

/// A small fact table with deliberate NULLs and duplicates.
fn ctx() -> Context {
    let mut c = Context::new();
    c.register_csv_str(
        "trips",
        "\
id,city,fare,rider,tip,day
1,Austin,12.50,ann,2.0,2024-01-05
2,Dallas,8.00,bob,,2024-01-06
3,Austin,20.00,cat,4.5,2024-01-06
4,Dallas,30.00,dan,0.0,2024-01-07
5,Austin,5.25,ann,1.0,2024-01-08
6,Houston,45.00,eve,9.0,2024-01-08
7,Dallas,,frank,3.0,2024-01-09
",
    )
    .unwrap();
    c.register_csv_str(
        "cities",
        "name,state,pop\nAustin,TX,961855\nDallas,TX,1304379\nSeattle,WA,749256\n",
    )
    .unwrap();
    c
}

/// Runs a query and returns its rows as strings.
fn rows(sql: &str) -> Vec<Vec<String>> {
    ctx().sql(sql).unwrap_or_else(|e| panic!("{sql}\n  failed: {e}")).rows()
}

fn one(sql: &str) -> String {
    let r = rows(sql);
    assert_eq!(r.len(), 1, "expected exactly one row from: {sql}\ngot {r:?}");
    assert_eq!(r[0].len(), 1, "expected exactly one column from: {sql}");
    r[0][0].clone()
}

// ---------------------------------------------------------------------------
// Projection and literals
// ---------------------------------------------------------------------------

#[test]
fn select_without_from_evaluates_literals() {
    assert_eq!(one("SELECT 1 + 2 * 3"), "7");
    assert_eq!(one("SELECT 10 / 4"), "2.5", "division must not truncate");
    assert_eq!(one("SELECT 'hello'"), "hello");
    assert_eq!(one("SELECT NOT (1 = 2)"), "true");
}

#[test]
fn wildcard_expands_to_all_columns() {
    let r = rows("SELECT * FROM trips");
    assert_eq!(r.len(), 7);
    assert_eq!(r[0].len(), 6);
}

#[test]
fn aliases_rename_output_columns() {
    let ctx = ctx();
    let r = ctx.sql("SELECT city AS c, fare AS f FROM trips LIMIT 1").unwrap();
    let names: Vec<String> = r.schema().fields().iter().map(|f| f.name.clone()).collect();
    assert_eq!(names, vec!["c", "f"]);
}

#[test]
fn arithmetic_propagates_nulls() {
    // Trip 7 has a NULL fare; NULL + anything is NULL.
    let r = rows("SELECT id, fare + tip AS total FROM trips WHERE id = 7");
    assert_eq!(r[0][1], "NULL");
}

#[test]
fn integer_arithmetic_stays_integral() {
    assert_eq!(one("SELECT 7 - 2"), "5");
    assert_eq!(one("SELECT 7 % 2"), "1");
    // ...but division is always float.
    assert_eq!(one("SELECT 7 / 2"), "3.5");
}

#[test]
fn division_by_zero_yields_null_not_a_crash() {
    assert_eq!(one("SELECT 1 / 0"), "NULL");
    assert_eq!(one("SELECT 1.0 / 0.0"), "NULL");
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

#[test]
fn where_filters_rows() {
    let r = rows("SELECT id FROM trips WHERE fare > 15 ORDER BY id");
    let ids: Vec<&str> = r.iter().map(|x| x[0].as_str()).collect();
    assert_eq!(ids, vec!["3", "4", "6"]);
}

#[test]
fn where_excludes_null_predicates() {
    // Trip 7's fare is NULL, so `fare > 0` is NULL, which is not TRUE and
    // therefore does not pass the filter.
    let r = rows("SELECT id FROM trips WHERE fare > 0");
    assert!(!r.iter().any(|x| x[0] == "7"), "a NULL predicate must not pass WHERE");
}

#[test]
fn is_null_finds_missing_values() {
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE fare IS NULL"), "1");
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE fare IS NOT NULL"), "6");
    // A NULL that came from the CSV's empty field, not a literal.
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE tip IS NULL"), "1");
}

#[test]
fn three_valued_logic_short_circuits() {
    // FALSE AND NULL is FALSE, not NULL -- so the row is excluded either way,
    // but `false AND (fare > 1)` must not error on the NULL row.
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE false AND fare > 1"), "0");
    // TRUE OR NULL is TRUE, so every row passes including the NULL-fare one.
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE true OR fare > 1"), "7");
}

#[test]
fn in_list_and_between_and_like() {
    let r = rows("SELECT id FROM trips WHERE city IN ('Austin', 'Houston') ORDER BY id");
    assert_eq!(r.len(), 4);

    let r = rows("SELECT id FROM trips WHERE fare BETWEEN 10 AND 25 ORDER BY id");
    let ids: Vec<&str> = r.iter().map(|x| x[0].as_str()).collect();
    assert_eq!(ids, vec!["1", "3"]);

    let r = rows("SELECT id FROM trips WHERE city NOT IN ('Austin') ORDER BY id");
    assert_eq!(r.len(), 4);

    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE city LIKE 'A%'"), "3");
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE city LIKE '%as%'"), "3");
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE rider LIKE '_nn'"), "2");
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE city NOT LIKE 'A%'"), "4");
}

#[test]
fn like_handles_adjacent_and_trailing_wildcards() {
    assert_eq!(one("SELECT 'abc' LIKE '%%c'"), "true");
    assert_eq!(one("SELECT 'abc' LIKE 'a%'"), "true");
    assert_eq!(one("SELECT 'abc' LIKE '%'"), "true");
    assert_eq!(one("SELECT 'abc' LIKE 'a_c'"), "true");
    assert_eq!(one("SELECT 'abc' LIKE 'a_d'"), "false");
    assert_eq!(one("SELECT 'aaa' LIKE '%a%a%'"), "true");
    // The classic backtracking case: a greedy `%` must give characters back.
    assert_eq!(one("SELECT 'aaaab' LIKE '%ab'"), "true");
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

#[test]
fn global_aggregates() {
    assert_eq!(one("SELECT COUNT(*) FROM trips"), "7");
    // COUNT(expr) skips NULLs, COUNT(*) does not -- the distinction matters.
    assert_eq!(one("SELECT COUNT(fare) FROM trips"), "6");
    assert_eq!(one("SELECT MIN(fare) FROM trips"), "5.25");
    assert_eq!(one("SELECT MAX(fare) FROM trips"), "45.0");
    assert_eq!(one("SELECT SUM(fare) FROM trips"), "120.75");
}

#[test]
fn avg_is_float_even_over_integers() {
    // AVG over integers that do not divide evenly must not truncate.
    let mut c = Context::new();
    c.register_csv_str("n", "x\n1\n2\n").unwrap();
    assert_eq!(c.sql("SELECT AVG(x) FROM n").unwrap().rows()[0][0], "1.5");
}

#[test]
fn group_by_partitions_rows() {
    let r = rows("SELECT city, COUNT(*) AS n FROM trips GROUP BY city ORDER BY city");
    assert_eq!(
        r,
        vec![
            vec!["Austin".to_string(), "3".to_string()],
            vec!["Dallas".to_string(), "3".to_string()],
            vec!["Houston".to_string(), "1".to_string()],
        ]
    );
}

#[test]
fn group_by_with_multiple_aggregates() {
    let r = rows(
        "SELECT city, COUNT(*) AS n, SUM(fare) AS total, MAX(fare) AS biggest
         FROM trips GROUP BY city ORDER BY city",
    );
    assert_eq!(r[0], vec!["Austin", "3", "37.75", "20.0"]);
    // Dallas has a NULL fare: SUM and MAX skip it, COUNT(*) does not.
    assert_eq!(r[1], vec!["Dallas", "3", "38.0", "30.0"]);
}

#[test]
fn sum_of_no_rows_is_null_not_zero() {
    assert_eq!(one("SELECT SUM(fare) FROM trips WHERE city = 'Nowhere'"), "NULL");
    // COUNT, by contrast, is 0.
    assert_eq!(one("SELECT COUNT(*) FROM trips WHERE city = 'Nowhere'"), "0");
}

#[test]
fn having_filters_groups() {
    let r = rows(
        "SELECT city, COUNT(*) AS n FROM trips GROUP BY city HAVING COUNT(*) > 1 ORDER BY city",
    );
    assert_eq!(r.len(), 2, "Houston has only one trip and must be excluded");
}

#[test]
fn count_distinct() {
    assert_eq!(one("SELECT COUNT(DISTINCT city) FROM trips"), "3");
    assert_eq!(one("SELECT COUNT(DISTINCT rider) FROM trips"), "6");
}

#[test]
fn aggregate_in_an_expression() {
    // The planner must lift SUM() into the aggregate node and leave the
    // surrounding arithmetic as a scalar projection above it.
    assert_eq!(one("SELECT SUM(fare) / COUNT(fare) FROM trips"), "20.125");
}

#[test]
fn selecting_a_non_grouped_column_is_rejected() {
    let err =
        ctx().sql("SELECT city, rider, COUNT(*) FROM trips GROUP BY city").unwrap_err().to_string();
    assert!(err.contains("GROUP BY"), "error should explain the GROUP BY requirement, got: {err}");
}

// ---------------------------------------------------------------------------
// Sorting, limit, distinct
// ---------------------------------------------------------------------------

#[test]
fn order_by_ascending_and_descending() {
    let r = rows("SELECT id FROM trips WHERE fare IS NOT NULL ORDER BY fare DESC");
    let ids: Vec<&str> = r.iter().map(|x| x[0].as_str()).collect();
    assert_eq!(ids, vec!["6", "4", "3", "1", "2", "5"]);
}

#[test]
fn order_by_sorts_nulls_last_ascending() {
    let r = rows("SELECT fare FROM trips ORDER BY fare ASC");
    assert_eq!(r.last().unwrap()[0], "NULL", "NULLs sort last in ascending order");
    let r = rows("SELECT fare FROM trips ORDER BY fare DESC");
    assert_eq!(r[0][0], "NULL", "NULLs sort first in descending order");
}

#[test]
fn order_by_multiple_keys_and_ordinal() {
    let r = rows("SELECT city, rider FROM trips ORDER BY city ASC, rider DESC");
    assert_eq!(r[0], vec!["Austin", "cat"]);
    assert_eq!(r[1], vec!["Austin", "ann"]);

    let by_ordinal = rows("SELECT city, rider FROM trips ORDER BY 1 ASC, 2 DESC");
    assert_eq!(r, by_ordinal, "ORDER BY 1 must mean the first select item");
}

#[test]
fn order_by_an_alias() {
    let r = rows("SELECT city, COUNT(*) AS n FROM trips GROUP BY city ORDER BY n DESC, city ASC");
    assert_eq!(r[0][1], "3");
    assert_eq!(r[2][0], "Houston");
}

#[test]
fn limit_and_offset() {
    let all = rows("SELECT id FROM trips ORDER BY id");
    let limited = rows("SELECT id FROM trips ORDER BY id LIMIT 3");
    assert_eq!(limited.len(), 3);
    assert_eq!(limited[0], all[0]);

    let offset = rows("SELECT id FROM trips ORDER BY id LIMIT 2 OFFSET 2");
    assert_eq!(offset, vec![all[2].clone(), all[3].clone()]);

    assert!(rows("SELECT id FROM trips LIMIT 0").is_empty());
}

#[test]
fn distinct_removes_duplicate_rows() {
    let r = rows("SELECT DISTINCT city FROM trips ORDER BY city");
    assert_eq!(r.len(), 3);
    // (Austin, ann) appears twice -- trips 1 and 5 -- so six pairs survive.
    let r = rows("SELECT DISTINCT city, rider FROM trips");
    assert_eq!(r.len(), 6);
}

// ---------------------------------------------------------------------------
// Joins
// ---------------------------------------------------------------------------

#[test]
fn inner_join_keeps_only_matches() {
    let r =
        rows("SELECT t.id, c.state FROM trips t JOIN cities c ON t.city = c.name ORDER BY t.id");
    // Houston is absent from `cities`, so trip 6 drops out.
    assert_eq!(r.len(), 6);
    assert!(r.iter().all(|x| x[1] == "TX"));
}

#[test]
fn left_join_null_extends_unmatched_rows() {
    let r = rows(
        "SELECT t.id, c.state FROM trips t LEFT JOIN cities c ON t.city = c.name ORDER BY t.id",
    );
    assert_eq!(r.len(), 7, "a LEFT JOIN must keep every left row");
    let houston = r.iter().find(|x| x[0] == "6").unwrap();
    assert_eq!(houston[1], "NULL");
}

#[test]
fn right_join_keeps_unmatched_right_rows() {
    let r = rows("SELECT t.id, c.name FROM trips t RIGHT JOIN cities c ON t.city = c.name");
    // Seattle has no trips but must still appear.
    assert!(r.iter().any(|x| x[1] == "Seattle" && x[0] == "NULL"));
}

#[test]
fn cross_join_is_the_cartesian_product() {
    let r = rows("SELECT t.id, c.name FROM trips t CROSS JOIN cities c");
    assert_eq!(r.len(), 7 * 3);
}

#[test]
fn join_with_a_non_equi_residual_predicate() {
    // `pop > 1000000` is not an equi-key, so it must be applied after the hash
    // join rather than silently dropped.
    let r = rows("SELECT t.id FROM trips t JOIN cities c ON t.city = c.name AND c.pop > 1000000");
    assert_eq!(r.len(), 3, "only Dallas exceeds a million");
}

#[test]
fn join_keys_with_nulls_never_match() {
    let mut c = Context::new();
    c.register_csv_str("a", "k,v\n1,x\n,y\n").unwrap();
    c.register_csv_str("b", "k,w\n1,p\n,q\n").unwrap();
    let r = c.sql("SELECT a.v, b.w FROM a JOIN b ON a.k = b.k").unwrap();
    assert_eq!(r.num_rows(), 1, "NULL must not equal NULL in a join key");
}

#[test]
fn aggregate_over_a_join() {
    let r = rows(
        "SELECT c.state, COUNT(*) AS n, SUM(t.fare) AS total
         FROM trips t JOIN cities c ON t.city = c.name
         GROUP BY c.state",
    );
    assert_eq!(r.len(), 1);
    assert_eq!(r[0][0], "TX");
    assert_eq!(r[0][1], "6");
}

// ---------------------------------------------------------------------------
// CASE, CAST, dates
// ---------------------------------------------------------------------------

#[test]
fn case_when_picks_the_first_matching_branch() {
    let r = rows(
        "SELECT id,
                CASE WHEN fare > 25 THEN 'high' WHEN fare > 10 THEN 'mid' ELSE 'low' END AS band
         FROM trips WHERE fare IS NOT NULL ORDER BY id",
    );
    assert_eq!(r[0], vec!["1", "mid"]);
    assert_eq!(r[1], vec!["2", "low"]);
    assert_eq!(r[2], vec!["3", "mid"]);
    assert_eq!(r[3], vec!["4", "high"]);
    assert_eq!(r[4], vec!["5", "low"]);
    assert_eq!(r[5], vec!["6", "high"]);
}

#[test]
fn case_without_else_yields_null() {
    assert_eq!(one("SELECT CASE WHEN 1 = 2 THEN 'x' END"), "NULL");
}

#[test]
fn cast_converts_and_fails_softly() {
    assert_eq!(one("SELECT CAST('42' AS INT)"), "42");
    assert_eq!(one("SELECT CAST(42 AS TEXT)"), "42");
    assert_eq!(one("SELECT CAST(3.9 AS INT)"), "3");
    // A value that cannot convert becomes NULL rather than failing the query.
    assert_eq!(one("SELECT CAST('banana' AS INT)"), "NULL");
}

#[test]
fn dates_are_inferred_and_comparable() {
    let ctx = ctx();
    let schema = ctx.schema_of("trips").unwrap();
    let day = schema.field(schema.index_of("day").unwrap());
    assert_eq!(day.data_type, quarry::DataType::Date32, "the day column should infer as a date");

    let r = rows("SELECT id FROM trips WHERE day > CAST('2024-01-07' AS DATE) ORDER BY id");
    let ids: Vec<&str> = r.iter().map(|x| x[0].as_str()).collect();
    assert_eq!(ids, vec!["5", "6", "7"]);
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn unknown_table_and_column_errors_are_helpful() {
    let err = ctx().sql("SELECT * FROM nope").unwrap_err().to_string();
    assert!(
        err.contains("nope") && err.contains("trips"),
        "error should name both the missing table and the available ones: {err}"
    );

    let err = ctx().sql("SELECT nope FROM trips").unwrap_err().to_string();
    assert!(
        err.contains("nope") && err.contains("city"),
        "error should list available columns: {err}"
    );
}

#[test]
fn type_errors_are_caught_at_planning_time() {
    for (sql, want) in [
        ("SELECT city + 1 FROM trips", "numeric"),
        ("SELECT * FROM trips WHERE city > 1", "compare"),
        ("SELECT * FROM trips WHERE fare", "boolean"),
        ("SELECT SUM(city) FROM trips", "numeric"),
    ] {
        let err = ctx().sql(sql).unwrap_err().to_string();
        assert!(
            err.to_lowercase().contains(want),
            "query {sql:?} should report a {want} problem, got: {err}"
        );
    }
}

#[test]
fn syntax_errors_report_a_position() {
    let err = ctx().sql("SELECT FROM").unwrap_err().to_string();
    assert!(
        err.contains("line") && err.contains("column"),
        "parse errors should carry a location: {err}"
    );
}

#[test]
fn value_accessor_returns_typed_values() {
    let c = ctx();
    let r = c.sql("SELECT COUNT(*) FROM trips").unwrap();
    assert_eq!(r.value(0, 0).unwrap(), Value::Int64(7));
    assert!(r.value(99, 0).is_err(), "out-of-range rows must error, not panic");
}

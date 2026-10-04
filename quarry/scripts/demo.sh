#!/usr/bin/env bash
# Runs a tour of quarry's features against the sample data.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./target/release/quarry
[[ -x "$BIN" ]] || cargo build --release

run() {
  echo
  echo "--- $1"
  shift
  "$BIN" -t trips=data/trips.csv -t cities=data/cities.csv -c "$1" 2>/dev/null
}

run "aggregate with GROUP BY and ORDER BY" \
  "SELECT city, COUNT(*) AS trips, AVG(fare) AS avg_fare, MAX(fare) AS max_fare
   FROM trips GROUP BY city ORDER BY trips DESC"

run "join across two tables, then aggregate" \
  "SELECT c.state, COUNT(*) AS n, AVG(t.fare) AS avg_fare
   FROM trips t JOIN cities c ON t.city = c.name
   WHERE t.fare > 10 GROUP BY c.state ORDER BY n DESC"

run "HAVING filters groups, not rows" \
  "SELECT rider, COUNT(*) AS n FROM trips
   GROUP BY rider HAVING COUNT(*) > 95 ORDER BY n DESC LIMIT 5"

run "CASE expressions and computed columns" \
  "SELECT city,
          CASE WHEN fare > 40 THEN 'high' WHEN fare > 15 THEN 'mid' ELSE 'low' END AS band,
          COUNT(*) AS n
   FROM trips GROUP BY city, band ORDER BY city, n DESC LIMIT 9"

run "NULL handling: tips are missing for some trips" \
  "SELECT COUNT(*) AS all_trips, COUNT(tip) AS with_tip, AVG(tip) AS avg_tip FROM trips"

run "LEFT JOIN keeps unmatched rows" \
  "SELECT c.name, COUNT(t.trip_id) AS n
   FROM cities c LEFT JOIN trips t ON c.name = t.city
   GROUP BY c.name ORDER BY n DESC"

run "dates are inferred and comparable" \
  "SELECT day, COUNT(*) AS n FROM trips
   WHERE day BETWEEN CAST('2024-03-01' AS DATE) AND CAST('2024-03-05' AS DATE)
   GROUP BY day ORDER BY day"

run "EXPLAIN shows what the optimizer did" \
  "EXPLAIN SELECT trip_id FROM trips WHERE fare > 90 AND city = 'Austin'"

echo
echo "done."

#!/usr/bin/env bash
# Brings up a three-node raftkv cluster on loopback, exercises it, kills the
# leader, and shows that acknowledged writes survived the failover.
set -euo pipefail

cd "$(dirname "$0")/.."

PEERS='1=http://127.0.0.1:9001,2=http://127.0.0.1:9002,3=http://127.0.0.1:9003'
ENDPOINTS='http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003'
DATA=${DATA_DIR:-./data}
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT

if [[ ! -x ./bin/raftkvd ]]; then
  echo "building..."
  make build >/dev/null
fi

rm -rf "$DATA"
mkdir -p "$DATA"

echo "==> starting a 3-node cluster"
for id in 1 2 3; do
  ./bin/raftkvd -id "$id" -peers "$PEERS" -listen ":900$id" \
    -data "$DATA/$id" -log-level error &
  PIDS+=($!)
done

echo "==> waiting for a leader"
for _ in $(seq 1 50); do
  if ./bin/raftkvctl -endpoints "$ENDPOINTS" status 2>/dev/null | grep -q leader; then
    break
  fi
  sleep 0.2
done
./bin/raftkvctl -endpoints "$ENDPOINTS" status

echo
echo "==> writing 5 keys"
for i in 1 2 3 4 5; do
  ./bin/raftkvctl -endpoints "$ENDPOINTS" put "key$i" "value$i" >/dev/null
done
./bin/raftkvctl -endpoints "$ENDPOINTS" keys

echo
echo "==> linearizable read of key3"
./bin/raftkvctl -endpoints "$ENDPOINTS" get key3

echo
echo "==> compare-and-swap: wrong expectation must fail"
./bin/raftkvctl -endpoints "$ENDPOINTS" cas key3 newvalue wrongvalue || echo "   (rejected, as expected)"
echo "==> compare-and-swap: correct expectation must succeed"
./bin/raftkvctl -endpoints "$ENDPOINTS" cas key3 newvalue value3
./bin/raftkvctl -endpoints "$ENDPOINTS" get key3

echo
echo "==> killing the leader"
# Fields are tab-separated, so match on the role column with awk rather than a
# grep pattern that assumes spaces.
LEADER_ID=$(./bin/raftkvctl -endpoints "$ENDPOINTS" status | awk '$3 == "leader" {print $2; exit}')
if [[ -z "$LEADER_ID" ]]; then
  echo "could not determine the leader" >&2
  exit 1
fi
echo "   leader was node $LEADER_ID"
kill "${PIDS[$((LEADER_ID-1))]}" 2>/dev/null || true

echo "==> waiting for a new leader"
sleep 3
./bin/raftkvctl -endpoints "$ENDPOINTS" status

echo
echo "==> every acknowledged write survived the failover:"
for i in 1 2 4 5; do
  printf "   key%s = %s\n" "$i" "$(./bin/raftkvctl -endpoints "$ENDPOINTS" get "key$i")"
done
printf "   key3 = %s\n" "$(./bin/raftkvctl -endpoints "$ENDPOINTS" get key3)"

echo
echo "==> the cluster still accepts new writes"
./bin/raftkvctl -endpoints "$ENDPOINTS" put after-failover yes
./bin/raftkvctl -endpoints "$ENDPOINTS" get after-failover
echo
echo "done."

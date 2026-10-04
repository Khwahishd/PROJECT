# raftkv

A distributed, strongly-consistent key–value store built on a from-scratch implementation of the
[Raft consensus algorithm](https://raft.github.io/raft.pdf) — with a deterministic fault-injection
simulator that verifies Raft's safety properties after *every state transition*, across thousands of
randomized failure histories.

```
go test ./...        # unit + integration + 200 randomized failure histories
make chaos           # 1,500 seeds of partitions, crashes, drops and reordering
```

[![CI](https://github.com/Khwahishd/raftkv/actions/workflows/ci.yml/badge.svg)](https://github.com/Khwahishd/raftkv/actions/workflows/ci.yml)
![Go](https://img.shields.io/badge/go-1.24-00ADD8)
![Coverage](https://img.shields.io/badge/coverage-82.6%25-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-0-blue)

---

## Why this exists

Most "I implemented Raft" projects demonstrate the happy path: three nodes start, one gets elected,
writes replicate. That is the easy 20%. The hard part — and the entire reason consensus algorithms
are difficult — is what happens when the network partitions mid-election, when a leader is deposed
between persisting an entry and acknowledging it, or when a follower's disk still holds a log suffix
that the cluster has since overwritten.

Those bugs are invisible on the happy path, non-deterministic under normal testing, and corrupt user
data silently when they do fire.

So the centerpiece of this project is not the algorithm — it is the **testing method**: a
deterministic simulator that runs an entire cluster inside one goroutine, with a seeded PRNG
controlling every message delay, drop, duplication, partition and crash, and an invariant checker
that asserts Raft's five safety properties after every single tick.

**It found three real bugs in this implementation.** They are documented below, because the bugs
found are more interesting than the code that survived.

---

## Architecture

The consensus core is a **pure state machine**. It owns no goroutines, no locks, no clock, and no
sockets. Time advances only via `Tick()`, input arrives only via `Step()`, and every side effect
leaves through `Ready()`:

```
                   ┌─────────────────────────────────────────┐
   Tick() ────────▶│                                         │
   Step(msg) ─────▶│   raft.Node  (pure, deterministic)      │──▶ Ready{
   Propose(cmd) ──▶│   no goroutines · no locks · no I/O     │      HardState  → fsync
                   └─────────────────────────────────────────┘      Entries    → WAL
                                                                    Messages   → network
                                                                    Committed  → state machine
                                                                    ReadStates → waiting readers
                                                                  }
```

This is the design etcd/raft popularized, and it is what makes the simulator possible: *the same
code* runs in production and under simulation, with only the driver differing.

| Layer | Package | Responsibility |
|---|---|---|
| Consensus | `internal/raft` | Elections, replication, commit rules, snapshots, membership, read-index |
| Simulation | `internal/raftsim` | Deterministic cluster + network + fault injection + invariant checker |
| Durability | `internal/storage` | CRC-checked WAL, atomic hard-state writes, snapshot files |
| State machine | `internal/kv` | Replicated map with CAS and client-request deduplication |
| Transport | `internal/transport` | HTTP peer messaging, intentionally lossy |
| Orchestration | `internal/server` | Single-goroutine driver enforcing the `Ready` contract |

**Zero third-party dependencies.** Everything — the binary wire codec, the WAL framing, the HTTP
layer — is standard library only. For a project about determinism, that is a feature: no vendored
map iteration or background goroutine can perturb a replay.

### The `Ready` contract

The ordering is not advisory. Violating it breaks safety:

1. Persist `HardState` and `Entries` durably.
2. **Only then** send `Messages`.
3. Apply `CommittedEntries`; release `ReadStates`.
4. Call `Advance`.

Sending before persisting lets a node vote, crash, restart with no record of the vote, and vote
again in the same term — electing two leaders.

---

## Features

**Consensus** — leader election with randomized timeouts · log replication with pipelining ·
the §5.4.2 current-term commit restriction (the Figure-8 case) · log compaction and snapshot
transfer · single-node-at-a-time membership changes.

**Beyond the basic paper** — **pre-vote** (§9.6): a partitioned node that rejoins cannot force a
term bump and depose a healthy leader · **check-quorum**: a leader that loses contact with a
majority steps down instead of serving stale reads indefinitely · **read-index** (§6.4):
linearizable reads confirmed by a heartbeat quorum, with no log write · **conflict-term backtracking**
(§5.3): a rejected append backs up a whole term per round trip, not one index.

**Production concerns** — CRC-checked WAL that discards torn tails · atomic `fsync`+rename for the
vote record · client-request deduplication so a retried write is never applied twice · leader
redirection via HTTP 421 · graceful shutdown that lets in-flight writes commit.

---

## The simulator

`internal/raftsim` runs a whole cluster — real `raft.Node` instances — inside a single goroutine,
driven by one seeded PRNG:

```go
s, _ := raftsim.New(raftsim.Options{
    Seed: 42, Peers: []raft.NodeID{1, 2, 3, 4, 5},
    DropRate: 0.12,        // lose 12% of messages
    DuplicateRate: 0.08,   // deliver 8% twice
    MaxDelay: 4,           // reorder by up to 4 ticks
    SnapshotThreshold: 15, // force compaction and snapshot transfer
})
checker := raftsim.NewChecker(s)

s.Partition([]raft.NodeID{1, 2}, []raft.NodeID{3, 4, 5})
s.Crash(4)
s.Restart(4)          // reloads from persisted state only

checker.RunChecked(5000)   // asserts all safety properties every tick
```

A crash discards all in-memory state; `Restart` rebuilds the node from its simulated disk alone.
That is what catches "we forgot to persist the vote" and "the on-disk log kept a stale suffix".

### Invariants checked every tick

| Property | What a violation means |
|---|---|
| **Election Safety** (§5.2) | Two leaders in one term — split-brain |
| **Leader Append-Only** (§5.3) | A leader rewrote history |
| **Log Matching** (§5.3) | Two logs agree at an index but differ earlier |
| **State Machine Safety** (§5.4.3) | **Two replicas applied different commands at the same index — silent data corruption** |
| **Commit Invariant** | A node applied an entry it had not committed |

Safety, not liveness, is asserted during the chaos phase: under an adversarial network Raft may make
no progress at all, and that is correct. Liveness is asserted separately, after the network heals —
the cluster must elect a leader, accept writes, and converge to byte-identical state.

### Determinism is a tested property

`TestDeterminism` runs the same seed twice and compares the full event trace. If it ever fails, every
other chaos failure becomes unreproducible — so it is guarded explicitly.

---

## Three bugs the simulator found

These are the honest payoff of the method. Each is a real defect that passed every hand-written test.

### 1. A stale on-disk log suffix survived truncation

`Ready()` offers unpersisted entries above a `persistedIndex` watermark. When a follower truncated a
conflicting suffix and the new leader rewrote those *same indices*, `persistedIndex` was still high —
so `Ready()` returned nothing, and the WAL kept the **old** entries at those indices.

Invisible while the node stayed up: the in-memory log was correct and every node agreed. The
divergence only surfaced after a crash reloaded the stale suffix.

> Caught as `Log Matching violated: nodes 1 and 3 agree at index 24 but differ at index 19`.
> Fixed by rolling `persistedIndex` back to the truncation point ([`raft.go`](internal/raft/raft.go)),
> with `maybeAppend` now reporting where it truncated.

### 2. A recovered follower could never catch up

Raft's happy path is purely reactive: a leader sends more only when a response tells it to. A
heartbeat is anchored at the follower's known `Match`, so it *always* succeeds — and reports nothing
new. A follower restarting at `Match=0` therefore answered every heartbeat successfully while the
leader concluded there was nothing to do. It stayed empty forever.

> Fixed by also sending when `Next <= lastIndex`, regardless of whether the response advanced `Match`.

### 3. A single lost packet could wedge a follower permanently

Two variants, same root cause — the reactive model has no recovery path when the message that would
have triggered the next send is the one that got dropped:

- A dropped **snapshot** leaves `Progress` in `StateSnapshot`, where `canSend` is false, waiting
  forever for a reply that will never come.
- A dropped **rejection** leaves `Progress` in `StateReplicate` with `Next` optimistically advanced
  past the end of the log, so nothing re-triggers a send while `Match` sits far behind.

In both cases heartbeats kept succeeding and the leader had no reason to act. No amount of
transport-level retrying would fix this — the stall is in the consensus state machine.

> Fixed by `recoverStalledFollowers`: if a follower is behind and its `Match` has not moved for three
> heartbeat rounds, rebuild its replication stream.

A fourth bug — pre-vote elections could never be won, because a pre-candidate adopted the higher term
carried by its own *granted* pre-vote responses and stepped down — was caught by a targeted unit test
before the chaos suite ran.

---

## Quick start

```bash
git clone https://github.com/Khwahishd/raftkv && cd raftkv
make build

# Three-node cluster on loopback
./scripts/demo.sh
```

Or manually:

```bash
PEERS='1=http://127.0.0.1:9001,2=http://127.0.0.1:9002,3=http://127.0.0.1:9003'
./bin/raftkvd -id 1 -peers "$PEERS" -listen :9001 -data ./data/1 &
./bin/raftkvd -id 2 -peers "$PEERS" -listen :9002 -data ./data/2 &
./bin/raftkvd -id 3 -peers "$PEERS" -listen :9003 -data ./data/3 &

ENDPOINTS='http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003'
./bin/raftkvctl -endpoints "$ENDPOINTS" put greeting hello
./bin/raftkvctl -endpoints "$ENDPOINTS" get greeting      # linearizable
./bin/raftkvctl -endpoints "$ENDPOINTS" status
```

The client finds the leader on its own: a non-leader answers `421 Misdirected Request` naming the
current leader, and `raftkvctl` follows it.

### Fault tolerance in 30 seconds

```bash
./bin/raftkvctl -endpoints "$ENDPOINTS" put durable yes
kill %1                                                    # kill the leader
sleep 3                                                    # a new one is elected
./bin/raftkvctl -endpoints "$ENDPOINTS" get durable        # -> yes
```

### HTTP API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/kv/{key}` | Linearizable read; `?stale=true` for a local read |
| `PUT` | `/kv/{key}` | Body is the value |
| `DELETE` | `/kv/{key}` | |
| `POST` | `/kv/{key}/cas` | `{"value","expect","expect_absent"}`; `409` if the precondition fails |
| `GET` | `/keys` · `/status` · `/health` | |
| `POST` | `/members` | `{"action":"add"\|"remove","node_id":N}` |

Send `X-Client-Id` and `X-Request-Id` on writes and the server deduplicates retries — a timed-out
write can be safely retried without being applied twice.

---

## Testing

```bash
make test        # everything, including 200 randomized histories
make race        # same, under -race
make chaos       # 1,500 seeds — the full sweep
make bench
```

| Suite | What it proves |
|---|---|
| `internal/raft` | Log mechanics in isolation: snapshot boundaries, conflict backtracking, the stale-append-truncates case, the election restriction |
| `internal/raftsim` | Cluster behaviour: elections, partitions, pre-vote, check-quorum, snapshot catch-up, membership changes, **plus 1,500 randomized failure histories** |
| `internal/storage` | Torn tails, bit-flip corruption, conflicting-suffix truncation, snapshot pruning |
| `internal/kv` | CAS semantics, retry deduplication, snapshot determinism, codec fuzzing against truncation |
| `internal/transport` | Wire round-trips including 1 MiB snapshots; unreachable peers never block the sender |
| `internal/server` | **Real 3-node clusters over real sockets**: leader failover with no acknowledged write lost, recovery from disk, 421 redirects, concurrent writers |

Statement coverage **82.6%** (cross-package). The full suite is race-clean.

### Benchmarks

Consensus core only — no disk, no network (Xeon @ 2.10GHz):

| Benchmark | ns/op | allocs/op |
|---|---|---|
| `Propose` (3 peers) | 230 | 2 |
| `Propose` (5 peers) | 347 | 2 |
| `Propose` (7 peers) | 520 | 2 |
| `AppendEntries` (batch=1) | 271 | 4 |
| `AppendEntries` (batch=128) | 49,940 | 4 |
| `MessageCodec` decode | 9,965 | 913 MB/s |

Two allocations per proposal, and a constant four per `AppendEntries` regardless of batch size —
batching amortizes everything but the entry copy itself.

---

## What I'd do next

Honest limitations, roughly in priority order:

- **Joint consensus.** Membership changes are one-node-at-a-time, which is safe but cannot
  atomically swap a cluster's membership.
- **Leadership transfer.** `MsgTimeoutNow` is wired through but not yet driven by a graceful
  step-down handshake, so planned restarts cost a full election timeout.
- **Follower reads.** `ReadIndex` only runs on the leader; followers could serve linearizable reads
  by fetching a read index from the leader and waiting for it locally.
- **Batched `fsync`.** Every `Ready` syncs independently; batching concurrent proposals into one
  sync is the single largest available throughput win.
- **A real linearizability checker.** The current checker verifies Raft's internal invariants.
  Checking client-observable histories against a sequential specification (Porcupine-style) would
  verify the property clients actually care about.

## References

- Ongaro & Ousterhout, [*In Search of an Understandable Consensus Algorithm*](https://raft.github.io/raft.pdf) (2014)
- Ongaro, [*Consensus: Bridging Theory and Practice*](https://github.com/ongardie/dissertation) (2014) — pre-vote (§9.6), read-index (§6.4), membership (§4.1)
- Howard et al., [*Raft Refloated: Do We Have Consensus?*](https://www.cl.cam.ac.uk/~ms705/pub/papers/2015-osr-raft.pdf) (2015)

## License

MIT — see [LICENSE](LICENSE).

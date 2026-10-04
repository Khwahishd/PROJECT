package raftsim_test

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/raftsim"
)

// defaultChaosSeeds is the number of independent randomized runs performed by
// `go test`. Each seed is a complete, reproducible history: when one fails,
// re-running with -run 'TestChaos/seed=N' replays exactly the same interleaving.
//
// CI raises this via CHAOS_SEEDS to sweep far more histories than a developer
// wants to wait for locally.
const defaultChaosSeeds = 200

// chaosSeeds resolves the seed count from the environment, falling back to the
// default. An unparseable or non-positive value is ignored rather than failing
// the run, so a typo in CI config cannot silently skip the whole suite.
func chaosSeeds() int {
	if v := os.Getenv("CHAOS_SEEDS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultChaosSeeds
}

// TestChaosNoSafetyViolations hammers a cluster with partitions, crashes,
// restarts, message loss, duplication and reordering, asserting Raft's safety
// properties after every single tick.
//
// Safety -- not liveness -- is the contract here: under a sufficiently hostile
// network Raft may make no progress at all, and that is allowed. What is never
// allowed is two leaders in a term, a committed entry being overwritten, or two
// nodes applying different commands at the same index.
func TestChaosNoSafetyViolations(t *testing.T) {
	seeds := chaosSeeds()
	if testing.Short() {
		seeds = 20
	}
	for seed := 0; seed < seeds; seed++ {
		seed := seed
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			t.Parallel()
			runChaos(t, int64(seed))
		})
	}
}

func runChaos(t *testing.T, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	peers := []raft.NodeID{1, 2, 3, 4, 5}

	s, err := raftsim.New(raftsim.Options{
		Seed:              seed,
		Peers:             peers,
		ElectionTimeout:   10,
		HeartbeatTimeout:  2,
		PreVote:           rng.Intn(2) == 0,
		CheckQuorum:       rng.Intn(2) == 0,
		DropRate:          rng.Float64() * 0.15,
		DuplicateRate:     rng.Float64() * 0.10,
		MaxDelay:          rng.Intn(4),
		SnapshotThreshold: []int{0, 15, 40}[rng.Intn(3)],
	})
	if err != nil {
		t.Fatalf("new sim: %v", err)
	}
	checker := raftsim.NewChecker(s)

	crashed := map[raft.NodeID]bool{}
	writes := 0

	for step := 0; step < 1200; step++ {
		// Injected faults are rare per tick but near-certain over a full run.
		switch {
		case rng.Float64() < 0.010:
			// Random partition into two groups.
			shuffled := append([]raft.NodeID(nil), peers...)
			rng.Shuffle(len(shuffled), func(i, j int) {
				shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
			})
			cut := 1 + rng.Intn(len(shuffled)-1)
			s.Partition(shuffled[:cut], shuffled[cut:])
		case rng.Float64() < 0.015:
			s.Heal()
		case rng.Float64() < 0.008:
			// Crash a node, but never enough to make the cluster unrecoverable
			// in a way that just stalls the test with nothing being exercised.
			victim := peers[rng.Intn(len(peers))]
			if !crashed[victim] && len(crashed) < 2 {
				s.Crash(victim)
				crashed[victim] = true
			}
		case rng.Float64() < 0.020:
			for id := range crashed {
				if err := s.Restart(id); err != nil {
					t.Fatalf("restart node %d: %v", id, err)
				}
				delete(crashed, id)
				break
			}
		case rng.Float64() < 0.25:
			if s.Propose(fmt.Sprintf("k%d", writes%7), fmt.Sprintf("v%d", writes)) {
				writes++
			}
		}

		s.Tick()
		if err := checker.Check(); err != nil {
			tail := s.Trace
			if len(tail) > 200 {
				tail = tail[len(tail)-200:]
			}
			t.Fatalf("seed %d: %v\n\ntrace:\n%s", seed, err, strings.Join(tail, "\n"))
		}
	}

	// Having survived the chaos, the cluster must recover once the network is
	// healthy again -- otherwise the run proved only that a dead cluster is
	// safe, which is true but useless.
	s.Heal()
	for id := range crashed {
		if err := s.Restart(id); err != nil {
			t.Fatalf("restart node %d: %v", id, err)
		}
	}
	recovered := false
	for i := 0; i < 2000; i++ {
		s.Tick()
		if err := checker.Check(); err != nil {
			t.Fatalf("seed %d (recovery phase): %v", seed, err)
		}
		if s.Leader() != nil {
			recovered = true
			break
		}
	}
	if !recovered {
		t.Fatalf("seed %d: cluster failed to elect a leader after the network healed", seed)
	}

	// And it must still accept writes.
	progressed := false
	for i := 0; i < 500; i++ {
		if s.Propose("final", "value") {
			progressed = true
			break
		}
		s.Tick()
	}
	if !progressed {
		t.Fatalf("seed %d: healed cluster cannot accept writes", seed)
	}
	for i := 0; i < 300; i++ {
		s.Tick()
		if err := checker.Check(); err != nil {
			t.Fatalf("seed %d (final write): %v", seed, err)
		}
	}

	// Finally, every live node must converge on byte-identical state. Comparing
	// before they have all applied the same index would be meaningless -- two
	// nodes at different commit points hold different values for the same key
	// and that is correct -- so wait for convergence first, and treat failure to
	// converge as its own bug.
	target := s.Leader().Status().Commit
	converged := false
	for i := 0; i < 3000 && !converged; i++ {
		s.Tick()
		if err := checker.Check(); err != nil {
			t.Fatalf("seed %d (convergence phase): %v", seed, err)
		}
		converged = true
		for _, n := range s.Nodes() {
			if n.Crashed() {
				continue
			}
			if n.Status().Applied < target {
				converged = false
				break
			}
		}
	}
	if !converged {
		var detail []string
		for _, n := range s.Nodes() {
			st := n.Status()
			detail = append(detail, fmt.Sprintf("node %d applied=%d commit=%d", n.ID, st.Applied, st.Commit))
		}
		t.Fatalf("seed %d: nodes did not converge to index %d: %s", seed, target, strings.Join(detail, ", "))
	}

	var ref *raftsim.Node
	for _, n := range s.Nodes() {
		if n.Crashed() {
			continue
		}
		if ref == nil {
			ref = n
			continue
		}
		want, got := ref.KV(), n.KV()
		if len(want) != len(got) {
			t.Fatalf("seed %d: node %d has %d keys, node %d has %d", seed, ref.ID, len(want), n.ID, len(got))
		}
		for k, v := range want {
			if got[k] != v {
				t.Fatalf("seed %d: nodes %d and %d diverged at key %s: %q vs %q",
					seed, ref.ID, n.ID, k, v, got[k])
			}
		}
	}
}

// TestDeterminism verifies the property the whole harness depends on: the same
// seed must produce a byte-identical history. If this ever fails, every other
// chaos failure becomes unreproducible.
func TestDeterminism(t *testing.T) {
	run := func() []string {
		s, err := raftsim.New(raftsim.Options{
			Seed: 42, Peers: []raft.NodeID{1, 2, 3},
			DropRate: 0.1, DuplicateRate: 0.05, MaxDelay: 3,
			SnapshotThreshold: 12, PreVote: true, CheckQuorum: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 400; i++ {
			if i%10 == 0 {
				s.Propose(fmt.Sprintf("k%d", i), fmt.Sprintf("v%d", i))
			}
			if i == 120 {
				s.Partition([]raft.NodeID{1}, []raft.NodeID{2, 3})
			}
			if i == 260 {
				s.Heal()
			}
			s.Tick()
		}
		var out []string
		for _, n := range s.Nodes() {
			st := n.Status()
			out = append(out, fmt.Sprintf("%d:%v:%d:%d:%d", n.ID, st.Role, st.Term, st.Commit, st.Applied))
		}
		return append(out, s.Trace...)
	}

	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("different history lengths: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("histories diverge at %d:\n  run A: %s\n  run B: %s", i, a[i], b[i])
		}
	}
}

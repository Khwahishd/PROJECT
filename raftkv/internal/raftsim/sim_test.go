package raftsim_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/raftsim"
)

func newSim(t *testing.T, opts raftsim.Options) *raftsim.Sim {
	t.Helper()
	if len(opts.Peers) == 0 {
		opts.Peers = []raft.NodeID{1, 2, 3}
	}
	s, err := raftsim.New(opts)
	if err != nil {
		t.Fatalf("new sim: %v", err)
	}
	return s
}

// dumpTrace attaches the simulation trace to a failure so the interleaving that
// produced it is visible without a re-run.
func dumpTrace(t *testing.T, s *raftsim.Sim) {
	t.Helper()
	tail := s.Trace
	if len(tail) > 120 {
		tail = tail[len(tail)-120:]
	}
	t.Logf("trace (last %d events):\n%s", len(tail), strings.Join(tail, "\n"))
}

// waitLeader advances the simulation until a leader emerges or ticks run out.
func waitLeader(t *testing.T, s *raftsim.Sim, maxTicks int) *raftsim.Node {
	t.Helper()
	for i := 0; i < maxTicks; i++ {
		s.Tick()
		if l := s.Leader(); l != nil {
			return l
		}
	}
	dumpTrace(t, s)
	t.Fatalf("no leader elected within %d ticks", maxTicks)
	return nil
}

func TestElectsSingleLeader(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 1})
	ldr := waitLeader(t, s, 200)

	leaders := 0
	for _, n := range s.Nodes() {
		if n.Status().Role == raft.Leader {
			leaders++
		}
	}
	if leaders != 1 {
		dumpTrace(t, s)
		t.Fatalf("expected exactly 1 leader, found %d", leaders)
	}
	if ldr.Status().Term == 0 {
		t.Error("leader elected at term 0")
	}
}

func TestSingleNodeClusterElectsItself(t *testing.T) {
	// A one-node cluster is its own quorum and must not need any messages.
	s := newSim(t, raftsim.Options{Seed: 2, Peers: []raft.NodeID{1}})
	ldr := waitLeader(t, s, 100)
	if ldr.ID != 1 {
		t.Fatalf("leader = %d, want 1", ldr.ID)
	}
	if !s.Propose("k", "v") {
		t.Fatal("single-node propose failed")
	}
	s.Run(5)
	if got := s.Node(1).KV()["k"]; got != "v" {
		t.Errorf("kv[k] = %q, want \"v\"", got)
	}
}

func TestReplicatesProposalsToAllNodes(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 3})
	waitLeader(t, s, 200)

	for i := 0; i < 10; i++ {
		if !s.Propose(fmt.Sprintf("key%d", i), fmt.Sprintf("val%d", i)) {
			dumpTrace(t, s)
			t.Fatalf("propose %d failed", i)
		}
		s.Run(2)
	}
	s.Run(40)

	for _, n := range s.Nodes() {
		kv := n.KV()
		for i := 0; i < 10; i++ {
			k, want := fmt.Sprintf("key%d", i), fmt.Sprintf("val%d", i)
			if kv[k] != want {
				dumpTrace(t, s)
				t.Fatalf("node %d: kv[%s] = %q, want %q", n.ID, k, kv[k], want)
			}
		}
	}
}

func TestMinorityPartitionCannotElectLeader(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 4, Peers: []raft.NodeID{1, 2, 3, 4, 5}})
	ldr := waitLeader(t, s, 200)

	// Keep the incumbent leader on the majority side, so that anything the
	// minority side claims must come from an election it won on its own.
	// (Without CheckQuorum a partitioned-away incumbent legitimately keeps
	// calling itself leader -- that case is covered by the CheckQuorum test.)
	majority := []raft.NodeID{ldr.ID}
	var minority []raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID == ldr.ID {
			continue
		}
		if len(majority) < 3 {
			majority = append(majority, n.ID)
		} else {
			minority = append(minority, n.ID)
		}
	}
	s.Partition(majority, minority)
	s.Run(400)

	// The minority can campaign forever; with 2 of 5 voters it can never reach
	// the 3-vote quorum.
	for _, id := range minority {
		if role := s.Node(id).Status().Role; role == raft.Leader {
			dumpTrace(t, s)
			t.Fatalf("node %d in the minority partition became leader", id)
		}
	}

	// The majority side must still be able to commit.
	if !s.Propose("during-partition", "ok") {
		dumpTrace(t, s)
		t.Fatal("majority partition could not commit a write")
	}
	s.Run(50)
	for _, id := range majority {
		if got := s.Node(id).KV()["during-partition"]; got != "ok" {
			dumpTrace(t, s)
			t.Fatalf("majority node %d did not commit the write (got %q)", id, got)
		}
	}
	for _, id := range minority {
		if _, ok := s.Node(id).KV()["during-partition"]; ok {
			dumpTrace(t, s)
			t.Fatalf("minority node %d observed a write it could not have replicated", id)
		}
	}
}

func TestLeaderPartitionedAwayLosesLeadership(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 5, CheckQuorum: true, PreVote: true})
	ldr := waitLeader(t, s, 200)
	old := ldr.ID

	// Cut the leader off from everyone else.
	var rest []raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != old {
			rest = append(rest, n.ID)
		}
	}
	s.Partition([]raft.NodeID{old}, rest)
	s.Run(400)

	// CheckQuorum must make the isolated leader step down rather than keep
	// serving stale reads indefinitely.
	if role := s.Node(old).Status().Role; role == raft.Leader {
		dumpTrace(t, s)
		t.Fatalf("isolated node %d is still leader (CheckQuorum did not fire)", old)
	}
	// The majority side elects a new leader at a higher term.
	var newLeader *raftsim.Node
	for _, id := range rest {
		if s.Node(id).Status().Role == raft.Leader {
			newLeader = s.Node(id)
		}
	}
	if newLeader == nil {
		dumpTrace(t, s)
		t.Fatal("majority partition failed to elect a new leader")
	}
}

func TestRejoiningNodeDoesNotDisruptLeaderWithPreVote(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 6, PreVote: true, CheckQuorum: true})
	ldr := waitLeader(t, s, 200)

	// Isolate a follower; it will spin on pre-votes, bumping nothing.
	var victim raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != ldr.ID {
			victim = n.ID
			break
		}
	}
	var rest []raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != victim {
			rest = append(rest, n.ID)
		}
	}
	s.Partition(rest, []raft.NodeID{victim})
	s.Run(500)

	isolatedTerm := s.Node(victim).Status().Term
	leaderTermBefore := s.Node(ldr.ID).Status().Term
	if isolatedTerm > leaderTermBefore {
		dumpTrace(t, s)
		t.Fatalf("isolated node ran its term up to %d (leader at %d): pre-vote should prevent this",
			isolatedTerm, leaderTermBefore)
	}

	// On rejoining, the leader must survive.
	s.Heal()
	s.Run(100)
	if s.Node(ldr.ID).Status().Role != raft.Leader {
		dumpTrace(t, s)
		t.Fatalf("leader %d was deposed by a rejoining node despite pre-vote", ldr.ID)
	}
}

func TestCrashedNodeRecoversAndCatchesUp(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 7})
	ldr := waitLeader(t, s, 200)

	var victim raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != ldr.ID {
			victim = n.ID
			break
		}
	}
	s.Crash(victim)

	for i := 0; i < 20; i++ {
		s.Propose(fmt.Sprintf("k%d", i), fmt.Sprintf("v%d", i))
		s.Run(2)
	}
	s.Run(50)

	if err := s.Restart(victim); err != nil {
		t.Fatalf("restart: %v", err)
	}
	s.Run(200)

	want := s.Leader()
	if want == nil {
		dumpTrace(t, s)
		t.Fatal("no leader after recovery")
	}
	got := s.Node(victim).KV()
	for k, v := range want.KV() {
		if got[k] != v {
			dumpTrace(t, s)
			t.Fatalf("recovered node %d: kv[%s] = %q, want %q", victim, k, got[k], v)
		}
	}
}

func TestSnapshotCatchesUpFarBehindFollower(t *testing.T) {
	// A low threshold forces compaction, so the lagging follower's required
	// entries are gone and the leader must fall back to a snapshot transfer.
	s := newSim(t, raftsim.Options{Seed: 8, SnapshotThreshold: 10})
	ldr := waitLeader(t, s, 200)

	var victim raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != ldr.ID {
			victim = n.ID
			break
		}
	}
	// Isolate rather than crash, so the node keeps running with a stale log.
	var rest []raft.NodeID
	for _, n := range s.Nodes() {
		if n.ID != victim {
			rest = append(rest, n.ID)
		}
	}
	s.Partition(rest, []raft.NodeID{victim})

	for i := 0; i < 60; i++ {
		s.Propose(fmt.Sprintf("k%d", i), fmt.Sprintf("v%d", i))
		s.Run(2)
	}
	s.Run(50)

	s.Heal()
	s.Run(400)

	leader := s.Leader()
	if leader == nil {
		dumpTrace(t, s)
		t.Fatal("no leader after healing")
	}
	got, want := s.Node(victim).KV(), leader.KV()
	if len(want) == 0 {
		t.Fatal("leader state machine is empty; test is not exercising anything")
	}
	for k, v := range want {
		if got[k] != v {
			dumpTrace(t, s)
			t.Fatalf("follower %d did not catch up via snapshot: kv[%s] = %q, want %q", victim, k, got[k], v)
		}
	}
	if !strings.Contains(strings.Join(s.Trace, "\n"), "installed snapshot") {
		t.Error("expected a snapshot install in the trace; the test may not be exercising snapshot transfer")
	}
}

func TestReadIndexConfirmsLinearizablePoint(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 9})
	ldr := waitLeader(t, s, 200)

	s.Propose("a", "1")
	s.Run(20)

	if !s.ReadIndex([]byte("req-1")) {
		dumpTrace(t, s)
		t.Fatal("ReadIndex was refused by the leader")
	}
	s.Run(20)

	states := s.Node(ldr.ID).ReadStates()
	if len(states) == 0 {
		dumpTrace(t, s)
		t.Fatal("no read state confirmed")
	}
	last := states[len(states)-1]
	if string(last.Context) != "req-1" {
		t.Errorf("context = %q, want \"req-1\"", last.Context)
	}
	// The confirmed index must cover the write that preceded the read.
	if last.Index < 1 {
		t.Errorf("read index %d does not cover the preceding write", last.Index)
	}
}

func TestConfChangeAddsAndRemovesVoters(t *testing.T) {
	s := newSim(t, raftsim.Options{Seed: 10, Peers: []raft.NodeID{1, 2, 3}})
	waitLeader(t, s, 200)
	s.Propose("before", "x")
	s.Run(20)

	if !s.ProposeConfChange(raft.ConfChange{Type: raft.ConfChangeRemoveNode, NodeID: 3}) {
		dumpTrace(t, s)
		t.Fatal("conf change proposal refused")
	}
	s.Run(100)

	ldr := s.Leader()
	if ldr == nil {
		dumpTrace(t, s)
		t.Fatal("lost leadership during conf change")
	}
	if ldr.Status().Config.Has(3) {
		dumpTrace(t, s)
		t.Fatalf("node 3 still a voter after removal: %v", ldr.Status().Config.Voters)
	}
	// The remaining two nodes must still be able to commit.
	if !s.Propose("after", "y") {
		dumpTrace(t, s)
		t.Fatal("cluster cannot commit after membership change")
	}
	s.Run(40)
	if ldr.KV()["after"] != "y" {
		dumpTrace(t, s)
		t.Error("write after conf change did not commit")
	}
}

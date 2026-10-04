// Package raftsim is a deterministic, single-goroutine simulator for a Raft
// cluster. It drives real raft.Node instances through a synthetic network that
// can partition, drop, delay, reorder and duplicate messages, and can crash and
// restart nodes with only their persisted state surviving.
//
// Everything is driven by one seeded PRNG. A failing run is reproducible from
// its seed alone, which is the difference between a flaky distributed-systems
// test and one you can actually debug.
package raftsim

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/Khwahishd/raftkv/internal/raft"
)

// Storage is the durable state of one node. It survives Crash and is reloaded
// by Restart, modelling a disk.
type Storage struct {
	HardState raft.HardState
	Entries   []raft.Entry
	Snapshot  *raft.Snapshot
}

// clone deep-copies the storage, so a crash cannot leak in-memory aliasing into
// what is nominally "on disk".
func (s *Storage) clone() *Storage {
	out := &Storage{HardState: s.HardState}
	out.Entries = append([]raft.Entry(nil), s.Entries...)
	if s.Snapshot != nil {
		snap := *s.Snapshot
		snap.Conf = s.Snapshot.Conf.Clone()
		out.Snapshot = &snap
	}
	return out
}

// append writes entries, truncating any conflicting suffix, mirroring what a
// real write-ahead log does.
func (s *Storage) append(entries []raft.Entry) {
	for _, e := range entries {
		idx := s.indexOf(e.Index)
		if idx >= 0 {
			s.Entries = s.Entries[:idx]
		}
		s.Entries = append(s.Entries, e)
	}
}

func (s *Storage) indexOf(index uint64) int {
	for i, e := range s.Entries {
		if e.Index == index {
			return i
		}
	}
	return -1
}

// Node couples a raft.Node with its durable storage and its applied state.
type Node struct {
	ID      raft.NodeID
	raft    *raft.Node
	storage *Storage

	// applied is the ordered sequence of normal entries handed to the state
	// machine. The invariant checker compares these across nodes.
	applied []raft.Entry
	// kv is a toy state machine so applied output is observable.
	kv map[string]string

	crashed bool
	// readStates collects confirmed linearizable read points.
	readStates []raft.ReadState
}

// Applied returns the entries this node's state machine has consumed.
func (n *Node) Applied() []raft.Entry { return n.applied }

// KV exposes the toy state machine.
func (n *Node) KV() map[string]string { return n.kv }

// Crashed reports whether the node is currently down.
func (n *Node) Crashed() bool { return n.crashed }

// Status proxies the underlying raft node's status.
func (n *Node) Status() raft.Status { return n.raft.Status() }

// envelope is one in-flight message with its scheduled delivery tick.
type envelope struct {
	at  int
	seq int // tiebreaker, so ordering is total and therefore deterministic
	msg raft.Message
}

// Options configure a simulation.
type Options struct {
	Seed             int64
	Peers            []raft.NodeID
	ElectionTimeout  int
	HeartbeatTimeout int
	PreVote          bool
	CheckQuorum      bool

	// DropRate is the probability that any given message is lost.
	DropRate float64
	// DuplicateRate is the probability that a message is delivered twice.
	DuplicateRate float64
	// MaxDelay is the maximum extra ticks a message may be held for. A value of
	// 0 delivers everything on the next tick, in order.
	MaxDelay int
	// SnapshotThreshold compacts the log once this many entries accumulate past
	// the last snapshot. 0 disables snapshotting.
	SnapshotThreshold int
}

func (o *Options) withDefaults() {
	if o.ElectionTimeout == 0 {
		o.ElectionTimeout = 10
	}
	if o.HeartbeatTimeout == 0 {
		o.HeartbeatTimeout = 2
	}
}

// Sim is a whole simulated cluster.
type Sim struct {
	opts  Options
	rng   *rand.Rand
	nodes map[raft.NodeID]*Node
	order []raft.NodeID

	queue []envelope
	seq   int
	tick  int

	// partitions maps each node to a partition id; nodes in different
	// partitions cannot exchange messages.
	partitions map[raft.NodeID]int

	// Trace records human-readable events for failure diagnosis.
	Trace []string
	// MaxTrace bounds the trace so long runs do not exhaust memory.
	MaxTrace int
}

// New builds a simulation with the given options.
func New(opts Options) (*Sim, error) {
	opts.withDefaults()
	s := &Sim{
		opts:       opts,
		rng:        rand.New(rand.NewSource(opts.Seed)),
		nodes:      map[raft.NodeID]*Node{},
		partitions: map[raft.NodeID]int{},
		MaxTrace:   20000,
	}
	for _, id := range opts.Peers {
		if err := s.spawn(id, &Storage{}); err != nil {
			return nil, err
		}
		s.order = append(s.order, id)
		s.partitions[id] = 0
	}
	return s, nil
}

func (s *Sim) spawn(id raft.NodeID, st *Storage) error {
	cfg := raft.NodeConfig{
		ID:               id,
		Peers:            s.opts.Peers,
		ElectionTimeout:  s.opts.ElectionTimeout,
		HeartbeatTimeout: s.opts.HeartbeatTimeout,
		PreVote:          s.opts.PreVote,
		CheckQuorum:      s.opts.CheckQuorum,
		// Each node gets its own stream, derived from the global seed, so that
		// election jitter differs per node but stays reproducible.
		Rand: rand.New(rand.NewSource(s.opts.Seed*1_000_003 + int64(id))),
	}
	if st.Snapshot != nil {
		cfg.SnapshotIndex = st.Snapshot.Index
		cfg.SnapshotTerm = st.Snapshot.Term
		conf := st.Snapshot.Conf.Clone()
		cfg.RestoredConf = &conf
		cfg.Applied = st.Snapshot.Index
	}
	rn, err := raft.NewNode(cfg)
	if err != nil {
		return err
	}
	rn.SetHardState(st.HardState)
	rn.RestoreEntries(st.Entries)

	node := &Node{ID: id, raft: rn, storage: st, kv: map[string]string{}}
	// Replay the snapshot into the toy state machine.
	if st.Snapshot != nil {
		node.kv = decodeKV(st.Snapshot.Data)
	}
	rn.SnapshotProvider = func() (*raft.Snapshot, error) {
		if node.storage.Snapshot == nil {
			return nil, fmt.Errorf("no snapshot available on node %d", id)
		}
		return node.storage.Snapshot, nil
	}
	s.nodes[id] = node
	return nil
}

func (s *Sim) tracef(format string, args ...any) {
	if len(s.Trace) >= s.MaxTrace {
		return
	}
	s.Trace = append(s.Trace, fmt.Sprintf("t=%-5d ", s.tick)+fmt.Sprintf(format, args...))
}

// Node returns the simulated node with the given id.
func (s *Sim) Node(id raft.NodeID) *Node { return s.nodes[id] }

// Nodes returns all nodes in a stable order.
func (s *Sim) Nodes() []*Node {
	out := make([]*Node, 0, len(s.order))
	for _, id := range s.order {
		out = append(out, s.nodes[id])
	}
	return out
}

// Tick advances the whole cluster by one logical tick: deliver due messages,
// tick every live node, then drain their Readys.
func (s *Sim) Tick() {
	s.tick++
	s.deliver()
	for _, id := range s.order {
		n := s.nodes[id]
		if n.crashed {
			continue
		}
		n.raft.Tick()
	}
	s.drain()
}

// Run advances the cluster by n ticks.
func (s *Sim) Run(n int) {
	for i := 0; i < n; i++ {
		s.Tick()
	}
}

// deliver hands every message whose scheduled tick has arrived to its target.
func (s *Sim) deliver() {
	if len(s.queue) == 0 {
		return
	}
	// Sorting by (at, seq) gives a total order independent of map iteration.
	sort.Slice(s.queue, func(i, j int) bool {
		if s.queue[i].at != s.queue[j].at {
			return s.queue[i].at < s.queue[j].at
		}
		return s.queue[i].seq < s.queue[j].seq
	})
	var remaining []envelope
	for _, env := range s.queue {
		if env.at > s.tick {
			remaining = append(remaining, env)
			continue
		}
		target := s.nodes[env.msg.To]
		if target == nil || target.crashed {
			continue
		}
		if !s.connected(env.msg.From, env.msg.To) {
			continue
		}
		if err := target.raft.Step(env.msg); err != nil {
			s.tracef("node %d step error: %v", env.msg.To, err)
		}
	}
	s.queue = remaining
}

// drain collects each live node's Ready, honouring the documented ordering:
// persist, then send, then apply, then advance.
func (s *Sim) drain() {
	// Repeat until quiescent: applying an entry can produce more work (e.g. a
	// conf change changing the quorum and unblocking a commit).
	for round := 0; round < 64; round++ {
		progressed := false
		for _, id := range s.order {
			n := s.nodes[id]
			if n.crashed || !n.raft.HasReady() {
				continue
			}
			progressed = true
			rd := n.raft.Ready()

			// 1. Persist.
			if !rd.HardState.IsEmpty() {
				n.storage.HardState = rd.HardState
			}
			if rd.Snapshot != nil {
				n.storage.Snapshot = rd.Snapshot
				n.storage.Entries = nil
				n.kv = decodeKV(rd.Snapshot.Data)
				s.tracef("node %d installed snapshot @%d", id, rd.Snapshot.Index)
			}
			n.storage.append(rd.Entries)

			// 2. Send (only after the above is durable).
			for _, m := range rd.Messages {
				s.enqueue(m)
			}

			// 3. Apply.
			for _, e := range rd.CommittedEntries {
				s.apply(n, e)
			}
			n.readStates = append(n.readStates, rd.ReadStates...)

			// 4. Acknowledge.
			n.raft.Advance(rd)

			s.maybeSnapshot(n)
		}
		if !progressed {
			return
		}
	}
}

func (s *Sim) apply(n *Node, e raft.Entry) {
	switch e.Type {
	case raft.EntryConfChange:
		cc, err := raft.DecodeConfChange(e.Data)
		if err != nil {
			s.tracef("node %d bad conf change at %d: %v", n.ID, e.Index, err)
			return
		}
		n.raft.ApplyConfChange(cc)
		s.tracef("node %d applied conf change %v node=%d @%d", n.ID, cc.Type, cc.NodeID, e.Index)
	case raft.EntryNormal:
		if len(e.Data) == 0 {
			return // the leader's no-op entry
		}
		k, v, ok := decodeCmd(e.Data)
		if ok {
			n.kv[k] = v
		}
	}
	n.applied = append(n.applied, e)
}

// maybeSnapshot compacts a node's log once it has grown past the threshold.
func (s *Sim) maybeSnapshot(n *Node) {
	if s.opts.SnapshotThreshold <= 0 {
		return
	}
	st := n.raft.Status()
	base := uint64(0)
	if n.storage.Snapshot != nil {
		base = n.storage.Snapshot.Index
	}
	if st.Applied < base+uint64(s.opts.SnapshotThreshold) {
		return
	}
	term := uint64(0)
	for _, e := range n.applied {
		if e.Index == st.Applied {
			term = e.Term
		}
	}
	if term == 0 {
		return
	}
	snap := &raft.Snapshot{
		Index: st.Applied,
		Term:  term,
		Conf:  st.Config.Clone(),
		Data:  encodeKV(n.kv),
	}
	n.storage.Snapshot = snap
	// Drop the compacted prefix from the simulated disk.
	kept := n.storage.Entries[:0]
	for _, e := range n.storage.Entries {
		if e.Index > snap.Index {
			kept = append(kept, e)
		}
	}
	n.storage.Entries = append([]raft.Entry(nil), kept...)
	n.raft.Compact(snap.Index, snap.Term)
	s.tracef("node %d compacted to %d", n.ID, snap.Index)
}

func (s *Sim) enqueue(m raft.Message) {
	if !s.connected(m.From, m.To) {
		return
	}
	if s.opts.DropRate > 0 && s.rng.Float64() < s.opts.DropRate {
		return
	}
	delay := 1
	if s.opts.MaxDelay > 0 {
		delay += s.rng.Intn(s.opts.MaxDelay + 1)
	}
	s.seq++
	s.queue = append(s.queue, envelope{at: s.tick + delay, seq: s.seq, msg: m})

	if s.opts.DuplicateRate > 0 && s.rng.Float64() < s.opts.DuplicateRate {
		s.seq++
		dupDelay := delay
		if s.opts.MaxDelay > 0 {
			dupDelay += s.rng.Intn(s.opts.MaxDelay + 1)
		}
		s.queue = append(s.queue, envelope{at: s.tick + dupDelay, seq: s.seq, msg: m})
	}
}

func (s *Sim) connected(a, b raft.NodeID) bool {
	pa, oka := s.partitions[a]
	pb, okb := s.partitions[b]
	if !oka || !okb {
		return false
	}
	return pa == pb
}

// Partition splits the cluster into disjoint groups. Nodes omitted from every
// group are isolated individually.
func (s *Sim) Partition(groups ...[]raft.NodeID) {
	next := map[raft.NodeID]int{}
	for i, g := range groups {
		for _, id := range g {
			next[id] = i
		}
	}
	isolated := len(groups)
	for _, id := range s.order {
		if _, ok := next[id]; !ok {
			next[id] = isolated
			isolated++
		}
	}
	s.partitions = next
	s.tracef("partition: %s", formatGroups(groups))
}

// Heal removes all partitions.
func (s *Sim) Heal() {
	for _, id := range s.order {
		s.partitions[id] = 0
	}
	s.tracef("partition healed")
}

// Crash takes a node down. Its in-memory state is lost; only Storage survives.
func (s *Sim) Crash(id raft.NodeID) {
	n := s.nodes[id]
	if n == nil || n.crashed {
		return
	}
	n.crashed = true
	n.storage = n.storage.clone()
	s.tracef("node %d crashed", id)
}

// Restart brings a crashed node back up, rebuilding it from persisted state
// alone -- the operation that catches "we forgot to persist the vote" bugs.
func (s *Sim) Restart(id raft.NodeID) error {
	n := s.nodes[id]
	if n == nil || !n.crashed {
		return nil
	}
	st := n.storage.clone()
	if err := s.spawn(id, st); err != nil {
		return err
	}
	// The in-memory state machine is gone, so the node replays its log from the
	// start. Its applied slice therefore restarts empty; the checker keeps the
	// cross-restart history globally and tolerates the reset.
	s.tracef("node %d restarted", id)
	return nil
}

// Leader returns the current leader if exactly one node in the majority
// partition believes it is leading at the highest term, else nil.
func (s *Sim) Leader() *Node {
	var best *Node
	for _, id := range s.order {
		n := s.nodes[id]
		if n.crashed {
			continue
		}
		st := n.raft.Status()
		if st.Role != raft.Leader {
			continue
		}
		if best == nil || st.Term > best.raft.Status().Term {
			best = n
		}
	}
	return best
}

// Propose submits a key/value write to the current leader, returning false if
// there is no leader to accept it.
func (s *Sim) Propose(key, value string) bool {
	ldr := s.Leader()
	if ldr == nil {
		return false
	}
	if _, _, err := ldr.raft.Propose(encodeCmd(key, value)); err != nil {
		return false
	}
	s.drain()
	return true
}

// ProposeConfChange submits a membership change to the current leader.
func (s *Sim) ProposeConfChange(cc raft.ConfChange) bool {
	ldr := s.Leader()
	if ldr == nil {
		return false
	}
	if _, err := ldr.raft.ProposeConfChange(cc); err != nil {
		return false
	}
	s.drain()
	return true
}

// ReadIndex issues a linearizable read request on the leader.
func (s *Sim) ReadIndex(ctx []byte) bool {
	ldr := s.Leader()
	if ldr == nil {
		return false
	}
	if err := ldr.raft.ReadIndex(ctx); err != nil {
		return false
	}
	s.drain()
	return true
}

// ReadStates returns confirmed read points observed on a node.
func (n *Node) ReadStates() []raft.ReadState { return n.readStates }

func formatGroups(groups [][]raft.NodeID) string {
	var parts []string
	for _, g := range groups {
		var ids []string
		for _, id := range g {
			ids = append(ids, fmt.Sprintf("%d", id))
		}
		parts = append(parts, "{"+strings.Join(ids, ",")+"}")
	}
	return strings.Join(parts, " | ")
}

// ---------------------------------------------------------------------------
// Toy state machine encoding
// ---------------------------------------------------------------------------

func encodeCmd(k, v string) []byte { return []byte(k + "\x00" + v) }

func decodeCmd(b []byte) (string, string, bool) {
	s := string(b)
	i := strings.IndexByte(s, 0)
	if i < 0 {
		return "", "", false
	}
	return s[:i], s[i+1:], true
}

func encodeKV(kv map[string]string) []byte {
	keys := make([]string, 0, len(kv))
	for k := range kv {
		keys = append(keys, k)
	}
	// Sorted so the encoding is deterministic -- otherwise two nodes with
	// identical state would produce different snapshot bytes.
	sort.Strings(keys)
	var sb strings.Builder
	for _, k := range keys {
		sb.WriteString(k)
		sb.WriteByte(0)
		sb.WriteString(kv[k])
		sb.WriteByte(1)
	}
	return []byte(sb.String())
}

func decodeKV(b []byte) map[string]string {
	out := map[string]string{}
	for _, rec := range strings.Split(string(b), "\x01") {
		if rec == "" {
			continue
		}
		k, v, ok := decodeCmd([]byte(rec))
		if ok {
			out[k] = v
		}
	}
	return out
}

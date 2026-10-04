// Package server wires the deterministic raft core to real storage, a real
// network and real clients.
//
// The central design decision is that raft.Node is driven by exactly one
// goroutine. Everything else -- HTTP handlers, the transport, the ticker --
// communicates with that goroutine over channels. This keeps the consensus core
// free of locks, makes the Ready ordering contract (persist, send, apply,
// advance) trivially enforceable, and means the production code path is the
// same single-threaded state machine the simulator exercises.
package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Khwahishd/raftkv/internal/kv"
	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/storage"
	"github.com/Khwahishd/raftkv/internal/transport"
)

// Errors surfaced to clients.
var (
	// ErrNotLeader means the request must be retried against the leader.
	ErrNotLeader = errors.New("not leader")
	// ErrTimeout means the request was accepted but not committed in time.
	ErrTimeout = errors.New("request timed out")
	// ErrLeaderChanged means a leader election happened while the request was
	// in flight, so its outcome is unknown and it must be retried.
	ErrLeaderChanged = errors.New("leader changed before commit")
	// ErrShuttingDown is returned once Close has been called.
	ErrShuttingDown = errors.New("server shutting down")
)

// Config configures a Server.
type Config struct {
	ID    raft.NodeID
	Peers map[raft.NodeID]string

	DataDir string

	// TickInterval is the wall-clock duration of one raft tick.
	TickInterval time.Duration
	// ElectionTicks and HeartbeatTicks are measured in TickIntervals.
	ElectionTicks  int
	HeartbeatTicks int

	// SnapshotThreshold is how many applied entries trigger a snapshot and log
	// compaction. Zero disables snapshotting.
	SnapshotThreshold uint64

	PreVote     bool
	CheckQuorum bool

	// RequestTimeout bounds how long a client write waits for commit.
	RequestTimeout time.Duration

	Logger *slog.Logger
}

func (c *Config) withDefaults() {
	if c.TickInterval == 0 {
		c.TickInterval = 50 * time.Millisecond
	}
	if c.ElectionTicks == 0 {
		c.ElectionTicks = 10
	}
	if c.HeartbeatTicks == 0 {
		c.HeartbeatTicks = 2
	}
	if c.RequestTimeout == 0 {
		c.RequestTimeout = 5 * time.Second
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// proposal is a client write waiting for its entry to be committed and applied.
type proposal struct {
	index uint64
	term  uint64
	done  chan proposalResult
}

type proposalResult struct {
	res kv.Result
	err error
}

// readRequest is a linearizable read waiting for a confirmed read index.
type readRequest struct {
	ctx  []byte
	done chan readResult
}

type readResult struct {
	index uint64
	err   error
}

// Server is a single raftkv node.
type Server struct {
	cfg Config
	log *slog.Logger

	node  *raft.Node
	store *storage.Storage
	fsm   *kv.FSM
	tr    *transport.HTTP

	// Channels into the single raft goroutine.
	msgCh     chan raft.Message
	proposeCh chan proposeRequest
	readCh    chan readRequest
	confCh    chan confRequest
	statusCh  chan chan Status

	// pending maps a log index to the client waiting on it. Only the raft
	// goroutine touches it.
	pending map[uint64]*proposal
	// pendingReads is keyed by the read context token.
	pendingReads map[string]*readRequest
	readSeq      uint64

	// appliedIndex is published for stale reads without taking the FSM lock.
	appliedIndex atomic.Uint64
	// leaderHint lets handlers answer "who is the leader" without a round trip
	// into the raft goroutine.
	leaderHint atomic.Uint64
	roleHint   atomic.Uint64

	// lastSnapshotIndex is the index of the most recent snapshot taken.
	lastSnapshotIndex uint64

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	closeOnce sync.Once
}

type proposeRequest struct {
	cmd  kv.Command
	done chan proposalResult
}

type confRequest struct {
	cc   raft.ConfChange
	done chan error
}

// Status is a snapshot of node state for the admin endpoint.
type Status struct {
	ID           raft.NodeID                         `json:"id"`
	Role         string                              `json:"role"`
	Term         uint64                              `json:"term"`
	Leader       raft.NodeID                         `json:"leader"`
	CommitIndex  uint64                              `json:"commit_index"`
	AppliedIndex uint64                              `json:"applied_index"`
	LastIndex    uint64                              `json:"last_index"`
	Voters       []raft.NodeID                       `json:"voters"`
	Keys         int                                 `json:"keys"`
	Peers        map[raft.NodeID]transport.PeerStats `json:"peers"`
	Progress     map[raft.NodeID]ProgressView        `json:"progress,omitempty"`
}

// ProgressView is the leader's replication state for one follower.
type ProgressView struct {
	Match uint64 `json:"match"`
	Next  uint64 `json:"next"`
	State string `json:"state"`
}

// New constructs a Server, restoring any state left by a previous run.
func New(cfg Config) (*Server, error) {
	cfg.withDefaults()

	store, err := storage.Open(cfg.DataDir)
	if err != nil {
		return nil, err
	}

	fsm := kv.New()

	// Recovery order matters: install the snapshot first so the state machine
	// has a base, then replay the WAL tail on top of it.
	nodeCfg := raft.NodeConfig{
		ID:               cfg.ID,
		ElectionTimeout:  cfg.ElectionTicks,
		HeartbeatTimeout: cfg.HeartbeatTicks,
		PreVote:          cfg.PreVote,
		CheckQuorum:      cfg.CheckQuorum,
	}
	for id := range cfg.Peers {
		nodeCfg.Peers = append(nodeCfg.Peers, id)
	}
	sortIDs(nodeCfg.Peers)

	snap, err := store.LoadSnapshot()
	switch {
	case err == nil:
		if err := fsm.Restore(snap.Data); err != nil {
			return nil, fmt.Errorf("restore snapshot: %w", err)
		}
		nodeCfg.SnapshotIndex = snap.Index
		nodeCfg.SnapshotTerm = snap.Term
		nodeCfg.Applied = snap.Index
		conf := snap.Conf.Clone()
		nodeCfg.RestoredConf = &conf
	case errors.Is(err, storage.ErrNotFound):
		// Fresh node.
	default:
		return nil, fmt.Errorf("load snapshot: %w", err)
	}

	entries, err := store.ReadEntries()
	if err != nil {
		return nil, fmt.Errorf("read wal: %w", err)
	}
	hs, err := store.LoadHardState()
	if err != nil {
		return nil, fmt.Errorf("load hard state: %w", err)
	}

	node, err := raft.NewNode(nodeCfg)
	if err != nil {
		return nil, err
	}
	node.RestoreEntries(entries)
	node.SetHardState(hs)

	tr := transport.NewHTTP(cfg.ID, peersExcept(cfg.Peers, cfg.ID), 2*time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	s := &Server{
		cfg:          cfg,
		log:          cfg.Logger.With("node", uint64(cfg.ID)),
		node:         node,
		store:        store,
		fsm:          fsm,
		tr:           tr,
		msgCh:        make(chan raft.Message, 1024),
		proposeCh:    make(chan proposeRequest, 256),
		readCh:       make(chan readRequest, 256),
		confCh:       make(chan confRequest, 8),
		statusCh:     make(chan chan Status, 16),
		pending:      map[uint64]*proposal{},
		pendingReads: map[string]*readRequest{},
		ctx:          ctx,
		cancel:       cancel,
	}
	if snap != nil {
		s.lastSnapshotIndex = snap.Index
		s.appliedIndex.Store(snap.Index)
	}

	// The leader serves snapshots to followers that have fallen off the end of
	// its compacted log.
	node.SnapshotProvider = func() (*raft.Snapshot, error) {
		return s.store.LoadSnapshot()
	}
	node.Logger = func(format string, args ...any) {
		s.log.Debug(fmt.Sprintf(format, args...))
	}

	tr.OnMessage(func(m raft.Message) {
		select {
		case s.msgCh <- m:
		case <-ctx.Done():
		default:
			// The raft goroutine is saturated. Dropping is safe and is what the
			// transport's own documentation promises.
		}
	})

	return s, nil
}

// Start launches the raft goroutine.
func (s *Server) Start() {
	s.wg.Add(1)
	go s.run()
}

// Close stops the server and releases its resources.
func (s *Server) Close() error {
	var err error
	s.closeOnce.Do(func() {
		s.cancel()
		s.wg.Wait()
		s.tr.Close()
		err = s.store.Close()
	})
	return err
}

// Transport exposes the peer-message handler for mounting on an HTTP mux.
func (s *Server) Transport() *transport.HTTP { return s.tr }

// run is the single goroutine that owns the raft node.
func (s *Server) run() {
	defer s.wg.Done()
	ticker := time.NewTicker(s.cfg.TickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			s.failAllPending(ErrShuttingDown)
			return

		case <-ticker.C:
			s.node.Tick()

		case m := <-s.msgCh:
			if err := s.node.Step(m); err != nil {
				s.log.Debug("step failed", "err", err, "type", m.Type.String())
			}

		case req := <-s.proposeCh:
			s.handlePropose(req)

		case req := <-s.readCh:
			s.handleRead(req)

		case req := <-s.confCh:
			_, err := s.node.ProposeConfChange(req.cc)
			req.done <- err

		case reply := <-s.statusCh:
			reply <- s.buildStatus()
		}

		s.processReady()
	}
}

func (s *Server) handlePropose(req proposeRequest) {
	index, term, err := s.node.Propose(kv.EncodeCommand(req.cmd))
	if err != nil {
		req.done <- proposalResult{err: mapRaftErr(err)}
		return
	}
	// A proposal may be lost if this node stops being leader before the entry
	// commits. Recording the term is what lets the apply loop tell "committed"
	// from "silently replaced by a new leader's entry at the same index".
	s.pending[index] = &proposal{index: index, term: term, done: req.done}
}

func (s *Server) handleRead(req readRequest) {
	if err := s.node.ReadIndex(req.ctx); err != nil {
		req.done <- readResult{err: mapRaftErr(err)}
		return
	}
	s.pendingReads[string(req.ctx)] = &req
}

// processReady performs one round of the Ready contract.
func (s *Server) processReady() {
	for s.node.HasReady() {
		rd := s.node.Ready()

		// 1. Persist, before anything leaves this node.
		if !rd.HardState.IsEmpty() {
			if err := s.store.SaveHardState(rd.HardState); err != nil {
				// Continuing past a failed durable write would risk electing two
				// leaders. There is no safe way to proceed.
				s.log.Error("fatal: cannot persist hard state", "err", err)
				s.cancel()
				return
			}
		}
		if rd.Snapshot != nil {
			if err := s.store.SaveSnapshot(rd.Snapshot); err != nil {
				s.log.Error("fatal: cannot persist snapshot", "err", err)
				s.cancel()
				return
			}
			if err := s.fsm.Restore(rd.Snapshot.Data); err != nil {
				s.log.Error("fatal: cannot restore snapshot into state machine", "err", err)
				s.cancel()
				return
			}
			if err := s.store.Compact(rd.Snapshot.Index); err != nil {
				s.log.Error("compact after snapshot install", "err", err)
			}
			s.lastSnapshotIndex = rd.Snapshot.Index
			s.appliedIndex.Store(rd.Snapshot.Index)
			// Entries below the snapshot can never be committed to a waiting
			// client now; fail them rather than leaking the waiter.
			s.failPendingBelow(rd.Snapshot.Index, ErrLeaderChanged)
		}
		if len(rd.Entries) > 0 {
			if err := s.store.Append(rd.Entries); err != nil {
				s.log.Error("fatal: cannot persist log entries", "err", err)
				s.cancel()
				return
			}
		}

		// 2. Send.
		for _, m := range rd.Messages {
			s.tr.Send(m)
		}

		// 3. Apply.
		for _, e := range rd.CommittedEntries {
			s.applyEntry(e)
		}
		for _, rs := range rd.ReadStates {
			if req, ok := s.pendingReads[string(rs.Context)]; ok {
				delete(s.pendingReads, string(rs.Context))
				req.done <- readResult{index: rs.Index}
			}
		}

		// 4. Acknowledge.
		s.node.Advance(rd)

		s.leaderHint.Store(uint64(rd.Leader))
		s.roleHint.Store(uint64(rd.Role))
		if rd.Role != raft.Leader {
			// We are not (or no longer) leader, so nothing we proposed is
			// guaranteed to commit. Telling clients now beats letting them
			// block until timeout on a request that will never land.
			s.failAllPending(ErrNotLeader)
		}

		s.maybeSnapshot()
	}
}

func (s *Server) applyEntry(e raft.Entry) {
	s.appliedIndex.Store(e.Index)

	switch e.Type {
	case raft.EntryConfChange:
		cc, err := raft.DecodeConfChange(e.Data)
		if err != nil {
			s.log.Error("undecodable conf change", "index", e.Index, "err", err)
			return
		}
		conf := s.node.ApplyConfChange(cc)
		s.log.Info("configuration changed", "voters", conf.Voters)
		return

	case raft.EntryNormal:
		if len(e.Data) == 0 {
			return // the leader's term no-op
		}
		cmd, err := kv.DecodeCommand(e.Data)
		if err != nil {
			s.log.Error("undecodable command", "index", e.Index, "err", err)
			return
		}
		res := s.fsm.Apply(e.Index, cmd)

		if p, ok := s.pending[e.Index]; ok {
			delete(s.pending, e.Index)
			// If the entry committed at this index came from a different term,
			// a new leader overwrote our proposal. The client's write may or
			// may not have happened, so it must retry rather than be told it
			// succeeded.
			if p.term != e.Term {
				p.done <- proposalResult{err: ErrLeaderChanged}
			} else {
				p.done <- proposalResult{res: res}
			}
		}
		// Any still-pending proposal at a lower index can never commit now.
		s.failPendingBelow(e.Index, ErrLeaderChanged)
	}
}

func (s *Server) failPendingBelow(index uint64, err error) {
	for i, p := range s.pending {
		if i <= index {
			delete(s.pending, i)
			p.done <- proposalResult{err: err}
		}
	}
}

func (s *Server) failAllPending(err error) {
	for i, p := range s.pending {
		delete(s.pending, i)
		p.done <- proposalResult{err: err}
	}
	for k, r := range s.pendingReads {
		delete(s.pendingReads, k)
		r.done <- readResult{err: err}
	}
}

// maybeSnapshot compacts the log once enough entries have been applied.
func (s *Server) maybeSnapshot() {
	if s.cfg.SnapshotThreshold == 0 {
		return
	}
	applied := s.fsm.AppliedIndex()
	if applied < s.lastSnapshotIndex+s.cfg.SnapshotThreshold {
		return
	}
	st := s.node.Status()
	if applied > st.Commit {
		return
	}
	term, ok := s.termAt(applied)
	if !ok {
		return
	}

	snap := &raft.Snapshot{
		Index: applied,
		Term:  term,
		Conf:  st.Config.Clone(),
		Data:  s.fsm.Snapshot(),
	}
	if err := s.store.SaveSnapshot(snap); err != nil {
		s.log.Error("save snapshot", "err", err)
		return
	}
	if err := s.store.Compact(applied); err != nil {
		s.log.Error("compact wal", "err", err)
		return
	}
	s.node.Compact(applied, term)
	s.lastSnapshotIndex = applied
	s.log.Info("snapshot taken", "index", applied, "keys", s.fsm.Len())
}

// termAt finds the term of an index by consulting the node's own log.
func (s *Server) termAt(index uint64) (uint64, bool) {
	return s.node.TermAt(index)
}

func (s *Server) buildStatus() Status {
	st := s.node.Status()
	out := Status{
		ID:           st.ID,
		Role:         st.Role.String(),
		Term:         st.Term,
		Leader:       st.Leader,
		CommitIndex:  st.Commit,
		AppliedIndex: st.Applied,
		LastIndex:    st.LastIndex,
		Voters:       st.Config.Voters,
		Keys:         s.fsm.Len(),
		Peers:        s.tr.Stats(),
	}
	if len(st.Progress) > 0 {
		out.Progress = map[raft.NodeID]ProgressView{}
		for id, pr := range st.Progress {
			out.Progress[id] = ProgressView{Match: pr.Match, Next: pr.Next, State: pr.State.String()}
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Client-facing API
// ---------------------------------------------------------------------------

// Apply proposes a command and waits for it to be committed and applied.
func (s *Server) Apply(ctx context.Context, cmd kv.Command) (kv.Result, error) {
	done := make(chan proposalResult, 1)
	select {
	case s.proposeCh <- proposeRequest{cmd: cmd, done: done}:
	case <-ctx.Done():
		return kv.Result{}, ctx.Err()
	case <-s.ctx.Done():
		return kv.Result{}, ErrShuttingDown
	}

	timer := time.NewTimer(s.cfg.RequestTimeout)
	defer timer.Stop()
	select {
	case r := <-done:
		return r.res, r.err
	case <-timer.C:
		// The entry may still commit later; the client must treat the outcome
		// as unknown and retry with the same RequestID, which the FSM
		// deduplicates.
		return kv.Result{}, ErrTimeout
	case <-ctx.Done():
		return kv.Result{}, ctx.Err()
	case <-s.ctx.Done():
		return kv.Result{}, ErrShuttingDown
	}
}

// Get performs a linearizable read: it obtains a read index from raft, waits
// for the state machine to catch up to it, and only then reads.
//
// This is strictly stronger than reading the local map. A node that has been
// partitioned away may still believe it is leader; without the read-index round
// trip it would happily serve values that a newer leader has already replaced.
func (s *Server) Get(ctx context.Context, key string) (string, error) {
	s.readSeq++
	token := fmt.Sprintf("%d-%d-%d", s.cfg.ID, time.Now().UnixNano(), s.readSeq)

	done := make(chan readResult, 1)
	select {
	case s.readCh <- readRequest{ctx: []byte(token), done: done}:
	case <-ctx.Done():
		return "", ctx.Err()
	case <-s.ctx.Done():
		return "", ErrShuttingDown
	}

	timer := time.NewTimer(s.cfg.RequestTimeout)
	defer timer.Stop()

	var readIndex uint64
	select {
	case r := <-done:
		if r.err != nil {
			return "", r.err
		}
		readIndex = r.index
	case <-timer.C:
		return "", ErrTimeout
	case <-ctx.Done():
		return "", ctx.Err()
	}

	if err := s.waitApplied(ctx, readIndex); err != nil {
		return "", err
	}
	return s.fsm.Get(key)
}

// GetStale reads the local state machine without any coordination. It is fast
// and may return stale data; use it only where that is acceptable.
func (s *Server) GetStale(key string) (string, error) { return s.fsm.Get(key) }

// waitApplied blocks until the state machine has applied through index.
func (s *Server) waitApplied(ctx context.Context, index uint64) error {
	if s.appliedIndex.Load() >= index {
		return nil
	}
	// Poll at a fraction of the tick interval: applies are driven by the raft
	// goroutine, and a condition variable here would mean taking a lock on the
	// hot apply path to serve a comparatively rare read.
	interval := s.cfg.TickInterval / 5
	if interval < time.Millisecond {
		interval = time.Millisecond
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	deadline := time.NewTimer(s.cfg.RequestTimeout)
	defer deadline.Stop()

	for {
		select {
		case <-ticker.C:
			if s.appliedIndex.Load() >= index {
				return nil
			}
		case <-deadline.C:
			return ErrTimeout
		case <-ctx.Done():
			return ctx.Err()
		case <-s.ctx.Done():
			return ErrShuttingDown
		}
	}
}

// Keys lists all keys in the local state machine.
func (s *Server) Keys() []string { return s.fsm.Keys() }

// Status returns the node's current state.
func (s *Server) Status(ctx context.Context) (Status, error) {
	reply := make(chan Status, 1)
	select {
	case s.statusCh <- reply:
	case <-ctx.Done():
		return Status{}, ctx.Err()
	case <-s.ctx.Done():
		return Status{}, ErrShuttingDown
	}
	select {
	case st := <-reply:
		return st, nil
	case <-ctx.Done():
		return Status{}, ctx.Err()
	case <-s.ctx.Done():
		return Status{}, ErrShuttingDown
	}
}

// ChangeMembership adds or removes a voting member.
func (s *Server) ChangeMembership(ctx context.Context, cc raft.ConfChange) error {
	done := make(chan error, 1)
	select {
	case s.confCh <- confRequest{cc: cc, done: done}:
	case <-ctx.Done():
		return ctx.Err()
	case <-s.ctx.Done():
		return ErrShuttingDown
	}
	select {
	case err := <-done:
		return mapRaftErr(err)
	case <-ctx.Done():
		return ctx.Err()
	}
}

// LeaderID returns the node's current belief about who leads, for client
// redirection. It may be stale, which is why clients must still handle a
// not-leader response from the node they are redirected to.
func (s *Server) LeaderID() raft.NodeID { return raft.NodeID(s.leaderHint.Load()) }

// IsLeader reports whether this node currently believes it is the leader.
func (s *Server) IsLeader() bool { return raft.Role(s.roleHint.Load()) == raft.Leader }

func mapRaftErr(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, raft.ErrNotLeader):
		return ErrNotLeader
	case errors.Is(err, raft.ErrProposalDropped):
		return fmt.Errorf("proposal dropped: %w", err)
	default:
		return err
	}
}

func peersExcept(peers map[raft.NodeID]string, self raft.NodeID) map[raft.NodeID]string {
	out := map[raft.NodeID]string{}
	for id, addr := range peers {
		if id != self {
			out[id] = addr
		}
	}
	return out
}

func sortIDs(ids []raft.NodeID) {
	for i := 1; i < len(ids); i++ {
		v := ids[i]
		j := i - 1
		for j >= 0 && ids[j] > v {
			ids[j+1] = ids[j]
			j--
		}
		ids[j+1] = v
	}
}

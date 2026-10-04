package raft

import (
	"errors"
	"fmt"
	"math/rand"
)

// Errors returned by Node operations.
var (
	// ErrNotLeader is returned when a write or read is issued to a node that is
	// not the leader. The caller should retry against Status().Leader.
	ErrNotLeader = errors.New("raft: not leader")
	// ErrProposalDropped is returned when a proposal cannot be accepted, e.g.
	// while a configuration change is already in flight.
	ErrProposalDropped = errors.New("raft: proposal dropped")
	// ErrUnknownNode is returned when a conf change targets a node in a way
	// that would leave the cluster inconsistent.
	ErrUnknownNode = errors.New("raft: unknown node")
)

// Config parameters for a Node. Timeouts are counted in Tick units; the caller
// decides how much wall-clock time a tick represents.
type NodeConfig struct {
	ID NodeID
	// Peers is the initial cluster membership, including ID. It is ignored when
	// restarting from a snapshot, which carries its own configuration.
	Peers []NodeID

	// ElectionTimeout is the base number of ticks without hearing from a leader
	// before a follower campaigns. The actual timeout for each election is
	// drawn uniformly from [ElectionTimeout, 2*ElectionTimeout) to make split
	// votes self-resolving.
	ElectionTimeout int
	// HeartbeatTimeout is the number of ticks between leader heartbeats. It must
	// be comfortably smaller than ElectionTimeout.
	HeartbeatTimeout int

	// MaxEntriesPerAppend bounds the entries carried by one AppendEntries.
	MaxEntriesPerAppend int
	// MaxCommittedEntriesPerReady bounds how much the state machine is asked to
	// apply in a single Ready, keeping the apply loop responsive.
	MaxCommittedEntriesPerReady int

	// PreVote enables the pre-vote protocol (thesis §9.6). A partitioned node
	// that rejoins will otherwise force a term bump and depose a healthy leader.
	PreVote bool

	// CheckQuorum makes a leader step down if it has not heard from a majority
	// within one election timeout, bounding how long a partitioned leader can
	// believe it is still in charge.
	CheckQuorum bool

	// Rand supplies election-timeout jitter. Tests inject a seeded source to
	// keep whole-cluster runs reproducible.
	Rand *rand.Rand

	// Applied is the index the state machine has already applied, used when
	// restarting so committed entries are not re-delivered.
	Applied uint64
	// Snapshot metadata to restore from, if any.
	SnapshotIndex uint64
	SnapshotTerm  uint64
	// Restored configuration; when set it overrides Peers.
	RestoredConf *Config
}

func (c *NodeConfig) validate() error {
	if c.ID == None {
		return errors.New("raft: node ID must be non-zero")
	}
	if c.ElectionTimeout <= 0 {
		return errors.New("raft: ElectionTimeout must be positive")
	}
	if c.HeartbeatTimeout <= 0 {
		return errors.New("raft: HeartbeatTimeout must be positive")
	}
	if c.HeartbeatTimeout >= c.ElectionTimeout {
		return errors.New("raft: HeartbeatTimeout must be less than ElectionTimeout")
	}
	if c.MaxEntriesPerAppend <= 0 {
		c.MaxEntriesPerAppend = 64
	}
	if c.MaxCommittedEntriesPerReady <= 0 {
		c.MaxCommittedEntriesPerReady = 256
	}
	if c.Rand == nil {
		c.Rand = rand.New(rand.NewSource(int64(c.ID)))
	}
	if c.RestoredConf == nil && len(c.Peers) == 0 {
		return errors.New("raft: no initial peers and no restored configuration")
	}
	return nil
}

// Ready is the batch of work the application must perform on behalf of a Node.
//
// The ordering contract is not advisory -- violating it breaks safety:
//
//  1. Persist HardState and Entries durably (and Snapshot, if present).
//  2. Only then send Messages.
//  3. Apply CommittedEntries to the state machine, and release ReadStates.
//  4. Call Advance.
//
// Sending before persisting would let a node vote, crash, restart with no
// record of the vote, and vote again in the same term -- electing two leaders.
type Ready struct {
	// HardState is empty (IsEmpty) when nothing durable changed.
	HardState HardState
	// Entries must be appended to stable storage.
	Entries []Entry
	// Snapshot, if non-nil, must be persisted and handed to the state machine
	// before CommittedEntries.
	Snapshot *Snapshot
	// Messages must be sent after Entries are durable.
	Messages []Message
	// CommittedEntries are ready to apply, oldest first.
	CommittedEntries []Entry
	// ReadStates are confirmed linearizable read points.
	ReadStates []ReadState

	// Role and Leader are informational, for observability and client routing.
	Role   Role
	Leader NodeID
}

// IsEmpty reports whether this Ready carries no work.
func (rd Ready) IsEmpty() bool {
	return rd.HardState.IsEmpty() && len(rd.Entries) == 0 && rd.Snapshot == nil &&
		len(rd.Messages) == 0 && len(rd.CommittedEntries) == 0 && len(rd.ReadStates) == 0
}

// Node is a single Raft peer. It is a deterministic state machine: identical
// sequences of Tick/Step/Propose calls always produce identical Readys.
//
// Node is NOT safe for concurrent use. The owning application is expected to
// drive it from a single goroutine (see internal/server).
type Node struct {
	cfg NodeConfig
	id  NodeID

	role Role
	term uint64
	vote NodeID
	lead NodeID

	log *raftLog

	conf     Config
	progress map[NodeID]*Progress

	// votes records responses in the current election; value is the grant.
	votes map[NodeID]bool

	electionElapsed  int
	heartbeatElapsed int
	// randomizedElectionTimeout is redrawn on every role change.
	randomizedElectionTimeout int

	readOnly readOnly

	// pendingConfIndex is the index of the most recent unapplied conf change.
	// Only one may be in flight at a time (§4.1).
	pendingConfIndex uint64

	// Outputs accumulated since the last Ready.
	msgs            []Message
	readStates      []ReadState
	pendingSnapshot *Snapshot

	// persistedIndex is the highest log index the application has confirmed
	// durable via Advance. Entries above it are re-offered in the next Ready.
	persistedIndex uint64

	// restoredCommit is the commit index read from persisted state. It is
	// re-applied after entries are restored, so SetHardState and RestoreEntries
	// may be called in either order without the commit index being clamped
	// against a log that has not been reloaded yet.
	restoredCommit uint64

	// prevHardState is used to emit HardState only when it actually changes.
	prevHardState HardState

	// SnapshotProvider is called when a follower has fallen behind the leader's
	// compacted log and must be caught up with a snapshot instead of entries.
	// It is a field rather than a NodeConfig entry because the state machine
	// that produces snapshots is typically constructed after the Node.
	SnapshotProvider func() (*Snapshot, error)

	// Logger is optional; when set it receives human-readable state transitions.
	Logger func(format string, args ...any)
}

// NewNode constructs a Node from cfg. The node starts as a follower at term 0
// (or at the restored state, if the caller supplies one via Restore).
func NewNode(cfg NodeConfig) (*Node, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	n := &Node{
		cfg:      cfg,
		id:       cfg.ID,
		role:     Follower,
		log:      newRaftLog(cfg.SnapshotIndex, cfg.SnapshotTerm),
		progress: map[NodeID]*Progress{},
	}
	if cfg.RestoredConf != nil {
		n.conf = cfg.RestoredConf.Clone()
	} else {
		n.conf = Config{Voters: append([]NodeID(nil), cfg.Peers...)}
	}
	if cfg.Applied > n.log.applied {
		n.log.appliedTo(cfg.Applied)
	}
	n.persistedIndex = cfg.SnapshotIndex
	n.resetProgress()
	n.resetElectionTimer()
	return n, nil
}

// SetHardState restores persisted term/vote/commit after a restart. It must be
// called before the node is driven, and is the mechanism that makes a restarted
// node remember the vote it already cast.
func (n *Node) SetHardState(hs HardState) {
	n.term = hs.Term
	n.vote = hs.Vote
	n.restoredCommit = max64(n.restoredCommit, hs.Commit)
	n.log.commitTo(n.restoredCommit)
	n.prevHardState = n.hardState()
}

// RestoreEntries reloads a persisted log tail after a restart.
func (n *Node) RestoreEntries(entries []Entry) {
	for _, e := range entries {
		if e.Index <= n.log.lastIndex() {
			continue
		}
		n.log.append(e)
	}
	n.persistedIndex = n.log.lastIndex()
	// Re-apply the persisted commit index now that the log is back: commitTo
	// clamps to the last index, so a call made before this point was a no-op.
	n.log.commitTo(n.restoredCommit)
	// Re-derive the configuration from any conf changes in the restored tail,
	// since they may post-date the snapshot's embedded configuration.
	for _, e := range entries {
		if e.Type == EntryConfChange && e.Index <= n.log.applied {
			if cc, err := DecodeConfChange(e.Data); err == nil {
				n.applyConfChange(cc)
			}
		}
	}
	n.resetProgress()
	n.prevHardState = n.hardState()
}

// ID returns this node's identifier.
func (n *Node) ID() NodeID { return n.id }

// Status returns a read-only view of the node's current state.
func (n *Node) Status() Status {
	st := Status{
		ID:        n.id,
		Role:      n.role,
		Term:      n.term,
		Leader:    n.lead,
		Commit:    n.log.committed,
		Applied:   n.log.applied,
		LastIndex: n.log.lastIndex(),
		Config:    n.conf.Clone(),
	}
	if n.role == Leader {
		st.Progress = make(map[NodeID]Progress, len(n.progress))
		for id, pr := range n.progress {
			st.Progress[id] = *pr
		}
	}
	return st
}

func (n *Node) logf(format string, args ...any) {
	if n.Logger != nil {
		n.Logger(format, args...)
	}
}

func (n *Node) hardState() HardState {
	return HardState{Term: n.term, Vote: n.vote, Commit: n.log.committed}
}

func (n *Node) quorum() int { return n.conf.Quorum() }

// resetProgress rebuilds the per-peer progress map to match the current
// configuration, preserving state for peers that remain.
func (n *Node) resetProgress() {
	next := map[NodeID]*Progress{}
	for _, id := range n.conf.Voters {
		if pr, ok := n.progress[id]; ok {
			next[id] = pr
			continue
		}
		next[id] = &Progress{Next: n.log.lastIndex() + 1, State: StateProbe}
	}
	n.progress = next
}

func (n *Node) resetElectionTimer() {
	n.electionElapsed = 0
	base := n.cfg.ElectionTimeout
	n.randomizedElectionTimeout = base + n.cfg.Rand.Intn(base)
}

// ---------------------------------------------------------------------------
// Role transitions
// ---------------------------------------------------------------------------

func (n *Node) becomeFollower(term uint64, lead NodeID) {
	if term > n.term {
		n.term = term
		n.vote = None
	}
	n.role = Follower
	n.lead = lead
	n.readOnly.reset()
	n.resetElectionTimer()
	n.heartbeatElapsed = 0
	n.logf("node %d became follower at term %d (leader=%d)", n.id, n.term, lead)
}

func (n *Node) becomePreCandidate() {
	// Deliberately does NOT increment the term or record a vote: a pre-candidate
	// is asking "would you vote for me?", not forcing a new term on the cluster.
	n.role = PreCandidate
	n.lead = None
	n.votes = map[NodeID]bool{n.id: true}
	n.readOnly.reset()
	n.resetElectionTimer()
	n.logf("node %d became pre-candidate at term %d", n.id, n.term)
}

func (n *Node) becomeCandidate() {
	n.term++
	n.role = Candidate
	n.vote = n.id
	n.lead = None
	n.votes = map[NodeID]bool{n.id: true}
	n.readOnly.reset()
	n.resetElectionTimer()
	n.logf("node %d became candidate at term %d", n.id, n.term)
}

func (n *Node) becomeLeader() {
	n.role = Leader
	n.lead = n.id
	n.heartbeatElapsed = 0
	n.electionElapsed = 0
	n.readOnly.reset()

	for id, pr := range n.progress {
		pr.Match = 0
		pr.Next = n.log.lastIndex() + 1
		pr.State = StateProbe
		pr.Paused = false
		pr.RecentActive = id == n.id
		if id == n.id {
			pr.Match = n.log.lastIndex()
		}
	}

	// Append an empty entry for the new term. Until an entry from the current
	// term is committed, the leader cannot know which earlier entries are
	// committed (§5.4.2), so this entry is what unblocks commit advancement and
	// linearizable reads.
	n.appendEntry(Entry{Type: EntryNormal})
	n.pendingConfIndex = n.log.lastIndex()

	n.logf("node %d became leader at term %d", n.id, n.term)
	n.broadcastAppend()
}

// ---------------------------------------------------------------------------
// Driving the node
// ---------------------------------------------------------------------------

// Tick advances the node's logical clock by one unit.
func (n *Node) Tick() {
	switch n.role {
	case Leader:
		n.tickLeader()
	default:
		n.tickElection()
	}
}

func (n *Node) tickElection() {
	n.electionElapsed++
	if n.electionElapsed < n.randomizedElectionTimeout {
		return
	}
	// A node that is no longer a voter (removed from the configuration) must
	// not campaign; it stays quiet until it is shut down.
	if !n.conf.Has(n.id) {
		n.electionElapsed = 0
		return
	}
	n.campaign()
}

func (n *Node) tickLeader() {
	n.heartbeatElapsed++
	n.electionElapsed++

	if n.cfg.CheckQuorum && n.electionElapsed >= n.randomizedElectionTimeout {
		n.electionElapsed = 0
		if !n.hasQuorumContact() {
			n.logf("node %d lost quorum contact, stepping down from term %d", n.id, n.term)
			n.becomeFollower(n.term, None)
			return
		}
		for id, pr := range n.progress {
			pr.RecentActive = id == n.id
		}
	}

	if n.heartbeatElapsed >= n.cfg.HeartbeatTimeout {
		n.heartbeatElapsed = 0
		n.recoverStalledFollowers()
		n.broadcastHeartbeat()
	}
}

// stallRounds is how many heartbeat rounds a follower may sit behind without
// its Match advancing before the leader rebuilds its replication stream.
const stallRounds = 3

// recoverStalledFollowers restarts replication to followers that are behind and
// making no progress.
//
// Raft's happy path is purely reactive: the leader sends more only when a
// response tells it to. That leaves two ways for a single lost packet to wedge
// a follower permanently, neither of which any amount of retrying at the
// transport layer would fix:
//
//   - A dropped snapshot leaves Progress in StateSnapshot, where canSend is
//     false, waiting forever for a reply that will never come.
//   - A dropped rejection leaves Progress in StateReplicate with Next optimistically
//     advanced past the end of the log, so nothing re-triggers a send, while
//     Match sits far behind.
//
// In both cases heartbeats still succeed (they are anchored at Match and so
// always pass the consistency check) and report nothing new, so the leader has
// no reason to act. Detecting the lack of progress directly is what closes the
// gap, and the chaos suite reproduces both cases within a few hundred seeds.
func (n *Node) recoverStalledFollowers() {
	last := n.log.lastIndex()
	for id, pr := range n.progress {
		if id == n.id {
			continue
		}
		if pr.Match >= last {
			pr.stalledRounds = 0
			pr.lastObservedMatch = pr.Match
			continue
		}
		if pr.Match != pr.lastObservedMatch {
			// Still making progress; leave the stream alone.
			pr.lastObservedMatch = pr.Match
			pr.stalledRounds = 0
			continue
		}
		pr.stalledRounds++
		if pr.stalledRounds < stallRounds {
			continue
		}
		pr.stalledRounds = 0
		pr.becomeProbe()
		n.sendAppend(id)
	}
}

func (n *Node) hasQuorumContact() bool {
	active := 0
	for _, id := range n.conf.Voters {
		if pr, ok := n.progress[id]; ok && (id == n.id || pr.RecentActive) {
			active++
		}
	}
	return active >= n.quorum()
}

// campaign starts a (pre-)election.
func (n *Node) campaign() {
	if n.cfg.PreVote {
		n.becomePreCandidate()
		n.sendVoteRequests(MsgPreVoteReq, n.term+1)
	} else {
		n.becomeCandidate()
		n.sendVoteRequests(MsgVoteReq, n.term)
	}
	// A single-node cluster wins immediately, with no messages to send.
	n.maybeWinElection()
}

func (n *Node) sendVoteRequests(t MessageType, term uint64) {
	for _, id := range n.conf.Voters {
		if id == n.id {
			continue
		}
		n.send(Message{
			Type:         t,
			To:           id,
			Term:         term,
			LastLogIndex: n.log.lastIndex(),
			LastLogTerm:  n.log.lastTerm(),
		})
	}
}

func (n *Node) maybeWinElection() bool {
	granted := 0
	for _, g := range n.votes {
		if g {
			granted++
		}
	}
	if granted < n.quorum() {
		return false
	}
	switch n.role {
	case PreCandidate:
		n.becomeCandidate()
		n.sendVoteRequests(MsgVoteReq, n.term)
		return n.maybeWinElection()
	case Candidate:
		n.becomeLeader()
		return true
	}
	return false
}

// maybeLoseElection reverts to follower once a majority has refused, rather
// than waiting out the full election timeout.
func (n *Node) maybeLoseElection() {
	rejected := 0
	for _, g := range n.votes {
		if !g {
			rejected++
		}
	}
	if rejected >= n.quorum() {
		n.becomeFollower(n.term, None)
	}
}

// Propose submits an application command. Only a leader may propose.
func (n *Node) Propose(data []byte) (index uint64, term uint64, err error) {
	if n.role != Leader {
		return 0, 0, ErrNotLeader
	}
	n.appendEntry(Entry{Type: EntryNormal, Data: data})
	n.broadcastAppend()
	return n.log.lastIndex(), n.term, nil
}

// ProposeConfChange submits a membership change. At most one may be in flight;
// applying them one at a time is what guarantees overlapping majorities.
func (n *Node) ProposeConfChange(cc ConfChange) (uint64, error) {
	if n.role != Leader {
		return 0, ErrNotLeader
	}
	if n.pendingConfIndex > n.log.applied {
		return 0, ErrProposalDropped
	}
	if cc.Type == ConfChangeRemoveNode && !n.conf.Has(cc.NodeID) {
		return 0, ErrUnknownNode
	}
	if cc.Type == ConfChangeRemoveNode && len(n.conf.Voters) == 1 {
		return 0, ErrProposalDropped
	}
	n.appendEntry(Entry{Type: EntryConfChange, Data: EncodeConfChange(cc)})
	n.pendingConfIndex = n.log.lastIndex()
	n.broadcastAppend()
	return n.log.lastIndex(), nil
}

// ReadIndex requests a linearizable read point tagged with ctx. The confirmed
// index is delivered later via Ready.ReadStates.
func (n *Node) ReadIndex(ctx []byte) error {
	if n.role != Leader {
		return ErrNotLeader
	}
	// A leader that has not yet committed an entry in its own term may have a
	// stale commit index; the no-op entry appended at becomeLeader will commit
	// shortly and the client should retry.
	if !n.committedInCurrentTerm() {
		return ErrProposalDropped
	}
	// A single-node cluster is its own quorum -- confirm immediately.
	if len(n.conf.Voters) == 1 {
		n.readStates = append(n.readStates, ReadState{Index: n.log.committed, Context: ctx})
		return nil
	}
	n.readOnly.add(n.log.committed, ctx, n.id)
	n.broadcastHeartbeat()
	return nil
}

func (n *Node) committedInCurrentTerm() bool {
	t, ok := n.log.term(n.log.committed)
	return ok && t == n.term
}

func (n *Node) appendEntry(e Entry) {
	e.Term = n.term
	e.Index = n.log.lastIndex() + 1
	n.log.append(e)
	if pr, ok := n.progress[n.id]; ok {
		pr.maybeUpdate(e.Index)
	}
	n.maybeCommit()
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

// Step feeds one received message into the node.
func (n *Node) Step(m Message) error {
	// Messages from a node no longer in the configuration are ignored, except
	// for responses, which may still be in flight from a node just removed.
	switch {
	case m.Term == 0:
		// Local / term-less message; handled below.
	case m.Term > n.term:
		if err := n.stepHigherTerm(m); err != nil {
			return err
		}
	case m.Term < n.term:
		return n.stepLowerTerm(m)
	}

	switch m.Type {
	case MsgPreVoteReq, MsgVoteReq:
		n.handleVoteRequest(m)
	case MsgPreVoteResp, MsgVoteResp:
		n.handleVoteResponse(m)
	case MsgAppendReq:
		n.handleAppend(m)
	case MsgAppendResp:
		n.handleAppendResponse(m)
	case MsgSnapshotReq:
		n.handleSnapshot(m)
	case MsgSnapshotResp:
		n.handleSnapshotResponse(m)
	case MsgTimeoutNow:
		// Only act if we are still a voter; campaign immediately.
		if n.conf.Has(n.id) {
			n.campaign()
		}
	default:
		return fmt.Errorf("raft: unhandled message type %s", m.Type)
	}
	return nil
}

// stepHigherTerm handles a message carrying a term greater than ours.
func (n *Node) stepHigherTerm(m Message) error {
	// A PreVote request carries term+1 without its sender having adopted that
	// term. Treating it as a term bump would defeat the entire purpose of
	// pre-vote, so we leave our term alone and let handleVoteRequest decide.
	if m.Type == MsgPreVoteReq {
		return nil
	}
	// A *granted* PreVote response echoes the prospective term we asked about,
	// which is by construction higher than ours. Stepping down here would mean
	// a pre-vote election could never be won by the node that called it.
	// The term is adopted later, in becomeCandidate, once a quorum has granted.
	if m.Type == MsgPreVoteResp && m.Granted {
		return nil
	}
	// Any other higher-term message (including a *rejected* pre-vote) means a
	// peer is genuinely ahead: adopt the term and remain a follower.
	lead := None
	if m.Type == MsgAppendReq || m.Type == MsgSnapshotReq {
		lead = m.From
	}
	n.becomeFollower(m.Term, lead)
	return nil
}

// stepLowerTerm handles a stale message.
func (n *Node) stepLowerTerm(m Message) error {
	switch m.Type {
	case MsgAppendReq, MsgSnapshotReq:
		// Tell the stale leader to step down by replying at our higher term.
		n.send(Message{Type: MsgAppendResp, To: m.From, Term: n.term, Reject: true})
	case MsgPreVoteReq:
		// Reject at our term so the pre-candidate learns it cannot win.
		n.send(Message{Type: MsgPreVoteResp, To: m.From, Term: n.term, Granted: false})
	default:
		// Drop silently; responding would amplify stale traffic.
	}
	return nil
}

func (n *Node) handleVoteRequest(m Message) {
	isPreVote := m.Type == MsgPreVoteReq
	respType := MsgVoteResp
	if isPreVote {
		respType = MsgPreVoteResp
	}

	// §5.4.1: never vote for a candidate whose log is behind ours.
	logOK := n.log.isUpToDate(m.LastLogIndex, m.LastLogTerm)

	var grant bool
	switch {
	case !logOK:
		grant = false
	case isPreVote:
		// Grant a pre-vote only if the candidate would actually win a real
		// election. Refusing while we still believe in a live leader is the
		// whole point of pre-vote: it stops a node that was partitioned away,
		// and has been campaigning into the void, from forcing a term bump and
		// deposing a perfectly healthy leader when it rejoins.
		hasLiveLeader := n.lead != None && n.electionElapsed < n.randomizedElectionTimeout
		grant = m.Term > n.term && !hasLiveLeader
	default:
		// Grant a real vote if we have not voted this term, or already voted
		// for this same candidate (making the RPC idempotent on retry).
		grant = n.vote == None || n.vote == m.From
	}

	if grant && !isPreVote {
		n.vote = m.From
		// Only reset the election timer when actually granting a vote, so that
		// a node spamming rejected requests cannot keep us from campaigning.
		n.resetElectionTimer()
	}

	respTerm := n.term
	if isPreVote && grant {
		// Echo the candidate's prospective term so it can match the response to
		// the election it is running.
		respTerm = m.Term
	}
	n.send(Message{Type: respType, To: m.From, Term: respTerm, Granted: grant})
}

func (n *Node) handleVoteResponse(m Message) {
	isPreVote := m.Type == MsgPreVoteResp
	if isPreVote && n.role != PreCandidate {
		return
	}
	if !isPreVote && n.role != Candidate {
		return
	}
	if n.votes == nil {
		n.votes = map[NodeID]bool{}
	}
	if _, seen := n.votes[m.From]; seen {
		return // duplicate response; must not double-count
	}
	n.votes[m.From] = m.Granted
	if !n.maybeWinElection() {
		n.maybeLoseElection()
	}
}

func (n *Node) handleAppend(m Message) {
	if n.role != Follower {
		// A candidate or pre-candidate that hears from a current-term leader
		// concedes the election.
		n.becomeFollower(m.Term, m.From)
	}
	n.lead = m.From
	n.resetElectionTimer()

	// A leader may be told about entries we have already compacted away. Those
	// are committed by definition, so acknowledge our snapshot boundary rather
	// than forcing a pointless snapshot transfer back to us.
	if m.PrevLogIndex < n.log.snapIndex {
		n.send(Message{
			Type: MsgAppendResp, To: m.From, Term: n.term,
			MatchIndex: n.log.snapIndex, Context: m.Context,
		})
		return
	}

	last, truncatedAt, ok := n.log.maybeAppend(m.PrevLogIndex, m.PrevLogTerm, m.Entries)
	if truncatedAt > 0 && n.persistedIndex >= truncatedAt {
		// Entries from truncatedAt onwards were rewritten, so the copies the
		// application already wrote to disk are stale. Rolling persistedIndex
		// back is what causes the next Ready to re-offer them.
		//
		// Omitting this is silent and extremely hard to spot: the in-memory log
		// is correct, every node agrees while it stays up, and the divergence
		// only surfaces after a crash and restart reloads the stale suffix.
		n.persistedIndex = truncatedAt - 1
	}
	if !ok {
		ci, ct := n.log.findConflict(m.PrevLogIndex)
		n.send(Message{
			Type: MsgAppendResp, To: m.From, Term: n.term,
			Reject: true, MatchIndex: m.PrevLogIndex,
			ConflictIndex: ci, ConflictTerm: ct, Context: m.Context,
		})
		return
	}

	// Never commit beyond what we have actually stored (§5.3).
	n.log.commitTo(min64(m.LeaderCommit, last))
	n.send(Message{
		Type: MsgAppendResp, To: m.From, Term: n.term,
		MatchIndex: last, Context: m.Context,
	})
}

func (n *Node) handleAppendResponse(m Message) {
	if n.role != Leader {
		return
	}
	pr, ok := n.progress[m.From]
	if !ok {
		return // response from a node removed from the configuration
	}
	pr.RecentActive = true
	pr.stalledRounds = 0

	if m.Reject {
		if pr.maybeDecrTo(m.MatchIndex, n.conflictHintToNext(m)) {
			if pr.State == StateReplicate {
				pr.becomeProbe()
			}
			n.sendAppend(m.From)
		}
		return
	}

	updated := pr.maybeUpdate(m.MatchIndex)
	if updated && pr.State == StateProbe {
		pr.becomeReplicate()
	}

	switch {
	case updated && n.maybeCommit():
		// Commit moved: push it out so followers can apply promptly rather than
		// waiting for the next heartbeat.
		n.broadcastAppend()
	case pr.Next <= n.log.lastIndex():
		// The follower is behind, so keep feeding it. This arm must be reached
		// even when maybeUpdate reported no change: a heartbeat anchored at a
		// follower's existing Match succeeds while telling the leader nothing
		// new, and if that were the end of it a restarted follower sitting at
		// Match=0 would never be probed again and could never catch up.
		n.sendAppend(m.From)
	}

	if len(m.Context) > 0 {
		if ready := n.readOnly.recvAck(m.From, m.Context, n.quorum()); len(ready) > 0 {
			n.readStates = append(n.readStates, ready...)
		}
	}
}

// conflictHintToNext converts a follower's conflict hint into the index the
// leader should probe next.
//
// If the leader has any entry in ConflictTerm, it can resume just after the
// last one; the follower is guaranteed to have no entry beyond that in that
// term. Otherwise it jumps straight to ConflictIndex, skipping the whole term.
func (n *Node) conflictHintToNext(m Message) uint64 {
	if m.ConflictTerm == 0 {
		return m.ConflictIndex
	}
	for i := n.log.lastIndex(); i >= n.log.firstIndex(); i-- {
		t, ok := n.log.term(i)
		if !ok {
			break
		}
		if t == m.ConflictTerm {
			return i + 1
		}
		if t < m.ConflictTerm {
			break
		}
	}
	return m.ConflictIndex
}

func (n *Node) handleSnapshot(m Message) {
	if n.role != Follower {
		n.becomeFollower(m.Term, m.From)
	}
	n.lead = m.From
	n.resetElectionTimer()

	snap := m.Snapshot
	if snap == nil || snap.Index <= n.log.committed {
		// Already at or past this snapshot; just acknowledge where we are.
		n.send(Message{Type: MsgSnapshotResp, To: m.From, Term: n.term, MatchIndex: n.log.committed})
		return
	}

	n.log.restore(snap)
	n.conf = snap.Conf.Clone()
	n.resetProgress()
	n.pendingSnapshot = snap
	n.logf("node %d restored snapshot at index %d term %d", n.id, snap.Index, snap.Term)

	n.send(Message{Type: MsgSnapshotResp, To: m.From, Term: n.term, MatchIndex: snap.Index})
}

func (n *Node) handleSnapshotResponse(m Message) {
	if n.role != Leader {
		return
	}
	pr, ok := n.progress[m.From]
	if !ok {
		return
	}
	pr.RecentActive = true
	pr.stalledRounds = 0
	pr.maybeUpdate(m.MatchIndex)
	pr.becomeProbe()
	n.sendAppend(m.From)
}

// ---------------------------------------------------------------------------
// Replication
// ---------------------------------------------------------------------------

func (n *Node) broadcastAppend() {
	for _, id := range n.conf.Voters {
		if id == n.id {
			continue
		}
		n.sendAppend(id)
	}
}

func (n *Node) broadcastHeartbeat() {
	ctx, _ := n.readOnly.lastPendingContext()
	for _, id := range n.conf.Voters {
		if id == n.id {
			continue
		}
		n.sendHeartbeat(id, ctx)
	}
}

func (n *Node) sendHeartbeat(to NodeID, ctx []byte) {
	pr, ok := n.progress[to]
	if !ok {
		return
	}
	// A heartbeat is an empty AppendEntries anchored at the follower's known
	// match point, so it can never fail the consistency check.
	prevIndex := pr.Match
	prevTerm, ok := n.log.term(prevIndex)
	if !ok {
		// The match point has been compacted; fall through to a real append,
		// which will trigger a snapshot if needed.
		n.sendAppend(to)
		return
	}
	n.send(Message{
		Type: MsgAppendReq, To: to, Term: n.term,
		PrevLogIndex: prevIndex, PrevLogTerm: prevTerm,
		LeaderCommit: min64(n.log.committed, pr.Match),
		Context:      ctx,
	})
	// Heartbeats re-open the probe window.
	if pr.State == StateProbe {
		pr.Paused = false
	}
}

func (n *Node) sendAppend(to NodeID) {
	pr, ok := n.progress[to]
	if !ok || !pr.canSend() {
		return
	}

	prevIndex := pr.Next - 1
	prevTerm, ok := n.log.term(prevIndex)
	if !ok {
		// The entries this follower needs are gone; send a snapshot instead.
		n.sendSnapshot(to, pr)
		return
	}

	entries := n.log.slice(pr.Next, n.log.lastIndex()+1, n.cfg.MaxEntriesPerAppend)
	n.send(Message{
		Type: MsgAppendReq, To: to, Term: n.term,
		PrevLogIndex: prevIndex, PrevLogTerm: prevTerm,
		Entries:      entries,
		LeaderCommit: n.log.committed,
	})

	switch pr.State {
	case StateReplicate:
		// Optimistically advance Next so the next append pipelines behind this
		// one instead of waiting for the acknowledgement.
		if len(entries) > 0 {
			pr.Next = entries[len(entries)-1].Index + 1
		}
	case StateProbe:
		pr.Paused = true
	}
}

func (n *Node) sendSnapshot(to NodeID, pr *Progress) {
	// The application supplies snapshot bytes through SnapshotProvider; when it
	// is unset we cannot help this follower until the log is available again.
	if n.SnapshotProvider == nil {
		n.logf("node %d cannot send snapshot to %d: no provider", n.id, to)
		return
	}
	snap, err := n.SnapshotProvider()
	if err != nil || snap == nil {
		n.logf("node %d snapshot unavailable for %d: %v", n.id, to, err)
		return
	}
	pr.becomeSnapshot(snap.Index)
	n.send(Message{Type: MsgSnapshotReq, To: to, Term: n.term, Snapshot: snap})
	n.logf("node %d sending snapshot at index %d to %d", n.id, snap.Index, to)
}

func (n *Node) send(m Message) {
	m.From = n.id
	if m.Term == 0 {
		m.Term = n.term
	}
	n.msgs = append(n.msgs, m)
}

// maybeCommit advances the commit index to the highest index replicated on a
// majority, subject to the current-term restriction of §5.4.2.
func (n *Node) maybeCommit() bool {
	if n.role != Leader {
		return false
	}
	matches := make([]uint64, 0, len(n.conf.Voters))
	for _, id := range n.conf.Voters {
		if pr, ok := n.progress[id]; ok {
			matches = append(matches, pr.Match)
		}
	}
	if len(matches) == 0 {
		return false
	}
	// The quorum-th largest match index is replicated on a majority.
	insertionSortDesc(matches)
	candidate := matches[n.quorum()-1]
	if candidate <= n.log.committed {
		return false
	}
	// A leader may only count replicas of entries from its *own* term. An entry
	// from a previous term replicated on a majority is still not safe to commit
	// -- Figure 8 in the paper shows exactly how doing so loses data.
	if t, ok := n.log.term(candidate); !ok || t != n.term {
		return false
	}
	return n.log.commitTo(candidate)
}

func insertionSortDesc(a []uint64) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] < v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}

// ---------------------------------------------------------------------------
// Configuration changes
// ---------------------------------------------------------------------------

// ApplyConfChange must be called by the application when it applies an
// EntryConfChange, so the node's membership tracks the state machine's.
func (n *Node) ApplyConfChange(cc ConfChange) Config {
	n.applyConfChange(cc)
	n.resetProgress()
	if n.role == Leader {
		// Membership changed: the quorum may have shrunk, which can make
		// already-replicated entries committable.
		n.maybeCommit()
	}
	return n.conf.Clone()
}

func (n *Node) applyConfChange(cc ConfChange) {
	switch cc.Type {
	case ConfChangeAddNode:
		if !n.conf.Has(cc.NodeID) {
			n.conf.Voters = append(n.conf.Voters, cc.NodeID)
		}
	case ConfChangeRemoveNode:
		out := n.conf.Voters[:0]
		for _, v := range n.conf.Voters {
			if v != cc.NodeID {
				out = append(out, v)
			}
		}
		n.conf.Voters = out
		delete(n.progress, cc.NodeID)
	}
}

// ---------------------------------------------------------------------------
// Ready / Advance
// ---------------------------------------------------------------------------

// Ready returns the work accumulated since the last Advance. The caller must
// honour the ordering contract documented on the Ready type.
func (n *Node) Ready() Ready {
	rd := Ready{
		Messages:   n.msgs,
		ReadStates: n.readStates,
		Role:       n.role,
		Leader:     n.lead,
		Snapshot:   n.pendingSnapshot,
	}
	if hs := n.hardState(); !hs.Equal(n.prevHardState) {
		rd.HardState = hs
	}
	// Unpersisted log tail. persistedIndex tracks what Advance has acknowledged;
	// slice clamps lo to firstIndex, so a compacted prefix is never re-offered.
	if lo := n.persistedIndex + 1; lo <= n.log.lastIndex() {
		rd.Entries = n.log.slice(lo, n.log.lastIndex()+1, 0)
	}
	rd.CommittedEntries = n.log.nextApplicable(n.cfg.MaxCommittedEntriesPerReady)
	return rd
}

// HasReady reports whether there is work pending, letting the driver loop avoid
// allocating a Ready when idle.
func (n *Node) HasReady() bool {
	if len(n.msgs) > 0 || len(n.readStates) > 0 || n.pendingSnapshot != nil {
		return true
	}
	if hs := n.hardState(); !hs.Equal(n.prevHardState) {
		return true
	}
	if n.persistedIndex < n.log.lastIndex() {
		return true
	}
	return n.log.applied < n.log.committed
}

// Advance acknowledges that the caller has completed the work in rd.
func (n *Node) Advance(rd Ready) {
	if !rd.HardState.IsEmpty() {
		n.prevHardState = rd.HardState
	}
	if rd.Snapshot != nil {
		n.persistedIndex = rd.Snapshot.Index
		if n.pendingSnapshot == rd.Snapshot {
			n.pendingSnapshot = nil
		}
	}
	if len(rd.Entries) > 0 {
		last := rd.Entries[len(rd.Entries)-1].Index
		if last > n.persistedIndex {
			n.persistedIndex = last
		}
	}
	if len(rd.CommittedEntries) > 0 {
		n.log.appliedTo(rd.CommittedEntries[len(rd.CommittedEntries)-1].Index)
	}
	// Drop exactly what was handed out; messages produced while the caller was
	// working (there are none in a single-threaded driver, but be explicit)
	// are retained.
	n.msgs = n.msgs[len(rd.Messages):]
	if len(n.msgs) == 0 {
		n.msgs = nil
	}
	n.readStates = n.readStates[len(rd.ReadStates):]
	if len(n.readStates) == 0 {
		n.readStates = nil
	}
}

// Compact discards log entries up to index, which the application must have
// captured in a snapshot first.
func (n *Node) Compact(index, term uint64) {
	n.log.compact(index, term)
	if n.persistedIndex < index {
		n.persistedIndex = index
	}
}

// TermAt returns the term of the entry at index, reporting false if that index
// has been compacted away or lies beyond the end of the log.
func (n *Node) TermAt(index uint64) (uint64, bool) { return n.log.term(index) }

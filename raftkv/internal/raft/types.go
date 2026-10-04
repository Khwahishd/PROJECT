// Package raft implements the Raft consensus algorithm as described in
// "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014),
// including the pre-vote and log-conflict optimisations from §4 of Ongaro's thesis.
//
// The implementation is a *pure state machine*: it owns no goroutines, no locks,
// no clock and no network. Time advances only through Tick, input arrives only
// through Step/Propose, and all side effects leave through Ready. Determinism is
// the point -- it is what lets the simulation harness in internal/raft/sim replay
// partitions, reorderings and crashes reproducibly from a single seed.
package raft

import "fmt"

// NodeID uniquely identifies a member of the cluster. Zero is reserved to mean
// "no node" (e.g. an empty vote).
type NodeID uint64

// None is the zero NodeID, used where a field is unset.
const None NodeID = 0

// Role is the Raft role a node currently occupies.
type Role uint8

const (
	// Follower passively replicates the leader's log.
	Follower Role = iota
	// PreCandidate is soliciting pre-votes. It has not incremented its term and
	// cannot disrupt a healthy leader (thesis §9.6).
	PreCandidate
	// Candidate has incremented its term and is soliciting real votes.
	Candidate
	// Leader accepts proposals and replicates them.
	Leader
)

func (r Role) String() string {
	switch r {
	case Follower:
		return "follower"
	case PreCandidate:
		return "pre-candidate"
	case Candidate:
		return "candidate"
	case Leader:
		return "leader"
	default:
		return fmt.Sprintf("role(%d)", uint8(r))
	}
}

// EntryType distinguishes replicated application commands from internal
// configuration changes.
type EntryType uint8

const (
	// EntryNormal carries an opaque application command.
	EntryNormal EntryType = iota
	// EntryConfChange carries a marshalled ConfChange.
	EntryConfChange
)

// Entry is a single record in the replicated log.
type Entry struct {
	Term  uint64
	Index uint64
	Type  EntryType
	Data  []byte
}

// MessageType enumerates the RPCs exchanged between peers. Request/response
// pairs are adjacent so that tests can assert on pairing.
type MessageType uint8

const (
	MsgVoteReq MessageType = iota
	MsgVoteResp
	MsgPreVoteReq
	MsgPreVoteResp
	MsgAppendReq
	MsgAppendResp
	MsgSnapshotReq
	MsgSnapshotResp
	// MsgReadIndexReq / MsgReadIndexResp implement linearizable reads without
	// writing to the log (thesis §6.4).
	MsgReadIndexReq
	MsgReadIndexResp
	// MsgTimeoutNow is sent by a leader stepping down to a chosen successor so
	// it can campaign immediately rather than waiting out an election timeout.
	MsgTimeoutNow
)

func (t MessageType) String() string {
	switch t {
	case MsgVoteReq:
		return "VoteReq"
	case MsgVoteResp:
		return "VoteResp"
	case MsgPreVoteReq:
		return "PreVoteReq"
	case MsgPreVoteResp:
		return "PreVoteResp"
	case MsgAppendReq:
		return "AppendReq"
	case MsgAppendResp:
		return "AppendResp"
	case MsgSnapshotReq:
		return "SnapshotReq"
	case MsgSnapshotResp:
		return "SnapshotResp"
	case MsgReadIndexReq:
		return "ReadIndexReq"
	case MsgReadIndexResp:
		return "ReadIndexResp"
	case MsgTimeoutNow:
		return "TimeoutNow"
	default:
		return fmt.Sprintf("msg(%d)", uint8(t))
	}
}

// Message is the single wire type for all peer-to-peer traffic. A single struct
// (rather than one per RPC) keeps the transport interface trivial and makes the
// simulation harness able to treat every message uniformly when it drops,
// delays or duplicates traffic.
type Message struct {
	Type MessageType
	From NodeID
	To   NodeID
	// Term is the sender's term. PreVote requests deliberately carry
	// CandidateTerm+1 without the sender having adopted that term.
	Term uint64

	// AppendReq fields.
	PrevLogIndex uint64
	PrevLogTerm  uint64
	Entries      []Entry
	LeaderCommit uint64

	// VoteReq / PreVoteReq fields.
	LastLogIndex uint64
	LastLogTerm  uint64

	// Response fields.
	Granted bool
	Reject  bool
	// MatchIndex is the highest index the responder has accepted (AppendResp).
	MatchIndex uint64
	// ConflictIndex / ConflictTerm let a leader skip a whole conflicting term in
	// one round trip instead of decrementing nextIndex entry by entry (§5.3).
	ConflictIndex uint64
	ConflictTerm  uint64

	// Snapshot transfer.
	Snapshot *Snapshot

	// ReadIndex correlation token, opaque to raft.
	Context []byte
	// ReadIndex is the commit index the leader confirmed for Context.
	ReadIndex uint64
}

// Snapshot is a compacted prefix of the log plus the cluster configuration at
// the moment it was taken.
type Snapshot struct {
	Index uint64
	Term  uint64
	// Conf is the membership in effect at Index. Snapshots must carry it:
	// a restoring node has no log entries left from which to rebuild it.
	Conf Config
	Data []byte
}

// Config is the set of voting members of the cluster.
type Config struct {
	Voters []NodeID
}

// Clone returns a deep copy, so callers can retain a configuration across
// subsequent mutations.
func (c Config) Clone() Config {
	out := Config{Voters: make([]NodeID, len(c.Voters))}
	copy(out.Voters, c.Voters)
	return out
}

// Has reports whether id is a voting member.
func (c Config) Has(id NodeID) bool {
	for _, v := range c.Voters {
		if v == id {
			return true
		}
	}
	return false
}

// Quorum is the smallest number of votes constituting a majority.
func (c Config) Quorum() int { return len(c.Voters)/2 + 1 }

// ConfChangeType is the kind of membership mutation requested.
type ConfChangeType uint8

const (
	// ConfChangeAddNode adds a voter.
	ConfChangeAddNode ConfChangeType = iota
	// ConfChangeRemoveNode removes a voter.
	ConfChangeRemoveNode
)

// ConfChange is the payload of an EntryConfChange entry. Changes are applied
// one node at a time, which is sufficient to guarantee that the old and new
// majorities always overlap (§4.1).
type ConfChange struct {
	Type   ConfChangeType
	NodeID NodeID
}

// HardState is the subset of node state that must survive a crash. It must be
// flushed to durable storage before any message in the same Ready is sent.
type HardState struct {
	Term   uint64
	Vote   NodeID
	Commit uint64
}

// IsEmpty reports whether hs carries no information worth persisting.
func (hs HardState) IsEmpty() bool {
	return hs.Term == 0 && hs.Vote == None && hs.Commit == 0
}

// Equal compares two HardStates field-wise.
func (hs HardState) Equal(other HardState) bool {
	return hs.Term == other.Term && hs.Vote == other.Vote && hs.Commit == other.Commit
}

// Status is a read-only snapshot of a node's state, for observability and tests.
type Status struct {
	ID        NodeID
	Role      Role
	Term      uint64
	Leader    NodeID
	Commit    uint64
	Applied   uint64
	LastIndex uint64
	Config    Config
	// Progress is populated only on a leader.
	Progress map[NodeID]Progress
}

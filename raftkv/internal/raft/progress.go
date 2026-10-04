package raft

import "fmt"

// ProgressState describes how a leader is currently feeding one follower.
type ProgressState uint8

const (
	// StateProbe sends at most one AppendEntries per heartbeat interval while
	// the leader searches for the follower's matching prefix. Without it, a
	// leader reconnecting to a far-behind follower would flood the link with
	// speculative appends that are all rejected.
	StateProbe ProgressState = iota
	// StateReplicate streams entries optimistically, advancing Next on send
	// rather than on acknowledgement, so the pipe stays full.
	StateReplicate
	// StateSnapshot pauses log replication while a snapshot is in flight.
	StateSnapshot
)

func (s ProgressState) String() string {
	switch s {
	case StateProbe:
		return "probe"
	case StateReplicate:
		return "replicate"
	case StateSnapshot:
		return "snapshot"
	default:
		return fmt.Sprintf("state(%d)", uint8(s))
	}
}

// Progress is the leader's view of one follower's replication state.
type Progress struct {
	// Match is the highest index known to be replicated on the follower.
	Match uint64
	// Next is the index of the next entry to send.
	Next uint64

	State ProgressState

	// Paused suppresses redundant probes between heartbeats.
	Paused bool
	// PendingSnapshot is the snapshot index being transferred in StateSnapshot.
	PendingSnapshot uint64
	// RecentActive is set when the follower responds and cleared each election
	// timeout; the leader uses it to detect that it has lost quorum contact.
	RecentActive bool

	// stalledRounds counts consecutive heartbeat rounds in which Match did not
	// move while the follower was known to be behind. See
	// Node.recoverStalledFollowers.
	stalledRounds int
	// lastObservedMatch is the Match value seen at the previous heartbeat round.
	lastObservedMatch uint64
}

func (pr *Progress) becomeProbe() {
	// If we were snapshotting, resume just past the snapshot we sent rather
	// than restarting the search from Match+1.
	if pr.State == StateSnapshot {
		pr.Next = max64(pr.Match+1, pr.PendingSnapshot+1)
	} else {
		pr.Next = pr.Match + 1
	}
	pr.State = StateProbe
	pr.PendingSnapshot = 0
	pr.Paused = false
}

func (pr *Progress) becomeReplicate() {
	pr.State = StateReplicate
	pr.PendingSnapshot = 0
	pr.Next = pr.Match + 1
	pr.Paused = false
}

func (pr *Progress) becomeSnapshot(index uint64) {
	pr.State = StateSnapshot
	pr.PendingSnapshot = index
	pr.Paused = true
}

// maybeUpdate advances Match/Next after a successful AppendEntries. It returns
// false for a stale or duplicated response, which must not move Match backwards.
func (pr *Progress) maybeUpdate(matchIndex uint64) bool {
	updated := false
	if pr.Match < matchIndex {
		pr.Match = matchIndex
		updated = true
		pr.Paused = false
	}
	if pr.Next < matchIndex+1 {
		pr.Next = matchIndex + 1
	}
	return updated
}

// maybeDecrTo backs Next up after a rejection, using the follower's conflict
// hint. It returns false if the rejection is stale and should be ignored.
func (pr *Progress) maybeDecrTo(rejectedIndex, conflictIndex uint64) bool {
	if pr.State == StateReplicate {
		// In replicate mode Next runs ahead of Match, so a rejection for an
		// index at or below Match is a duplicate of one already handled.
		if rejectedIndex <= pr.Match {
			return false
		}
		pr.Next = pr.Match + 1
		return true
	}
	// In probe mode Next-1 is exactly the index we probed; anything else is stale.
	if pr.Next-1 != rejectedIndex {
		return false
	}
	pr.Next = max64(min64(rejectedIndex, conflictIndex), 1)
	pr.Paused = false
	return true
}

// canSend reports whether the leader may send to this follower right now.
func (pr *Progress) canSend() bool {
	switch pr.State {
	case StateProbe:
		return !pr.Paused
	case StateReplicate:
		return true
	case StateSnapshot:
		return false
	default:
		return false
	}
}

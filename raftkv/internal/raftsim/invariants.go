package raftsim

import (
	"fmt"

	"github.com/Khwahishd/raftkv/internal/raft"
)

// Violation describes a breach of one of Raft's safety properties.
type Violation struct {
	Property string
	Detail   string
	Tick     int
}

func (v Violation) Error() string {
	return fmt.Sprintf("t=%d: %s violated: %s", v.Tick, v.Property, v.Detail)
}

// history records, per log index, the entry that some node committed there.
// Any disagreement is a State Machine Safety violation.
type history struct {
	committed map[uint64]raft.Entry
	// maxTermLeaders records the leader elected for each term, to check that no
	// term ever has two.
	leaders map[uint64]raft.NodeID
}

func newHistory() *history {
	return &history{
		committed: map[uint64]raft.Entry{},
		leaders:   map[uint64]raft.NodeID{},
	}
}

// Checker continuously verifies the safety properties of a running simulation.
//
// These are the five properties from Figure 3 of the Raft paper. Liveness is
// deliberately not checked here: under an adversarial network Raft is not
// guaranteed to make progress, so liveness is asserted separately by tests that
// heal the network first.
type Checker struct {
	sim  *Sim
	hist *history
	// lastAppliedLen tracks how far each node's applied log has been checked,
	// so each entry is examined once.
	lastAppliedLen map[raft.NodeID]int
}

// NewChecker builds a checker bound to a simulation.
func NewChecker(s *Sim) *Checker {
	return &Checker{
		sim:            s,
		hist:           newHistory(),
		lastAppliedLen: map[raft.NodeID]int{},
	}
}

// Check runs every invariant and returns the first violation found, or nil.
// Call it after each tick for the tightest possible failure localisation.
func (c *Checker) Check() error {
	for _, fn := range []func() error{
		c.checkElectionSafety,
		c.checkLeaderAppendOnly,
		c.checkLogMatching,
		c.checkStateMachineSafety,
		c.checkCommitMonotonic,
	} {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}

// Election Safety: at most one leader can be elected in a given term (§5.2).
func (c *Checker) checkElectionSafety() error {
	for _, n := range c.sim.Nodes() {
		if n.crashed {
			continue
		}
		st := n.raft.Status()
		if st.Role != raft.Leader {
			continue
		}
		if prev, ok := c.hist.leaders[st.Term]; ok && prev != n.ID {
			return Violation{
				Property: "Election Safety",
				Tick:     c.sim.tick,
				Detail: fmt.Sprintf("term %d has two leaders: node %d and node %d",
					st.Term, prev, n.ID),
			}
		}
		c.hist.leaders[st.Term] = n.ID
	}
	return nil
}

// Leader Append-Only: a leader never overwrites or deletes entries in its log,
// it only appends (§5.3). We approximate this by checking that each node's
// applied sequence only ever grows and never rewrites an earlier position.
func (c *Checker) checkLeaderAppendOnly() error {
	for _, n := range c.sim.Nodes() {
		applied := n.applied
		for i := 1; i < len(applied); i++ {
			if applied[i].Index <= applied[i-1].Index {
				return Violation{
					Property: "Leader Append-Only",
					Tick:     c.sim.tick,
					Detail: fmt.Sprintf("node %d applied index %d after %d (non-monotonic)",
						n.ID, applied[i].Index, applied[i-1].Index),
				}
			}
		}
	}
	return nil
}

// Log Matching: if two logs contain an entry with the same index and term, then
// the logs are identical in all entries up through that index (§5.3).
func (c *Checker) checkLogMatching() error {
	nodes := c.sim.Nodes()
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			a, b := nodes[i], nodes[j]
			if err := c.compareLogs(a, b); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *Checker) compareLogs(a, b *Node) error {
	ia := map[uint64]raft.Entry{}
	for _, e := range a.storage.Entries {
		ia[e.Index] = e
	}
	// Find the highest index where both have an entry with a matching term.
	var anchor uint64
	for _, eb := range b.storage.Entries {
		if ea, ok := ia[eb.Index]; ok && ea.Term == eb.Term && eb.Index > anchor {
			anchor = eb.Index
		}
	}
	if anchor == 0 {
		return nil
	}
	for _, eb := range b.storage.Entries {
		if eb.Index > anchor {
			continue
		}
		ea, ok := ia[eb.Index]
		if !ok {
			continue // compacted away on a; the prefix is still implied equal
		}
		if ea.Term != eb.Term || string(ea.Data) != string(eb.Data) {
			return Violation{
				Property: "Log Matching",
				Tick:     c.sim.tick,
				Detail: fmt.Sprintf("nodes %d and %d agree at index %d but differ at index %d (terms %d vs %d)",
					a.ID, b.ID, anchor, eb.Index, ea.Term, eb.Term),
			}
		}
	}
	return nil
}

// State Machine Safety: if a server has applied a log entry at a given index to
// its state machine, no other server will ever apply a different log entry for
// the same index (§5.4.3). This is the property that actually protects user
// data, and the one a subtly wrong commit rule breaks.
func (c *Checker) checkStateMachineSafety() error {
	for _, n := range c.sim.Nodes() {
		start := c.lastAppliedLen[n.ID]
		if start > len(n.applied) {
			// The node restarted and is replaying its log, so its applied slice
			// shrank. Re-verify from the beginning; the global history below is
			// what carries the pre-crash record.
			start = 0
		}
		for i := start; i < len(n.applied); i++ {
			e := n.applied[i]
			prev, ok := c.hist.committed[e.Index]
			if !ok {
				c.hist.committed[e.Index] = e
				continue
			}
			if prev.Term != e.Term || string(prev.Data) != string(e.Data) {
				return Violation{
					Property: "State Machine Safety",
					Tick:     c.sim.tick,
					Detail: fmt.Sprintf("index %d applied as (term %d, %q) elsewhere but as (term %d, %q) on node %d",
						e.Index, prev.Term, prev.Data, e.Term, e.Data, n.ID),
				}
			}
		}
		c.lastAppliedLen[n.ID] = len(n.applied)
	}
	return nil
}

// Commit indices must never move backwards on a live node.
func (c *Checker) checkCommitMonotonic() error {
	for _, n := range c.sim.Nodes() {
		if n.crashed {
			continue
		}
		st := n.raft.Status()
		if st.Applied > st.Commit {
			return Violation{
				Property: "Commit Invariant",
				Tick:     c.sim.tick,
				Detail:   fmt.Sprintf("node %d applied %d > commit %d", n.ID, st.Applied, st.Commit),
			}
		}
	}
	return nil
}

// RunChecked advances the simulation, verifying every invariant after each
// tick. On violation it returns the error along with the recorded trace, so a
// failure report contains the exact interleaving that produced it.
func (c *Checker) RunChecked(ticks int) error {
	for i := 0; i < ticks; i++ {
		c.sim.Tick()
		if err := c.Check(); err != nil {
			return err
		}
	}
	return nil
}

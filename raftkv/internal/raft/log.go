package raft

import "fmt"

// raftLog holds the in-memory tail of the replicated log together with the
// metadata of the snapshot that replaced its prefix.
//
// Index space is global and 1-based: index 0 is the "empty" position before the
// first entry. After compaction the log keeps snapIndex/snapTerm so that an
// AppendEntries whose PrevLogIndex lands exactly on the snapshot boundary can
// still be matched without the entry itself being present.
type raftLog struct {
	// entries holds indices (snapIndex, snapIndex+len(entries)].
	entries []Entry

	snapIndex uint64
	snapTerm  uint64

	committed uint64
	applied   uint64
}

func newRaftLog(snapIndex, snapTerm uint64) *raftLog {
	return &raftLog{
		snapIndex: snapIndex,
		snapTerm:  snapTerm,
		committed: snapIndex,
		applied:   snapIndex,
	}
}

// firstIndex is the lowest index still retrievable from the log.
func (l *raftLog) firstIndex() uint64 { return l.snapIndex + 1 }

// lastIndex is the highest index in the log, or snapIndex if it is empty.
func (l *raftLog) lastIndex() uint64 {
	if n := len(l.entries); n > 0 {
		return l.entries[n-1].Index
	}
	return l.snapIndex
}

// term returns the term of the entry at index i.
//
// It reports ok=false when i has been compacted away or is beyond the end of
// the log -- callers must distinguish "term 0" from "unknown", since a wrong
// answer here silently breaks the log matching property.
func (l *raftLog) term(i uint64) (uint64, bool) {
	switch {
	case i == l.snapIndex:
		return l.snapTerm, true
	case i < l.snapIndex || i > l.lastIndex():
		return 0, false
	default:
		return l.entries[i-l.snapIndex-1].Term, true
	}
}

// lastTerm is the term of the final entry, or the snapshot term if empty.
func (l *raftLog) lastTerm() uint64 {
	t, ok := l.term(l.lastIndex())
	if !ok {
		return 0
	}
	return t
}

// slice returns entries in [lo, hi), capped at maxEntries. The returned slice
// shares no backing array with the log, so callers may retain it.
func (l *raftLog) slice(lo, hi uint64, maxEntries int) []Entry {
	if lo < l.firstIndex() {
		lo = l.firstIndex()
	}
	if hi > l.lastIndex()+1 {
		hi = l.lastIndex() + 1
	}
	if lo >= hi {
		return nil
	}
	if maxEntries > 0 && hi-lo > uint64(maxEntries) {
		hi = lo + uint64(maxEntries)
	}
	out := make([]Entry, hi-lo)
	copy(out, l.entries[lo-l.snapIndex-1:hi-l.snapIndex-1])
	return out
}

// isUpToDate implements the election restriction of §5.4.1: a candidate's log
// must be at least as up to date as the voter's, comparing last term first and
// breaking ties on length.
func (l *raftLog) isUpToDate(lastIndex, lastTerm uint64) bool {
	myTerm := l.lastTerm()
	if lastTerm != myTerm {
		return lastTerm > myTerm
	}
	return lastIndex >= l.lastIndex()
}

// matches reports whether the log contains an entry at index with the given
// term -- the consistency check performed on every AppendEntries.
func (l *raftLog) matches(index, term uint64) bool {
	t, ok := l.term(index)
	return ok && t == term
}

// append adds entries to the end of the log. Entries must be contiguous with
// the current last index; the caller (maybeAppend / propose) guarantees this.
func (l *raftLog) append(entries ...Entry) {
	if len(entries) == 0 {
		return
	}
	if want := l.lastIndex() + 1; entries[0].Index != want {
		panic(fmt.Sprintf("raft: non-contiguous append at %d, want %d", entries[0].Index, want))
	}
	l.entries = append(l.entries, entries...)
}

// maybeAppend performs the follower side of AppendEntries.
//
// It returns the index of the last entry now in the log, the index at which the
// log was truncated (0 if it was not), and whether the consistency check
// passed. The truncation point matters to the caller: any index at or above it
// has been *rewritten*, so a previously-persisted copy on disk is now stale and
// must be written again.
//
// Entries that already match are skipped rather than rewritten. This matters
// for correctness, not just efficiency: a delayed duplicate of an older
// AppendEntries must not truncate committed entries that a later request
// already added (the classic "stale append truncates the log" bug).
func (l *raftLog) maybeAppend(prevIndex, prevTerm uint64, entries []Entry) (last, truncatedAt uint64, ok bool) {
	if !l.matches(prevIndex, prevTerm) {
		return 0, 0, false
	}

	// Skip the prefix that is already present with the same term.
	i := 0
	for ; i < len(entries); i++ {
		e := entries[i]
		if e.Index > l.lastIndex() {
			break
		}
		t, ok := l.term(e.Index)
		if !ok || t != e.Term {
			// Conflict: truncate from here. Truncating at or below the commit
			// index would mean a committed entry was overwritten, which Raft
			// forbids; assert rather than corrupt the state machine.
			if e.Index <= l.committed {
				panic(fmt.Sprintf("raft: attempted to truncate committed index %d (commit=%d)", e.Index, l.committed))
			}
			l.truncateFrom(e.Index)
			truncatedAt = e.Index
			break
		}
	}
	if i < len(entries) {
		l.append(entries[i:]...)
	}
	return prevIndex + uint64(len(entries)), truncatedAt, true
}

// truncateFrom discards all entries with index >= from.
func (l *raftLog) truncateFrom(from uint64) {
	if from <= l.snapIndex {
		l.entries = nil
		return
	}
	if from > l.lastIndex() {
		return
	}
	l.entries = l.entries[:from-l.snapIndex-1]
}

// commitTo advances the commit index. It never moves backwards and never
// advances past the end of the local log, which bounds a leader's LeaderCommit
// to what this node has actually stored.
func (l *raftLog) commitTo(i uint64) bool {
	if i <= l.committed {
		return false
	}
	if last := l.lastIndex(); i > last {
		i = last
	}
	if i <= l.committed {
		return false
	}
	l.committed = i
	return true
}

// nextApplicable returns committed-but-unapplied entries, oldest first.
func (l *raftLog) nextApplicable(maxEntries int) []Entry {
	lo := max64(l.applied+1, l.firstIndex())
	if lo > l.committed {
		return nil
	}
	return l.slice(lo, l.committed+1, maxEntries)
}

// appliedTo records that the state machine has consumed through index i.
func (l *raftLog) appliedTo(i uint64) {
	if i > l.applied {
		l.applied = i
	}
}

// compact discards entries up to and including index, recording the snapshot
// metadata that replaces them. It is a no-op if the log is already compacted
// past index.
func (l *raftLog) compact(index, term uint64) {
	if index <= l.snapIndex {
		return
	}
	if index > l.lastIndex() {
		l.entries = nil
	} else {
		keep := l.entries[index-l.snapIndex:]
		trimmed := make([]Entry, len(keep))
		copy(trimmed, keep)
		l.entries = trimmed
	}
	l.snapIndex = index
	l.snapTerm = term
	if l.committed < index {
		l.committed = index
	}
	if l.applied < index {
		l.applied = index
	}
}

// restore replaces the entire log with the state captured by a snapshot.
func (l *raftLog) restore(snap *Snapshot) {
	l.entries = nil
	l.snapIndex = snap.Index
	l.snapTerm = snap.Term
	l.committed = snap.Index
	l.applied = snap.Index
}

// findConflict locates the hint a follower returns when AppendEntries fails, so
// the leader can back up by a whole term instead of one index per round trip.
//
// If the follower's log is simply too short, it reports its own length. If it
// has a conflicting entry at prevIndex, it reports the first index of that
// conflicting term, letting the leader skip the term wholesale.
func (l *raftLog) findConflict(prevIndex uint64) (conflictIndex, conflictTerm uint64) {
	if prevIndex > l.lastIndex() {
		return l.lastIndex() + 1, 0
	}
	t, ok := l.term(prevIndex)
	if !ok {
		// Compacted away; ask for everything we no longer have.
		return l.firstIndex(), 0
	}
	idx := prevIndex
	for idx > l.firstIndex() {
		pt, ok := l.term(idx - 1)
		if !ok || pt != t {
			break
		}
		idx--
	}
	return idx, t
}

func max64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

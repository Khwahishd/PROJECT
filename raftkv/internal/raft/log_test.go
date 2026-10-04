package raft

import "testing"

func entries(spec ...[2]uint64) []Entry {
	out := make([]Entry, 0, len(spec))
	for _, s := range spec {
		out = append(out, Entry{Index: s[0], Term: s[1]})
	}
	return out
}

func TestLogTermAcrossSnapshotBoundary(t *testing.T) {
	l := newRaftLog(5, 3)
	l.append(entries([2]uint64{6, 4}, [2]uint64{7, 4})...)

	cases := []struct {
		index    uint64
		wantTerm uint64
		wantOK   bool
	}{
		{4, 0, false}, // compacted away
		{5, 3, true},  // the snapshot boundary itself is still addressable
		{6, 4, true},
		{7, 4, true},
		{8, 0, false}, // past the end
	}
	for _, tc := range cases {
		got, ok := l.term(tc.index)
		if ok != tc.wantOK || got != tc.wantTerm {
			t.Errorf("term(%d) = (%d, %v), want (%d, %v)", tc.index, got, ok, tc.wantTerm, tc.wantOK)
		}
	}
}

func TestIsUpToDateElectionRestriction(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 2})...)

	cases := []struct {
		name      string
		idx, term uint64
		want      bool
	}{
		{"higher term wins even with shorter log", 1, 3, true},
		{"lower term loses even with longer log", 9, 1, false},
		{"same term, longer log wins", 3, 2, true},
		{"same term, equal length is up to date", 2, 2, true},
		{"same term, shorter log loses", 1, 2, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := l.isUpToDate(tc.idx, tc.term); got != tc.want {
				t.Errorf("isUpToDate(%d,%d) = %v, want %v", tc.idx, tc.term, got, tc.want)
			}
		})
	}
}

func TestMaybeAppendSkipsMatchingPrefix(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 1}, [2]uint64{3, 1})...)
	l.commitTo(3)

	// A delayed duplicate of an older AppendEntries arrives. Its entries match
	// what we already have, so it must not truncate anything -- truncating here
	// would discard committed entries.
	last, truncatedAt, ok := l.maybeAppend(0, 0, entries([2]uint64{1, 1}, [2]uint64{2, 1}))
	if !ok {
		t.Fatal("maybeAppend rejected a matching prefix")
	}
	if last != 2 {
		t.Errorf("last = %d, want 2", last)
	}
	if truncatedAt != 0 {
		t.Errorf("truncatedAt = %d, want 0: a matching prefix must not truncate", truncatedAt)
	}
	if got := l.lastIndex(); got != 3 {
		t.Fatalf("stale append truncated the log: lastIndex = %d, want 3", got)
	}
}

func TestMaybeAppendTruncatesOnConflict(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 1}, [2]uint64{3, 1})...)

	// Entry 2 arrives with a different term: everything from index 2 on is
	// uncommitted and must be replaced.
	last, truncatedAt, ok := l.maybeAppend(1, 1, entries([2]uint64{2, 2}))
	if !ok {
		t.Fatal("maybeAppend rejected a valid append")
	}
	if last != 2 || l.lastIndex() != 2 {
		t.Fatalf("last = %d, lastIndex = %d, want 2 and 2", last, l.lastIndex())
	}
	// The caller needs the truncation point to invalidate what it already wrote
	// to disk at indices 2 and 3.
	if truncatedAt != 2 {
		t.Errorf("truncatedAt = %d, want 2", truncatedAt)
	}
	if term, _ := l.term(2); term != 2 {
		t.Errorf("term at 2 = %d, want 2", term)
	}
}

func TestMaybeAppendRejectsOnFailedConsistencyCheck(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1})...)

	if _, _, ok := l.maybeAppend(1, 9, entries([2]uint64{2, 2})); ok {
		t.Error("maybeAppend accepted an entry whose prevLogTerm does not match")
	}
	if _, _, ok := l.maybeAppend(5, 1, entries([2]uint64{6, 2})); ok {
		t.Error("maybeAppend accepted an entry beyond the end of the log")
	}
}

func TestCommitToNeverExceedsLocalLog(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 1})...)

	// A leader reports a commit index ahead of what we have stored. We may only
	// commit what we actually hold.
	l.commitTo(99)
	if l.committed != 2 {
		t.Fatalf("committed = %d, want 2", l.committed)
	}
	// Commit must never regress.
	l.commitTo(1)
	if l.committed != 2 {
		t.Fatalf("commit regressed to %d", l.committed)
	}
}

func TestFindConflictSkipsWholeTerm(t *testing.T) {
	l := newRaftLog(0, 0)
	// Indices 1..3 are term 1, 4..6 are term 5.
	l.append(entries(
		[2]uint64{1, 1}, [2]uint64{2, 1}, [2]uint64{3, 1},
		[2]uint64{4, 5}, [2]uint64{5, 5}, [2]uint64{6, 5},
	)...)

	ci, ct := l.findConflict(6)
	if ct != 5 || ci != 4 {
		t.Errorf("findConflict(6) = (%d, %d), want (4, 5): should report the first index of the conflicting term", ci, ct)
	}

	// Beyond the end of the log, the follower just reports its length.
	ci, ct = l.findConflict(10)
	if ci != 7 || ct != 0 {
		t.Errorf("findConflict(10) = (%d, %d), want (7, 0)", ci, ct)
	}
}

func TestCompactPreservesSuffix(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 1}, [2]uint64{3, 2}, [2]uint64{4, 2})...)
	l.commitTo(4)
	l.appliedTo(4)

	l.compact(2, 1)
	if l.firstIndex() != 3 {
		t.Errorf("firstIndex = %d, want 3", l.firstIndex())
	}
	if l.lastIndex() != 4 {
		t.Errorf("lastIndex = %d, want 4", l.lastIndex())
	}
	if _, ok := l.term(1); ok {
		t.Error("term(1) should be unavailable after compaction")
	}
	if term, ok := l.term(2); !ok || term != 1 {
		t.Errorf("term(2) = (%d,%v), want (1,true): the snapshot boundary stays addressable", term, ok)
	}
	if term, ok := l.term(3); !ok || term != 2 {
		t.Errorf("term(3) = (%d,%v), want (2,true)", term, ok)
	}
}

func TestSliceReturnsIndependentCopy(t *testing.T) {
	l := newRaftLog(0, 0)
	l.append(entries([2]uint64{1, 1}, [2]uint64{2, 1}, [2]uint64{3, 1})...)

	got := l.slice(1, 4, 0)
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	got[0].Term = 99
	if term, _ := l.term(1); term != 1 {
		t.Error("mutating the returned slice corrupted the log")
	}

	if n := len(l.slice(1, 4, 2)); n != 2 {
		t.Errorf("maxEntries ignored: got %d entries, want 2", n)
	}
}

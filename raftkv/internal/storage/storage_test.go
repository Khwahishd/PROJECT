package storage_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/storage"
)

func mkEntries(from, to uint64, term uint64) []raft.Entry {
	var out []raft.Entry
	for i := from; i <= to; i++ {
		out = append(out, raft.Entry{Index: i, Term: term, Data: []byte{byte(i)}})
	}
	return out
}

func open(t *testing.T, dir string) *storage.Storage {
	t.Helper()
	s, err := storage.Open(dir)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestAppendAndReloadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.Append(mkEntries(1, 50, 3)); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	reopened := open(t, dir)
	got, err := reopened.ReadEntries()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 50 {
		t.Fatalf("got %d entries, want 50", len(got))
	}
	for i, e := range got {
		if e.Index != uint64(i+1) || e.Term != 3 {
			t.Fatalf("entry %d = {index %d, term %d}, want {%d, 3}", i, e.Index, e.Term, i+1)
		}
	}
}

func TestAppendTruncatesConflictingSuffix(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.Append(mkEntries(1, 10, 1)); err != nil {
		t.Fatal(err)
	}
	// A new leader overwrites from index 6 with a higher term.
	if err := s.Append(mkEntries(6, 8, 2)); err != nil {
		t.Fatal(err)
	}

	got, err := s.ReadEntries()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 8 {
		t.Fatalf("got %d entries, want 8 (entries 9 and 10 must be gone)", len(got))
	}
	for _, e := range got {
		wantTerm := uint64(1)
		if e.Index >= 6 {
			wantTerm = 2
		}
		if e.Term != wantTerm {
			t.Errorf("index %d has term %d, want %d", e.Index, e.Term, wantTerm)
		}
	}
}

func TestTornTailIsDiscarded(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.Append(mkEntries(1, 20, 1)); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// Simulate a crash mid-append by lopping bytes off the end of the WAL.
	path := filepath.Join(dir, "raft.wal")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-7); err != nil {
		t.Fatal(err)
	}

	reopened := open(t, dir)
	got, err := reopened.ReadEntries()
	if err != nil {
		t.Fatalf("reload after torn write: %v", err)
	}
	if len(got) != 19 {
		t.Fatalf("got %d entries, want 19: the partially written record must be dropped", len(got))
	}
	// And the log must be usable again: the next append lands at 20, not after
	// the garbage.
	if err := reopened.Append(mkEntries(20, 22, 2)); err != nil {
		t.Fatalf("append after recovery: %v", err)
	}
	got, err = reopened.ReadEntries()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 22 || got[len(got)-1].Index != 22 {
		t.Fatalf("after recovery got %d entries ending at %d, want 22 ending at 22",
			len(got), got[len(got)-1].Index)
	}
}

func TestCorruptRecordStopsReplay(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.Append(mkEntries(1, 10, 1)); err != nil {
		t.Fatal(err)
	}
	s.Close()

	// Flip a bit in the middle of the file; the CRC must catch it.
	path := filepath.Join(dir, "raft.wal")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	b[len(b)/2] ^= 0xFF
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}

	reopened := open(t, dir)
	got, err := reopened.ReadEntries()
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(got) >= 10 {
		t.Fatalf("got %d entries; corruption should have stopped replay short of 10", len(got))
	}
}

func TestHardStateSurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)

	// A node with no prior state must report the zero value, not an error.
	hs, err := s.LoadHardState()
	if err != nil {
		t.Fatalf("load on empty dir: %v", err)
	}
	if !hs.IsEmpty() {
		t.Fatalf("expected empty hard state, got %+v", hs)
	}

	want := raft.HardState{Term: 7, Vote: 3, Commit: 42}
	if err := s.SaveHardState(want); err != nil {
		t.Fatal(err)
	}
	s.Close()

	got, err := open(t, dir).LoadHardState()
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("got %+v, want %+v: a lost vote lets a node vote twice in one term", got, want)
	}
}

func TestHardStateRejectsCorruption(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.SaveHardState(raft.HardState{Term: 2, Vote: 1, Commit: 9}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "hardstate")
	b, _ := os.ReadFile(path)
	b[len(b)-1] ^= 0xFF
	os.WriteFile(path, b, 0o644)

	if _, err := open(t, dir).LoadHardState(); err == nil {
		t.Fatal("corrupt hard state was accepted; it must be reported, not silently used")
	}
}

func TestSnapshotRoundTripAndPruning(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)

	if _, err := s.LoadSnapshot(); err == nil {
		t.Fatal("expected ErrNotFound on an empty directory")
	}

	for i := uint64(1); i <= 6; i++ {
		snap := &raft.Snapshot{
			Index: i * 100, Term: i,
			Conf: raft.Config{Voters: []raft.NodeID{1, 2, 3}},
			Data: []byte{byte(i)},
		}
		if err := s.SaveSnapshot(snap); err != nil {
			t.Fatal(err)
		}
	}

	got, err := s.LoadSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got.Index != 600 || got.Term != 6 {
		t.Fatalf("loaded snapshot {%d,%d}, want {600,6}", got.Index, got.Term)
	}
	if len(got.Conf.Voters) != 3 {
		t.Errorf("snapshot lost its configuration: %v", got.Conf.Voters)
	}

	// Only the most recent few are retained.
	ents, _ := os.ReadDir(dir)
	count := 0
	for _, e := range ents {
		if len(e.Name()) > 8 && e.Name()[:8] == "snapshot" {
			count++
		}
	}
	if count > 3 {
		t.Errorf("kept %d snapshots, want at most 3", count)
	}
}

func TestCompactDropsPrefix(t *testing.T) {
	dir := t.TempDir()
	s := open(t, dir)
	if err := s.Append(mkEntries(1, 100, 1)); err != nil {
		t.Fatal(err)
	}
	if err := s.Compact(60); err != nil {
		t.Fatalf("compact: %v", err)
	}
	got, err := s.ReadEntries()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 40 || got[0].Index != 61 {
		t.Fatalf("after compact got %d entries starting at %d, want 40 starting at 61",
			len(got), got[0].Index)
	}
	// The compacted log must still accept appends.
	if err := s.Append(mkEntries(101, 105, 2)); err != nil {
		t.Fatalf("append after compact: %v", err)
	}
	got, _ = s.ReadEntries()
	if got[len(got)-1].Index != 105 {
		t.Fatalf("last index %d, want 105", got[len(got)-1].Index)
	}
}

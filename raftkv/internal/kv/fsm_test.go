package kv_test

import (
	"errors"
	"testing"

	"github.com/Khwahishd/raftkv/internal/kv"
)

func TestPutGetDelete(t *testing.T) {
	f := kv.New()
	f.Apply(1, kv.Command{Op: kv.OpPut, Key: "a", Value: "1"})

	got, err := f.Get("a")
	if err != nil || got != "1" {
		t.Fatalf("Get(a) = (%q, %v), want (\"1\", nil)", got, err)
	}

	res := f.Apply(2, kv.Command{Op: kv.OpDelete, Key: "a"})
	if !res.Applied || res.Previous != "1" {
		t.Fatalf("delete result = %+v, want applied with previous \"1\"", res)
	}
	if _, err := f.Get("a"); !errors.Is(err, kv.ErrNotFound) {
		t.Fatalf("Get after delete returned %v, want ErrNotFound", err)
	}

	// Deleting a missing key is not an error but reports that nothing changed.
	if res := f.Apply(3, kv.Command{Op: kv.OpDelete, Key: "ghost"}); res.Applied {
		t.Error("deleting a missing key reported Applied")
	}
}

func TestCompareAndSwap(t *testing.T) {
	f := kv.New()

	// Create-if-absent.
	if res := f.Apply(1, kv.Command{Op: kv.OpCAS, Key: "k", Value: "v1", ExpectAbsent: true}); !res.Applied {
		t.Fatal("CAS with ExpectAbsent failed on a missing key")
	}
	// The same CAS must now fail -- this is what makes it usable as a lock.
	if res := f.Apply(2, kv.Command{Op: kv.OpCAS, Key: "k", Value: "v2", ExpectAbsent: true}); res.Applied {
		t.Fatal("CAS with ExpectAbsent succeeded on an existing key")
	}
	if res := f.Apply(3, kv.Command{Op: kv.OpCAS, Key: "k", Value: "v2", Expect: "wrong"}); res.Applied {
		t.Fatal("CAS succeeded with a mismatched expectation")
	}
	if res := f.Apply(4, kv.Command{Op: kv.OpCAS, Key: "k", Value: "v2", Expect: "v1"}); !res.Applied {
		t.Fatal("CAS failed with a correct expectation")
	}
	if got, _ := f.Get("k"); got != "v2" {
		t.Fatalf("value = %q, want \"v2\"", got)
	}
}

func TestRetriedRequestIsNotAppliedTwice(t *testing.T) {
	f := kv.New()
	cmd := kv.Command{Op: kv.OpPut, Key: "k", Value: "v1", ClientID: 7, RequestID: 1}
	first := f.Apply(1, cmd)

	// The client timed out and retried; raft faithfully replicated it again.
	second := f.Apply(2, cmd)
	if second != first {
		t.Fatalf("retry returned %+v, want the original result %+v", second, first)
	}

	// A CAS makes double-application observable: applying it twice would
	// succeed the first time and fail the second.
	f.Apply(3, kv.Command{Op: kv.OpPut, Key: "c", Value: "0"})
	casCmd := kv.Command{Op: kv.OpCAS, Key: "c", Value: "1", Expect: "0", ClientID: 7, RequestID: 2}
	r1 := f.Apply(4, casCmd)
	r2 := f.Apply(5, casCmd)
	if !r1.Applied {
		t.Fatal("first CAS should have applied")
	}
	if !r2.Applied {
		t.Fatal("retried CAS returned a different answer; deduplication failed")
	}
}

func TestSnapshotRestoreRoundTrip(t *testing.T) {
	f := kv.New()
	for i := 0; i < 100; i++ {
		f.Apply(uint64(i+1), kv.Command{
			Op: kv.OpPut, Key: string(rune('a' + i%26)), Value: string(rune('A' + i%26)),
			ClientID: uint64(i%5) + 1, RequestID: uint64(i),
		})
	}
	snap := f.Snapshot()

	restored := kv.New()
	if err := restored.Restore(snap); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if restored.AppliedIndex() != f.AppliedIndex() {
		t.Fatalf("applied index %d, want %d", restored.AppliedIndex(), f.AppliedIndex())
	}
	if restored.Len() != f.Len() {
		t.Fatalf("restored %d keys, want %d", restored.Len(), f.Len())
	}
	for _, k := range f.Keys() {
		want, _ := f.Get(k)
		got, err := restored.Get(k)
		if err != nil || got != want {
			t.Fatalf("restored[%s] = (%q, %v), want %q", k, got, err, want)
		}
	}
}

func TestSnapshotIsDeterministic(t *testing.T) {
	// Two FSMs reaching the same state by different insertion orders must
	// produce identical snapshot bytes, or replicas will appear to diverge.
	build := func(order []int) []byte {
		f := kv.New()
		for i, k := range order {
			f.Apply(uint64(i+1), kv.Command{Op: kv.OpPut, Key: string(rune('a' + k)), Value: "x"})
		}
		return f.Snapshot()
	}
	a := build([]int{0, 1, 2, 3, 4, 5, 6, 7})
	b := build([]int{7, 3, 1, 5, 0, 6, 2, 4})
	if string(a) != string(b) {
		t.Fatal("snapshot bytes depend on insertion order; map iteration has leaked into the encoding")
	}
}

func TestCommandCodecRoundTrip(t *testing.T) {
	cases := []kv.Command{
		{Op: kv.OpPut, Key: "k", Value: "v"},
		{Op: kv.OpDelete, Key: "gone"},
		{Op: kv.OpCAS, Key: "k", Value: "new", Expect: "old", ClientID: 9, RequestID: 42},
		{Op: kv.OpCAS, Key: "k", Value: "new", ExpectAbsent: true},
		{Op: kv.OpPut, Key: "", Value: ""},
	}
	for _, want := range cases {
		got, err := kv.DecodeCommand(kv.EncodeCommand(want))
		if err != nil {
			t.Fatalf("decode %+v: %v", want, err)
		}
		if got != want {
			t.Errorf("round trip = %+v, want %+v", got, want)
		}
	}
}

func TestDecodeRejectsTruncatedInput(t *testing.T) {
	full := kv.EncodeCommand(kv.Command{Op: kv.OpPut, Key: "key", Value: "value"})
	for n := 2; n < len(full); n++ {
		if _, err := kv.DecodeCommand(full[:n]); err == nil {
			t.Errorf("decoding %d of %d bytes succeeded; truncation must be detected", n, len(full))
		}
	}
}

func TestRestoreRejectsCorruptSnapshot(t *testing.T) {
	f := kv.New()
	f.Apply(1, kv.Command{Op: kv.OpPut, Key: "k", Value: "v"})
	snap := f.Snapshot()

	if err := kv.New().Restore(snap[:len(snap)/2]); err == nil {
		t.Error("truncated snapshot was accepted")
	}
	if err := kv.New().Restore(nil); err == nil {
		t.Error("empty snapshot was accepted")
	}
}

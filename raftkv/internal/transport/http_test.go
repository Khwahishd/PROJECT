package transport_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/transport"
)

func TestRoundTripPreservesMessage(t *testing.T) {
	received := make(chan raft.Message, 1)

	receiver := transport.NewHTTP(2, nil, time.Second)
	receiver.OnMessage(func(m raft.Message) { received <- m })
	srv := httptest.NewServer(receiver)
	defer srv.Close()

	sender := transport.NewHTTP(1, map[raft.NodeID]string{2: srv.URL}, time.Second)
	defer sender.Close()

	want := raft.Message{
		Type: raft.MsgAppendReq, From: 1, To: 2, Term: 9,
		PrevLogIndex: 4, PrevLogTerm: 3, LeaderCommit: 4,
		Entries: []raft.Entry{
			{Index: 5, Term: 9, Type: raft.EntryNormal, Data: []byte("hello")},
			{Index: 6, Term: 9, Type: raft.EntryConfChange, Data: []byte("cc")},
		},
		Context: []byte("read-token"),
	}
	sender.Send(want)

	select {
	case got := <-received:
		if got.Type != want.Type || got.From != want.From || got.Term != want.Term {
			t.Fatalf("header mismatch: got %+v", got)
		}
		if len(got.Entries) != 2 {
			t.Fatalf("got %d entries, want 2", len(got.Entries))
		}
		if string(got.Entries[0].Data) != "hello" || got.Entries[1].Type != raft.EntryConfChange {
			t.Fatalf("entries corrupted: %+v", got.Entries)
		}
		if string(got.Context) != "read-token" {
			t.Errorf("context = %q, want \"read-token\"", got.Context)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("message never arrived")
	}
}

func TestSnapshotRoundTrip(t *testing.T) {
	received := make(chan raft.Message, 1)
	receiver := transport.NewHTTP(2, nil, time.Second)
	receiver.OnMessage(func(m raft.Message) { received <- m })
	srv := httptest.NewServer(receiver)
	defer srv.Close()

	sender := transport.NewHTTP(1, map[raft.NodeID]string{2: srv.URL}, 5*time.Second)
	defer sender.Close()

	payload := bytes.Repeat([]byte("x"), 1<<20)
	sender.Send(raft.Message{
		Type: raft.MsgSnapshotReq, From: 1, To: 2, Term: 4,
		Snapshot: &raft.Snapshot{
			Index: 500, Term: 3,
			Conf: raft.Config{Voters: []raft.NodeID{1, 2, 3}},
			Data: payload,
		},
	})

	select {
	case got := <-received:
		if got.Snapshot == nil {
			t.Fatal("snapshot was dropped in transit")
		}
		if got.Snapshot.Index != 500 || len(got.Snapshot.Data) != len(payload) {
			t.Fatalf("snapshot corrupted: index=%d len=%d", got.Snapshot.Index, len(got.Snapshot.Data))
		}
		if len(got.Snapshot.Conf.Voters) != 3 {
			t.Errorf("snapshot lost its configuration: %v", got.Snapshot.Conf.Voters)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("snapshot never arrived")
	}
}

func TestSendToUnknownPeerIsDropped(t *testing.T) {
	tr := transport.NewHTTP(1, map[raft.NodeID]string{2: "http://127.0.0.1:1"}, 50*time.Millisecond)
	defer tr.Close()
	// Must not panic or block; an unknown peer is simply ignored.
	tr.Send(raft.Message{Type: raft.MsgAppendReq, To: 99})
}

func TestUnreachablePeerDoesNotBlockSender(t *testing.T) {
	// Port 1 on loopback refuses connections immediately on Linux; the point is
	// that Send never blocks the caller regardless of what the peer does.
	tr := transport.NewHTTP(1, map[raft.NodeID]string{2: "http://127.0.0.1:1"}, 50*time.Millisecond)
	defer tr.Close()

	done := make(chan struct{})
	go func() {
		for i := 0; i < 200; i++ {
			tr.Send(raft.Message{Type: raft.MsgAppendReq, To: 2, Term: uint64(i)})
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Send blocked on an unreachable peer")
	}

	// Eventually every one of those sends is accounted for as dropped.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if tr.Stats()[2].Dropped > 0 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Error("no drops recorded for an unreachable peer")
}

func TestRejectsMalformedRequests(t *testing.T) {
	var mu sync.Mutex
	delivered := 0
	receiver := transport.NewHTTP(2, nil, time.Second)
	receiver.OnMessage(func(raft.Message) {
		mu.Lock()
		delivered++
		mu.Unlock()
	})
	srv := httptest.NewServer(receiver)
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("GET returned %d, want 405", resp.StatusCode)
	}

	resp, err = http.Post(srv.URL, "application/octet-stream", bytes.NewReader([]byte{0x01, 0x02}))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("truncated payload returned %d, want 400", resp.StatusCode)
	}

	mu.Lock()
	defer mu.Unlock()
	if delivered != 0 {
		t.Errorf("%d malformed messages reached the raft node", delivered)
	}
}

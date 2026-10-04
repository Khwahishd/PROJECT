// Package transport carries Raft messages between peers over HTTP.
//
// Raft tolerates a lossy, reordering, duplicating network by design, so the
// transport deliberately does not retry, buffer or reorder: a failed send is
// simply dropped and the sender's own retry logic (heartbeats, probes, the
// stall detector) recovers. Adding reliability here would only add latency and
// hide the very failure modes the consensus layer already handles.
package transport

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/Khwahishd/raftkv/internal/raft"
)

// MessagePath is the endpoint peers POST raft messages to.
const MessagePath = "/raft/message"

// Handler receives a message that arrived from a peer.
type Handler func(raft.Message)

// HTTP is an HTTP-based raft transport.
type HTTP struct {
	self  raft.NodeID
	peers map[raft.NodeID]string // node id -> base URL

	client *http.Client

	mu      sync.RWMutex
	handler Handler

	// inflight bounds concurrent sends per peer so that one unreachable node
	// cannot accumulate unbounded goroutines.
	inflight map[raft.NodeID]chan struct{}

	// Metrics, read by the status endpoint.
	sent    map[raft.NodeID]*counter
	dropped map[raft.NodeID]*counter
}

type counter struct {
	mu sync.Mutex
	n  uint64
}

func (c *counter) inc() {
	c.mu.Lock()
	c.n++
	c.mu.Unlock()
}

func (c *counter) load() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}

// NewHTTP builds a transport for self, able to reach the given peers.
func NewHTTP(self raft.NodeID, peers map[raft.NodeID]string, timeout time.Duration) *HTTP {
	t := &HTTP{
		self:  self,
		peers: peers,
		client: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConnsPerHost: 8,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		inflight: map[raft.NodeID]chan struct{}{},
		sent:     map[raft.NodeID]*counter{},
		dropped:  map[raft.NodeID]*counter{},
	}
	for id := range peers {
		// A small window per peer: enough to pipeline, small enough that a
		// black-holed peer stops consuming resources almost immediately.
		t.inflight[id] = make(chan struct{}, 4)
		t.sent[id] = &counter{}
		t.dropped[id] = &counter{}
	}
	return t
}

// OnMessage registers the callback invoked for each inbound message.
func (t *HTTP) OnMessage(h Handler) {
	t.mu.Lock()
	t.handler = h
	t.mu.Unlock()
}

// Send delivers m to its target, dropping it if the peer is unknown,
// unreachable, or already has too many sends in flight.
func (t *HTTP) Send(m raft.Message) {
	url, ok := t.peers[m.To]
	if !ok {
		return
	}
	slot := t.inflight[m.To]
	select {
	case slot <- struct{}{}:
	default:
		// Peer is backed up. Dropping is the correct behaviour: raft will
		// retry, and queueing here would only deliver stale messages later.
		t.dropped[m.To].inc()
		return
	}

	body := raft.EncodeMessage(m)
	go func() {
		defer func() { <-slot }()
		req, err := http.NewRequest(http.MethodPost, url+MessagePath, bytes.NewReader(body))
		if err != nil {
			t.dropped[m.To].inc()
			return
		}
		req.Header.Set("Content-Type", "application/octet-stream")
		resp, err := t.client.Do(req)
		if err != nil {
			t.dropped[m.To].inc()
			return
		}
		// The body is empty but must be drained for the connection to be reused.
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		t.sent[m.To].inc()
	}()
}

// ServeHTTP accepts an inbound raft message.
func (t *HTTP) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	// Bound the read so a malformed or hostile peer cannot exhaust memory.
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		http.Error(w, "read body", http.StatusBadRequest)
		return
	}
	m, err := raft.DecodeMessage(body)
	if err != nil {
		http.Error(w, fmt.Sprintf("decode: %v", err), http.StatusBadRequest)
		return
	}
	t.mu.RLock()
	h := t.handler
	t.mu.RUnlock()
	if h != nil {
		h(m)
	}
	w.WriteHeader(http.StatusNoContent)
}

// Stats reports per-peer send and drop counts.
func (t *HTTP) Stats() map[raft.NodeID]PeerStats {
	out := map[raft.NodeID]PeerStats{}
	for id := range t.peers {
		out[id] = PeerStats{Sent: t.sent[id].load(), Dropped: t.dropped[id].load()}
	}
	return out
}

// PeerStats is per-peer transport telemetry.
type PeerStats struct {
	Sent    uint64 `json:"sent"`
	Dropped uint64 `json:"dropped"`
}

// Close releases the underlying connections.
func (t *HTTP) Close() {
	t.client.CloseIdleConnections()
}

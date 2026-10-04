package server_test

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Khwahishd/raftkv/internal/kv"
	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/server"
)

// cluster is a set of real raftkv nodes talking to each other over loopback
// HTTP. These tests are slower than the simulator but prove something it
// cannot: that the storage, transport and server layers correctly honour the
// contract the consensus core depends on.
type cluster struct {
	t       *testing.T
	nodes   map[raft.NodeID]*server.Server
	servers map[raft.NodeID]*httptest.Server
	dirs    map[raft.NodeID]string
	peers   map[raft.NodeID]string
	mu      sync.Mutex
}

func newCluster(t *testing.T, n int) *cluster {
	t.Helper()
	c := &cluster{
		t:       t,
		nodes:   map[raft.NodeID]*server.Server{},
		servers: map[raft.NodeID]*httptest.Server{},
		dirs:    map[raft.NodeID]string{},
		peers:   map[raft.NodeID]string{},
	}

	// Reserve listeners first so every node knows the whole peer table before
	// any of them starts.
	listeners := map[raft.NodeID]net.Listener{}
	for i := 1; i <= n; i++ {
		id := raft.NodeID(i)
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("listen: %v", err)
		}
		listeners[id] = ln
		c.peers[id] = "http://" + ln.Addr().String()
		c.dirs[id] = t.TempDir()
	}

	for id, ln := range listeners {
		c.start(id, ln)
	}
	t.Cleanup(c.stopAll)
	return c
}

func (c *cluster) start(id raft.NodeID, ln net.Listener) {
	c.t.Helper()
	srv, err := server.New(server.Config{
		ID:      id,
		Peers:   c.peers,
		DataDir: c.dirs[id],
		// Short ticks keep the tests fast while leaving the heartbeat interval
		// comfortably below the election timeout.
		TickInterval:      10 * time.Millisecond,
		ElectionTicks:     10,
		HeartbeatTicks:    2,
		SnapshotThreshold: 0,
		PreVote:           true,
		CheckQuorum:       true,
		RequestTimeout:    4 * time.Second,
		Logger:            slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	if err != nil {
		c.t.Fatalf("new server %d: %v", id, err)
	}
	srv.Start()

	hs := &httptest.Server{
		Listener: ln,
		Config:   &http.Server{Handler: srv.Mux()},
	}
	hs.Start()

	c.mu.Lock()
	c.nodes[id] = srv
	c.servers[id] = hs
	c.mu.Unlock()
}

func (c *cluster) stop(id raft.NodeID) {
	c.mu.Lock()
	hs, srv := c.servers[id], c.nodes[id]
	delete(c.servers, id)
	delete(c.nodes, id)
	c.mu.Unlock()
	if hs != nil {
		hs.Close()
	}
	if srv != nil {
		srv.Close()
	}
}

func (c *cluster) stopAll() {
	c.mu.Lock()
	ids := make([]raft.NodeID, 0, len(c.nodes))
	for id := range c.nodes {
		ids = append(ids, id)
	}
	c.mu.Unlock()
	for _, id := range ids {
		c.stop(id)
	}
}

// restart brings a stopped node back up on a fresh listener, keeping its data
// directory -- so recovery really does come from disk.
func (c *cluster) restart(id raft.NodeID) {
	c.t.Helper()
	addr := strings.TrimPrefix(c.peers[id], "http://")
	var ln net.Listener
	var err error
	// The old port may linger briefly in TIME_WAIT.
	for i := 0; i < 50; i++ {
		ln, err = net.Listen("tcp", addr)
		if err == nil {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if err != nil {
		c.t.Fatalf("relisten on %s: %v", addr, err)
	}
	c.start(id, ln)
}

// leader polls until exactly one node reports itself leader.
func (c *cluster) leader(timeout time.Duration) (raft.NodeID, *server.Server) {
	c.t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		var found raft.NodeID
		count := 0
		for id, srv := range c.nodes {
			if srv.IsLeader() {
				found = id
				count++
			}
		}
		c.mu.Unlock()
		if count == 1 {
			return found, c.nodes[found]
		}
		time.Sleep(10 * time.Millisecond)
	}
	c.t.Fatalf("no unique leader within %s", timeout)
	return 0, nil
}

func (c *cluster) node(id raft.NodeID) *server.Server {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.nodes[id]
}

// putWithRetry writes through whichever node is currently leading, retrying
// across elections the way a real client must.
func (c *cluster) putWithRetry(key, value string, timeout time.Duration) error {
	c.t.Helper()
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		_, srv := c.leader(2 * time.Second)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, err := srv.Apply(ctx, kv.Command{Op: kv.OpPut, Key: key, Value: value})
		cancel()
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(20 * time.Millisecond)
	}
	return fmt.Errorf("put %s timed out: %w", key, lastErr)
}

func TestClusterElectsLeaderAndReplicates(t *testing.T) {
	c := newCluster(t, 3)
	id, srv := c.leader(10 * time.Second)
	t.Logf("leader is node %d", id)

	ctx := context.Background()
	for i := 0; i < 25; i++ {
		if _, err := srv.Apply(ctx, kv.Command{
			Op: kv.OpPut, Key: fmt.Sprintf("k%d", i), Value: fmt.Sprintf("v%d", i),
		}); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}

	// A linearizable read on the leader must see every write.
	for i := 0; i < 25; i++ {
		got, err := srv.Get(ctx, fmt.Sprintf("k%d", i))
		if err != nil {
			t.Fatalf("get k%d: %v", i, err)
		}
		if want := fmt.Sprintf("v%d", i); got != want {
			t.Fatalf("k%d = %q, want %q", i, got, want)
		}
	}

	// And every follower must converge on the same data.
	deadline := time.Now().Add(5 * time.Second)
	for {
		converged := true
		for pid := raft.NodeID(1); pid <= 3; pid++ {
			if c.node(pid).Keys() == nil || len(c.node(pid).Keys()) != 25 {
				converged = false
			}
		}
		if converged || time.Now().After(deadline) {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	for pid := raft.NodeID(1); pid <= 3; pid++ {
		keys := c.node(pid).Keys()
		if len(keys) != 25 {
			t.Fatalf("node %d has %d keys, want 25", pid, len(keys))
		}
		for i := 0; i < 25; i++ {
			got, err := c.node(pid).GetStale(fmt.Sprintf("k%d", i))
			if err != nil || got != fmt.Sprintf("v%d", i) {
				t.Fatalf("node %d: k%d = (%q, %v)", pid, i, got, err)
			}
		}
	}
}

func TestWritesSurviveLeaderFailure(t *testing.T) {
	c := newCluster(t, 3)
	oldLeader, srv := c.leader(10 * time.Second)

	ctx := context.Background()
	for i := 0; i < 10; i++ {
		if _, err := srv.Apply(ctx, kv.Command{
			Op: kv.OpPut, Key: fmt.Sprintf("pre%d", i), Value: "x",
		}); err != nil {
			t.Fatalf("pre-failure put: %v", err)
		}
	}

	// Kill the leader. The surviving two nodes still form a quorum.
	c.stop(oldLeader)

	newLeader, newSrv := c.leader(15 * time.Second)
	if newLeader == oldLeader {
		t.Fatalf("stopped node %d is still reported as leader", oldLeader)
	}
	t.Logf("failed over from node %d to node %d", oldLeader, newLeader)

	// Every acknowledged pre-failure write must have survived: this is the
	// durability promise the whole system exists to make.
	for i := 0; i < 10; i++ {
		got, err := newSrv.Get(ctx, fmt.Sprintf("pre%d", i))
		if err != nil || got != "x" {
			t.Fatalf("committed write pre%d lost after failover: (%q, %v)", i, got, err)
		}
	}

	// And the cluster must still accept new writes.
	if err := c.putWithRetry("post", "y", 10*time.Second); err != nil {
		t.Fatalf("write after failover: %v", err)
	}
}

func TestRestartedNodeRecoversFromDisk(t *testing.T) {
	c := newCluster(t, 3)
	_, srv := c.leader(10 * time.Second)

	ctx := context.Background()
	for i := 0; i < 30; i++ {
		if _, err := srv.Apply(ctx, kv.Command{
			Op: kv.OpPut, Key: fmt.Sprintf("k%d", i), Value: fmt.Sprintf("v%d", i),
		}); err != nil {
			t.Fatalf("put: %v", err)
		}
	}

	// Restart a follower; it must rebuild its state machine purely from the
	// WAL on disk.
	leaderID, _ := c.leader(5 * time.Second)
	victim := raft.NodeID(1)
	if victim == leaderID {
		victim = 2
	}
	c.stop(victim)
	time.Sleep(100 * time.Millisecond)
	c.restart(victim)

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if len(c.node(victim).Keys()) == 30 {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	for i := 0; i < 30; i++ {
		got, err := c.node(victim).GetStale(fmt.Sprintf("k%d", i))
		if err != nil || got != fmt.Sprintf("v%d", i) {
			t.Fatalf("restarted node %d: k%d = (%q, %v), want %q",
				victim, i, got, err, fmt.Sprintf("v%d", i))
		}
	}
}

func TestFollowerRejectsWritesAndRedirects(t *testing.T) {
	c := newCluster(t, 3)
	leaderID, _ := c.leader(10 * time.Second)

	var followerID raft.NodeID
	for id := raft.NodeID(1); id <= 3; id++ {
		if id != leaderID {
			followerID = id
			break
		}
	}

	// A write through the follower's HTTP API must be refused with a redirect,
	// not silently accepted locally.
	req, _ := http.NewRequest(http.MethodPut, c.peers[followerID]+"/kv/x", strings.NewReader("1"))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("put to follower: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMisdirectedRequest {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("follower returned %d, want 421; body: %s", resp.StatusCode, body)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "leader") {
		t.Errorf("redirect response does not name the leader: %s", body)
	}
}

func TestHTTPEndToEnd(t *testing.T) {
	c := newCluster(t, 3)
	leaderID, _ := c.leader(10 * time.Second)
	base := c.peers[leaderID]

	put := func(key, val string) *http.Response {
		req, _ := http.NewRequest(http.MethodPut, base+"/kv/"+key, strings.NewReader(val))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("put %s: %v", key, err)
		}
		return resp
	}

	resp := put("alpha", "one")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT returned %d", resp.StatusCode)
	}
	resp.Body.Close()

	resp, err := http.Get(base + "/kv/alpha")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(body), `"value":"one"`) {
		t.Fatalf("GET returned %s", body)
	}

	// A missing key is a 404, not an empty 200.
	resp, _ = http.Get(base + "/kv/missing")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("GET missing key returned %d, want 404", resp.StatusCode)
	}
	resp.Body.Close()

	// CAS with a wrong expectation is a 409, distinguishing a failed
	// precondition from a server error.
	resp, err = http.Post(base+"/kv/alpha/cas", "application/json",
		strings.NewReader(`{"value":"two","expect":"wrong"}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("failed CAS returned %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()

	resp, err = http.Post(base+"/kv/alpha/cas", "application/json",
		strings.NewReader(`{"value":"two","expect":"one"}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("successful CAS returned %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp, _ = http.Get(base + "/status")
	body, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	for _, want := range []string{`"role":"leader"`, `"voters"`, `"commit_index"`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("status response missing %s: %s", want, body)
		}
	}

	req, _ := http.NewRequest(http.MethodDelete, base+"/kv/alpha", nil)
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	resp, _ = http.Get(base + "/kv/alpha")
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("deleted key still readable (status %d)", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestConcurrentWritesAllCommit(t *testing.T) {
	c := newCluster(t, 3)
	_, srv := c.leader(10 * time.Second)

	const writers, perWriter = 8, 20
	var wg sync.WaitGroup
	errCh := make(chan error, writers*perWriter)

	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			ctx := context.Background()
			for i := 0; i < perWriter; i++ {
				key := fmt.Sprintf("w%d-k%d", w, i)
				if _, err := srv.Apply(ctx, kv.Command{Op: kv.OpPut, Key: key, Value: "v"}); err != nil {
					errCh <- fmt.Errorf("%s: %w", key, err)
					return
				}
			}
		}(w)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatalf("concurrent write failed: %v", err)
	}

	if got := len(srv.Keys()); got != writers*perWriter {
		t.Fatalf("committed %d keys, want %d", got, writers*perWriter)
	}
}

func TestMain(m *testing.M) {
	// Keep test output readable; the servers log to io.Discard anyway.
	slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	os.Exit(m.Run())
}

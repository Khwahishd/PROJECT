// Command raftkvd runs a single raftkv node.
//
// A three-node cluster on one machine:
//
//	raftkvd -id 1 -peers '1=http://127.0.0.1:9001,2=http://127.0.0.1:9002,3=http://127.0.0.1:9003' -listen :9001 -data ./data/1
//	raftkvd -id 2 -peers '...' -listen :9002 -data ./data/2
//	raftkvd -id 3 -peers '...' -listen :9003 -data ./data/3
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/server"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "raftkvd: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		id         = flag.Uint64("id", 0, "this node's id (required, non-zero)")
		peersFlag  = flag.String("peers", "", "comma-separated id=url list for the whole cluster (required)")
		listen     = flag.String("listen", ":9001", "address to listen on")
		dataDir    = flag.String("data", "", "data directory (required)")
		tick       = flag.Duration("tick", 50*time.Millisecond, "raft tick interval")
		electionT  = flag.Int("election-ticks", 10, "election timeout, in ticks")
		heartbeatT = flag.Int("heartbeat-ticks", 2, "heartbeat interval, in ticks")
		snapEvery  = flag.Uint64("snapshot-threshold", 10000, "entries between snapshots (0 disables)")
		preVote    = flag.Bool("prevote", true, "enable the pre-vote protocol")
		checkQ     = flag.Bool("check-quorum", true, "step down when quorum contact is lost")
		reqTimeout = flag.Duration("request-timeout", 5*time.Second, "client request timeout")
		logLevel   = flag.String("log-level", "info", "debug, info, warn or error")
	)
	flag.Parse()

	if *id == 0 {
		return errors.New("-id is required and must be non-zero")
	}
	if *dataDir == "" {
		return errors.New("-data is required")
	}
	peers, err := parsePeers(*peersFlag)
	if err != nil {
		return err
	}
	if _, ok := peers[raft.NodeID(*id)]; !ok {
		return fmt.Errorf("-peers does not include this node's id %d", *id)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: parseLevel(*logLevel)}))

	srv, err := server.New(server.Config{
		ID:                raft.NodeID(*id),
		Peers:             peers,
		DataDir:           *dataDir,
		TickInterval:      *tick,
		ElectionTicks:     *electionT,
		HeartbeatTicks:    *heartbeatT,
		SnapshotThreshold: *snapEvery,
		PreVote:           *preVote,
		CheckQuorum:       *checkQ,
		RequestTimeout:    *reqTimeout,
		Logger:            logger,
	})
	if err != nil {
		return err
	}
	srv.Start()

	httpSrv := &http.Server{
		Addr:              *listen,
		Handler:           srv.Mux(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		logger.Info("listening", "addr", *listen, "id", *id, "peers", len(peers))
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		srv.Close()
		return err
	case s := <-sig:
		logger.Info("shutting down", "signal", s.String())
	}

	// Stop accepting requests before stopping raft, so in-flight writes get a
	// chance to commit rather than being failed on the floor.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		logger.Warn("http shutdown", "err", err)
	}
	return srv.Close()
}

// parsePeers parses "1=http://host:port,2=http://host:port".
func parsePeers(s string) (map[raft.NodeID]string, error) {
	if strings.TrimSpace(s) == "" {
		return nil, errors.New("-peers is required, e.g. '1=http://127.0.0.1:9001,2=http://127.0.0.1:9002'")
	}
	out := map[raft.NodeID]string{}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			return nil, fmt.Errorf("malformed peer %q, want id=url", part)
		}
		id, err := strconv.ParseUint(strings.TrimSpace(k), 10, 64)
		if err != nil {
			return nil, fmt.Errorf("malformed peer id in %q: %w", part, err)
		}
		if id == 0 {
			return nil, fmt.Errorf("peer id must be non-zero in %q", part)
		}
		url := strings.TrimSuffix(strings.TrimSpace(v), "/")
		if url == "" {
			return nil, fmt.Errorf("empty url for peer %d", id)
		}
		if _, dup := out[raft.NodeID(id)]; dup {
			return nil, fmt.Errorf("duplicate peer id %d", id)
		}
		out[raft.NodeID(id)] = url
	}
	if len(out) == 0 {
		return nil, errors.New("-peers produced no entries")
	}
	return out, nil
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

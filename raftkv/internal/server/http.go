package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/Khwahishd/raftkv/internal/kv"
	"github.com/Khwahishd/raftkv/internal/raft"
	"github.com/Khwahishd/raftkv/internal/transport"
)

// Mux returns an http.Handler exposing both the peer transport and the client
// API.
//
//	POST   /raft/message        peer-to-peer raft traffic
//	GET    /kv/{key}            linearizable read (?stale=true for a local read)
//	PUT    /kv/{key}            write; body is the value
//	DELETE /kv/{key}            delete
//	POST   /kv/{key}/cas        compare-and-swap
//	GET    /keys                list keys
//	GET    /status              node and cluster state
//	POST   /members             add or remove a voter
func (s *Server) Mux() http.Handler {
	mux := http.NewServeMux()
	mux.Handle(transport.MessagePath, s.tr)
	mux.HandleFunc("/kv/", s.handleKV)
	mux.HandleFunc("/keys", s.handleKeys)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/members", s.handleMembers)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "id": s.cfg.ID})
	})
	return mux
}

type errorResponse struct {
	Error  string      `json:"error"`
	Leader raft.NodeID `json:"leader,omitempty"`
	// LeaderAddr lets a client follow the redirect without its own peer table.
	LeaderAddr string `json:"leader_addr,omitempty"`
}

func (s *Server) writeErr(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	resp := errorResponse{Error: err.Error()}

	switch {
	case errors.Is(err, ErrNotLeader), errors.Is(err, ErrLeaderChanged):
		// 421 Misdirected Request is the honest code here: the request was
		// well-formed, but this node cannot serve it.
		status = http.StatusMisdirectedRequest
		if l := s.LeaderID(); l != raft.None {
			resp.Leader = l
			resp.LeaderAddr = s.cfg.Peers[l]
		}
	case errors.Is(err, kv.ErrNotFound):
		status = http.StatusNotFound
	case errors.Is(err, ErrTimeout):
		status = http.StatusGatewayTimeout
	case errors.Is(err, ErrShuttingDown):
		status = http.StatusServiceUnavailable
	}
	writeJSON(w, status, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// clientMeta extracts the deduplication identifiers a client sends so that a
// retried write is not applied twice.
func clientMeta(r *http.Request) (clientID, requestID uint64) {
	clientID, _ = strconv.ParseUint(r.Header.Get("X-Client-Id"), 10, 64)
	requestID, _ = strconv.ParseUint(r.Header.Get("X-Request-Id"), 10, 64)
	return
}

func (s *Server) handleKV(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/kv/")
	if path == "" {
		http.Error(w, "missing key", http.StatusBadRequest)
		return
	}

	if strings.HasSuffix(path, "/cas") {
		s.handleCAS(w, r, strings.TrimSuffix(path, "/cas"))
		return
	}
	key := path

	switch r.Method {
	case http.MethodGet:
		var (
			val string
			err error
		)
		if r.URL.Query().Get("stale") == "true" {
			val, err = s.GetStale(key)
		} else {
			val, err = s.Get(r.Context(), key)
		}
		if err != nil {
			s.writeErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"key": key, "value": val})

	case http.MethodPut:
		body, err := io.ReadAll(io.LimitReader(r.Body, 4<<20))
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}
		cid, rid := clientMeta(r)
		res, err := s.Apply(r.Context(), kv.Command{
			Op: kv.OpPut, Key: key, Value: string(body),
			ClientID: cid, RequestID: rid,
		})
		if err != nil {
			s.writeErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"key": key, "applied": res.Applied, "previous": res.Previous, "existed": res.Existed,
		})

	case http.MethodDelete:
		cid, rid := clientMeta(r)
		res, err := s.Apply(r.Context(), kv.Command{
			Op: kv.OpDelete, Key: key, ClientID: cid, RequestID: rid,
		})
		if err != nil {
			s.writeErr(w, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"key": key, "deleted": res.Applied})

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

type casRequest struct {
	Value        string `json:"value"`
	Expect       string `json:"expect"`
	ExpectAbsent bool   `json:"expect_absent"`
}

func (s *Server) handleCAS(w http.ResponseWriter, r *http.Request, key string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req casRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid json body", http.StatusBadRequest)
		return
	}
	cid, rid := clientMeta(r)
	res, err := s.Apply(r.Context(), kv.Command{
		Op: kv.OpCAS, Key: key, Value: req.Value,
		Expect: req.Expect, ExpectAbsent: req.ExpectAbsent,
		ClientID: cid, RequestID: rid,
	})
	if err != nil {
		s.writeErr(w, err)
		return
	}
	status := http.StatusOK
	if !res.Applied {
		// The request was processed correctly; the precondition simply did not
		// hold. 409 distinguishes that from an error.
		status = http.StatusConflict
	}
	writeJSON(w, status, map[string]any{
		"key": key, "applied": res.Applied, "previous": res.Previous, "existed": res.Existed,
	})
}

func (s *Server) handleKeys(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"keys": s.Keys()})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	st, err := s.Status(r.Context())
	if err != nil {
		s.writeErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, st)
}

type membersRequest struct {
	Action string `json:"action"` // "add" or "remove"
	NodeID uint64 `json:"node_id"`
}

func (s *Server) handleMembers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req membersRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid json body", http.StatusBadRequest)
		return
	}
	var cc raft.ConfChange
	switch req.Action {
	case "add":
		cc = raft.ConfChange{Type: raft.ConfChangeAddNode, NodeID: raft.NodeID(req.NodeID)}
	case "remove":
		cc = raft.ConfChange{Type: raft.ConfChangeRemoveNode, NodeID: raft.NodeID(req.NodeID)}
	default:
		http.Error(w, `action must be "add" or "remove"`, http.StatusBadRequest)
		return
	}
	if err := s.ChangeMembership(r.Context(), cc); err != nil {
		s.writeErr(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

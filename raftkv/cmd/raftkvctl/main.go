// Command raftkvctl is a command-line client for a raftkv cluster.
//
// It implements the client-side half of the consistency story: it discovers the
// leader, follows redirects when it guesses wrong, and tags every write with a
// stable request ID so that retrying after a timeout cannot apply the write
// twice.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "raftkvctl: %v\n", err)
		os.Exit(1)
	}
}

const usage = `raftkvctl - client for a raftkv cluster

usage:
  raftkvctl -endpoints <urls> <command> [args]

commands:
  get <key>                 linearizable read
  get -stale <key>          local read, may be stale
  put <key> <value>         write a value
  del <key>                 delete a key
  cas <key> <new> <expect>  compare-and-swap (use -absent for create-if-missing)
  keys                      list all keys
  status                    show cluster status
  member add|remove <id>    change cluster membership

flags:
`

func run(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("raftkvctl", flag.ContinueOnError)
	fs.SetOutput(out)
	endpoints := fs.String("endpoints", "http://127.0.0.1:9001", "comma-separated node URLs")
	stale := fs.Bool("stale", false, "allow a stale local read (get only)")
	absent := fs.Bool("absent", false, "for cas: require the key to be absent")
	timeout := fs.Duration("timeout", 10*time.Second, "overall request timeout")
	retries := fs.Int("retries", 5, "how many times to retry on a redirect or transient failure")
	fs.Usage = func() {
		fmt.Fprint(out, usage)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) == 0 {
		fs.Usage()
		return errors.New("no command given")
	}

	c := &client{
		urls:    splitEndpoints(*endpoints),
		http:    &http.Client{Timeout: *timeout},
		retries: *retries,
		// A random client ID per process, combined with a monotonic request
		// counter, is what makes retries idempotent server-side.
		clientID: rand.Uint64() | 1,
		out:      out,
	}
	if len(c.urls) == 0 {
		return errors.New("-endpoints is empty")
	}

	switch rest[0] {
	case "get":
		if len(rest) != 2 {
			return errors.New("usage: get <key>")
		}
		return c.get(rest[1], *stale)
	case "put":
		if len(rest) != 3 {
			return errors.New("usage: put <key> <value>")
		}
		return c.put(rest[1], rest[2])
	case "del", "delete":
		if len(rest) != 2 {
			return errors.New("usage: del <key>")
		}
		return c.del(rest[1])
	case "cas":
		if *absent {
			if len(rest) != 3 {
				return errors.New("usage: cas -absent <key> <value>")
			}
			return c.cas(rest[1], rest[2], "", true)
		}
		if len(rest) != 4 {
			return errors.New("usage: cas <key> <new-value> <expected-value>")
		}
		return c.cas(rest[1], rest[2], rest[3], false)
	case "keys":
		return c.simple(http.MethodGet, "/keys", nil)
	case "status":
		return c.status()
	case "member":
		if len(rest) != 3 {
			return errors.New("usage: member add|remove <id>")
		}
		return c.member(rest[1], rest[2])
	default:
		fs.Usage()
		return fmt.Errorf("unknown command %q", rest[0])
	}
}

type client struct {
	urls     []string
	http     *http.Client
	retries  int
	clientID uint64
	reqSeq   uint64
	out      io.Writer
}

func splitEndpoints(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSuffix(strings.TrimSpace(p), "/")
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// do issues a request, following the cluster's leader redirects.
//
// A 421 response names the current leader, so a misdirected request costs one
// extra round trip rather than a failure. Writes carry X-Client-Id and
// X-Request-Id, which the server uses to collapse duplicate deliveries of a
// retried request into a single application.
func (c *client) do(method, path string, body []byte, isWrite bool) ([]byte, error) {
	if isWrite {
		c.reqSeq++
	}
	target := c.urls[0]
	var lastErr error

	for attempt := 0; attempt <= c.retries; attempt++ {
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(body)
		}
		req, err := http.NewRequest(method, target+path, rdr)
		if err != nil {
			return nil, err
		}
		if body != nil {
			req.Header.Set("Content-Type", "application/json")
		}
		if isWrite {
			req.Header.Set("X-Client-Id", fmt.Sprint(c.clientID))
			req.Header.Set("X-Request-Id", fmt.Sprint(c.reqSeq))
		}

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			target = c.urls[(attempt+1)%len(c.urls)]
			time.Sleep(backoff(attempt))
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		switch resp.StatusCode {
		case http.StatusOK, http.StatusConflict, http.StatusNotFound:
			if resp.StatusCode == http.StatusNotFound {
				return raw, fmt.Errorf("not found")
			}
			return raw, nil

		case http.StatusMisdirectedRequest:
			var e struct {
				LeaderAddr string `json:"leader_addr"`
			}
			if json.Unmarshal(raw, &e) == nil && e.LeaderAddr != "" {
				target = strings.TrimSuffix(e.LeaderAddr, "/")
			} else {
				target = c.urls[(attempt+1)%len(c.urls)]
			}
			lastErr = fmt.Errorf("not leader")
			time.Sleep(backoff(attempt))

		case http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			// The write may or may not have committed. Retrying is safe
			// precisely because of the request ID above.
			lastErr = fmt.Errorf("server busy: %s", strings.TrimSpace(string(raw)))
			target = c.urls[(attempt+1)%len(c.urls)]
			time.Sleep(backoff(attempt))

		default:
			return nil, fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, strings.TrimSpace(string(raw)))
		}
	}
	return nil, fmt.Errorf("gave up after %d attempts: %w", c.retries+1, lastErr)
}

func backoff(attempt int) time.Duration {
	d := time.Duration(50*(1<<attempt)) * time.Millisecond
	if d > time.Second {
		d = time.Second
	}
	return d
}

func (c *client) get(key string, stale bool) error {
	path := "/kv/" + key
	if stale {
		path += "?stale=true"
	}
	raw, err := c.do(http.MethodGet, path, nil, false)
	if err != nil {
		return err
	}
	var r struct{ Value string }
	if err := json.Unmarshal(raw, &r); err != nil {
		return err
	}
	fmt.Fprintln(c.out, r.Value)
	return nil
}

func (c *client) put(key, value string) error {
	_, err := c.do(http.MethodPut, "/kv/"+key, []byte(value), true)
	if err != nil {
		return err
	}
	fmt.Fprintln(c.out, "OK")
	return nil
}

func (c *client) del(key string) error {
	raw, err := c.do(http.MethodDelete, "/kv/"+key, nil, true)
	if err != nil {
		return err
	}
	var r struct{ Deleted bool }
	json.Unmarshal(raw, &r)
	if r.Deleted {
		fmt.Fprintln(c.out, "OK")
	} else {
		fmt.Fprintln(c.out, "key did not exist")
	}
	return nil
}

func (c *client) cas(key, value, expect string, absent bool) error {
	body, _ := json.Marshal(map[string]any{
		"value": value, "expect": expect, "expect_absent": absent,
	})
	raw, err := c.do(http.MethodPost, "/kv/"+key+"/cas", body, true)
	if err != nil {
		return err
	}
	var r struct {
		Applied  bool   `json:"applied"`
		Previous string `json:"previous"`
	}
	json.Unmarshal(raw, &r)
	if !r.Applied {
		return fmt.Errorf("precondition failed (current value is %q)", r.Previous)
	}
	fmt.Fprintln(c.out, "OK")
	return nil
}

func (c *client) simple(method, path string, body []byte) error {
	raw, err := c.do(method, path, body, false)
	if err != nil {
		return err
	}
	return c.printJSON(raw)
}

func (c *client) status() error {
	// Status is per-node, so query every endpoint rather than following a
	// redirect: seeing all of them is the point.
	for _, u := range c.urls {
		resp, err := c.http.Get(u + "/status")
		if err != nil {
			fmt.Fprintf(c.out, "%s\tUNREACHABLE (%v)\n", u, err)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		var st struct {
			ID           uint64 `json:"id"`
			Role         string `json:"role"`
			Term         uint64 `json:"term"`
			CommitIndex  uint64 `json:"commit_index"`
			AppliedIndex uint64 `json:"applied_index"`
			Keys         int    `json:"keys"`
		}
		if err := json.Unmarshal(raw, &st); err != nil {
			fmt.Fprintf(c.out, "%s\tBAD RESPONSE\n", u)
			continue
		}
		fmt.Fprintf(c.out, "node %d\t%-9s term=%-4d commit=%-6d applied=%-6d keys=%-6d %s\n",
			st.ID, st.Role, st.Term, st.CommitIndex, st.AppliedIndex, st.Keys, u)
	}
	return nil
}

func (c *client) member(action, id string) error {
	if action != "add" && action != "remove" {
		return fmt.Errorf("action must be add or remove, got %q", action)
	}
	var nodeID uint64
	if _, err := fmt.Sscanf(id, "%d", &nodeID); err != nil || nodeID == 0 {
		return fmt.Errorf("invalid node id %q", id)
	}
	body, _ := json.Marshal(map[string]any{"action": action, "node_id": nodeID})
	if _, err := c.do(http.MethodPost, "/members", body, true); err != nil {
		return err
	}
	fmt.Fprintln(c.out, "OK")
	return nil
}

func (c *client) printJSON(raw []byte) error {
	var v any
	if err := json.Unmarshal(raw, &v); err != nil {
		fmt.Fprintln(c.out, string(raw))
		return nil
	}
	enc := json.NewEncoder(c.out)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

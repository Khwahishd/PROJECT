// Package kv implements the replicated key-value state machine that sits on top
// of Raft.
//
// The state machine is deliberately simple, but it is strict about one thing:
// Apply must be deterministic. Every replica applies the same commands in the
// same order and must reach byte-identical state, or snapshots taken on one
// node will not restore correctly on another.
package kv

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
	"sync"
)

// Op is the kind of mutation a command performs.
type Op uint8

const (
	// OpPut sets a key.
	OpPut Op = iota
	// OpDelete removes a key.
	OpDelete
	// OpCAS sets a key only if its current value matches an expectation,
	// giving clients a compare-and-swap primitive without a separate
	// transaction layer.
	OpCAS
)

// Command is one replicated mutation.
type Command struct {
	Op    Op
	Key   string
	Value string
	// Expect is the required current value for OpCAS. ExpectAbsent selects the
	// "key must not exist" case, which cannot be expressed with Expect alone.
	Expect       string
	ExpectAbsent bool
	// RequestID deduplicates retried client requests. A client that times out
	// and retries must not apply its write twice; the FSM remembers the last
	// request it saw from each client.
	ClientID  uint64
	RequestID uint64
}

// Result is the outcome of applying a Command.
type Result struct {
	// Applied is false when a CAS precondition failed.
	Applied bool
	// Previous is the value the key held before the command.
	Previous string
	// Existed reports whether the key existed before the command.
	Existed bool
}

// ErrNotFound is returned by Get for a missing key.
var ErrNotFound = errors.New("kv: key not found")

// FSM is the replicated key-value store.
//
// It is safe for concurrent reads while Apply runs on a single goroutine, which
// matches how the server drives it: one apply loop, many request handlers.
type FSM struct {
	mu   sync.RWMutex
	data map[string]string

	// dedupe maps a client ID to the highest request ID it has applied, and the
	// result produced, so a retry returns the original answer rather than
	// re-applying.
	dedupe map[uint64]dedupeEntry

	// appliedIndex is the raft index of the last command applied.
	appliedIndex uint64
}

type dedupeEntry struct {
	requestID uint64
	result    Result
}

// New returns an empty FSM.
func New() *FSM {
	return &FSM{
		data:   map[string]string{},
		dedupe: map[uint64]dedupeEntry{},
	}
}

// Apply executes a command. It must be called with commands in raft log order.
func (f *FSM) Apply(index uint64, cmd Command) Result {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.appliedIndex = index

	// Replay protection. Without it, a client that retries after a timeout can
	// apply the same increment or delete twice, which no amount of consensus
	// correctness would prevent.
	if cmd.ClientID != 0 {
		if prev, ok := f.dedupe[cmd.ClientID]; ok {
			if cmd.RequestID <= prev.requestID {
				return prev.result
			}
		}
	}

	var res Result
	cur, existed := f.data[cmd.Key]
	res.Previous, res.Existed = cur, existed

	switch cmd.Op {
	case OpPut:
		f.data[cmd.Key] = cmd.Value
		res.Applied = true
	case OpDelete:
		delete(f.data, cmd.Key)
		res.Applied = existed
	case OpCAS:
		ok := (cmd.ExpectAbsent && !existed) || (!cmd.ExpectAbsent && existed && cur == cmd.Expect)
		if ok {
			f.data[cmd.Key] = cmd.Value
		}
		res.Applied = ok
	}

	if cmd.ClientID != 0 {
		f.dedupe[cmd.ClientID] = dedupeEntry{requestID: cmd.RequestID, result: res}
	}
	return res
}

// Get reads a key. Callers needing linearizability must first obtain a read
// index from raft and wait for AppliedIndex to reach it.
func (f *FSM) Get(key string) (string, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	v, ok := f.data[key]
	if !ok {
		return "", fmt.Errorf("%w: %s", ErrNotFound, key)
	}
	return v, nil
}

// Keys returns all keys in sorted order.
func (f *FSM) Keys() []string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make([]string, 0, len(f.data))
	for k := range f.data {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Len returns the number of keys.
func (f *FSM) Len() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.data)
}

// AppliedIndex returns the raft index of the last applied command.
func (f *FSM) AppliedIndex() uint64 {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.appliedIndex
}

// Snapshot serialises the whole state machine.
//
// Keys are emitted in sorted order so that two replicas holding identical state
// produce identical bytes. That is not merely tidy: it makes snapshots
// comparable in tests, which is how state divergence gets caught early.
func (f *FSM) Snapshot() []byte {
	f.mu.RLock()
	defer f.mu.RUnlock()

	keys := make([]string, 0, len(f.data))
	for k := range f.data {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	buf := binary.LittleEndian.AppendUint64(nil, f.appliedIndex)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(keys)))
	for _, k := range keys {
		buf = appendString(buf, k)
		buf = appendString(buf, f.data[k])
	}

	clients := make([]uint64, 0, len(f.dedupe))
	for c := range f.dedupe {
		clients = append(clients, c)
	}
	sort.Slice(clients, func(i, j int) bool { return clients[i] < clients[j] })
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(clients)))
	for _, c := range clients {
		buf = binary.LittleEndian.AppendUint64(buf, c)
		buf = binary.LittleEndian.AppendUint64(buf, f.dedupe[c].requestID)
	}
	return buf
}

// Restore replaces the FSM's contents with a snapshot.
func (f *FSM) Restore(b []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	d := &reader{buf: b}
	appliedIndex := d.u64()
	n := d.u32()
	if d.err != nil {
		return d.err
	}
	data := make(map[string]string, n)
	for i := uint32(0); i < n; i++ {
		k := d.str()
		v := d.str()
		if d.err != nil {
			return d.err
		}
		data[k] = v
	}
	dedupe := map[uint64]dedupeEntry{}
	if len(d.buf) >= 4 {
		m := d.u32()
		for i := uint32(0); i < m && d.err == nil; i++ {
			c := d.u64()
			r := d.u64()
			dedupe[c] = dedupeEntry{requestID: r}
		}
	}
	if d.err != nil {
		return d.err
	}

	f.data = data
	f.dedupe = dedupe
	f.appliedIndex = appliedIndex
	return nil
}

func appendString(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint32(b, uint32(len(s)))
	return append(b, s...)
}

type reader struct {
	buf []byte
	err error
}

func (r *reader) u32() uint32 {
	if len(r.buf) < 4 {
		r.err = errors.New("kv: truncated snapshot")
		return 0
	}
	v := binary.LittleEndian.Uint32(r.buf)
	r.buf = r.buf[4:]
	return v
}

func (r *reader) u64() uint64 {
	if len(r.buf) < 8 {
		r.err = errors.New("kv: truncated snapshot")
		return 0
	}
	v := binary.LittleEndian.Uint64(r.buf)
	r.buf = r.buf[8:]
	return v
}

func (r *reader) str() string {
	n := r.u32()
	if r.err != nil {
		return ""
	}
	if uint64(n) > uint64(len(r.buf)) {
		r.err = errors.New("kv: snapshot string length out of range")
		return ""
	}
	s := string(r.buf[:n])
	r.buf = r.buf[n:]
	return s
}

// EncodeCommand serialises a command for storage in a raft log entry.
func EncodeCommand(c Command) []byte {
	buf := []byte{byte(c.Op)}
	if c.ExpectAbsent {
		buf = append(buf, 1)
	} else {
		buf = append(buf, 0)
	}
	buf = binary.LittleEndian.AppendUint64(buf, c.ClientID)
	buf = binary.LittleEndian.AppendUint64(buf, c.RequestID)
	buf = appendString(buf, c.Key)
	buf = appendString(buf, c.Value)
	buf = appendString(buf, c.Expect)
	return buf
}

// DecodeCommand parses a command from a raft log entry.
func DecodeCommand(b []byte) (Command, error) {
	if len(b) < 2 {
		return Command{}, errors.New("kv: truncated command")
	}
	c := Command{Op: Op(b[0]), ExpectAbsent: b[1] == 1}
	r := &reader{buf: b[2:]}
	c.ClientID = r.u64()
	c.RequestID = r.u64()
	c.Key = r.str()
	c.Value = r.str()
	c.Expect = r.str()
	return c, r.err
}

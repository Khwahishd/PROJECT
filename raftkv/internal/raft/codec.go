package raft

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// This file defines the wire and on-disk encodings for Raft's durable and
// network types. A hand-rolled binary format is used rather than JSON or a
// generated codec for three reasons: the records are tiny and read on every
// append, the format must be stable across restarts, and keeping the whole
// repository dependency-free makes the fault-injection harness trivially
// deterministic (no third-party map iteration or time sources sneaking in).
//
// All integers are little-endian. Byte slices are length-prefixed with a uint32.

// ErrCorrupt indicates a malformed record.
var ErrCorrupt = errors.New("raft: corrupt record")

type encoder struct{ buf []byte }

func (e *encoder) u8(v uint8)   { e.buf = append(e.buf, v) }
func (e *encoder) u64(v uint64) { e.buf = binary.LittleEndian.AppendUint64(e.buf, v) }
func (e *encoder) u32(v uint32) { e.buf = binary.LittleEndian.AppendUint32(e.buf, v) }

func (e *encoder) bytes(b []byte) {
	e.u32(uint32(len(b)))
	e.buf = append(e.buf, b...)
}

type decoder struct {
	buf []byte
	err error
}

func (d *decoder) fail(format string, args ...any) {
	if d.err == nil {
		d.err = fmt.Errorf("%w: %s", ErrCorrupt, fmt.Sprintf(format, args...))
	}
}

func (d *decoder) u8() uint8 {
	if len(d.buf) < 1 {
		d.fail("truncated uint8")
		return 0
	}
	v := d.buf[0]
	d.buf = d.buf[1:]
	return v
}

func (d *decoder) u32() uint32 {
	if len(d.buf) < 4 {
		d.fail("truncated uint32")
		return 0
	}
	v := binary.LittleEndian.Uint32(d.buf)
	d.buf = d.buf[4:]
	return v
}

func (d *decoder) u64() uint64 {
	if len(d.buf) < 8 {
		d.fail("truncated uint64")
		return 0
	}
	v := binary.LittleEndian.Uint64(d.buf)
	d.buf = d.buf[8:]
	return v
}

func (d *decoder) bytes() []byte {
	n := d.u32()
	if d.err != nil {
		return nil
	}
	if uint64(n) > uint64(len(d.buf)) {
		d.fail("length prefix %d exceeds remaining %d bytes", n, len(d.buf))
		return nil
	}
	if n == 0 {
		return nil
	}
	out := make([]byte, n)
	copy(out, d.buf[:n])
	d.buf = d.buf[n:]
	return out
}

// EncodeConfChange serialises a ConfChange for storage in a log entry.
func EncodeConfChange(cc ConfChange) []byte {
	e := &encoder{}
	e.u8(uint8(cc.Type))
	e.u64(uint64(cc.NodeID))
	return e.buf
}

// DecodeConfChange parses a ConfChange from an EntryConfChange payload.
func DecodeConfChange(b []byte) (ConfChange, error) {
	d := &decoder{buf: b}
	cc := ConfChange{Type: ConfChangeType(d.u8()), NodeID: NodeID(d.u64())}
	if d.err != nil {
		return ConfChange{}, d.err
	}
	if cc.Type != ConfChangeAddNode && cc.Type != ConfChangeRemoveNode {
		return ConfChange{}, fmt.Errorf("%w: unknown conf change type %d", ErrCorrupt, cc.Type)
	}
	return cc, nil
}

// EncodeEntry serialises a log entry.
func EncodeEntry(e Entry) []byte {
	enc := &encoder{}
	enc.u64(e.Term)
	enc.u64(e.Index)
	enc.u8(uint8(e.Type))
	enc.bytes(e.Data)
	return enc.buf
}

// DecodeEntry parses a log entry.
func DecodeEntry(b []byte) (Entry, error) {
	d := &decoder{buf: b}
	e := Entry{Term: d.u64(), Index: d.u64(), Type: EntryType(d.u8())}
	e.Data = d.bytes()
	return e, d.err
}

// EncodeHardState serialises the durable term/vote/commit triple.
func EncodeHardState(hs HardState) []byte {
	e := &encoder{}
	e.u64(hs.Term)
	e.u64(uint64(hs.Vote))
	e.u64(hs.Commit)
	return e.buf
}

// DecodeHardState parses a HardState.
func DecodeHardState(b []byte) (HardState, error) {
	d := &decoder{buf: b}
	hs := HardState{Term: d.u64(), Vote: NodeID(d.u64()), Commit: d.u64()}
	return hs, d.err
}

// EncodeSnapshot serialises snapshot metadata, membership and payload.
func EncodeSnapshot(s *Snapshot) []byte {
	e := &encoder{}
	e.u64(s.Index)
	e.u64(s.Term)
	e.u32(uint32(len(s.Conf.Voters)))
	for _, v := range s.Conf.Voters {
		e.u64(uint64(v))
	}
	e.bytes(s.Data)
	return e.buf
}

// DecodeSnapshot parses a snapshot.
func DecodeSnapshot(b []byte) (*Snapshot, error) {
	d := &decoder{buf: b}
	s := &Snapshot{Index: d.u64(), Term: d.u64()}
	n := d.u32()
	if d.err != nil {
		return nil, d.err
	}
	// Guard against a corrupt length driving a huge allocation.
	if uint64(n)*8 > uint64(len(d.buf)) {
		return nil, fmt.Errorf("%w: voter count %d exceeds remaining bytes", ErrCorrupt, n)
	}
	s.Conf.Voters = make([]NodeID, n)
	for i := range s.Conf.Voters {
		s.Conf.Voters[i] = NodeID(d.u64())
	}
	s.Data = d.bytes()
	return s, d.err
}

// EncodeMessage serialises a peer-to-peer message for the network transport.
func EncodeMessage(m Message) []byte {
	e := &encoder{}
	e.u8(uint8(m.Type))
	e.u64(uint64(m.From))
	e.u64(uint64(m.To))
	e.u64(m.Term)
	e.u64(m.PrevLogIndex)
	e.u64(m.PrevLogTerm)
	e.u64(m.LeaderCommit)
	e.u64(m.LastLogIndex)
	e.u64(m.LastLogTerm)
	e.u64(m.MatchIndex)
	e.u64(m.ConflictIndex)
	e.u64(m.ConflictTerm)
	e.u64(m.ReadIndex)
	var flags uint8
	if m.Granted {
		flags |= 1
	}
	if m.Reject {
		flags |= 2
	}
	if m.Snapshot != nil {
		flags |= 4
	}
	e.u8(flags)
	e.bytes(m.Context)
	e.u32(uint32(len(m.Entries)))
	for _, en := range m.Entries {
		e.bytes(EncodeEntry(en))
	}
	if m.Snapshot != nil {
		e.bytes(EncodeSnapshot(m.Snapshot))
	}
	return e.buf
}

// DecodeMessage parses a peer-to-peer message.
func DecodeMessage(b []byte) (Message, error) {
	d := &decoder{buf: b}
	m := Message{
		Type:          MessageType(d.u8()),
		From:          NodeID(d.u64()),
		To:            NodeID(d.u64()),
		Term:          d.u64(),
		PrevLogIndex:  d.u64(),
		PrevLogTerm:   d.u64(),
		LeaderCommit:  d.u64(),
		LastLogIndex:  d.u64(),
		LastLogTerm:   d.u64(),
		MatchIndex:    d.u64(),
		ConflictIndex: d.u64(),
		ConflictTerm:  d.u64(),
		ReadIndex:     d.u64(),
	}
	flags := d.u8()
	m.Granted = flags&1 != 0
	m.Reject = flags&2 != 0
	hasSnap := flags&4 != 0
	m.Context = d.bytes()

	n := d.u32()
	if d.err != nil {
		return Message{}, d.err
	}
	// Each encoded entry is at least 21 bytes plus its 4-byte frame, so a count
	// larger than the remaining buffer can hold is corrupt.
	if uint64(n)*4 > uint64(len(d.buf)) {
		return Message{}, fmt.Errorf("%w: entry count %d exceeds remaining bytes", ErrCorrupt, n)
	}
	if n > 0 {
		m.Entries = make([]Entry, n)
		for i := range m.Entries {
			raw := d.bytes()
			if d.err != nil {
				return Message{}, d.err
			}
			en, err := DecodeEntry(raw)
			if err != nil {
				return Message{}, err
			}
			m.Entries[i] = en
		}
	}
	if hasSnap {
		raw := d.bytes()
		if d.err != nil {
			return Message{}, d.err
		}
		s, err := DecodeSnapshot(raw)
		if err != nil {
			return Message{}, err
		}
		m.Snapshot = s
	}
	return m, d.err
}

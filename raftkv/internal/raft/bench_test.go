package raft

import (
	"fmt"
	"math/rand"
	"testing"
)

func benchNode(b *testing.B, peers int) *Node {
	b.Helper()
	ids := make([]NodeID, peers)
	for i := range ids {
		ids[i] = NodeID(i + 1)
	}
	n, err := NewNode(NodeConfig{
		ID: 1, Peers: ids,
		ElectionTimeout: 10, HeartbeatTimeout: 2,
		Rand: rand.New(rand.NewSource(1)),
	})
	if err != nil {
		b.Fatal(err)
	}
	// Drive it to leadership so proposals are accepted.
	for i := 0; i < 40 && n.Status().Role != Leader; i++ {
		n.Tick()
		for _, m := range n.Ready().Messages {
			if m.Type == MsgVoteReq {
				n.Step(Message{Type: MsgVoteResp, From: m.To, To: 1, Term: m.Term, Granted: true})
			}
		}
		rd := n.Ready()
		n.Advance(rd)
	}
	if n.Status().Role != Leader {
		b.Fatal("failed to become leader")
	}
	return n
}

// BenchmarkPropose measures the cost of the consensus core alone: appending an
// entry and generating the outbound AppendEntries for every follower, with no
// disk or network involved.
func BenchmarkPropose(b *testing.B) {
	for _, peers := range []int{3, 5, 7} {
		b.Run(fmt.Sprintf("peers=%d", peers), func(b *testing.B) {
			n := benchNode(b, peers)
			payload := make([]byte, 128)
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, _, err := n.Propose(payload); err != nil {
					b.Fatal(err)
				}
				rd := n.Ready()
				n.Advance(rd)
			}
		})
	}
}

// BenchmarkAppendEntries measures the follower-side hot path: the consistency
// check plus appending a batch of entries.
func BenchmarkAppendEntries(b *testing.B) {
	for _, batch := range []int{1, 16, 128} {
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			n, err := NewNode(NodeConfig{
				ID: 2, Peers: []NodeID{1, 2, 3},
				ElectionTimeout: 10, HeartbeatTimeout: 2,
				Rand: rand.New(rand.NewSource(1)),
			})
			if err != nil {
				b.Fatal(err)
			}
			payload := make([]byte, 128)
			var next uint64 = 1
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				ents := make([]Entry, batch)
				for j := range ents {
					ents[j] = Entry{Index: next + uint64(j), Term: 1, Data: payload}
				}
				prevIndex := next - 1
				n.Step(Message{
					Type: MsgAppendReq, From: 1, To: 2, Term: 1,
					PrevLogIndex: prevIndex, PrevLogTerm: termFor(prevIndex),
					Entries: ents, LeaderCommit: prevIndex,
				})
				next += uint64(batch)
				rd := n.Ready()
				n.Advance(rd)
			}
		})
	}
}

func termFor(index uint64) uint64 {
	if index == 0 {
		return 0
	}
	return 1
}

// BenchmarkLogSlice measures reads out of the in-memory log, which a leader
// performs on every replication round.
func BenchmarkLogSlice(b *testing.B) {
	l := newRaftLog(0, 0)
	for i := uint64(1); i <= 100_000; i++ {
		l.append(Entry{Index: i, Term: 1, Data: make([]byte, 64)})
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		lo := uint64(i%90_000) + 1
		_ = l.slice(lo, lo+64, 64)
	}
}

// BenchmarkMessageCodec measures the wire encoding, which runs once per message
// on both the send and receive path.
func BenchmarkMessageCodec(b *testing.B) {
	ents := make([]Entry, 32)
	for i := range ents {
		ents[i] = Entry{Index: uint64(i + 1), Term: 3, Data: make([]byte, 256)}
	}
	m := Message{
		Type: MsgAppendReq, From: 1, To: 2, Term: 3,
		PrevLogIndex: 0, PrevLogTerm: 0, Entries: ents, LeaderCommit: 0,
	}

	b.Run("encode", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = EncodeMessage(m)
		}
	})
	b.Run("decode", func(b *testing.B) {
		buf := EncodeMessage(m)
		b.ReportAllocs()
		b.SetBytes(int64(len(buf)))
		for i := 0; i < b.N; i++ {
			if _, err := DecodeMessage(buf); err != nil {
				b.Fatal(err)
			}
		}
	})
}

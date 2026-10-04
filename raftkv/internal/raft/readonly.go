package raft

// ReadState is a confirmed linearizable read point: once the state machine has
// applied through Index, a read tagged with Context is guaranteed to observe
// every write that completed before the read was issued.
type ReadState struct {
	Index   uint64
	Context []byte
}

// readIndexStatus tracks the heartbeat quorum confirming one read request.
type readIndexStatus struct {
	index uint64
	ctx   []byte
	acks  map[NodeID]struct{}
}

// readOnly accumulates in-flight ReadIndex requests on a leader.
//
// Serving reads from the leader's local state is not safe on its own: a
// deposed leader may not yet know it has been deposed. ReadIndex (§6.4) makes
// reads linearizable without appending to the log, by recording the current
// commit index and confirming leadership with a heartbeat round before
// releasing the read.
type readOnly struct {
	pending []*readIndexStatus
}

func (ro *readOnly) add(index uint64, ctx []byte, self NodeID) {
	st := &readIndexStatus{index: index, ctx: ctx, acks: map[NodeID]struct{}{}}
	st.acks[self] = struct{}{} // the leader's own implicit ack
	ro.pending = append(ro.pending, st)
}

// recvAck records a heartbeat acknowledgement and returns every read request
// that has now reached quorum, in issue order.
//
// Requests are confirmed in order: an ack for a later request implies the
// earlier ones are confirmed too, since they were broadcast earlier and carry
// smaller indices.
func (ro *readOnly) recvAck(from NodeID, ctx []byte, quorum int) []ReadState {
	matched := -1
	for i, st := range ro.pending {
		if string(st.ctx) != string(ctx) {
			continue
		}
		st.acks[from] = struct{}{}
		if len(st.acks) >= quorum {
			matched = i
		}
		break
	}
	if matched < 0 {
		return nil
	}
	ready := make([]ReadState, 0, matched+1)
	for _, st := range ro.pending[:matched+1] {
		ready = append(ready, ReadState{Index: st.index, Context: st.ctx})
	}
	ro.pending = ro.pending[matched+1:]
	return ready
}

// reset drops all pending reads, used when a node ceases to be leader. The
// client retries; silently serving them would be unsafe.
func (ro *readOnly) reset() { ro.pending = nil }

// lastPendingContext returns the context of the most recent request, which the
// leader piggybacks on its next heartbeat round.
func (ro *readOnly) lastPendingContext() ([]byte, bool) {
	if len(ro.pending) == 0 {
		return nil, false
	}
	return ro.pending[len(ro.pending)-1].ctx, true
}

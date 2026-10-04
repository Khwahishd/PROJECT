// Package storage provides crash-safe durable storage for a Raft node: a
// segmented write-ahead log for entries, a small record for the hard state, and
// snapshot files.
//
// The correctness requirement that drives the design is narrow but absolute:
// when Node.Ready hands over entries and a hard state, they must be on disk
// before any message in that Ready is sent. Everything here exists to make that
// guarantee cheap enough to do on every append.
package storage

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/Khwahishd/raftkv/internal/raft"
)

// ErrNotFound indicates a missing file or record.
var ErrNotFound = errors.New("storage: not found")

const (
	walFileName      = "raft.wal"
	hardStateFile    = "hardstate"
	snapshotPrefix   = "snapshot-"
	snapshotSuffix   = ".snap"
	recordHeaderSize = 8 // uint32 length + uint32 crc
)

// Storage is a node's durable state directory.
type Storage struct {
	dir string

	wal    *os.File
	walBuf *bufio.Writer

	// firstIndex is the lowest index present in the WAL. Entries below it have
	// been compacted into a snapshot.
	firstIndex uint64
	lastIndex  uint64

	// offsets maps a log index to its byte offset in the WAL, so truncation can
	// seek directly instead of rescanning.
	offsets map[uint64]int64
	walSize int64

	// SyncWrites controls whether every append is fsynced. It defaults to true;
	// disabling it trades durability for throughput and is only appropriate for
	// benchmarks, never for a real deployment.
	SyncWrites bool
}

// Open opens (creating if necessary) the storage directory at dir.
func Open(dir string) (*Storage, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("storage: create dir: %w", err)
	}
	s := &Storage{
		dir:        dir,
		offsets:    map[uint64]int64{},
		SyncWrites: true,
	}
	f, err := os.OpenFile(filepath.Join(dir, walFileName), os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("storage: open wal: %w", err)
	}
	s.wal = f
	s.walBuf = bufio.NewWriterSize(f, 64*1024)
	return s, nil
}

// Close flushes and closes the storage.
func (s *Storage) Close() error {
	if s.wal == nil {
		return nil
	}
	if err := s.walBuf.Flush(); err != nil {
		return err
	}
	err := s.wal.Close()
	s.wal = nil
	return err
}

// ---------------------------------------------------------------------------
// Write-ahead log
// ---------------------------------------------------------------------------

// writeRecord frames a payload as [len][crc][payload].
//
// The CRC is what makes a torn tail detectable. A crash mid-append leaves a
// partial record; on reload, ReadEntries stops at the first record whose length
// runs past the end of the file or whose checksum does not match, and treats
// everything after it as never having been written. That is exactly the right
// semantics: an entry that was not fully persisted was never acknowledged.
func (s *Storage) writeRecord(payload []byte) error {
	var hdr [recordHeaderSize]byte
	binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint32(hdr[4:8], crc32.ChecksumIEEE(payload))
	if _, err := s.walBuf.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := s.walBuf.Write(payload); err != nil {
		return err
	}
	return nil
}

// Append writes entries to the WAL, truncating any existing suffix that they
// overwrite. It is the durable counterpart of raft's log truncation.
func (s *Storage) Append(entries []raft.Entry) error {
	if len(entries) == 0 {
		return nil
	}
	first := entries[0].Index

	// Overwriting an index we already hold means the leader changed and this
	// node's uncommitted tail is being replaced. Physically truncate so a
	// restart cannot resurrect the stale suffix.
	if first <= s.lastIndex && s.lastIndex > 0 {
		off, ok := s.offsets[first]
		if !ok {
			return fmt.Errorf("storage: no offset recorded for index %d", first)
		}
		if err := s.walBuf.Flush(); err != nil {
			return err
		}
		if err := s.wal.Truncate(off); err != nil {
			return fmt.Errorf("storage: truncate wal: %w", err)
		}
		if _, err := s.wal.Seek(off, io.SeekStart); err != nil {
			return err
		}
		s.walBuf.Reset(s.wal)
		s.walSize = off
		for i := first; i <= s.lastIndex; i++ {
			delete(s.offsets, i)
		}
		s.lastIndex = first - 1
	}

	for _, e := range entries {
		s.offsets[e.Index] = s.walSize
		payload := raft.EncodeEntry(e)
		if err := s.writeRecord(payload); err != nil {
			return fmt.Errorf("storage: append entry %d: %w", e.Index, err)
		}
		s.walSize += int64(recordHeaderSize + len(payload))
		s.lastIndex = e.Index
		if s.firstIndex == 0 {
			s.firstIndex = e.Index
		}
	}
	return s.flush()
}

func (s *Storage) flush() error {
	if err := s.walBuf.Flush(); err != nil {
		return err
	}
	if s.SyncWrites {
		return s.wal.Sync()
	}
	return nil
}

// ReadEntries loads every intact entry from the WAL, stopping at the first
// damaged or partial record.
func (s *Storage) ReadEntries() ([]raft.Entry, error) {
	if err := s.walBuf.Flush(); err != nil {
		return nil, err
	}
	if _, err := s.wal.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	r := bufio.NewReaderSize(s.wal, 64*1024)

	var (
		out    []raft.Entry
		offset int64
	)
	s.offsets = map[uint64]int64{}
	for {
		var hdr [recordHeaderSize]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break // clean end, or a header torn by a crash
			}
			return nil, err
		}
		length := binary.LittleEndian.Uint32(hdr[0:4])
		want := binary.LittleEndian.Uint32(hdr[4:8])

		payload := make([]byte, length)
		if _, err := io.ReadFull(r, payload); err != nil {
			break // torn payload: the entry was never durable
		}
		if crc32.ChecksumIEEE(payload) != want {
			break // corrupt record: everything from here on is suspect
		}
		e, err := raft.DecodeEntry(payload)
		if err != nil {
			break
		}
		out = append(out, e)
		s.offsets[e.Index] = offset
		offset += int64(recordHeaderSize + len(payload))
	}

	// Discard anything after the damaged point so later appends start clean.
	if err := s.wal.Truncate(offset); err != nil {
		return nil, fmt.Errorf("storage: truncate damaged tail: %w", err)
	}
	if _, err := s.wal.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}
	s.walBuf.Reset(s.wal)
	s.walSize = offset

	if len(out) > 0 {
		s.firstIndex = out[0].Index
		s.lastIndex = out[len(out)-1].Index
	}
	return out, nil
}

// Compact drops WAL records at or below index, after the caller has written a
// snapshot covering them.
//
// It rewrites the WAL to a temporary file and renames it into place, so a crash
// part-way through leaves the original intact rather than a half-rewritten log.
func (s *Storage) Compact(index uint64) error {
	entries, err := s.ReadEntries()
	if err != nil {
		return err
	}
	kept := make([]raft.Entry, 0, len(entries))
	for _, e := range entries {
		if e.Index > index {
			kept = append(kept, e)
		}
	}
	if len(kept) == len(entries) {
		return nil
	}

	tmpPath := filepath.Join(s.dir, walFileName+".compact")
	tmp, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(tmp, 64*1024)
	newOffsets := map[uint64]int64{}
	var size int64
	for _, e := range kept {
		payload := raft.EncodeEntry(e)
		var hdr [recordHeaderSize]byte
		binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(payload)))
		binary.LittleEndian.PutUint32(hdr[4:8], crc32.ChecksumIEEE(payload))
		if _, err := w.Write(hdr[:]); err != nil {
			tmp.Close()
			return err
		}
		if _, err := w.Write(payload); err != nil {
			tmp.Close()
			return err
		}
		newOffsets[e.Index] = size
		size += int64(recordHeaderSize + len(payload))
	}
	if err := w.Flush(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := s.wal.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, filepath.Join(s.dir, walFileName)); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(s.dir, walFileName), os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	s.wal = f
	s.walBuf = bufio.NewWriterSize(f, 64*1024)
	s.offsets = newOffsets
	s.walSize = size
	s.firstIndex = index + 1
	if len(kept) > 0 {
		s.lastIndex = kept[len(kept)-1].Index
	} else {
		s.lastIndex = index
	}
	return syncDir(s.dir)
}

// ---------------------------------------------------------------------------
// Hard state
// ---------------------------------------------------------------------------

// SaveHardState durably records term, vote and commit index.
//
// This is the single most safety-critical write in the system: losing a
// recorded vote lets a node vote twice in one term, which elects two leaders.
// The write therefore goes to a temp file, is fsynced, and is then atomically
// renamed, so a crash leaves either the old state or the new one -- never a
// torn mixture.
func (s *Storage) SaveHardState(hs raft.HardState) error {
	payload := raft.EncodeHardState(hs)
	buf := make([]byte, 4, 4+len(payload))
	binary.LittleEndian.PutUint32(buf[0:4], crc32.ChecksumIEEE(payload))
	buf = append(buf, payload...)
	return writeFileAtomic(filepath.Join(s.dir, hardStateFile), buf)
}

// LoadHardState reads the persisted hard state, returning the zero value if
// none has been written yet.
func (s *Storage) LoadHardState() (raft.HardState, error) {
	b, err := os.ReadFile(filepath.Join(s.dir, hardStateFile))
	if errors.Is(err, os.ErrNotExist) {
		return raft.HardState{}, nil
	}
	if err != nil {
		return raft.HardState{}, err
	}
	if len(b) < 4 {
		return raft.HardState{}, fmt.Errorf("storage: hardstate too short")
	}
	if crc32.ChecksumIEEE(b[4:]) != binary.LittleEndian.Uint32(b[0:4]) {
		return raft.HardState{}, fmt.Errorf("storage: hardstate checksum mismatch")
	}
	return raft.DecodeHardState(b[4:])
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

// SaveSnapshot writes a snapshot atomically and prunes older ones.
func (s *Storage) SaveSnapshot(snap *raft.Snapshot) error {
	payload := raft.EncodeSnapshot(snap)
	buf := make([]byte, 4, 4+len(payload))
	binary.LittleEndian.PutUint32(buf[0:4], crc32.ChecksumIEEE(payload))
	buf = append(buf, payload...)

	name := fmt.Sprintf("%s%020d%s", snapshotPrefix, snap.Index, snapshotSuffix)
	if err := writeFileAtomic(filepath.Join(s.dir, name), buf); err != nil {
		return err
	}
	return s.pruneSnapshots(3)
}

// LoadSnapshot returns the most recent intact snapshot.
//
// Older snapshots are tried in turn if the newest fails its checksum, which
// keeps a single bad write from making the node unrecoverable.
func (s *Storage) LoadSnapshot() (*raft.Snapshot, error) {
	names, err := s.snapshotNames()
	if err != nil {
		return nil, err
	}
	for i := len(names) - 1; i >= 0; i-- {
		b, err := os.ReadFile(filepath.Join(s.dir, names[i]))
		if err != nil || len(b) < 4 {
			continue
		}
		if crc32.ChecksumIEEE(b[4:]) != binary.LittleEndian.Uint32(b[0:4]) {
			continue
		}
		snap, err := raft.DecodeSnapshot(b[4:])
		if err != nil {
			continue
		}
		return snap, nil
	}
	return nil, ErrNotFound
}

func (s *Storage) snapshotNames() ([]string, error) {
	ents, err := os.ReadDir(s.dir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range ents {
		n := e.Name()
		if strings.HasPrefix(n, snapshotPrefix) && strings.HasSuffix(n, snapshotSuffix) {
			names = append(names, n)
		}
	}
	// Zero-padded indices make lexical order the same as numeric order.
	sort.Strings(names)
	return names, nil
}

func (s *Storage) pruneSnapshots(keep int) error {
	names, err := s.snapshotNames()
	if err != nil {
		return err
	}
	for i := 0; i < len(names)-keep; i++ {
		if err := os.Remove(filepath.Join(s.dir, names[i])); err != nil {
			return err
		}
	}
	return nil
}

// FirstIndex and LastIndex report the range currently held in the WAL.
func (s *Storage) FirstIndex() uint64 { return s.firstIndex }

// LastIndex returns the highest index in the WAL.
func (s *Storage) LastIndex() uint64 { return s.lastIndex }

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// writeFileAtomic writes via a temp file + fsync + rename, then fsyncs the
// directory. Without the directory fsync the rename itself can be lost on a
// crash, leaving the old file in place despite the new one being durable.
func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	return syncDir(filepath.Dir(path))
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

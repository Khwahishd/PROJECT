"""Chunking: offsets, overlap, and the boundaries that matter."""

from __future__ import annotations

from itertools import pairwise

import pytest

from lens.chunking import chunk_documents, fixed_window, normalize_whitespace, recursive_split
from lens.types import Document


def doc(text: str, doc_id: str = "d1") -> Document:
    return Document(doc_id=doc_id, text=text)


class TestOffsets:
    def test_offsets_index_back_into_the_source(self):
        """A citation is only verifiable if its offsets are correct."""
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        for chunk in recursive_split(doc(text), size=25, overlap=5):
            assert text[chunk.start : chunk.end] == chunk.text, (
                f"offsets {chunk.start}-{chunk.end} do not reproduce the chunk text"
            )

    def test_offsets_are_correct_with_fixed_windows_too(self):
        text = "x" * 50 + " " + "y" * 50
        for chunk in fixed_window(doc(text), size=30, overlap=10):
            assert text[chunk.start : chunk.end] == chunk.text

    def test_offsets_survive_leading_whitespace(self):
        text = "\n\n   First sentence here.   \n\n  Second one follows.  "
        for chunk in recursive_split(doc(text), size=30, overlap=0):
            assert text[chunk.start : chunk.end] == chunk.text
            assert not chunk.text.startswith(" ")
            assert not chunk.text.endswith(" ")


class TestOverlap:
    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = recursive_split(doc(text), size=200, overlap=50)
        assert len(chunks) > 1
        for prev, nxt in pairwise(chunks):
            assert nxt.start < prev.end, "chunks should overlap, not merely abut"

    def test_overlap_does_not_start_mid_word(self):
        """A chunk beginning 'ily, and performs...' is a token nothing matches."""
        text = " ".join(f"supercalifragilistic{i}" for i in range(40))
        for chunk in recursive_split(doc(text), size=150, overlap=40):
            if chunk.start == 0:
                continue
            preceding = text[chunk.start - 1]
            assert preceding.isspace(), (
                f"chunk starts mid-word after {preceding!r}: {chunk.text[:40]!r}"
            )

    def test_zero_overlap_produces_disjoint_chunks(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = recursive_split(doc(text), size=100, overlap=0)
        for prev, nxt in pairwise(chunks):
            assert nxt.start >= prev.end


class TestBoundaries:
    def test_recursive_split_prefers_paragraph_boundaries(self):
        text = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        chunks = recursive_split(doc(text), size=30, overlap=0)
        # Each paragraph fits in the window, so none should be cut in half.
        for chunk in chunks:
            assert "paragraph" in chunk.text

    def test_a_single_oversized_token_is_hard_cut(self):
        """Every separator exhausted: a hard cut beats blowing the window."""
        text = "a" * 300
        chunks = recursive_split(doc(text), size=100, overlap=0)
        assert len(chunks) >= 3
        assert all(len(c.text) <= 100 for c in chunks)

    def test_chunks_respect_the_size_budget(self):
        text = ". ".join(f"Sentence number {i} with some words" for i in range(60))
        for chunk in recursive_split(doc(text), size=200, overlap=30):
            # Overlap extends a chunk backwards, so the budget applies to the
            # packed body rather than the final span.
            assert len(chunk.text) <= 200 + 30 + 10


class TestEdgeCases:
    def test_empty_and_whitespace_documents_produce_nothing(self):
        assert recursive_split(doc("")) == []
        assert recursive_split(doc("   \n\n  \t ")) == []
        assert fixed_window(doc("")) == []

    def test_a_document_shorter_than_the_window_is_one_chunk(self):
        chunks = recursive_split(doc("short text"), size=500)
        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_chunk_ids_are_unique_across_a_corpus(self):
        docs = [
            doc("Some text here that is long enough to split up nicely.", f"d{i}") for i in range(5)
        ]
        chunks = chunk_documents(docs, size=20, overlap=5)
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_metadata_is_carried_onto_chunks(self):
        d = Document("d1", "Some text here.", {"title": "T", "year": 2024})
        for chunk in recursive_split(d, size=10, overlap=0):
            assert chunk.metadata["title"] == "T"
            assert chunk.metadata["year"] == 2024

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)],
    )
    def test_invalid_parameters_are_rejected(self, size, overlap):
        """overlap >= size would make the window never advance -- an infinite loop."""
        with pytest.raises(ValueError):
            recursive_split(doc("some text"), size=size, overlap=overlap)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown chunking strategy"):
            chunk_documents([doc("text")], strategy="magic")

    def test_empty_doc_id_is_rejected(self):
        with pytest.raises(ValueError, match="doc_id"):
            Document(doc_id="", text="x")


def test_normalize_whitespace_preserves_paragraphs():
    assert normalize_whitespace("a  \t b\n\n\n\nc") == "a b\n\nc"

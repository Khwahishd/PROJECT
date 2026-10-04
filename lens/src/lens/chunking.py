"""Document chunking.

Chunking is the most consequential and least glamorous part of a retrieval
system. A chunk that is too large dilutes its embedding with unrelated content
so it matches nothing well; one that is too small loses the context that makes
it answerable. Overlap exists because the sentence that answers a question is
frequently the one straddling a naive boundary.

Two strategies are provided, with the same interface:

* :func:`fixed_window` -- deterministic character windows with overlap.
* :func:`recursive_split` -- splits on the largest natural boundary that fits
  (paragraph, then sentence, then word), falling back to a hard cut only when
  a single token exceeds the window.

``recursive_split`` is the default because it keeps semantic units intact
whenever it can, which measurably improves retrieval on prose.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence

from .types import Chunk, Document

__all__ = ["Chunker", "chunk_documents", "fixed_window", "recursive_split"]

#: Separators tried in order, from the largest natural boundary to the smallest.
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

#: A chunker turns one document into its chunks.
Chunker = Callable[[Document], list[Chunk]]


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= size:
        # With overlap >= size the window never advances, so chunking would
        # loop forever producing the same span.
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk size ({size})")


def fixed_window(doc: Document, size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Splits ``doc`` into fixed-size character windows with ``overlap``."""
    _validate(size, overlap)
    text = doc.text
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        span = text[start:end]
        if span.strip():
            chunks.append(_make_chunk(doc, span, start, end, len(chunks)))
        if end == len(text):
            break
        start += step
    return chunks


def recursive_split(
    doc: Document,
    size: int = 512,
    overlap: int = 64,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Splits ``doc`` on the largest natural boundary that fits within ``size``."""
    _validate(size, overlap)
    if not doc.text.strip():
        return []

    pieces = list(_split_recursive(doc.text, 0, size, tuple(separators)))
    merged = _merge_with_overlap(doc.text, pieces, size, overlap)
    return [
        _make_chunk(doc, doc.text[s:e], s, e, i)
        for i, (s, e) in enumerate(merged)
        if doc.text[s:e].strip()
    ]


def _split_recursive(
    text: str, offset: int, size: int, separators: tuple[str, ...]
) -> Iterator[tuple[int, int]]:
    """Yields ``(start, end)`` spans no longer than ``size`` where possible."""
    if len(text) <= size:
        if text:
            yield (offset, offset + len(text))
        return

    if not separators:
        # Every separator has been exhausted and the text is still too long --
        # a single unbroken token. A hard cut is the only option left, and is
        # better than emitting a span that blows the window.
        for i in range(0, len(text), size):
            yield (offset + i, offset + min(i + size, len(text)))
        return

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        yield from _split_recursive(text, offset, size, rest)
        return

    cursor = 0
    for part in text.split(sep):
        part_start = cursor
        part_end = cursor + len(part)
        if part.strip():
            yield from _split_recursive(part, offset + part_start, size, rest)
        cursor = part_end + len(sep)


def _merge_with_overlap(
    text: str, pieces: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int]]:
    """Greedily packs adjacent pieces up to ``size``, then applies overlap.

    Splitting alone produces fragments far smaller than the target; packing them
    back up to the window is what makes the chunks useful. The overlap is then
    taken from the *preceding* text, so each chunk begins with the tail of its
    neighbour.
    """
    if not pieces:
        return []

    packed: list[tuple[int, int]] = []
    cur_start, cur_end = pieces[0]
    for start, end in pieces[1:]:
        # Measure from the current chunk's start to the candidate's end, so the
        # whitespace between pieces counts toward the window too.
        if end - cur_start <= size:
            cur_end = end
        else:
            packed.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    packed.append((cur_start, cur_end))

    if overlap == 0:
        return packed

    out: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(packed):
        if i == 0:
            out.append((start, end))
            continue
        # Extend backwards into the previous chunk, but never past the start of
        # the document or beyond the previous chunk's own start.
        back = max(0, start - overlap, packed[i - 1][0])
        back = _snap_to_word(text, back, limit=start)
        out.append((back, end))
    return out


def _snap_to_word(text: str, position: int, *, limit: int) -> int:
    """Moves ``position`` forward to the next word boundary, at most to ``limit``.

    Overlap is measured in characters, so it lands wherever it lands -- very
    often mid-word. A chunk beginning ``"ily, and performs a best-first..."``
    is not just untidy: that fragment is a token the embedder has never
    usefully seen, and it shows up verbatim in any answer that quotes the
    chunk.
    """
    if position <= 0 or position >= limit:
        return position
    if text[position - 1].isspace():
        return position
    while position < limit and not text[position].isspace():
        position += 1
    while position < limit and text[position].isspace():
        position += 1
    return position


def _make_chunk(doc: Document, text: str, start: int, end: int, ordinal: int) -> Chunk:
    # Trim surrounding whitespace while keeping offsets honest, so a citation
    # highlights the text and not the padding around it.
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    real_start = start + lead
    real_end = end - trail
    return Chunk(
        chunk_id=f"{doc.doc_id}::{ordinal}",
        doc_id=doc.doc_id,
        text=text.strip(),
        start=real_start,
        end=real_end,
        metadata=dict(doc.metadata),
    )


def chunk_documents(
    docs: Sequence[Document],
    size: int = 512,
    overlap: int = 64,
    strategy: str = "recursive",
) -> list[Chunk]:
    """Chunks a corpus with the named strategy."""
    if strategy not in ("recursive", "fixed"):
        raise ValueError(f"unknown chunking strategy {strategy!r}; use 'recursive' or 'fixed'")

    out: list[Chunk] = []
    for doc in docs:
        if strategy == "recursive":
            out.extend(recursive_split(doc, size=size, overlap=overlap))
        else:
            out.extend(fixed_window(doc, size=size, overlap=overlap))
    return out


def normalize_whitespace(text: str) -> str:
    """Collapses runs of whitespace, preserving paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

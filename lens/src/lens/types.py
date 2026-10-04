"""Core data types shared across retrieval, generation and evaluation.

Everything here is a frozen dataclass. Retrieval results get passed between a
retriever, a reranker, a generator and an evaluator, and making them immutable
means a component cannot quietly mutate a score another component already
recorded -- a class of bug that is painful to find because the numbers still
look plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "QueryExample",
    "RetrievalResult",
    "ScoredChunk",
]


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable span of a document.

    ``start`` and ``end`` are character offsets into the parent document's
    text. Keeping them lets a citation point at the exact span rather than at
    the whole document, which is the difference between a verifiable answer and
    one the user has to take on trust.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of the chunk in characters."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk together with the score that retrieved it."""

    chunk: Chunk
    score: float
    #: Where the score came from: ``"dense"``, ``"lexical"``, ``"hybrid"`` or
    #: the name of a reranker. Carrying it makes a fused result explainable.
    source: str = "unknown"
    #: Per-retriever contributions, populated by hybrid fusion.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """The underlying chunk's id."""
        return self.chunk.chunk_id


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The ranked output of a single query, with timing."""

    query: str
    chunks: tuple[ScoredChunk, ...]
    #: Wall-clock retrieval latency in milliseconds.
    latency_ms: float = 0.0
    #: Free-form diagnostics, e.g. per-retriever candidate counts.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def chunk_ids(self) -> list[str]:
        """The retrieved chunk ids, in rank order."""
        return [c.chunk_id for c in self.chunks]

    @property
    def doc_ids(self) -> list[str]:
        """The retrieved document ids, in rank order, deduplicated.

        A document is as relevant as its best-ranked chunk, so the first
        occurrence wins and later chunks of the same document are dropped.
        Evaluating recall against documents rather than chunks is usually what
        a labelled dataset supports.
        """
        seen: list[str] = []
        for c in self.chunks:
            if c.chunk.doc_id not in seen:
                seen.append(c.chunk.doc_id)
        return seen


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer with the context it was grounded in."""

    question: str
    text: str
    citations: tuple[ScoredChunk, ...] = ()
    #: Which generator produced it, for reproducibility.
    model: str = "unknown"
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryExample:
    """One labelled example in an evaluation set."""

    query_id: str
    query: str
    #: Document ids that answer the query. Order is irrelevant.
    relevant_doc_ids: frozenset[str] = frozenset()
    #: Optional graded relevance, doc_id -> gain. Used by nDCG; a document
    #: absent from this mapping but present in ``relevant_doc_ids`` has gain 1.
    gains: dict[str, float] = field(default_factory=dict)
    #: Optional reference answer, for answer-quality metrics.
    reference_answer: str | None = None

    def gain(self, doc_id: str) -> float:
        """The graded gain of a document, defaulting to binary relevance."""
        if doc_id in self.gains:
            return self.gains[doc_id]
        return 1.0 if doc_id in self.relevant_doc_ids else 0.0

    @property
    def all_relevant(self) -> frozenset[str]:
        """Every document with non-zero gain."""
        graded = {d for d, g in self.gains.items() if g > 0}
        return frozenset(self.relevant_doc_ids | graded)

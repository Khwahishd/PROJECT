"""Retrievers: dense, lexical, and the hybrid that combines them."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np

from ..index.bm25 import BM25Index
from ..index.hnsw import BruteForceIndex, HNSWIndex, Metric
from ..providers.base import Embedder
from ..types import Chunk, RetrievalResult, ScoredChunk
from .fusion import reciprocal_rank_fusion, weighted_fusion

__all__ = ["DenseRetriever", "HybridRetriever", "LexicalRetriever", "Retriever"]

FusionMethod = Literal["rrf", "weighted"]


class Retriever(Protocol):
    """Anything that turns a query into ranked chunks."""

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        ...

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Returns the top ``k`` chunks for ``query``."""
        ...


class _ChunkStore:
    """Chunk storage shared by the retrievers."""

    __slots__ = ("by_id", "chunks")

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique")
        self.chunks = list(chunks)
        self.by_id = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)


class DenseRetriever:
    """Embedding-based retrieval over an ANN index."""

    __slots__ = ("_embedder", "_exact", "_index", "_name", "_store")

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        metric: Metric = "cosine",
        exact: bool | None = None,
        hnsw_m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        fit_idf: bool = True,
    ) -> None:
        self._store = _ChunkStore(chunks)
        self._embedder = embedder

        texts = [c.text for c in self._store.chunks]
        # Give the embedder corpus statistics when it can use them, so a
        # document's vector is not dominated by its stop words.
        if fit_idf and hasattr(embedder, "fit_idf"):
            embedder.fit_idf(texts)  # type: ignore[attr-defined]

        # Below a few thousand vectors a contiguous matrix multiply beats
        # pointer-chasing through a graph, and exact search removes recall as a
        # variable. Above it, the graph is the only thing that scales.
        self._exact = exact if exact is not None else len(self._store) < 2000
        vectors = embedder.embed(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)

        if self._exact:
            index: BruteForceIndex | HNSWIndex = BruteForceIndex(metric=metric)
        else:
            index = HNSWIndex(
                metric=metric, M=hnsw_m, ef_construction=ef_construction, ef_search=ef_search
            )
        if texts:
            index.add([c.chunk_id for c in self._store.chunks], vectors)
        self._index = index
        self._name = f"dense[{embedder.name},{'exact' if self._exact else 'hnsw'}]"

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        return self._name

    @property
    def is_exact(self) -> bool:
        """Whether search is exact rather than approximate."""
        return self._exact

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Returns the top ``k`` chunks by embedding similarity."""
        start = time.perf_counter()
        if not self._store.chunks:
            return RetrievalResult(query=query, chunks=(), latency_ms=0.0)
        vector = self._embedder.embed_query(query)
        hits = self._index.search(vector, k=k)
        scored = tuple(
            ScoredChunk(chunk=self._store.by_id[cid], score=float(s), source="dense")
            for cid, s in hits
        )
        return RetrievalResult(
            query=query,
            chunks=scored,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            diagnostics={"index": "exact" if self._exact else "hnsw", "candidates": len(hits)},
        )


class LexicalRetriever:
    """BM25 retrieval."""

    __slots__ = ("_index", "_name", "_store")

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._store = _ChunkStore(chunks)
        self._index = BM25Index(k1=k1, b=b).fit(
            [c.chunk_id for c in self._store.chunks], [c.text for c in self._store.chunks]
        )
        self._name = f"bm25[k1={k1},b={b}]"

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        return self._name

    @property
    def index(self) -> BM25Index:
        """The underlying BM25 index, for explanations."""
        return self._index

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Returns the top ``k`` chunks by BM25 score."""
        start = time.perf_counter()
        hits = self._index.score(query, top_k=k)
        scored = tuple(
            ScoredChunk(chunk=self._store.by_id[cid], score=float(s), source="lexical")
            for cid, s in hits
        )
        return RetrievalResult(
            query=query,
            chunks=scored,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            diagnostics={"matched": len(hits)},
        )


class HybridRetriever:
    """Fuses dense and lexical retrieval."""

    __slots__ = ("_dense", "_lexical", "_method", "_name", "_overfetch", "_rrf_k", "_weights")

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        *,
        method: FusionMethod = "rrf",
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: float = 60.0,
        overfetch: int = 3,
    ) -> None:
        if overfetch < 1:
            raise ValueError(f"overfetch must be at least 1, got {overfetch}")
        self._dense = dense
        self._lexical = lexical
        self._method: FusionMethod = method
        self._weights = {"dense": dense_weight, "lexical": lexical_weight}
        self._rrf_k = rrf_k
        # Each retriever is asked for more than k. Fusion reorders, so a chunk
        # ranked 15th by one retriever and 3rd by the other can legitimately
        # finish in the top 10 -- but only if it was fetched at all.
        self._overfetch = overfetch
        self._name = f"hybrid[{method},d={dense_weight},l={lexical_weight}]"

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        return self._name

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Returns the top ``k`` chunks after fusing both retrievers."""
        start = time.perf_counter()
        fetch = k * self._overfetch

        dense_result = self._dense.retrieve(query, k=fetch)
        lexical_result = self._lexical.retrieve(query, k=fetch)

        rankings = {
            "dense": [(c.chunk_id, c.score) for c in dense_result.chunks],
            "lexical": [(c.chunk_id, c.score) for c in lexical_result.chunks],
        }
        if self._method == "rrf":
            fused = reciprocal_rank_fusion(rankings, k=self._rrf_k, weights=self._weights, top_k=k)
        else:
            fused = weighted_fusion(rankings, weights=self._weights, top_k=k)

        lookup = {c.chunk_id: c.chunk for c in dense_result.chunks}
        lookup.update({c.chunk_id: c.chunk for c in lexical_result.chunks})

        scored = tuple(
            ScoredChunk(chunk=lookup[cid], score=float(score), source="hybrid", components=comp)
            for cid, score, comp in fused
            if cid in lookup
        )
        return RetrievalResult(
            query=query,
            chunks=scored,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            diagnostics={
                "dense_candidates": len(dense_result),
                "lexical_candidates": len(lexical_result),
                "dense_ms": round(dense_result.latency_ms, 3),
                "lexical_ms": round(lexical_result.latency_ms, 3),
                "fusion": self._method,
                # How much the two retrievers agreed. Near 1.0 means hybrid is
                # buying nothing; near 0 means they are finding different
                # things and fusion is doing real work.
                "overlap": _overlap(dense_result.chunk_ids, lexical_result.chunk_ids),
            },
        )


def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard overlap between two result sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

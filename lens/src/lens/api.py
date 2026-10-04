"""HTTP API.

The service holds one immutable index built at startup. Retrieval is CPU-bound
and the index is read-only, so handlers are plain ``def`` rather than ``async
def``: FastAPI runs those in a thread pool, which keeps a slow query from
blocking the event loop. Declaring a CPU-bound handler ``async`` would do the
opposite and serialize every request behind the slowest one.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunking import chunk_documents
from .eval import compare, evaluate_retriever
from .eval.datasets import load_jsonl
from .pipeline import RAGPipeline
from .providers import ExtractiveGenerator, HashingEmbedder
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever
from .types import Document, QueryExample

__all__ = ["app", "create_app"]


class AppState:
    """The service's in-memory index and derived retrievers."""

    def __init__(
        self,
        docs: list[Document],
        queries: list[QueryExample],
        *,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        dim: int = 512,
    ) -> None:
        self.docs = docs
        self.queries = queries
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap)
        self.embedder = HashingEmbedder(dim=dim)
        self.dense = DenseRetriever(self.chunks, self.embedder)
        self.lexical = LexicalRetriever(self.chunks)
        self.hybrid = HybridRetriever(self.dense, self.lexical)
        self.generator = ExtractiveGenerator()
        self.built_at = time.time()

    def retriever(self, name: str) -> Any:
        """Looks up a retriever by name."""
        table = {"dense": self.dense, "lexical": self.lexical, "hybrid": self.hybrid}
        if name not in table:
            raise HTTPException(
                status_code=400,
                detail=f"unknown retriever {name!r}; choose one of {sorted(table)}",
            )
        return table[name]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChunkOut(BaseModel):
    """A retrieved chunk as returned over the wire."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    score: float
    source: str
    components: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """A ranked search result."""

    query: str
    retriever: str
    results: list[ChunkOut]
    latency_ms: float
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """A generated answer with its citations."""

    question: str
    answer: str
    citations: list[ChunkOut]
    model: str
    latency_ms: float
    usage: dict[str, int] = Field(default_factory=dict)
    prompt: str | None = None


class MetricRow(BaseModel):
    """One configuration's aggregated metrics."""

    name: str
    means: dict[str, float]
    latency: dict[str, float]
    num_queries: int
    skipped: int


class QueryOutcomeOut(BaseModel):
    """One query's result within an evaluation run."""

    query_id: str
    query: str
    retrieved: list[str]
    relevant: list[str]
    latency_ms: float
    scores: dict[str, float]


class EvalResponse(BaseModel):
    """A single configuration's evaluation."""

    report: MetricRow
    outcomes: list[QueryOutcomeOut]
    failures: list[QueryOutcomeOut]


class SweepResponse(BaseModel):
    """A comparison across configurations."""

    rows: list[MetricRow]
    table: str
    primary: str
    best: str


class StatusResponse(BaseModel):
    """Index and corpus statistics."""

    documents: int
    chunks: int
    queries: int
    chunk_size: int
    chunk_overlap: int
    embedding_dim: int
    vocabulary_size: int
    mean_chunk_chars: float
    dense_index: str


def _to_chunk_out(sc: Any) -> ChunkOut:
    return ChunkOut(
        chunk_id=sc.chunk.chunk_id,
        doc_id=sc.chunk.doc_id,
        text=sc.chunk.text,
        start=sc.chunk.start,
        end=sc.chunk.end,
        score=round(float(sc.score), 6),
        source=sc.source,
        components={k: round(float(v), 6) for k, v in sc.components.items()},
        metadata=sc.chunk.metadata,
    )


def _to_outcome(o: Any) -> QueryOutcomeOut:
    return QueryOutcomeOut(
        query_id=o.query_id,
        query=o.query,
        retrieved=o.retrieved,
        relevant=o.relevant,
        latency_ms=round(o.latency_ms, 3),
        scores={k: round(v, 6) for k, v in o.scores.items()},
    )


def _to_row(report: Any) -> MetricRow:
    return MetricRow(
        name=report.name,
        means={k: round(v, 6) for k, v in report.means.items()},
        latency={k: round(v, 3) for k, v in report.latency.items()},
        num_queries=report.num_queries,
        skipped=report.skipped,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def create_app(
    corpus_path: str | Path | None = None,
    queries_path: str | Path | None = None,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    dim: int = 512,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Builds the application, loading and indexing the corpus at startup."""
    corpus = Path(corpus_path or os.environ.get("LENS_CORPUS", "data/corpus.jsonl"))
    queries = Path(queries_path or os.environ.get("LENS_QUERIES", "data/queries.jsonl"))

    state: dict[str, AppState] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        # Index at startup, not per request. Building it lazily would make the
        # first request after a deploy pay the whole cost, which is exactly
        # when a health check is most likely to time out.
        docs, qs = load_jsonl(corpus, queries)
        state["app"] = AppState(
            docs, qs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, dim=dim
        )
        yield
        state.clear()

    app = FastAPI(
        title="lens",
        description="Hybrid retrieval and RAG evaluation",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def st() -> AppState:
        if "app" not in state:
            raise HTTPException(status_code=503, detail="index is still building")
        return state["app"]

    @app.get("/api/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        """Corpus and index statistics."""
        s = st()
        lengths = [len(c.text) for c in s.chunks] or [0]
        return StatusResponse(
            documents=len(s.docs),
            chunks=len(s.chunks),
            queries=len(s.queries),
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            embedding_dim=s.embedder.dim,
            vocabulary_size=s.lexical.index.vocabulary_size,
            mean_chunk_chars=round(sum(lengths) / len(lengths), 1),
            dense_index="exact" if s.dense.is_exact else "hnsw",
        )

    @app.get("/api/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1, description="the query"),
        retriever: Literal["dense", "lexical", "hybrid"] = "hybrid",
        k: int = Query(10, ge=1, le=100),
    ) -> SearchResponse:
        """Runs one query against the chosen retriever."""
        s = st()
        result = s.retriever(retriever).retrieve(q, k=k)
        return SearchResponse(
            query=q,
            retriever=retriever,
            results=[_to_chunk_out(c) for c in result.chunks],
            latency_ms=round(result.latency_ms, 3),
            diagnostics=result.diagnostics,
        )

    @app.get("/api/ask", response_model=AskResponse)
    def ask(
        q: str = Query(..., min_length=1),
        retriever: Literal["dense", "lexical", "hybrid"] = "hybrid",
        k: int = Query(5, ge=1, le=50),
        include_prompt: bool = False,
    ) -> AskResponse:
        """Runs the RAG pipeline and returns a cited answer."""
        s = st()
        pipeline = RAGPipeline(s.retriever(retriever), s.generator, k=k)
        answer = pipeline.answer(q)
        return AskResponse(
            question=answer.question,
            answer=answer.text,
            citations=[_to_chunk_out(c) for c in answer.citations],
            model=answer.model,
            latency_ms=round(answer.latency_ms, 3),
            usage=answer.usage,
            prompt=pipeline.build_prompt(q) if include_prompt else None,
        )

    @app.get("/api/eval", response_model=EvalResponse)
    def evaluate(
        retriever: Literal["dense", "lexical", "hybrid"] = "hybrid",
        k: int = Query(10, ge=1, le=100),
    ) -> EvalResponse:
        """Evaluates one configuration over the labelled query set."""
        s = st()
        report = evaluate_retriever(s.retriever(retriever), s.queries, k=k)
        return EvalResponse(
            report=_to_row(report),
            outcomes=[_to_outcome(o) for o in report.outcomes],
            failures=[_to_outcome(o) for o in report.failures(f"recall@{k}")],
        )

    @app.get("/api/sweep", response_model=SweepResponse)
    def sweep(
        k: int = Query(10, ge=1, le=100),
        primary: str = "ndcg@10",
    ) -> SweepResponse:
        """Compares every retriever and several fusion weightings."""
        s = st()
        reports = [
            evaluate_retriever(s.dense, s.queries, k=k, name="dense"),
            evaluate_retriever(s.lexical, s.queries, k=k, name="bm25"),
        ]
        for dw, lw in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]:
            hybrid = HybridRetriever(s.dense, s.lexical, dense_weight=dw, lexical_weight=lw)
            reports.append(
                evaluate_retriever(hybrid, s.queries, k=k, name=f"hybrid(d={dw:g},l={lw:g})")
            )
        best = max(reports, key=lambda r: r.means.get(primary, 0.0))
        return SweepResponse(
            rows=[_to_row(r) for r in reports],
            table=compare(reports, primary=primary),
            primary=primary,
            best=best.name,
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Liveness probe. Reports readiness separately from liveness."""
        return {"ok": True, "indexed": "app" in state}

    return app


#: Module-level app for ``uvicorn lens.api:app``.
app = create_app()

"""lens -- hybrid retrieval and RAG evaluation.

A retrieval stack built from first principles: HNSW and BM25 implemented
directly, rank fusion, a RAG pipeline, and an evaluation harness that runs
deterministically with no API key.

Example::

    from lens import build_index, HashingEmbedder
    from lens.eval.datasets import load_jsonl
    from lens.eval import evaluate_retriever, compare

    docs, queries = load_jsonl("data/corpus.jsonl", "data/queries.jsonl")
    dense, lexical, hybrid = build_index(docs, HashingEmbedder())
    reports = [evaluate_retriever(r, queries) for r in (dense, lexical, hybrid)]
    print(compare(reports))
"""

from __future__ import annotations

from collections.abc import Sequence

from .chunking import chunk_documents
from .eval import EvalReport, compare, evaluate_retriever
from .pipeline import RAGPipeline
from .providers import EchoGenerator, Embedder, ExtractiveGenerator, Generator, HashingEmbedder
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever, Retriever
from .types import Answer, Chunk, Document, QueryExample, RetrievalResult, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "DenseRetriever",
    "Document",
    "EchoGenerator",
    "Embedder",
    "EvalReport",
    "ExtractiveGenerator",
    "Generator",
    "HashingEmbedder",
    "HybridRetriever",
    "LexicalRetriever",
    "QueryExample",
    "RAGPipeline",
    "RetrievalResult",
    "Retriever",
    "ScoredChunk",
    "__version__",
    "build_index",
    "chunk_documents",
    "compare",
    "evaluate_retriever",
]


def build_index(
    docs: Sequence[Document],
    embedder: Embedder | None = None,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    strategy: str = "recursive",
    fusion: str = "rrf",
) -> tuple[DenseRetriever, LexicalRetriever, HybridRetriever]:
    """Chunks a corpus and builds all three retrievers over it.

    Returns dense, lexical and hybrid retrievers sharing one set of chunks, so
    a comparison between them is a comparison of retrieval strategy and not of
    chunking.
    """
    chunks = chunk_documents(docs, size=chunk_size, overlap=chunk_overlap, strategy=strategy)
    emb = embedder if embedder is not None else HashingEmbedder()
    dense = DenseRetriever(chunks, emb)
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical, method=fusion)  # type: ignore[arg-type]
    return dense, lexical, hybrid

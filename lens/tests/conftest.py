"""Shared fixtures.

Most fixtures load the *shipped* dataset rather than a synthetic one. That
makes the test suite double as a check that `data/` is valid and that the
corpus actually exhibits the property the project is built around: dense and
lexical retrieval failing on different queries. A synthetic corpus small enough
to inline is also small enough for both retrievers to ace, which would quietly
stop testing anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lens.chunking import chunk_documents
from lens.providers import HashingEmbedder
from lens.retrieval import DenseRetriever, HybridRetriever, LexicalRetriever
from lens.types import Document, QueryExample

#: A deliberately tiny corpus for tests that only need *a* corpus.
DOCS = [
    Document(
        "graph",
        "Layered proximity graphs support approximate nearest neighbour lookup. The efSearch setting trades recall for speed. Error ANN-4417 means the entry point is missing.",
    ),
    Document(
        "lexical",
        "Okapi BM25 weights terms by inverse document frequency and saturates term frequency. The k1 and b parameters control saturation and length normalization.",
    ),
    Document(
        "dense",
        "A bi-encoder maps questions and passages into one vector space independently, so it matches rephrased wording that shares no words with the query.",
    ),
    Document(
        "fusion",
        "Reciprocal rank fusion merges ranked lists by position rather than score, which avoids comparing incompatible score scales.",
    ),
    Document(
        "chunks",
        "Splitting documents into overlapping windows keeps the sentence that answers a question from being cut at a boundary.",
    ),
    Document(
        "rerank",
        "A cross-encoder reads query and passage together and is far more accurate but much slower, so it reranks a shortlist.",
    ),
]

QUERIES = [
    QueryExample("q1", "efSearch", frozenset({"graph"})),
    QueryExample("q2", "ANN-4417", frozenset({"graph"})),
    QueryExample("q3", "k1 and b parameters", frozenset({"lexical"})),
    QueryExample("q4", "matching rephrased wording with no shared words", frozenset({"dense"})),
    QueryExample("q5", "merging ranked lists by position", frozenset({"fusion"})),
    QueryExample("q6", "windows that overlap so sentences stay whole", frozenset({"chunks"})),
    QueryExample("q7", "slow but accurate second stage scoring", frozenset({"rerank"})),
]


#: Repository root, so fixtures do not depend on the working directory.
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def shipped() -> tuple[list[Document], list[QueryExample]]:
    """The dataset in `data/`, loaded once."""
    from lens.eval.datasets import load_jsonl

    return load_jsonl(ROOT / "data" / "corpus.jsonl", ROOT / "data" / "queries.jsonl")


@pytest.fixture(scope="session")
def docs(shipped) -> list[Document]:
    return shipped[0]


@pytest.fixture(scope="session")
def queries(shipped) -> list[QueryExample]:
    return shipped[1]


@pytest.fixture(scope="session")
def tiny_docs() -> list[Document]:
    """A small inline corpus, for tests that do not need a realistic one."""
    return list(DOCS)


@pytest.fixture(scope="session")
def tiny_queries() -> list[QueryExample]:
    return list(QUERIES)


@pytest.fixture(scope="session")
def chunks(docs):
    return chunk_documents(docs, size=400, overlap=60)


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder(dim=512)


@pytest.fixture(scope="session")
def dense(chunks, embedder):
    return DenseRetriever(chunks, embedder)


@pytest.fixture(scope="session")
def lexical(chunks):
    return LexicalRetriever(chunks)


@pytest.fixture(scope="session")
def hybrid(dense, lexical):
    return HybridRetriever(dense, lexical)

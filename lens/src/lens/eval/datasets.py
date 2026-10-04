"""Dataset loading and the built-in evaluation corpus.

Datasets load from JSONL via :func:`load_jsonl`, in the format BEIR and most
RAG benchmarks use. A small labelled corpus ships in ``data/`` so the harness
is runnable, testable and demonstrable with no download, no API key and no
setup -- the evaluation machinery can then be verified independently of any
particular dataset.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..types import Document, QueryExample

__all__ = ["load_jsonl", "save_jsonl"]


def load_jsonl(
    corpus_path: str | Path, queries_path: str | Path
) -> tuple[list[Document], list[QueryExample]]:
    """Loads a corpus and query set from JSONL files.

    Corpus lines: ``{"doc_id": ..., "text": ..., "metadata": {...}}``.
    Query lines: ``{"query_id": ..., "query": ..., "relevant_doc_ids": [...],
    "gains": {...}, "reference_answer": ...}``.
    """
    docs: list[Document] = []
    for line_no, line in _read_lines(corpus_path):
        obj = _parse(line, corpus_path, line_no)
        if "doc_id" not in obj or "text" not in obj:
            raise ValueError(f"{corpus_path}:{line_no}: corpus rows need 'doc_id' and 'text'")
        docs.append(
            Document(
                doc_id=str(obj["doc_id"]),
                text=str(obj["text"]),
                metadata=dict(obj.get("metadata") or {}),
            )
        )

    known = {d.doc_id for d in docs}
    queries: list[QueryExample] = []
    for line_no, line in _read_lines(queries_path):
        obj = _parse(line, queries_path, line_no)
        if "query" not in obj:
            raise ValueError(f"{queries_path}:{line_no}: query rows need a 'query' field")
        relevant = {str(d) for d in (obj.get("relevant_doc_ids") or [])}
        gains = {str(k): float(v) for k, v in (obj.get("gains") or {}).items()}

        # A label pointing at a document that is not in the corpus caps recall
        # below 1.0 for reasons that have nothing to do with the retriever.
        # Failing loudly here beats puzzling over the number later.
        missing = (relevant | set(gains)) - known
        if missing:
            raise ValueError(
                f"{queries_path}:{line_no}: labels reference documents absent from the corpus: "
                f"{sorted(missing)}"
            )

        queries.append(
            QueryExample(
                query_id=str(obj.get("query_id", f"q{line_no}")),
                query=str(obj["query"]),
                relevant_doc_ids=frozenset(relevant),
                gains=gains,
                reference_answer=obj.get("reference_answer"),
            )
        )
    return docs, queries


def _read_lines(path: str | Path) -> Iterable[tuple[int, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if line.strip():
                yield i, line


def _parse(line: str, path: str | Path, line_no: int) -> dict:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}:{line_no}: expected a JSON object, got {type(obj).__name__}")
    return obj


def save_jsonl(
    docs: Iterable[Document], queries: Iterable[QueryExample], directory: str | Path
) -> None:
    """Writes a corpus and query set as JSONL."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    with (d / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(
                json.dumps({"doc_id": doc.doc_id, "text": doc.text, "metadata": doc.metadata})
                + "\n"
            )
    with (d / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "query": q.query,
                        "relevant_doc_ids": sorted(q.relevant_doc_ids),
                        "gains": q.gains,
                        "reference_answer": q.reference_answer,
                    }
                )
                + "\n"
            )

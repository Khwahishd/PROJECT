"""Command line interface.

Four subcommands, each answering a question someone actually asks of a
retrieval system:

``search``   what comes back for this query, and why?
``ask``      what answer does the full RAG pipeline produce, with citations?
``eval``     how good is this configuration, per query and in aggregate?
``sweep``    which configuration is best, and by how much?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .chunking import chunk_documents
from .eval import compare, evaluate_retriever
from .eval.datasets import load_jsonl
from .pipeline import RAGPipeline
from .providers import ExtractiveGenerator, HashingEmbedder
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever

__all__ = ["main"]


def _load(args: argparse.Namespace) -> tuple[Any, Any]:
    corpus = Path(args.corpus)
    queries = (
        Path(args.queries) if getattr(args, "queries", None) else corpus.parent / "queries.jsonl"
    )
    return load_jsonl(corpus, queries)


def _retrievers(docs: Any, args: argparse.Namespace) -> dict[str, Any]:
    chunks = chunk_documents(docs, size=args.chunk_size, overlap=args.chunk_overlap)
    embedder = HashingEmbedder(dim=args.dim)
    dense = DenseRetriever(chunks, embedder)
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical, method=args.fusion)
    return {"dense": dense, "lexical": lexical, "hybrid": hybrid}


def cmd_search(args: argparse.Namespace) -> int:
    """Runs one query and prints the ranked chunks."""
    docs, _ = _load(args)
    retrievers = _retrievers(docs, args)
    retriever = retrievers[args.retriever]
    result = retriever.retrieve(args.query, k=args.k)

    print(f"query: {args.query!r}   ({result.latency_ms:.2f}ms, {len(result)} results)")
    if result.diagnostics:
        print(f"  {json.dumps(result.diagnostics, default=str)}")
    print()
    for rank, sc in enumerate(result.chunks, start=1):
        print(f"{rank:>2}. [{sc.score:.4f}] {sc.chunk.doc_id}  ({sc.chunk.start}-{sc.chunk.end})")
        if sc.components:
            parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(sc.components.items()))
            print(f"    from: {parts}")
        text = sc.chunk.text.replace("\n", " ")
        print(f"    {text[:160]}{'…' if len(text) > 160 else ''}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Runs the full RAG pipeline on one question."""
    docs, _ = _load(args)
    retrievers = _retrievers(docs, args)
    pipeline = RAGPipeline(retrievers[args.retriever], ExtractiveGenerator(), k=args.k)

    if args.show_prompt:
        print(pipeline.build_prompt(args.query))
        print("\n" + "=" * 70 + "\n")

    answer = pipeline.answer(args.query)
    print(answer.text)
    print()
    print(f"-- {answer.model}, {answer.latency_ms:.1f}ms, {len(answer.citations)} citation(s)")
    for i, sc in enumerate(answer.citations, start=1):
        print(f"   [{i}] {sc.chunk.doc_id} chars {sc.chunk.start}-{sc.chunk.end}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluates one configuration and prints per-query failures."""
    docs, queries = _load(args)
    retrievers = _retrievers(docs, args)
    report = evaluate_retriever(retrievers[args.retriever], queries, k=args.k)

    print(report.summary())

    failures = report.failures(f"recall@{args.k}", threshold=0.0)
    if failures:
        print(f"\n{len(failures)} queries found nothing relevant:")
        for outcome in failures:
            print(f"  {outcome.query_id}: {outcome.query!r}")
            print(f"    wanted {outcome.relevant}, got {outcome.retrieved[:5]}")
    else:
        print(f"\nevery query retrieved a relevant document within k={args.k}")

    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Compares many configurations and reports the best."""
    docs, queries = _load(args)
    reports = []

    for size, overlap in [(200, 40), (400, 60), (800, 100)]:
        chunks = chunk_documents(docs, size=size, overlap=overlap)
        embedder = HashingEmbedder(dim=args.dim)
        dense = DenseRetriever(chunks, embedder)
        lexical = LexicalRetriever(chunks)

        reports.append(evaluate_retriever(dense, queries, k=args.k, name=f"dense(chunk={size})"))
        reports.append(evaluate_retriever(lexical, queries, k=args.k, name=f"bm25(chunk={size})"))
        for dw, lw in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]:
            hybrid = HybridRetriever(dense, lexical, dense_weight=dw, lexical_weight=lw)
            reports.append(
                evaluate_retriever(
                    hybrid, queries, k=args.k, name=f"hybrid(chunk={size},d={dw:g},l={lw:g})"
                )
            )

    print(compare(reports, primary=args.primary))
    if args.json:
        payload = [r.to_dict() for r in reports]
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def _common_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by every subcommand."""
    parser.add_argument("--corpus", default="data/corpus.jsonl", help="corpus JSONL path")
    parser.add_argument("--queries", default=None, help="queries JSONL path")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    parser.add_argument("--dim", type=int, default=512, help="embedding dimensionality")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("-k", type=int, default=10, help="number of results")


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser.

    Common options are attached to the top-level parser *and*, via a shared
    parent, to each subcommand. argparse otherwise accepts them only before the
    subcommand, so `lens search "q" -k 3` -- which is how anyone would
    reasonably type it -- would be rejected.
    """
    parser = argparse.ArgumentParser(prog="lens", description=__doc__.split("\n")[0])
    _common_options(parser)

    common = argparse.ArgumentParser(add_help=False)
    _common_options(common)
    # The subparser's defaults must not clobber a value given before the
    # subcommand, so suppress them and let the top-level parse supply them.
    for action in common._actions:
        action.default = argparse.SUPPRESS

    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)

    p = sub.add_parser("search", help="run one query", parents=[common])
    p.add_argument("query")
    p.add_argument("--retriever", choices=["dense", "lexical", "hybrid"], default="hybrid")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="run the RAG pipeline on one question", parents=[common])
    p.add_argument("query")
    p.add_argument("--retriever", choices=["dense", "lexical", "hybrid"], default="hybrid")
    p.add_argument("--show-prompt", action="store_true", help="print the prompt that is sent")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="evaluate one configuration", parents=[common])
    p.add_argument("--retriever", choices=["dense", "lexical", "hybrid"], default="hybrid")
    p.add_argument("--json", help="write the full report to this path")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sweep", help="compare many configurations", parents=[common])
    p.add_argument("--primary", default="ndcg@10", help="metric to rank configurations by")
    p.add_argument("--json", help="write all reports to this path")
    p.set_defaults(func=cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"lens: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

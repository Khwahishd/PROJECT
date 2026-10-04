"""The evaluation harness.

The design goal is that a retrieval change can be *judged* rather than argued
about. That requires three things, each of which this module enforces:

1. **Determinism.** With the local providers the whole run is reproducible, so
   a difference between two runs is a difference between two *configurations*,
   not noise.
2. **Per-query results, not just averages.** A mean recall of 0.8 is consistent
   with "every query is mediocre" and with "most are perfect and a few fail
   completely"; those call for opposite fixes. The harness keeps every query's
   outcome so the distribution is inspectable.
3. **Honest aggregation.** Queries with no relevant documents are excluded
   from recall-style metrics rather than scored as 0 or 1, both of which would
   be arbitrary. The count of excluded queries is reported.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..types import QueryExample, RetrievalResult
from . import metrics as M  # noqa: N812 -- a terse alias keeps the metric calls readable

__all__ = ["DEFAULT_CUTOFFS", "EvalReport", "QueryOutcome", "compare", "evaluate_retriever"]

#: Cutoffs reported by default. 1 and 3 matter because a RAG prompt usually
#: carries only a handful of chunks; 10 and 20 show whether the answer is
#: merely ranked badly rather than genuinely missing.
DEFAULT_CUTOFFS: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """Everything measured for one query."""

    query_id: str
    query: str
    retrieved: list[str]
    relevant: list[str]
    latency_ms: float
    scores: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def found_any(self) -> bool:
        """Whether any relevant document was retrieved at all."""
        rel = set(self.relevant)
        return any(d in rel for d in self.retrieved)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregated results for one configuration."""

    name: str
    #: Mean of each metric over the queries that could be scored.
    means: dict[str, float]
    #: Per-query outcomes, in input order.
    outcomes: list[QueryOutcome]
    #: Latency percentiles in milliseconds.
    latency: dict[str, float]
    #: Queries skipped because they had no labelled relevant document.
    skipped: int
    #: Total wall-clock time for the run, in seconds.
    duration_s: float

    @property
    def num_queries(self) -> int:
        """Number of scored queries."""
        return len(self.outcomes) - self.skipped

    def failures(self, metric: str = "recall@10", threshold: float = 0.0) -> list[QueryOutcome]:
        """Queries scoring at or below ``threshold``.

        The most useful output of the whole harness: an aggregate tells you
        *that* something is wrong, this tells you *which* queries to read.
        """
        return [o for o in self.outcomes if o.scores.get(metric, 0.0) <= threshold]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view, for storing runs or sending to a UI."""
        return {
            "name": self.name,
            "means": self.means,
            "latency": self.latency,
            "skipped": self.skipped,
            "num_queries": self.num_queries,
            "duration_s": self.duration_s,
            "outcomes": [asdict(o) for o in self.outcomes],
        }

    def summary(self) -> str:
        """A one-line-per-metric text summary."""
        lines = [f"{self.name}  ({self.num_queries} queries, {self.duration_s:.2f}s)"]
        for key in sorted(self.means):
            lines.append(f"  {key:<16} {self.means[key]:.4f}")
        lines.append(
            f"  latency          p50={self.latency['p50']:.2f}ms p95={self.latency['p95']:.2f}ms"
        )
        if self.skipped:
            lines.append(f"  (skipped {self.skipped} queries with no labelled relevant documents)")
        return "\n".join(lines)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99 plus mean and max.

    Averages hide tail latency, and the tail is what users experience. A p50 of
    5ms with a p99 of 400ms is a different system from a uniform 12ms, and
    reporting only the mean makes them look identical.
    """
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        # Nearest-rank, which never interpolates between observed values --
        # a reported percentile is always a latency that actually happened.
        idx = min(len(ordered) - 1, max(0, math_ceil(p / 100.0 * len(ordered)) - 1))
        return ordered[idx]

    return {
        "mean": statistics.fmean(ordered),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "max": ordered[-1],
    }


def math_ceil(x: float) -> int:
    """Ceiling, kept local so the module has no import-time surprises."""
    import math

    return math.ceil(x)


def evaluate_retriever(
    retriever: Any,
    examples: Sequence[QueryExample],
    *,
    k: int = 10,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    name: str | None = None,
    granularity: str = "doc",
) -> EvalReport:
    """Runs ``retriever`` over ``examples`` and scores the results.

    ``granularity`` selects whether retrieved chunks are scored as documents
    (the default, matching how datasets are usually labelled) or as chunks.
    """
    if granularity not in ("doc", "chunk"):
        raise ValueError(f"granularity must be 'doc' or 'chunk', got {granularity!r}")

    label = name or getattr(retriever, "name", retriever.__class__.__name__)
    # Fetch enough for the largest cutoff; scoring recall@20 against 10
    # results would silently cap the metric at half its range.
    fetch = max(k, max(cutoffs))

    outcomes: list[QueryOutcome] = []
    latencies: list[float] = []
    skipped = 0
    started = time.perf_counter()

    for example in examples:
        result: RetrievalResult = retriever.retrieve(example.query, k=fetch)
        latencies.append(result.latency_ms)

        ranked = result.doc_ids if granularity == "doc" else result.chunk_ids
        relevant = example.all_relevant

        if not relevant:
            skipped += 1
            outcomes.append(
                QueryOutcome(
                    query_id=example.query_id,
                    query=example.query,
                    retrieved=list(ranked),
                    relevant=[],
                    latency_ms=result.latency_ms,
                    scores={},
                    diagnostics=dict(result.diagnostics),
                )
            )
            continue

        scores: dict[str, float] = {}
        ideal = sorted((example.gain(d) for d in relevant), reverse=True)
        for cut in cutoffs:
            scores[f"recall@{cut}"] = M.recall_at_k(ranked, relevant, cut)
            scores[f"precision@{cut}"] = M.precision_at_k(ranked, relevant, cut)
            scores[f"hit@{cut}"] = M.hit_rate_at_k(ranked, relevant, cut)
            scores[f"ndcg@{cut}"] = M.ndcg_at_k(ranked, example.gain, cut, ideal_gains=ideal)
        scores["mrr"] = M.reciprocal_rank(ranked, relevant)
        scores["map"] = M.average_precision(ranked, relevant)

        outcomes.append(
            QueryOutcome(
                query_id=example.query_id,
                query=example.query,
                retrieved=list(ranked),
                relevant=sorted(relevant),
                latency_ms=result.latency_ms,
                scores=scores,
                diagnostics=dict(result.diagnostics),
            )
        )

    duration = time.perf_counter() - started

    # Average only over the queries that were actually scored.
    scored = [o for o in outcomes if o.scores]
    means: dict[str, float] = {}
    if scored:
        for key in scored[0].scores:
            means[key] = statistics.fmean(o.scores[key] for o in scored)

    return EvalReport(
        name=label,
        means=means,
        outcomes=outcomes,
        latency=_percentiles(latencies),
        skipped=skipped,
        duration_s=duration,
    )


def compare(
    reports: Sequence[EvalReport],
    *,
    primary: str = "ndcg@10",
) -> str:
    """Renders several reports as a comparison table, best-first on ``primary``."""
    if not reports:
        return "(no reports)"

    keys = [k for k in ("recall@5", "recall@10", "mrr", "ndcg@10", "map") if k in reports[0].means]
    ordered = sorted(reports, key=lambda r: -r.means.get(primary, 0.0))
    best = ordered[0]

    name_w = max(len(r.name) for r in reports) + 2
    header = f"{'configuration':<{name_w}}" + "".join(f"{k:>11}" for k in keys) + f"{'p95 ms':>10}"
    lines = [header, "-" * len(header)]

    for report in ordered:
        row = f"{report.name:<{name_w}}"
        for key in keys:
            row += f"{report.means.get(key, 0.0):>11.4f}"
        row += f"{report.latency['p95']:>10.2f}"
        if report is best and len(ordered) > 1:
            row += "  <- best"
        lines.append(row)

    # A ranking without the gap is hard to act on: a 0.3% difference over 20
    # queries is noise, a 30% difference is a decision.
    if len(ordered) > 1:
        gap = best.means.get(primary, 0.0) - ordered[-1].means.get(primary, 0.0)
        lines.append("")
        lines.append(
            f"spread on {primary}: {gap:.4f} "
            f"({best.name} over {ordered[-1].name}, {best.num_queries} queries)"
        )
    return "\n".join(lines)

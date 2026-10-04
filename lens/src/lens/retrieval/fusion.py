"""Combining rankings from multiple retrievers.

Dense and lexical retrieval fail in complementary ways. Embeddings capture
paraphrase ("car" ≈ "automobile") but blur exact tokens; BM25 nails exact terms
but misses any rewording. Hybrid retrieval is worth doing because the union of
their results is reliably better than either alone -- but only if they are
combined correctly.

Two fusion strategies are provided, and the difference between them matters:

* :func:`reciprocal_rank_fusion` combines by **rank**. It needs no calibration
  and no tuning, which is its whole appeal: BM25 scores are unbounded sums of
  IDF terms while cosine similarities live in [-1, 1], so the two are simply
  not comparable as numbers.
* :func:`weighted_fusion` combines by **normalized score**, which preserves the
  *margin* between a strong and a weak match -- information RRF discards -- at
  the cost of needing the normalization to be meaningful.

RRF is the default. It is harder to get wrong, and the usual failure of
weighted fusion is that one retriever's score distribution shifts (a different
corpus, a different embedding model) and silently swamps the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["normalize_scores", "reciprocal_rank_fusion", "weighted_fusion"]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    k: float = 60.0,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuses ranked lists by reciprocal rank (Cormack et al., 2009).

    Each list contributes ``weight / (k + rank)`` for each item, with ``rank``
    starting at 1.

    ``k`` damps the influence of the very top ranks. With ``k=0`` the first
    result would contribute 1.0 and the second 0.5 -- a single retriever's
    top hit would dominate everything. The conventional ``k=60`` flattens that
    curve so agreement *across* retrievers outweighs confidence *within* one,
    which is the entire point of fusing.

    Returns ``(id, fused_score, per_retriever_contributions)``, best first.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}

    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight == 0.0:
            continue
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            contribution = weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + contribution
            components.setdefault(doc_id, {})[name] = contribution

    # Sort by score, breaking ties on id so the output is deterministic --
    # otherwise two runs over the same data can disagree, which makes an
    # evaluation irreproducible for no reason.
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [(doc_id, score, components.get(doc_id, {})) for doc_id, score in ordered]


def normalize_scores(scores: Sequence[float], method: str = "minmax") -> list[float]:
    """Maps scores onto a comparable range.

    ``minmax`` rescales to [0, 1]; ``zscore`` centres and scales by standard
    deviation. A degenerate input (every score identical) maps to all ones
    rather than dividing by zero -- if a retriever cannot distinguish its own
    results, every one of them is equally good as far as it knows.
    """
    if not scores:
        return []
    values = list(scores)

    if method == "minmax":
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    if method == "zscore":
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        sd = var**0.5
        if sd < 1e-12:
            return [0.0] * len(values)
        return [(v - mean) / sd for v in values]

    raise ValueError(f"unknown normalization {method!r}; use 'minmax' or 'zscore'")


def weighted_fusion(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    weights: Mapping[str, float] | None = None,
    normalization: str = "minmax",
    top_k: int | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuses ranked lists by weighted, normalized score.

    An item missing from a retriever's list contributes nothing from that
    retriever, rather than contributing its minimum score. Treating "not
    retrieved" as "scored lowest" would be wrong: the retriever did not rank it
    last, it never considered it at all, and conflating the two systematically
    penalises items that only one retriever surfaced -- which are precisely the
    items hybrid retrieval exists to rescue.
    """
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}

    for name, ranking in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        if weight == 0.0 or not ranking:
            continue
        normalized = normalize_scores([s for _, s in ranking], normalization)
        for (doc_id, _), value in zip(ranking, normalized, strict=True):
            contribution = weight * value
            fused[doc_id] = fused.get(doc_id, 0.0) + contribution
            components.setdefault(doc_id, {})[name] = contribution

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[:top_k]
    return [(doc_id, score, components.get(doc_id, {})) for doc_id, score in ordered]

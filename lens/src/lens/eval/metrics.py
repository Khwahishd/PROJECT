"""Retrieval and answer quality metrics.

Each metric answers a different question, and using the wrong one produces
confident, meaningless numbers:

* **Recall@k** -- *did we find the answer at all?* The ceiling on everything
  downstream: a generator cannot cite a passage that was never retrieved.
* **Precision@k** -- *how much of what we retrieved was useful?* Matters
  because every irrelevant chunk consumes context window and invites the model
  to ground its answer in the wrong place.
* **MRR** -- *how far down was the first good result?* Rewards getting one
  right answer to the top. The metric to watch when exactly one passage can
  answer the query.
* **nDCG@k** -- *how good is the whole ordering?* The only one here that uses
  graded relevance and discounts by position, so it distinguishes "relevant
  result at rank 1" from "relevant result at rank 9".
* **Hit rate@k** -- the blunt instrument: did *anything* relevant appear?

All of them take ranked ids and a relevance judgement, so they work equally
over chunk ids or document ids.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

__all__ = [
    "answer_contains_citation",
    "average_precision",
    "exact_match",
    "f1",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "token_f1",
]


def _truncate(ranked: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return ranked[:k]


def recall_at_k(ranked: Sequence[str], relevant: set[str] | frozenset[str], k: int) -> float:
    """Fraction of relevant items appearing in the top ``k``.

    Returns 0.0 when nothing is relevant. The alternative -- returning 1.0 on
    the grounds that all zero relevant items were found -- would quietly
    inflate a corpus-level average with queries that have no answer, so such
    queries are excluded by the aggregator instead.
    """
    if not relevant:
        return 0.0
    found = sum(1 for item in _truncate(ranked, k) if item in relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str] | frozenset[str], k: int) -> float:
    """Fraction of the top ``k`` that is relevant.

    The denominator is ``k``, not the number of results returned. A retriever
    that returns 3 results when asked for 10 is penalised for the 7 it did not
    supply, which is the honest accounting -- those slots were available and
    went unused.
    """
    top = _truncate(ranked, k)
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str] | frozenset[str], k: int) -> float:
    """1.0 if any relevant item is in the top ``k``, else 0.0."""
    return 1.0 if any(item in relevant for item in _truncate(ranked, k)) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str] | frozenset[str]) -> float:
    """``1 / rank`` of the first relevant item, or 0.0 if there is none."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence[str], relevant: set[str] | frozenset[str]) -> float:
    """Mean of the precisions measured at each relevant item's rank.

    Unlike precision@k this needs no cutoff, and unlike MRR it accounts for
    *every* relevant item rather than just the first -- the right choice when a
    query has several valid answers scattered through the ranking.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(
    ranked: Sequence[str],
    gain: Callable[[str], float],
    k: int,
    *,
    ideal_gains: Sequence[float] | None = None,
) -> float:
    """Normalized discounted cumulative gain at ``k``.

    Gains are discounted by ``1 / log2(rank + 1)``, so a relevant result at
    rank 1 is worth roughly three times one at rank 8. Normalizing against the
    ideal ordering is what makes scores comparable across queries that have
    different numbers of relevant documents.

    ``ideal_gains`` supplies the full set of achievable gains. Without it the
    ideal is computed from the retrieved items alone, which would score a
    retriever 1.0 for perfectly ordering the two relevant documents it happened
    to find while missing eight others.
    """
    top = _truncate(ranked, k)
    dcg = sum(gain(item) / math.log2(rank + 1) for rank, item in enumerate(top, start=1))

    if ideal_gains is None:
        ideal = sorted((gain(item) for item in ranked), reverse=True)[:k]
    else:
        ideal = sorted(ideal_gains, reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, start=1))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _normalize_answer(text: str) -> list[str]:
    """Lowercases, strips punctuation and articles, and splits into tokens.

    The standard SQuAD normalization. Without it, "The Eiffel Tower." and
    "eiffel tower" score as a complete mismatch, which says nothing about the
    answer and everything about the punctuation.
    """
    import re

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between a predicted and a reference answer.

    Deliberately shallow: it measures lexical overlap, not meaning, and will
    score a correct paraphrase poorly. It is included because it is cheap,
    deterministic and the standard baseline -- not because it is sufficient.
    Semantic scoring needs a model, which reintroduces exactly the
    non-determinism this harness is built to avoid.
    """
    pred_tokens = _normalize_answer(prediction)
    ref_tokens = _normalize_answer(reference)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    from collections import Counter

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return f1(precision, recall)


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalized answers are identical."""
    return float(_normalize_answer(prediction) == _normalize_answer(reference))


def answer_contains_citation(answer: str, contexts: Sequence[str], *, min_span: int = 24) -> float:
    """Fraction of an answer's sentences traceable to the retrieved context.

    A cheap, model-free groundedness proxy: it checks whether a long enough
    span of each sentence appears verbatim in some retrieved chunk. It cannot
    detect a *paraphrased* hallucination, which is its main limitation, but it
    reliably catches an answer invented wholesale -- the failure mode that
    matters most in RAG.
    """
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    joined = " ".join(contexts).lower()

    grounded = 0
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence.lower())
        if len(normalized) < min_span:
            # Too short to check meaningfully; require the whole thing.
            if normalized in joined:
                grounded += 1
            continue
        # Slide a window so a sentence that merely extends a quoted span
        # still counts as grounded.
        if any(
            normalized[i : i + min_span] in joined
            for i in range(0, len(normalized) - min_span + 1, max(1, min_span // 2))
        ):
            grounded += 1
    return grounded / len(sentences)

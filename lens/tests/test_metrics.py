"""Evaluation metrics: values verified by hand."""

from __future__ import annotations

import pytest

from lens.eval.metrics import (
    answer_contains_citation,
    average_precision,
    exact_match,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)

RANKED = ["a", "x", "b", "y", "c"]
RELEVANT = {"a", "b", "z"}


class TestRecallAndPrecision:
    @pytest.mark.parametrize(("k", "expected"), [(1, 1 / 3), (3, 2 / 3), (5, 2 / 3)])
    def test_recall(self, k, expected):
        assert recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)

    @pytest.mark.parametrize(("k", "expected"), [(1, 1.0), (3, 2 / 3), (5, 0.4)])
    def test_precision(self, k, expected):
        assert precision_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)

    def test_precision_divides_by_k_not_by_results_returned(self):
        """A retriever asked for 10 and returning 2 is penalised for the 8 it
        left on the table, which is the honest accounting."""
        assert precision_at_k(["a"], {"a"}, 10) == pytest.approx(0.1)

    def test_no_relevant_documents_scores_zero(self):
        assert recall_at_k(RANKED, set(), 5) == 0.0

    def test_hit_rate_is_binary(self):
        assert hit_rate_at_k(RANKED, RELEVANT, 1) == 1.0
        assert hit_rate_at_k(["x", "y"], RELEVANT, 2) == 0.0

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_is_rejected(self, k):
        with pytest.raises(ValueError):
            recall_at_k(RANKED, RELEVANT, k)


class TestRankMetrics:
    def test_reciprocal_rank_uses_the_first_hit(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert reciprocal_rank(["x", "y", "z", "a"], {"a"}) == 0.25
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_average_precision_accounts_for_every_relevant_item(self):
        # Hits at ranks 1 and 3: (1/1 + 2/3) / 3 relevant items.
        assert average_precision(RANKED, RELEVANT) == pytest.approx((1.0 + 2 / 3) / 3)

    def test_average_precision_is_1_for_a_perfect_ranking(self):
        assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)


class TestNDCG:
    def test_a_perfect_ordering_scores_one(self):
        gains = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(
            ["a", "b", "c"], gains.get, 3, ideal_gains=[3.0, 2.0, 1.0]
        ) == pytest.approx(1.0)  # type: ignore[arg-type]

    def test_position_matters(self):
        gain = {"a": 1.0}
        early = ndcg_at_k(["a", "x", "y"], lambda d: gain.get(d, 0.0), 3, ideal_gains=[1.0])
        late = ndcg_at_k(["x", "y", "a"], lambda d: gain.get(d, 0.0), 3, ideal_gains=[1.0])
        assert early > late, "a relevant result at rank 1 must beat the same at rank 3"

    def test_ideal_gains_prevent_a_misleading_perfect_score(self):
        """Without the full ideal, a retriever that perfectly orders the two
        documents it found -- while missing eight -- would score 1.0."""
        gain = {"a": 1.0, "b": 1.0}
        found_two_of_ten = ndcg_at_k(
            ["a", "b"], lambda d: gain.get(d, 0.0), 10, ideal_gains=[1.0] * 10
        )
        assert found_two_of_ten < 0.5

        naive = ndcg_at_k(["a", "b"], lambda d: gain.get(d, 0.0), 10)
        assert naive == pytest.approx(1.0), "this is the misleading value being guarded against"

    def test_nothing_relevant_scores_zero(self):
        assert ndcg_at_k(["x", "y"], lambda _: 0.0, 2) == 0.0


class TestAnswerMetrics:
    def test_token_f1_ignores_articles_and_punctuation(self):
        assert token_f1("The Eiffel Tower.", "eiffel tower") == pytest.approx(1.0)

    def test_token_f1_rewards_partial_overlap(self):
        score = token_f1("the cat sat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_token_f1_is_zero_with_no_overlap(self):
        assert token_f1("completely different", "nothing alike") == 0.0

    def test_exact_match_normalizes_before_comparing(self):
        assert exact_match("The Answer!", "answer") == 1.0
        assert exact_match("a different answer", "answer") == 0.0

    def test_groundedness_detects_a_verbatim_quote(self):
        context = ["The cat sat on the mat and then slept for a while."]
        assert answer_contains_citation("The cat sat on the mat and then slept.", context) == 1.0

    def test_groundedness_detects_wholesale_invention(self):
        context = ["The cat sat on the mat and then slept for a while."]
        invented = "The dog flew to Mars on a rocket powered by cheese sandwiches."
        assert answer_contains_citation(invented, context) == 0.0

    def test_groundedness_is_fractional_for_a_mixed_answer(self):
        context = ["The cat sat on the mat and then slept for a while."]
        mixed = "The cat sat on the mat and then slept. The dog flew to Mars on a cheese rocket."
        score = answer_contains_citation(mixed, context)
        assert 0.0 < score < 1.0

    def test_an_empty_answer_is_not_grounded(self):
        assert answer_contains_citation("", ["some context"]) == 0.0

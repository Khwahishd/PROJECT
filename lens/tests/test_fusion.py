"""Rank fusion: the properties that make hybrid retrieval work."""

from __future__ import annotations

import pytest

from lens.retrieval.fusion import normalize_scores, reciprocal_rank_fusion, weighted_fusion


class TestReciprocalRankFusion:
    def test_agreement_across_retrievers_beats_confidence_within_one(self):
        """The defining property of RRF, and the reason to use it."""
        rankings = {
            # "b" is second for both; "a" is first for one and absent from the other.
            "x": [("a", 99.0), ("b", 1.0)],
            "y": [("c", 99.0), ("b", 1.0)],
        }
        fused = reciprocal_rank_fusion(rankings)
        assert fused[0][0] == "b", f"the consensus result should win: {fused}"

    def test_score_scales_are_irrelevant(self):
        """BM25 scores are unbounded; cosine lives in [-1, 1]. RRF ignores both."""
        tiny = {"a": [("x", 0.001), ("y", 0.0005)], "b": [("y", 0.002), ("x", 0.0001)]}
        huge = {"a": [("x", 1000.0), ("y", 500.0)], "b": [("y", 2000.0), ("x", 100.0)]}
        assert [i for i, _, _ in reciprocal_rank_fusion(tiny)] == [
            i for i, _, _ in reciprocal_rank_fusion(huge)
        ]

    def test_k_damps_the_influence_of_top_ranks(self):
        rankings = {"a": [("x", 1.0), ("y", 0.9)], "b": [("y", 1.0), ("z", 0.9)]}
        sharp = dict((i, s) for i, s, _ in reciprocal_rank_fusion(rankings, k=0))
        flat = dict((i, s) for i, s, _ in reciprocal_rank_fusion(rankings, k=1000))
        # With k=0 the rank-1 item dominates; with large k everything flattens.
        assert sharp["x"] / sharp["y"] > flat["x"] / flat["y"]

    def test_weights_shift_the_balance(self):
        rankings = {"dense": [("d", 1.0)], "lexical": [("l", 1.0)]}
        dense_heavy = reciprocal_rank_fusion(rankings, weights={"dense": 10.0, "lexical": 1.0})
        assert dense_heavy[0][0] == "d"
        lex_heavy = reciprocal_rank_fusion(rankings, weights={"dense": 1.0, "lexical": 10.0})
        assert lex_heavy[0][0] == "l"

    def test_a_zero_weight_removes_a_retriever(self):
        rankings = {"dense": [("d", 1.0)], "lexical": [("l", 1.0)]}
        fused = reciprocal_rank_fusion(rankings, weights={"dense": 0.0, "lexical": 1.0})
        assert [i for i, _, _ in fused] == ["l"]

    def test_components_attribute_the_score(self):
        rankings = {"dense": [("x", 1.0)], "lexical": [("x", 5.0)]}
        _, score, components = reciprocal_rank_fusion(rankings)[0]
        assert set(components) == {"dense", "lexical"}
        assert sum(components.values()) == pytest.approx(score)

    def test_ties_break_deterministically(self):
        """Two runs disagreeing would make any evaluation irreproducible."""
        rankings = {"a": [("z", 1.0), ("y", 1.0), ("x", 1.0)]}
        first = [i for i, _, _ in reciprocal_rank_fusion(rankings)]
        second = [i for i, _, _ in reciprocal_rank_fusion(rankings)]
        assert first == second

    def test_empty_and_truncated_inputs(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"a": []}) == []
        rankings = {"a": [(f"x{i}", 1.0) for i in range(20)]}
        assert len(reciprocal_rank_fusion(rankings, top_k=5)) == 5

    def test_negative_k_is_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion({"a": [("x", 1.0)]}, k=-1)


class TestWeightedFusion:
    def test_preserves_the_margin_between_strong_and_weak_matches(self):
        """The information RRF throws away."""
        rankings = {"a": [("x", 100.0), ("y", 1.0)]}
        fused = dict((i, s) for i, s, _ in weighted_fusion(rankings))
        assert fused["x"] > fused["y"] * 2

    def test_a_missing_item_contributes_nothing_rather_than_the_minimum(self):
        """Treating "not retrieved" as "scored lowest" would penalise exactly
        the single-retriever finds that hybrid search exists to rescue."""
        rankings = {"a": [("x", 1.0), ("y", 0.5)], "b": [("z", 1.0)]}
        components = {i: c for i, _, c in weighted_fusion(rankings)}
        assert "b" not in components["x"], "x was never ranked by b"
        assert set(components["z"]) == {"b"}

    def test_identical_scores_normalize_without_dividing_by_zero(self):
        rankings = {"a": [("x", 5.0), ("y", 5.0), ("z", 5.0)]}
        fused = weighted_fusion(rankings)
        assert len(fused) == 3
        assert all(s == pytest.approx(1.0) for _, s, _ in fused)


class TestNormalization:
    def test_minmax_maps_onto_the_unit_interval(self):
        assert normalize_scores([1.0, 5.0, 3.0]) == [0.0, 1.0, 0.5]

    def test_zscore_centres_and_scales(self):
        out = normalize_scores([1.0, 2.0, 3.0], "zscore")
        assert out[1] == pytest.approx(0.0)
        assert out[0] == pytest.approx(-out[2])

    def test_degenerate_input_does_not_divide_by_zero(self):
        assert normalize_scores([7.0, 7.0, 7.0]) == [1.0, 1.0, 1.0]
        assert normalize_scores([7.0, 7.0], "zscore") == [0.0, 0.0]

    def test_empty_input(self):
        assert normalize_scores([]) == []

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown normalization"):
            normalize_scores([1.0], "magic")

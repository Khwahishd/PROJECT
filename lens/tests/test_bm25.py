"""BM25: the scoring function's defining behaviours."""

from __future__ import annotations

import pytest

from lens.index.bm25 import BM25Index, tokenize


@pytest.fixture
def index() -> BM25Index:
    docs = {
        "d1": "the cat sat on the mat",
        "d2": "the dog sat on the log",
        "d3": "cats and dogs are common pets in many households",
        "d4": "quantum chromodynamics describes the strong interaction",
        "d5": "the cat the cat the cat the cat the cat",
    }
    return BM25Index().fit(list(docs), list(docs.values()))


class TestTokenization:
    def test_lowercases_and_splits(self):
        assert tokenize("The Quick BROWN fox") == ["the", "quick", "brown", "fox"]

    def test_keeps_intra_word_punctuation(self):
        """Splitting these would leave fragments that match nothing."""
        assert tokenize("state-of-the-art") == ["state-of-the-art"]
        assert tokenize("snake_case_name") == ["snake_case_name"]
        assert tokenize("don't") == ["don't"]

    def test_drops_surrounding_punctuation(self):
        assert tokenize("Hello, world! (really)") == ["hello", "world", "really"]

    def test_keeps_alphanumeric_identifiers(self):
        """Product codes and error numbers are exactly what BM25 is for."""
        assert "ann-4417" in tokenize("error ANN-4417 was raised")
        assert "gpt4" in tokenize("the GPT4 model")

    def test_stopword_removal_is_opt_in(self):
        assert "the" in tokenize("the cat")
        assert "the" not in tokenize("the cat", remove_stopwords=True)


class TestScoring:
    def test_ranks_the_matching_document_first(self, index):
        ranked = index.score("cat mat")
        assert ranked[0][0] == "d1"

    def test_a_query_with_no_matches_returns_nothing(self, index):
        assert index.score("zebra giraffe hippopotamus") == []

    def test_an_empty_query_returns_nothing(self, index):
        assert index.score("") == []
        assert index.score("   ") == []

    def test_scores_are_sorted_descending(self, index):
        scores = [s for _, s in index.score("cat dog sat")]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self, index):
        assert len(index.score("the", top_k=2)) <= 2

    def test_term_frequency_saturates(self, index):
        """d5 repeats 'cat' five times; it must not score five times d1."""
        scores = dict(index.score("cat"))
        assert scores["d5"] < 5 * scores["d1"], "term frequency is not saturating"

    def test_idf_never_goes_negative(self):
        """The classic IDF formula penalises a document for matching.

        Without the ``+1`` inside the logarithm, a term in more than half the
        corpus scores negatively, so containing a query term would push a
        document *down*. This checks the whole df range, including the region
        where the unsmoothed form is negative.
        """
        import math

        n = 5
        for df in range(1, n + 1):
            texts = ["term filler" if i < df else "other words" for i in range(n)]
            idx = BM25Index().fit([str(i) for i in range(n)], texts)
            assert idx.idf("term") >= 0.0, f"IDF went negative at df={df}"
            if df > n / 2:
                classic = math.log((n - df + 0.5) / (df + 0.5))
                assert classic < 0, "the test's own premise about the classic form is wrong"

    def test_idf_decreases_as_a_term_becomes_more_common(self):
        n = 5
        previous = float("inf")
        for df in range(1, n + 1):
            texts = ["term filler" if i < df else "other words" for i in range(n)]
            idx = BM25Index().fit([str(i) for i in range(n)], texts)
            current = idx.idf("term")
            assert current < previous, f"IDF did not decrease at df={df}"
            previous = current

    def test_a_rare_term_outweighs_a_common_one(self, index):
        assert index.idf("quantum") > index.idf("cat")

    def test_length_normalization_responds_to_b(self):
        short, long = "cat", "cat " + " ".join(f"filler{i}" for i in range(50))
        ids, texts = ["s", "l"], [short, long]

        no_norm = dict(BM25Index(b=0.0).fit(ids, texts).score("cat"))
        full_norm = dict(BM25Index(b=1.0).fit(ids, texts).score("cat"))

        assert no_norm["s"] == pytest.approx(no_norm["l"]), "b=0 should ignore length"
        assert full_norm["s"] > full_norm["l"], "b=1 should favour the shorter document"

    def test_repeating_a_query_term_weights_it(self, index):
        once = dict(index.score("cat"))
        twice = dict(index.score("cat cat"))
        assert twice["d1"] == pytest.approx(2 * once["d1"])


class TestExplain:
    def test_breaks_a_score_into_per_term_contributions(self, index):
        parts = index.explain("cat mat", "d1")
        assert set(parts) == {"cat", "mat"}
        assert all(v > 0 for v in parts.values())
        total = sum(parts.values())
        assert total == pytest.approx(dict(index.score("cat mat"))["d1"])

    def test_a_term_absent_from_the_document_contributes_zero(self, index):
        assert index.explain("cat zebra", "d1")["zebra"] == 0.0

    def test_an_unknown_document_raises(self, index):
        with pytest.raises(KeyError, match="no document"):
            index.explain("cat", "nope")


class TestValidation:
    def test_mismatched_ids_and_texts(self):
        with pytest.raises(ValueError, match="ids but"):
            BM25Index().fit(["a", "b"], ["only one"])

    def test_duplicate_ids(self):
        with pytest.raises(ValueError, match="unique"):
            BM25Index().fit(["a", "a"], ["x", "y"])

    @pytest.mark.parametrize(("k1", "b"), [(-1.0, 0.75), (1.5, -0.1), (1.5, 1.1)])
    def test_out_of_range_parameters(self, k1, b):
        with pytest.raises(ValueError):
            BM25Index(k1=k1, b=b)

    def test_an_empty_index_scores_nothing(self):
        assert BM25Index().fit([], []).score("anything") == []

    def test_statistics_are_exposed(self, index):
        assert len(index) == 5
        assert index.vocabulary_size > 0
        assert index.average_document_length > 0
        # "cats" in d3 is a distinct token: there is no stemming, by design.
        assert index.document_frequency("cat") == 2
        assert index.document_frequency("cats") == 1

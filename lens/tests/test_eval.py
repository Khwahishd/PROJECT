"""The evaluation harness, and the claims it is built to check."""

from __future__ import annotations

import json

import pytest

from lens.eval import compare, evaluate_retriever
from lens.eval.datasets import load_jsonl, save_jsonl
from lens.retrieval import HybridRetriever
from lens.types import QueryExample


class TestHarness:
    def test_reports_aggregate_and_per_query_results(self, hybrid, queries):
        report = evaluate_retriever(hybrid, queries, k=5)
        assert report.num_queries == len(queries)
        assert len(report.outcomes) == len(queries)
        for metric in ("recall@5", "precision@5", "mrr", "ndcg@5", "map"):
            assert 0.0 <= report.means[metric] <= 1.0

    def test_keeps_every_query_so_the_distribution_is_inspectable(self, hybrid, queries):
        """A mean of 0.8 is consistent with 'all mediocre' and with 'mostly
        perfect, a few catastrophic'. Those need opposite fixes."""
        report = evaluate_retriever(hybrid, queries, k=5)
        for outcome in report.outcomes:
            assert outcome.query_id
            assert outcome.retrieved is not None
            assert "recall@5" in outcome.scores

    def test_failures_surfaces_the_queries_to_read(self, dense, queries):
        report = evaluate_retriever(dense, queries, k=5)
        failures = report.failures("recall@5", threshold=0.0)
        assert all(f.scores["recall@5"] == 0.0 for f in failures)

    def test_unlabelled_queries_are_skipped_not_scored(self, hybrid, queries):
        """Scoring them 0 would depress the mean; scoring them 1 would inflate
        it. Both are arbitrary, so they are excluded and counted."""
        padded = [*queries, QueryExample("unlabelled", "no labels here", frozenset())]
        report = evaluate_retriever(hybrid, padded, k=5)
        assert report.skipped == 1
        assert report.num_queries == len(queries)

        labelled_only = evaluate_retriever(hybrid, queries, k=5)
        assert report.means["recall@5"] == pytest.approx(labelled_only.means["recall@5"])

    def test_fetches_enough_for_the_largest_cutoff(self, hybrid, queries):
        """Scoring recall@20 against 10 retrieved results silently caps it."""
        report = evaluate_retriever(hybrid, queries, k=3, cutoffs=(1, 3, 10))
        assert report.means["recall@10"] >= report.means["recall@3"]

    def test_latency_percentiles_are_reported(self, hybrid, queries):
        report = evaluate_retriever(hybrid, queries, k=5)
        latency = report.latency
        assert latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["max"]

    def test_chunk_granularity_is_available(self, hybrid, queries):
        doc_level = evaluate_retriever(hybrid, queries, k=5, granularity="doc")
        chunk_level = evaluate_retriever(hybrid, queries, k=5, granularity="chunk")
        assert doc_level.means["recall@5"] >= 0.0
        assert chunk_level.means["recall@5"] >= 0.0

    def test_invalid_granularity_is_rejected(self, hybrid, queries):
        with pytest.raises(ValueError, match="granularity"):
            evaluate_retriever(hybrid, queries, granularity="paragraph")

    def test_report_serializes_to_json(self, hybrid, queries):
        report = evaluate_retriever(hybrid, queries, k=5)
        payload = json.dumps(report.to_dict())
        assert json.loads(payload)["name"] == report.name

    def test_summary_is_human_readable(self, hybrid, queries):
        summary = evaluate_retriever(hybrid, queries, k=5).summary()
        assert "recall@5" in summary and "latency" in summary


class TestClaims:
    """The claims the project makes, checked rather than asserted in prose."""

    def test_complementary_failure(self, dense, lexical, queries):
        """Dense and lexical retrieval must fail on *different* queries.

        This is the entire premise of hybrid retrieval. If one retriever
        dominated on every query, fusing them would be pointless -- and the
        corpus would not be testing anything.
        """
        dense_report = evaluate_retriever(dense, queries, k=3)
        lexical_report = evaluate_retriever(lexical, queries, k=3)

        by_query_dense = {o.query_id: o.scores.get("mrr", 0.0) for o in dense_report.outcomes}
        by_query_lexical = {o.query_id: o.scores.get("mrr", 0.0) for o in lexical_report.outcomes}

        dense_wins = [q for q in by_query_dense if by_query_dense[q] > by_query_lexical[q]]
        lexical_wins = [q for q in by_query_dense if by_query_lexical[q] > by_query_dense[q]]

        assert dense_wins, f"no query where dense beats lexical: {by_query_dense}"
        assert lexical_wins, f"no query where lexical beats dense: {by_query_lexical}"

    def test_hybrid_is_at_least_as_good_as_its_worse_component(self, dense, lexical, queries):
        """A fusion that scores below both inputs is broken, not a tradeoff."""
        hybrid = HybridRetriever(dense, lexical)
        scores = {
            r.name: r.means["ndcg@5"]
            for r in (
                evaluate_retriever(dense, queries, k=5),
                evaluate_retriever(lexical, queries, k=5),
                evaluate_retriever(hybrid, queries, k=5),
            )
        }
        hybrid_score = scores[hybrid.name]
        assert hybrid_score >= min(scores.values()), scores

    def test_evaluation_is_reproducible(self, hybrid, queries):
        """Without this, a difference between runs could be noise rather than
        a difference between configurations -- and the harness is useless."""
        first = evaluate_retriever(hybrid, queries, k=5)
        second = evaluate_retriever(hybrid, queries, k=5)
        assert first.means == second.means
        assert [o.retrieved for o in first.outcomes] == [o.retrieved for o in second.outcomes]


class TestCompare:
    def test_ranks_configurations_and_marks_the_best(self, dense, lexical, hybrid, queries):
        reports = [evaluate_retriever(r, queries, k=5) for r in (dense, lexical, hybrid)]
        table = compare(reports, primary="ndcg@5")
        assert "<- best" in table
        assert "spread on ndcg@5" in table, "a ranking without the gap is hard to act on"

    def test_handles_an_empty_list(self):
        assert compare([]) == "(no reports)"


class TestDatasets:
    def test_round_trips_through_jsonl(self, tmp_path, docs, queries):
        save_jsonl(docs, queries, tmp_path)
        loaded_docs, loaded_queries = load_jsonl(
            tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl"
        )
        assert [d.doc_id for d in loaded_docs] == [d.doc_id for d in docs]
        assert [q.query_id for q in loaded_queries] == [q.query_id for q in queries]
        assert loaded_queries[0].relevant_doc_ids == queries[0].relevant_doc_ids

    def test_a_label_pointing_outside_the_corpus_is_rejected(self, tmp_path):
        """Such a label caps recall below 1.0 for reasons that have nothing to
        do with the retriever. Failing loudly beats puzzling over the number."""
        (tmp_path / "corpus.jsonl").write_text('{"doc_id": "a", "text": "x"}\n')
        (tmp_path / "queries.jsonl").write_text('{"query": "q", "relevant_doc_ids": ["ghost"]}\n')
        with pytest.raises(ValueError, match="absent from the corpus"):
            load_jsonl(tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl")

    def test_malformed_json_names_the_line(self, tmp_path):
        (tmp_path / "corpus.jsonl").write_text('{"doc_id": "a", "text": "x"}\nnot json\n')
        (tmp_path / "queries.jsonl").write_text('{"query": "q"}\n')
        with pytest.raises(ValueError, match=":2:"):
            load_jsonl(tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl")

    def test_missing_required_fields_are_reported(self, tmp_path):
        (tmp_path / "corpus.jsonl").write_text('{"doc_id": "a"}\n')
        (tmp_path / "queries.jsonl").write_text('{"query": "q"}\n')
        with pytest.raises(ValueError, match="doc_id"):
            load_jsonl(tmp_path / "corpus.jsonl", tmp_path / "queries.jsonl")

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nope.jsonl", tmp_path / "also-nope.jsonl")

    def test_graded_gains_override_binary_relevance(self):
        example = QueryExample("q", "text", frozenset({"a"}), gains={"a": 3.0, "b": 1.0})
        assert example.gain("a") == 3.0
        assert example.gain("b") == 1.0
        assert example.gain("c") == 0.0
        assert example.all_relevant == {"a", "b"}

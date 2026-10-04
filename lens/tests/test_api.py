"""The HTTP API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lens.api import create_app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    app = create_app(ROOT / "data" / "corpus.jsonl", ROOT / "data" / "queries.jsonl")
    with TestClient(app) as c:
        yield c


class TestStatus:
    def test_reports_corpus_and_index_statistics(self, client):
        body = client.get("/api/status").json()
        assert body["documents"] > 0
        assert body["chunks"] >= body["documents"]
        assert body["queries"] > 0
        assert body["embedding_dim"] > 0
        assert body["dense_index"] in ("exact", "hnsw")

    def test_health_is_separate_from_readiness(self, client):
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["indexed"] is True


class TestSearch:
    def test_returns_ranked_results(self, client):
        body = client.get("/api/search", params={"q": "reciprocal rank fusion", "k": 3}).json()
        assert len(body["results"]) <= 3
        assert body["results"][0]["doc_id"] == "hybrid"
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_results_carry_offsets_for_verifiable_citation(self, client):
        body = client.get("/api/search", params={"q": "BM25", "k": 1}).json()
        result = body["results"][0]
        assert result["end"] > result["start"] >= 0
        assert len(result["text"]) > 0

    @pytest.mark.parametrize("retriever", ["dense", "lexical", "hybrid"])
    def test_every_retriever_is_reachable(self, client, retriever):
        body = client.get(
            "/api/search", params={"q": "vector quantization", "retriever": retriever, "k": 2}
        ).json()
        assert body["retriever"] == retriever
        assert body["latency_ms"] >= 0

    def test_hybrid_results_carry_per_retriever_attribution(self, client):
        body = client.get("/api/search", params={"q": "efSearch parameter", "k": 5}).json()
        assert any(r["components"] for r in body["results"])

    def test_an_unknown_retriever_is_a_client_error(self, client):
        assert client.get("/api/search", params={"q": "x", "retriever": "magic"}).status_code == 422

    def test_an_empty_query_is_rejected(self, client):
        assert client.get("/api/search", params={"q": ""}).status_code == 422

    def test_k_is_bounded(self, client):
        assert client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422
        assert client.get("/api/search", params={"q": "x", "k": 1000}).status_code == 422


class TestAsk:
    def test_returns_an_answer_with_citations(self, client):
        body = client.get("/api/ask", params={"q": "what is reciprocal rank fusion", "k": 3}).json()
        assert body["answer"]
        assert body["citations"]
        assert body["model"] == "extractive"

    def test_the_prompt_is_available_for_debugging(self, client):
        body = client.get("/api/ask", params={"q": "what is BM25", "include_prompt": True}).json()
        assert "Question: what is BM25" in body["prompt"]

    def test_the_prompt_is_omitted_by_default(self, client):
        body = client.get("/api/ask", params={"q": "what is BM25"}).json()
        assert body["prompt"] is None


class TestEval:
    def test_returns_aggregate_and_per_query_results(self, client):
        body = client.get("/api/eval", params={"retriever": "hybrid", "k": 10}).json()
        assert 0.0 <= body["report"]["means"]["recall@10"] <= 1.0
        assert len(body["outcomes"]) == body["report"]["num_queries"] + body["report"]["skipped"]
        for outcome in body["outcomes"]:
            assert outcome["query_id"]
            assert "relevant" in outcome

    def test_failures_are_surfaced_separately(self, client):
        body = client.get("/api/eval", params={"retriever": "dense", "k": 5}).json()
        for failure in body["failures"]:
            assert failure["scores"]["recall@5"] == 0.0


class TestSweep:
    def test_compares_configurations_and_names_the_best(self, client):
        body = client.get("/api/sweep", params={"k": 10}).json()
        assert len(body["rows"]) >= 3
        assert body["best"] in [r["name"] for r in body["rows"]]
        assert "<- best" in body["table"]

    def test_the_best_row_actually_has_the_highest_primary_metric(self, client):
        body = client.get("/api/sweep", params={"k": 10, "primary": "ndcg@10"}).json()
        rows = {r["name"]: r["means"]["ndcg@10"] for r in body["rows"]}
        assert rows[body["best"]] == max(rows.values())

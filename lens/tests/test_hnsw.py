"""HNSW: correctness against exact search, and the recall/latency dial."""

from __future__ import annotations

import numpy as np
import pytest

from lens.index.hnsw import BruteForceIndex, HNSWIndex


def clustered(n: int, dim: int, clusters: int = 10, seed: int = 0) -> np.ndarray:
    """Clustered vectors, which is what real embeddings look like.

    Uniformly random vectors are nearly equidistant in high dimensions, so a
    graph index looks good on them almost regardless of how its edges are
    chosen. Clustered data is where a bad neighbour-selection heuristic
    actually traps the search.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)) * 3.0
    return np.vstack([centers[i % clusters] + rng.normal(size=dim) * 0.5 for i in range(n)]).astype(
        np.float32
    )


def recall_against_exact(
    index: HNSWIndex, exact: BruteForceIndex, queries: np.ndarray, k: int, ef: int | None = None
) -> float:
    hits = 0
    for q in queries:
        got = {i for i, _ in index.search(q, k=k, ef=ef)}
        want = {i for i, _ in exact.search(q, k=k)}
        hits += len(got & want)
    return hits / (len(queries) * k)


@pytest.fixture(scope="module")
def corpus() -> tuple[list[str], np.ndarray]:
    vectors = clustered(1200, 32)
    return [f"v{i}" for i in range(len(vectors))], vectors


@pytest.fixture(scope="module")
def built(corpus) -> tuple[HNSWIndex, BruteForceIndex, np.ndarray]:
    ids, vectors = corpus
    hnsw = HNSWIndex(M=16, ef_construction=200, seed=7)
    hnsw.add(ids, vectors)
    exact = BruteForceIndex()
    exact.add(ids, vectors)

    # Queries drawn from the data distribution, which is what a real query
    # looks like relative to its corpus.
    rng = np.random.default_rng(99)
    idx = rng.choice(len(ids), 40, replace=False)
    queries = (vectors[idx] + rng.normal(size=(40, vectors.shape[1])) * 0.3).astype(np.float32)
    return hnsw, exact, queries


class TestExactSearch:
    def test_finds_an_exact_duplicate_first(self):
        vectors = clustered(100, 16)
        index = BruteForceIndex()
        index.add([f"v{i}" for i in range(100)], vectors)
        hits = index.search(vectors[42], k=1)
        assert hits[0][0] == "v42"

    def test_results_are_sorted_by_score(self):
        vectors = clustered(100, 16)
        index = BruteForceIndex()
        index.add([f"v{i}" for i in range(100)], vectors)
        scores = [s for _, s in index.search(vectors[0], k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_an_empty_index_returns_nothing(self):
        assert BruteForceIndex().search(np.zeros(8, dtype=np.float32)) == []

    def test_k_larger_than_the_corpus_is_clamped(self):
        index = BruteForceIndex()
        index.add(["a", "b"], np.eye(2, dtype=np.float32))
        assert len(index.search(np.array([1.0, 0.0], dtype=np.float32), k=100)) == 2

    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
    def test_every_metric_finds_the_duplicate(self, metric):
        vectors = clustered(80, 16)
        index = BruteForceIndex(metric=metric)
        index.add([f"v{i}" for i in range(80)], vectors)
        assert index.search(vectors[7], k=1)[0][0] == "v7"

    def test_a_zero_vector_does_not_produce_nan(self):
        """Normalizing a zero vector would divide by zero and poison every score."""
        index = BruteForceIndex(metric="cosine")
        vectors = np.zeros((3, 4), dtype=np.float32)
        vectors[1, 0] = 1.0
        index.add(["z", "a", "z2"], vectors)
        scores = [s for _, s in index.search(np.array([1.0, 0, 0, 0], dtype=np.float32), k=3)]
        assert all(np.isfinite(s) for s in scores)


class TestHNSWCorrectness:
    def test_finds_an_exact_duplicate_first(self, built, corpus):
        hnsw, _, _ = built
        _, vectors = corpus
        assert hnsw.search(vectors[500], k=1)[0][0] == "v500"

    def test_recall_is_high_against_exact_search(self, built):
        hnsw, exact, queries = built
        recall = recall_against_exact(hnsw, exact, queries, k=10, ef=64)
        assert recall >= 0.95, f"recall@10 was only {recall:.3f}"

    def test_ef_trades_recall_for_work(self, built):
        """The whole point of ef: tunable at query time, without rebuilding."""
        hnsw, exact, queries = built
        measurements = []
        for ef in (8, 16, 64, 200):
            hnsw.reset_counters()
            recall = recall_against_exact(hnsw, exact, queries, k=10, ef=ef)
            measurements.append((ef, recall, hnsw.distance_computations))

        recalls = [r for _, r, _ in measurements]
        work = [w for _, _, w in measurements]
        assert recalls == sorted(recalls), f"recall should not fall as ef rises: {measurements}"
        assert work == sorted(work), f"work should rise with ef: {measurements}"

    def test_search_is_far_cheaper_than_a_full_scan(self, built, corpus):
        hnsw, _, queries = built
        ids, _ = corpus
        hnsw.reset_counters()
        for q in queries:
            hnsw.search(q, k=10, ef=64)
        per_query = hnsw.distance_computations / len(queries)
        assert per_query < len(ids) / 2, (
            f"{per_query:.0f} distance computations per query against a corpus of "
            f"{len(ids)} is no better than brute force"
        )

    def test_the_graph_is_layered(self, built):
        hnsw, _, _ = built
        stats = hnsw.stats()
        assert stats["layers"] >= 2, "a corpus this size should produce multiple layers"
        sizes = stats["layer_sizes"]
        assert sizes == sorted(sizes, reverse=True), "layers should shrink going up"
        assert sizes[0] == len(hnsw), "the base layer must hold every node"

    def test_degree_stays_within_budget(self, built):
        """Unbounded degree is what the pruning step exists to prevent."""
        hnsw, _, _ = built
        assert hnsw.stats()["max_degree"] <= hnsw.M0

    def test_construction_is_reproducible(self, corpus):
        """Level assignment is random; an unseeded index is not reproducible."""
        ids, vectors = corpus
        small_ids, small_vecs = ids[:300], vectors[:300]
        a, b = HNSWIndex(seed=3), HNSWIndex(seed=3)
        a.add(small_ids, small_vecs)
        b.add(small_ids, small_vecs)
        for q in small_vecs[:10]:
            assert a.search(q, k=5) == b.search(q, k=5)


class TestHNSWEdgeCases:
    def test_an_empty_index_returns_nothing(self):
        assert HNSWIndex().search(np.zeros(8, dtype=np.float32)) == []

    def test_a_single_vector_index_works(self):
        index = HNSWIndex()
        index.add(["only"], np.ones((1, 4), dtype=np.float32))
        assert index.search(np.ones(4, dtype=np.float32), k=5) == [("only", pytest.approx(1.0))]

    def test_k_larger_than_the_corpus_is_clamped(self):
        index = HNSWIndex()
        index.add(["a", "b", "c"], clustered(3, 8))
        assert len(index.search(clustered(1, 8)[0], k=50)) == 3

    def test_ef_below_k_is_raised_to_k(self):
        """Otherwise the candidate list caps the result below k."""
        index = HNSWIndex()
        index.add([f"v{i}" for i in range(50)], clustered(50, 8))
        assert len(index.search(clustered(1, 8)[0], k=20, ef=1)) == 20

    def test_duplicate_ids_are_rejected(self):
        index = HNSWIndex()
        index.add(["a"], np.ones((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="duplicate"):
            index.add(["a"], np.zeros((1, 4), dtype=np.float32))

    def test_dimension_mismatch_is_rejected(self):
        index = HNSWIndex()
        index.add(["a"], np.ones((1, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            index.add(["b"], np.ones((1, 8), dtype=np.float32))
        with pytest.raises(ValueError, match="dimension"):
            index.search(np.ones(8, dtype=np.float32))

    def test_mismatched_ids_and_vectors_are_rejected(self):
        with pytest.raises(ValueError, match="ids but"):
            HNSWIndex().add(["a", "b"], np.ones((1, 4), dtype=np.float32))

    def test_adding_in_batches_matches_adding_at_once(self):
        vectors = clustered(200, 16, seed=5)
        ids = [f"v{i}" for i in range(200)]
        one_shot = HNSWIndex(seed=11)
        one_shot.add(ids, vectors)
        batched = HNSWIndex(seed=11)
        for start in range(0, 200, 50):
            batched.add(ids[start : start + 50], vectors[start : start + 50])
        assert len(one_shot) == len(batched) == 200
        # Both must find an exact duplicate; graph shape may differ slightly
        # because insert order into a layer is not identical.
        for i in (0, 77, 199):
            assert one_shot.search(vectors[i], k=1)[0][0] == f"v{i}"
            assert batched.search(vectors[i], k=1)[0][0] == f"v{i}"

    @pytest.mark.parametrize(("m", "ef"), [(1, 200), (16, 4)])
    def test_invalid_parameters_are_rejected(self, m, ef):
        with pytest.raises(ValueError):
            HNSWIndex(M=m, ef_construction=ef)

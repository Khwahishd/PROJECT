"""HNSW: Hierarchical Navigable Small World graphs for approximate nearest
neighbour search.

Implemented from scratch (Malkov & Yashunin, 2016) rather than wrapping a
library, because the structure is the interesting part.

**The problem.** Exact nearest-neighbour search over N vectors costs O(N)
distance computations per query. At a million chunks and 768 dimensions that is
hundreds of millions of multiply-adds for a single query. Tree structures
(k-d trees, ball trees) degrade to brute force above roughly 20 dimensions --
the curse of dimensionality makes every point nearly equidistant.

**The idea.** Build a navigable graph where greedy traversal converges on the
nearest neighbour. A single-layer small-world graph works, but greedy search
takes many hops to cross it. HNSW stacks graphs: the top layer is sparse and
long-range, each layer below is denser and shorter-range, and search descends
through them. This is a skip list generalized from one dimension to many --
coarse layers cover distance quickly, fine layers refine locally. Search cost
drops to roughly O(log N).

**The parameters.**

* ``M`` -- edges per node per layer. Higher means better recall and more memory.
* ``ef_construction`` -- candidate-list size while inserting. Higher builds a
  better graph, more slowly.
* ``ef_search`` -- candidate-list size at query time. The recall/latency dial,
  tunable *after* the index is built, which is what makes it practical.

The neighbour-selection heuristic (Algorithm 4 in the paper) is what makes the
graph navigable rather than merely connected: instead of keeping the M closest
candidates, it keeps candidates that are closer to the new node than to any
already-selected neighbour. That preserves long-range links into sparse
regions; keeping only the nearest neighbours produces tight clusters with no
bridges between them, and greedy search gets trapped in the first cluster it
enters.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np

__all__ = ["BruteForceIndex", "HNSWIndex", "Metric"]

Metric = Literal["cosine", "l2", "dot"]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """Scales rows to unit length, leaving zero rows untouched."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # A zero vector has no direction; dividing would produce NaN and poison
    # every subsequent distance, so leave it as-is.
    norms = np.where(norms == 0.0, 1.0, norms)
    # numpy's stubs type arithmetic as Any; assert the element type at the
    # boundary rather than disabling the check for the whole module.
    return cast("np.ndarray", vectors / norms)


class BruteForceIndex:
    """Exact search by full scan.

    Kept deliberately: it is the ground truth the HNSW tests measure recall
    against, and below roughly ten thousand vectors it is genuinely faster than
    a graph, because a contiguous matrix multiply beats pointer chasing.
    """

    __slots__ = ("_dim", "_ids", "_vectors", "metric")

    def __init__(self, metric: Metric = "cosine") -> None:
        self.metric: Metric = metric
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._dim: int | None = None

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int | None:
        """Vector dimensionality, or ``None`` before anything is added."""
        return self._dim

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Adds vectors to the index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"got {len(ids)} ids but {vectors.shape[0]} vectors")
        if self._dim is not None and vectors.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-dimensional vectors, got {vectors.shape[1]}")

        self._dim = vectors.shape[1]
        if self.metric == "cosine":
            vectors = _normalize(vectors)
        self._ids.extend(ids)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """Returns the ``k`` nearest ids with their similarity scores."""
        if self._vectors is None or not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._dim is not None and q.shape[0] != self._dim:
            raise ValueError(f"query has dimension {q.shape[0]}, index has {self._dim}")

        if self.metric == "cosine":
            q = _normalize(q.reshape(1, -1)).reshape(-1)
            scores = self._vectors @ q
        elif self.metric == "dot":
            scores = self._vectors @ q
        else:  # l2 -- negate so that larger is still better
            scores = -np.linalg.norm(self._vectors - q, axis=1)

        k = min(k, len(self._ids))
        # argpartition finds the top k in O(n) rather than sorting all n.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]


class HNSWIndex:
    """A hierarchical navigable small world graph index."""

    __slots__ = (
        "M",
        "M0",
        "_dim",
        "_distance_count",
        "_entry_point",
        "_graph",
        "_id_to_node",
        "_ids",
        "_level_mult",
        "_matrix",
        "_max_level",
        "_rng",
        "_size",
        "ef_construction",
        "ef_search",
        "metric",
    )

    def __init__(
        self,
        metric: Metric = "cosine",
        M: int = 16,  # noqa: N803 -- M is the name used throughout the literature
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if M < 2:
            raise ValueError(f"M must be at least 2, got {M}")
        if ef_construction < M:
            raise ValueError(f"ef_construction ({ef_construction}) must be at least M ({M})")

        self.metric: Metric = metric
        self.M = M
        # The base layer gets twice the connectivity. It holds every node and
        # is where search terminates, so its recall matters most; the paper
        # recommends 2*M there.
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # A seeded generator keeps index construction reproducible. Level
        # assignment is random, so without this two builds of the same data
        # produce different graphs and different recall.
        self._rng = random.Random(seed)
        # 1/ln(M) makes the expected number of layers ~log_M(N), matching the
        # skip-list analogy: each layer holds roughly 1/M of the one below.
        self._level_mult = 1.0 / math.log(M)

        self._ids: list[str] = []
        self._id_to_node: dict[str, int] = {}
        # Vectors live in one contiguous matrix that grows geometrically, not
        # in a Python list. Fancy-indexing it with an array of neighbour ids
        # lets a whole expansion step become a single matrix-vector product
        # instead of one dot product per edge -- the difference between the
        # inner loop running in NumPy and running in the interpreter.
        self._matrix: np.ndarray | None = None
        self._size = 0
        # _graph[level][node] -> list of neighbour nodes
        self._graph: list[dict[int, list[int]]] = []
        self._entry_point: int | None = None
        self._max_level = -1
        self._dim: int | None = None
        self._distance_count = 0

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int | None:
        """Vector dimensionality."""
        return self._dim

    @property
    def num_layers(self) -> int:
        """How many layers the graph currently has."""
        return self._max_level + 1

    @property
    def distance_computations(self) -> int:
        """Total distance evaluations since construction.

        The honest cost measure for an ANN index: wall-clock time depends on
        the machine, but distance computations are what the algorithm actually
        controls, and comparing them against brute force shows the speedup
        directly.
        """
        return self._distance_count

    def reset_counters(self) -> None:
        """Zeroes the distance counter."""
        self._distance_count = 0

    # -- distance ---------------------------------------------------------

    def _vec(self, node: int) -> np.ndarray:
        """The stored vector for ``node``."""
        assert self._matrix is not None
        return cast("np.ndarray", self._matrix[node])

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance where *smaller is closer*, whatever the metric."""
        self._distance_count += 1
        if self.metric in ("cosine", "dot"):
            # Vectors are normalized on insert for cosine, so the dot product
            # is the cosine similarity; negate to turn it into a distance.
            return float(-np.dot(a, b))
        return float(np.linalg.norm(a - b))

    def _distances_to(self, query: np.ndarray, nodes: Sequence[int]) -> np.ndarray:
        """Distances from ``query`` to many nodes at once."""
        if not nodes:
            return np.empty(0, dtype=np.float32)
        self._distance_count += len(nodes)
        assert self._matrix is not None
        block = self._matrix[np.asarray(nodes, dtype=np.intp)]
        if self.metric in ("cosine", "dot"):
            return cast("np.ndarray", -(block @ query))
        return cast("np.ndarray", np.linalg.norm(block - query, axis=1))

    def _reserve(self, extra: int) -> None:
        """Grows the vector matrix to hold ``extra`` more rows."""
        assert self._dim is not None
        needed = self._size + extra
        if self._matrix is None:
            self._matrix = np.empty((max(needed, 16), self._dim), dtype=np.float32)
        elif needed > self._matrix.shape[0]:
            # Double rather than grow by one: amortizes the copy to O(1) per
            # insert instead of O(n).
            capacity = max(needed, self._matrix.shape[0] * 2)
            grown = np.empty((capacity, self._dim), dtype=np.float32)
            grown[: self._size] = self._matrix[: self._size]
            self._matrix = grown

    def _random_level(self) -> int:
        """Draws a node's top layer from an exponentially decaying distribution."""
        return int(-math.log(self._rng.random()) * self._level_mult)

    # -- construction -----------------------------------------------------

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Inserts vectors into the graph."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(f"got {len(ids)} ids but {vectors.shape[0]} vectors")
        if self._dim is not None and vectors.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-dimensional vectors, got {vectors.shape[1]}")
        self._dim = vectors.shape[1]

        if self.metric == "cosine":
            vectors = _normalize(vectors)

        self._reserve(vectors.shape[0])
        for vid, vec in zip(ids, vectors, strict=True):
            self._insert(vid, vec)

    def _insert(self, vid: str, vector: np.ndarray) -> None:
        if vid in self._id_to_node:
            raise ValueError(f"duplicate id {vid!r}")

        node = self._size
        self._ids.append(vid)
        self._id_to_node[vid] = node
        assert self._matrix is not None
        self._matrix[node] = vector
        self._size += 1

        level = self._random_level()
        while len(self._graph) <= level:
            self._graph.append({})
        for lv in range(level + 1):
            self._graph[lv][node] = []

        if self._entry_point is None:
            self._entry_point = node
            self._max_level = level
            return

        ep = self._entry_point
        # Phase 1: descend from the top, greedily, to the layer just above the
        # new node's own top layer. One nearest neighbour is enough here --
        # these layers only need to get us into the right neighbourhood.
        for lv in range(self._max_level, level, -1):
            ep = self._greedy_descend(vector, ep, lv)

        # Phase 2: from the node's top layer down to 0, do a proper search and
        # wire up bidirectional edges.
        entries = [ep]
        for lv in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(vector, entries, self.ef_construction, lv)
            m = self.M0 if lv == 0 else self.M
            neighbours = self._select_neighbours(vector, candidates, m)

            self._graph[lv][node] = list(neighbours)
            for nb in neighbours:
                self._graph[lv][nb].append(node)
                # The back-edge may push a neighbour over its degree budget.
                # Pruning with the same heuristic keeps the graph navigable
                # instead of letting hub nodes accumulate unbounded degree.
                if len(self._graph[lv][nb]) > m:
                    nbrs = self._graph[lv][nb]
                    dists = self._distances_to(self._vec(nb), nbrs)
                    pruned = self._select_neighbours(
                        self._vec(nb),
                        list(zip(dists.tolist(), nbrs, strict=True)),
                        m,
                    )
                    self._graph[lv][nb] = list(pruned)

            # Carry the whole candidate set down rather than just its closest
            # member. The lower layer is denser, and seeding its search from
            # several entry points explores more of it for the same ef.
            if candidates:
                entries = [n for _, n in sorted(candidates)[: self.M]]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = node

    def _greedy_descend(self, query: np.ndarray, entry: int, level: int) -> int:
        """Walks greedily to a local minimum on one layer."""
        current = entry
        current_dist = self._distance(query, self._vec(current))
        improved = True
        while improved:
            improved = False
            nbrs = self._graph[level].get(current, ())
            if not nbrs:
                break
            dists = self._distances_to(query, list(nbrs))
            best = int(np.argmin(dists))
            if float(dists[best]) < current_dist:
                current, current_dist = nbrs[best], float(dists[best])
                improved = True
        return current

    def _search_layer(
        self, query: np.ndarray, entries: list[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """Best-first search on one layer, returning up to ``ef`` candidates.

        Two heaps are maintained: a min-heap of candidates still to expand, and
        a max-heap of the best results found. The search stops as soon as the
        closest unexpanded candidate is farther than the worst result kept --
        at that point no unexplored node can improve the answer, because the
        graph is navigable.
        """
        visited: set[int] = set(entries)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []

        entry_dists = self._distances_to(query, entries)
        for e, d in zip(entries, entry_dists.tolist(), strict=True):
            heapq.heappush(candidates, (d, e))
            heapq.heappush(results, (-d, e))

        while candidates:
            dist, node = heapq.heappop(candidates)
            worst = -results[0][0]
            if dist > worst and len(results) >= ef:
                # No unexpanded candidate can improve the result set, because
                # the graph is navigable: anything reachable from here is at
                # least this far away.
                break

            fresh = [nb for nb in self._graph[level].get(node, ()) if nb not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            # One matrix-vector product for the whole expansion.
            dists = self._distances_to(query, fresh)
            for nb, d in zip(fresh, dists.tolist(), strict=True):
                if len(results) < ef or d < -results[0][0]:
                    heapq.heappush(candidates, (d, nb))
                    heapq.heappush(results, (-d, nb))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-d, n) for d, n in results]

    def _select_neighbours(
        self, base: np.ndarray, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Algorithm 4: the heuristic that keeps the graph navigable.

        A candidate is kept only if it is closer to ``base`` than to any
        neighbour already selected. The effect is to preserve edges that reach
        into *different* regions rather than piling up edges into whichever
        cluster happens to be nearest.

        Taking the M nearest candidates instead produces a graph of tight,
        well-connected clusters with no bridges between them -- and greedy
        search, which can only follow edges, gets stuck in the first cluster it
        reaches. That failure mode is invisible on uniformly random data and
        severe on real clustered embeddings, which is precisely what retrieval
        corpora are.
        """
        if not candidates:
            return []

        ordered = sorted(candidates)
        selected: list[int] = []
        for dist, cand in ordered:
            if len(selected) >= m:
                break
            # Scalar with an early exit, deliberately: this loop almost always
            # rejects on its first comparison, and batching it into NumPy would
            # pay array-construction overhead to compute distances that the
            # early exit never needs. The opposite is true in _search_layer,
            # where every neighbour's distance is required.
            keep = True
            cand_vec = self._vec(cand)
            for chosen in selected:
                if self._distance(cand_vec, self._vec(chosen)) < dist:
                    keep = False
                    break
            if keep:
                selected.append(cand)

        # If the heuristic was very selective, top up with the nearest
        # remaining candidates so the degree budget is actually used.
        if len(selected) < m:
            for _, cand in ordered:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= m:
                    break
        return selected

    # -- query ------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int = 10, ef: int | None = None
    ) -> list[tuple[str, float]]:
        """Returns the approximate ``k`` nearest ids with similarity scores.

        ``ef`` overrides ``ef_search`` for this query, trading latency for
        recall without rebuilding anything.
        """
        if self._entry_point is None:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._dim is not None and q.shape[0] != self._dim:
            raise ValueError(f"query has dimension {q.shape[0]}, index has {self._dim}")
        if self.metric == "cosine":
            q = _normalize(q.reshape(1, -1)).reshape(-1)

        # ef below k would cap the result list before k items are found.
        ef = max(ef if ef is not None else self.ef_search, k)

        ep = self._entry_point
        for lv in range(self._max_level, 0, -1):
            ep = self._greedy_descend(q, ep, lv)

        found = self._search_layer(q, [ep], ef, 0)
        found.sort()
        # Internally distance means "smaller is closer"; callers expect a score
        # where larger is better, so negate. For cosine and dot that recovers
        # the similarity exactly; for L2 it yields a negative distance, which
        # orders correctly even though its magnitude is not a similarity.
        return [(self._ids[node], float(-dist)) for dist, node in found[:k]]

    def stats(self) -> dict[str, float]:
        """Structural statistics, useful for tuning and for tests."""
        degrees = [len(nbrs) for layer in self._graph for nbrs in layer.values()]
        layer_sizes = [len(layer) for layer in self._graph]
        return {
            "nodes": float(len(self._ids)),
            "layers": float(self.num_layers),
            "mean_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
            "max_degree": float(max(degrees)) if degrees else 0.0,
            "layer_sizes": layer_sizes,  # type: ignore[dict-item]
            "distance_computations": float(self._distance_count),
        }

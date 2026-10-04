"""BM25 lexical retrieval, implemented directly.

BM25 is in every hybrid retrieval system for a reason that dense embeddings
have not displaced: it matches *exact terms*. A query containing a product code,
an error number, a surname or an API method name needs the document containing
that literal string, and an embedding of a rare token is close to noise.

The scoring function (Robertson & Walker, Okapi BM25) is::

    score(q, d) = Σ_t  IDF(t) · (f(t,d) · (k1 + 1)) / (f(t,d) + k1 · (1 - b + b · |d|/avgdl))

with the two free parameters doing specific jobs:

* ``k1`` controls term-frequency saturation. A term appearing 20 times does not
  make a document 20 times more relevant, and ``k1`` sets how quickly the
  benefit flattens.
* ``b`` controls length normalization. ``b=0`` ignores document length
  entirely; ``b=1`` normalizes fully. The default 0.75 is the usual compromise,
  since long documents genuinely do contain more, but not proportionally more.

IDF uses the probabilistic form with ``+0.5`` smoothing and, crucially, a
``+1`` inside the logarithm (the variant Lucene uses). Without that ``+1`` the
classic form goes *negative* for any term appearing in more than half the
corpus -- on a five-document corpus, a term in three of them scores -0.34 --
so a matching document would be actively pushed down the ranking for matching.
With it, IDF decreases monotonically toward zero but never crosses it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

__all__ = ["STOPWORDS", "BM25Index", "tokenize"]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_'][a-z0-9]+)*")

#: A deliberately small stop list. Aggressive stop-word removal hurts BM25 more
#: than it helps, because IDF already discounts common terms -- and dropping
#: them breaks queries where they carry meaning ("the who", "to be or not").
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]
)


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """Lowercases and splits text into alphanumeric terms.

    Hyphens, underscores and apostrophes inside a word are kept, so
    ``state-of-the-art``, ``snake_case`` and ``don't`` survive as single terms
    rather than fragmenting into pieces that match nothing.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class BM25Index:
    """An in-memory BM25 index over a fixed corpus.

    The index is built once from all documents, because IDF needs corpus-wide
    document frequencies and the average document length. Adding documents
    incrementally would invalidate every previously computed score.
    """

    __slots__ = (
        "_avgdl",
        "_doc_ids",
        "_doc_len",
        "_idf",
        "_n_docs",
        "_postings",
        "_remove_stopwords",
        "b",
        "k1",
    )

    def __init__(self, k1: float = 1.5, b: float = 0.75, *, remove_stopwords: bool = False) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self.k1 = k1
        self.b = b
        self._remove_stopwords = remove_stopwords
        self._doc_ids: list[str] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        # term -> {document position -> term frequency}
        self._postings: dict[str, dict[int, int]] = {}
        self._idf: dict[str, float] = {}
        self._n_docs = 0

    def __len__(self) -> int:
        return self._n_docs

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct terms in the index."""
        return len(self._postings)

    @property
    def average_document_length(self) -> float:
        """Mean document length in tokens."""
        return self._avgdl

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        """Builds the index. Returns ``self`` so it can be chained."""
        if len(doc_ids) != len(texts):
            raise ValueError(f"got {len(doc_ids)} ids but {len(texts)} texts")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("document ids must be unique")

        self._doc_ids = list(doc_ids)
        self._n_docs = len(doc_ids)
        self._postings = {}
        self._doc_len = []

        for pos, text in enumerate(texts):
            tokens = tokenize(text, remove_stopwords=self._remove_stopwords)
            self._doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self._postings.setdefault(term, {})[pos] = freq

        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        """Precomputes IDF per term.

        Done once at build time rather than per query: IDF depends only on the
        corpus, and recomputing a logarithm per query term per document would
        dominate scoring.
        """
        n = self._n_docs
        self._idf = {}
        for term, postings in self._postings.items():
            df = len(postings)
            # Robertson/Sparck-Jones probabilistic IDF with the Lucene +1.
            # That +1 is what keeps the value non-negative; the max() below is
            # belt-and-braces against a pathological corpus.
            value = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            self._idf[term] = max(value, 0.0)

    def score(self, query: str, *, top_k: int | None = None) -> list[tuple[str, float]]:
        """Scores the corpus against ``query``, best first.

        Only documents containing at least one query term are scored. Walking
        the postings lists rather than the corpus makes the cost proportional
        to the number of matching documents, not to the corpus size -- the
        reason an inverted index exists at all.
        """
        terms = tokenize(query, remove_stopwords=self._remove_stopwords)
        if not terms or self._n_docs == 0:
            return []

        scores: dict[int, float] = {}
        for term, qf in Counter(terms).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            for pos, tf in postings.items():
                norm = 1.0 - self.b + self.b * (self._doc_len[pos] / self._avgdl)
                contribution = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                # Repeating a term in the query weights it proportionally,
                # which matters for queries built by concatenating fields.
                scores[pos] = scores.get(pos, 0.0) + contribution * qf

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._doc_ids[kv[0]]))
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(self._doc_ids[pos], score) for pos, score in ranked]

    def explain(self, query: str, doc_id: str) -> dict[str, float]:
        """Returns each query term's contribution to one document's score.

        Retrieval failures are usually a question of *why* a document did or
        did not match; a per-term breakdown answers it directly.
        """
        try:
            pos = self._doc_ids.index(doc_id)
        except ValueError as exc:
            raise KeyError(f"no document with id {doc_id!r} in the index") from exc

        out: dict[str, float] = {}
        for term, qf in Counter(tokenize(query, remove_stopwords=self._remove_stopwords)).items():
            postings = self._postings.get(term)
            if not postings or pos not in postings:
                out[term] = 0.0
                continue
            tf = postings[pos]
            idf = self._idf.get(term, 0.0)
            norm = 1.0 - self.b + self.b * (self._doc_len[pos] / self._avgdl)
            out[term] = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm) * qf
        return out

    def document_frequency(self, term: str) -> int:
        """How many documents contain ``term``."""
        return len(self._postings.get(term.lower(), {}))

    def idf(self, term: str) -> float:
        """The inverse document frequency of ``term``."""
        return self._idf.get(term.lower(), 0.0)


def build_bm25(doc_ids: Iterable[str], texts: Iterable[str], **kwargs: float) -> BM25Index:
    """Convenience constructor."""
    return BM25Index(**kwargs).fit(list(doc_ids), list(texts))  # type: ignore[arg-type]

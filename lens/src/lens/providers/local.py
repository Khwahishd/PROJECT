"""Deterministic local providers.

These are not toys standing in for "real" models -- they are the reason the
project is testable. A hashing embedder has a genuine mathematical basis (the
hashing trick, Weinberger et al. 2009), produces identical vectors on every
machine and every run, and needs no network. That makes retrieval behaviour
reproducible and lets the evaluation harness be tested for correctness
independently of any model's quality.

They are not a substitute for a trained embedding model in production. What
they *are* is the control condition: if a tuned pipeline cannot beat hashed
bag-of-words on your corpus, the problem is not the embedding model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import cast

import numpy as np

from ..index.bm25 import tokenize
from .base import GenerationResult

__all__ = ["EchoGenerator", "ExtractiveGenerator", "HashingEmbedder"]


class HashingEmbedder:
    """Embeds text by hashing n-grams into a fixed-dimensional space.

    Each token is hashed to a coordinate and accumulated with a signed weight.
    The sign comes from a second hash and is what makes the estimator unbiased:
    without it, hash collisions always add, so unrelated documents drift toward
    each other and every similarity creeps upward. With signed weights,
    collisions cancel in expectation.

    Character n-grams are mixed in alongside word tokens, which gives some
    robustness to typos and morphology -- ``retrieval`` and ``retrieve`` share
    most of their 4-grams even though they are distinct word tokens.

    Term weights are sublinear in frequency (``1 + log tf``), the same
    saturation idea BM25 applies, so one repeated word cannot dominate a
    vector's direction.
    """

    __slots__ = ("_char_ngrams", "_dim", "_idf", "_name", "_seed", "_use_idf")

    def __init__(
        self,
        dim: int = 384,
        *,
        seed: int = 0,
        char_ngrams: int = 4,
        use_idf: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if char_ngrams < 0:
            raise ValueError(f"char_ngrams must be non-negative, got {char_ngrams}")
        self._dim = dim
        self._seed = seed
        self._char_ngrams = char_ngrams
        self._use_idf = use_idf
        self._idf: dict[str, float] = {}
        self._name = f"hashing-{dim}d"

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    def name(self) -> str:
        """Provider identifier."""
        return self._name

    def fit_idf(self, corpus: list[str]) -> HashingEmbedder:
        """Learns IDF weights from a corpus.

        Optional but worthwhile: without it every term contributes equally, so
        a document's vector points mostly in the direction of its stop words.
        """
        if not corpus:
            return self
        df: Counter[str] = Counter()
        for text in corpus:
            df.update(set(self._features(text)))
        n = len(corpus)
        self._idf = {term: math.log(1.0 + n / (1.0 + count)) for term, count in df.items()}
        return self

    def _features(self, text: str) -> list[str]:
        """Word tokens plus character n-grams."""
        words = tokenize(text)
        feats = list(words)
        if self._char_ngrams > 0:
            cleaned = re.sub(r"\s+", " ", text.lower())
            n = self._char_ngrams
            feats.extend(cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1)))
        return feats

    def _hash(self, token: str) -> tuple[int, float]:
        """Maps a token to a coordinate and a sign."""
        # blake2b with a fixed seed: stable across processes and Python
        # versions, unlike the built-in hash(), which is randomized per run and
        # would make vectors differ between invocations.
        digest = hashlib.blake2b(
            token.encode("utf-8"), digest_size=8, key=str(self._seed).encode()
        ).digest()
        value = int.from_bytes(digest, "little")
        index = value % self._dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds a batch of texts."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._features(text))
            for token, tf in counts.items():
                index, sign = self._hash(token)
                # Sublinear term frequency, as in BM25: the tenth occurrence of
                # a word says much less than the second.
                weight = 1.0 + math.log(tf)
                if self._use_idf and self._idf:
                    weight *= self._idf.get(token, 1.0)
                out[row, index] += sign * weight

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return cast("np.ndarray", out / norms)

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds one query."""
        return cast("np.ndarray", self.embed([text])[0])


class ExtractiveGenerator:
    """Answers by extracting the most query-relevant sentences from context.

    Deliberately extractive, which gives it one property no generative model
    has: **it cannot hallucinate**. Every sentence it emits is copied verbatim
    from the retrieved context. That makes it the right default for testing a
    RAG pipeline, because any factual error in the output is necessarily a
    *retrieval* failure rather than a generation one -- which is exactly the
    variable the evaluation harness is trying to isolate.
    """

    __slots__ = ("_max_sentences", "_name")

    def __init__(self, max_sentences: int = 3) -> None:
        self._max_sentences = max_sentences
        self._name = "extractive"

    @property
    def name(self) -> str:
        """Provider identifier."""
        return self._name

    def generate(self, prompt: str, *, max_tokens: int = 512) -> GenerationResult:
        """Selects the sentences in the prompt's context that best match its question."""
        question, context = _split_prompt(prompt)
        sentences = _split_sentences(context)
        if not sentences:
            return GenerationResult(
                "I don't have enough information in the provided context to answer that.",
                self._name,
                {"prompt_chars": len(prompt)},
            )

        q_terms = set(tokenize(question, remove_stopwords=True))
        scored: list[tuple[float, int, str]] = []
        for i, sentence in enumerate(sentences):
            terms = set(tokenize(sentence, remove_stopwords=True))
            if not terms:
                continue
            overlap = len(q_terms & terms)
            # Normalize by sentence length so a long sentence does not win on
            # term overlap alone, and keep the index to break ties by document
            # order -- which keeps the output deterministic.
            score = overlap / math.sqrt(len(terms))
            scored.append((score, -i, sentence))

        if not scored or max(s for s, _, _ in scored) == 0.0:
            return GenerationResult(
                "I don't have enough information in the provided context to answer that.",
                self._name,
                {"prompt_chars": len(prompt)},
            )

        scored.sort(reverse=True)
        chosen = scored[: self._max_sentences]
        # Restore document order, so the answer reads as prose rather than as a
        # relevance-ranked list.
        chosen.sort(key=lambda t: -t[1])
        text = " ".join(s for _, _, s in chosen)
        return GenerationResult(
            text,
            self._name,
            {"prompt_chars": len(prompt), "sentences_considered": len(sentences)},
        )


class EchoGenerator:
    """Returns a fixed string. Used to isolate retrieval metrics from generation."""

    __slots__ = ("_name", "_reply")

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self._name = "echo"

    @property
    def name(self) -> str:
        """Provider identifier."""
        return self._name

    def generate(self, prompt: str, *, max_tokens: int = 512) -> GenerationResult:
        """Returns the fixed reply."""
        return GenerationResult(self._reply, self._name, {"prompt_chars": len(prompt)})


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    """Splits text into sentences on terminal punctuation."""
    parts = [p.strip() for p in _SENTENCE_RE.split(text)]
    return [p for p in parts if p]


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Separates the question from the retrieved context in a built prompt.

    The context is the text *between* the ``Context:`` and ``Question:``
    headers. Taking everything before ``Question:`` would include the
    instruction block, and the extractive generator would then happily quote
    "If the context does not contain the answer, say so" back as its answer --
    a sentence that scores well on term overlap with almost any question.

    Falls back to treating the whole prompt as both, so a caller passing raw
    text still gets sensible behaviour rather than an empty answer.
    """
    question = prompt
    if "Question:" in prompt:
        question = prompt.rpartition("Question:")[2].strip()

    body = prompt.rpartition("Question:")[0] if "Question:" in prompt else prompt
    if "Context:" in body:
        body = body.partition("Context:")[2]

    # Drop the citation markers so they are not quoted into the answer.
    context = re.sub(r"^\s*\[\d+\]\s*", "", body, flags=re.MULTILINE)
    return question, context.strip()

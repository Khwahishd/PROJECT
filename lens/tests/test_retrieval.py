"""Retrievers end to end, and the behaviours that distinguish them."""

from __future__ import annotations

import numpy as np
import pytest

from lens.providers import EchoGenerator, ExtractiveGenerator, HashingEmbedder
from lens.retrieval import DenseRetriever, HybridRetriever


class TestDenseRetriever:
    def test_retrieves_the_right_document(self, dense):
        assert "hybrid" in dense.retrieve("merging ranked lists", k=3).doc_ids

    def test_matches_a_paraphrase(self, dense):
        """The thing dense retrieval exists for."""
        result = dense.retrieve("why does splitting text into pieces that overlap help", k=3)
        assert "chunking" in result.doc_ids[:2]

    def test_respects_k(self, dense):
        assert len(dense.retrieve("graphs", k=2).chunks) <= 2

    def test_reports_latency_and_diagnostics(self, dense):
        result = dense.retrieve("graphs", k=3)
        assert result.latency_ms >= 0.0
        assert "index" in result.diagnostics

    def test_doc_ids_are_deduplicated_keeping_the_best_rank(self, dense):
        result = dense.retrieve("graphs", k=10)
        assert len(result.doc_ids) == len(set(result.doc_ids))

    def test_an_empty_corpus_returns_nothing(self, embedder):
        retriever = DenseRetriever([], embedder)
        assert retriever.retrieve("anything", k=5).chunks == ()

    def test_switches_to_an_approximate_index_when_large(self, embedder):
        from lens.types import Chunk

        chunks = [
            Chunk(f"c{i}", f"d{i}", f"document number {i} about topic {i % 50}", 0, 10)
            for i in range(2100)
        ]
        retriever = DenseRetriever(chunks, HashingEmbedder(dim=32))
        assert not retriever.is_exact, "a large corpus should use the graph index"
        assert len(retriever.retrieve("topic 7", k=5).chunks) == 5


class TestLexicalRetriever:
    def test_matches_a_rare_literal_token(self, lexical):
        """The thing BM25 exists for."""
        assert lexical.retrieve("ANN-4417", k=3).doc_ids[0] == "ann-hnsw"

    def test_a_query_sharing_no_terms_returns_nothing(self, lexical):
        """BM25 matches surface forms; with no shared term there is no signal.

        This is the complementary failure that makes hybrid retrieval worth
        building -- see test_complementary_failure in test_eval.py for the
        corpus-level version of the claim.
        """
        assert lexical.retrieve("zebra giraffe hippopotamus xylophone", k=3).chunks == ()

    def test_exposes_its_index_for_explanation(self, lexical):
        top = lexical.retrieve("k1 and b parameters", k=1).chunks[0]
        parts = lexical.index.explain("k1 and b parameters", top.chunk_id)
        assert parts["k1"] > 0, parts


class TestHybridRetriever:
    def test_combines_both_signals(self, hybrid):
        for query, expected in [
            ("ANN-4417", "ann-hnsw"),
            ("why does splitting text into pieces that overlap help", "chunking"),
            ("reciprocal rank fusion", "hybrid"),
            ("shrinking memory by storing fewer bits per dimension", "quantization"),
        ]:
            assert expected in hybrid.retrieve(query, k=3).doc_ids, f"failed on {query!r}"

    def test_results_carry_per_retriever_attribution(self, hybrid):
        result = hybrid.retrieve("k1 and b parameters", k=5)
        assert any(c.components for c in result.chunks)
        for chunk in result.chunks:
            assert set(chunk.components) <= {"dense", "lexical"}

    def test_reports_how_much_the_retrievers_agreed(self, hybrid):
        """Overlap near 1.0 means hybrid is buying nothing."""
        result = hybrid.retrieve("graphs", k=5)
        assert 0.0 <= result.diagnostics["overlap"] <= 1.0

    def test_overfetches_so_fusion_can_promote(self, dense, lexical):
        """A chunk ranked 15th by one retriever and 3rd by the other can
        legitimately finish top-10 -- but only if it was fetched."""
        retriever = HybridRetriever(dense, lexical, overfetch=3)
        result = retriever.retrieve("graphs", k=2)
        assert result.diagnostics["dense_candidates"] > 2

    def test_weights_shift_which_retriever_dominates(self, dense, lexical):
        query = "why does splitting text into pieces that overlap help"
        dense_heavy = HybridRetriever(dense, lexical, dense_weight=10.0, lexical_weight=0.0)
        assert dense_heavy.retrieve(query, k=1).doc_ids == ["chunking"]

    def test_weighted_fusion_is_available(self, dense, lexical):
        retriever = HybridRetriever(dense, lexical, method="weighted")
        assert len(retriever.retrieve("graphs", k=3).chunks) > 0

    def test_invalid_overfetch_is_rejected(self, dense, lexical):
        with pytest.raises(ValueError, match="overfetch"):
            HybridRetriever(dense, lexical, overfetch=0)


class TestEmbedder:
    def test_vectors_are_unit_length(self, embedder):
        vectors = embedder.embed(["hello world", "another document here"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_embeddings_are_deterministic_across_instances(self):
        """Python's built-in hash is randomized per process; blake2b is not.

        Without this the same corpus would embed differently on every run and
        no evaluation would be reproducible.
        """
        a = HashingEmbedder(dim=128).embed(["stable text"])
        b = HashingEmbedder(dim=128).embed(["stable text"])
        assert np.array_equal(a, b)

    def test_similar_text_is_closer_than_unrelated_text(self, embedder):
        vectors = embedder.embed(
            ["the cat sat on the mat", "a cat sitting on a mat", "quantum chromodynamics"]
        )
        assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])

    def test_character_ngrams_give_robustness_to_morphology(self):
        emb = HashingEmbedder(dim=512, char_ngrams=4, use_idf=False)
        with_ngrams = emb.embed(["retrieval system", "retrieving systems"])
        word_only = HashingEmbedder(dim=512, char_ngrams=0, use_idf=False).embed(
            ["retrieval system", "retrieving systems"]
        )
        assert float(with_ngrams[0] @ with_ngrams[1]) > float(word_only[0] @ word_only[1])

    def test_empty_text_does_not_produce_nan(self, embedder):
        vector = embedder.embed([""])
        assert np.all(np.isfinite(vector))

    def test_query_and_document_encoding_agree(self, embedder):
        batched = embedder.embed(["a query"])[0]
        single = embedder.embed_query("a query")
        assert np.allclose(batched, single)

    def test_invalid_dimension_is_rejected(self):
        with pytest.raises(ValueError):
            HashingEmbedder(dim=0)


class TestGenerators:
    def test_extractive_generator_only_quotes_the_context(self):
        """It cannot hallucinate, which is the point: any factual error in the
        output is necessarily a retrieval failure."""
        context = "The cat sat on the mat. Dogs are loyal. The sky is blue."
        prompt = f"Instructions here.\n\nContext:\n{context}\n\nQuestion: where did the cat sit?"
        answer = ExtractiveGenerator(max_sentences=1).generate(prompt).text
        assert answer in context or answer.strip(".") in context

    def test_extractive_generator_does_not_quote_the_instructions(self):
        """The instruction block overlaps heavily with almost any question."""
        prompt = (
            "Answer using only the context. If the context does not contain the "
            "answer, say so explicitly rather than guessing.\n\n"
            "Context:\n[1] Penguins are flightless birds native to the southern hemisphere.\n\n"
            "Question: what does the context say about the answer"
        )
        answer = ExtractiveGenerator().generate(prompt).text
        assert "say so explicitly" not in answer

    def test_extractive_generator_declines_with_no_context(self):
        prompt = "Instructions.\n\nContext:\n\n\nQuestion: anything?"
        assert "don't have enough information" in ExtractiveGenerator().generate(prompt).text

    def test_echo_generator_isolates_retrieval_metrics(self):
        assert EchoGenerator("fixed").generate("anything").text == "fixed"

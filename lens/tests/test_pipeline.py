"""The RAG pipeline: prompt construction, citations, and context budgets."""

from __future__ import annotations

import pytest

from lens.pipeline import DEFAULT_TEMPLATE, PromptTemplate, RAGPipeline, extract_citations
from lens.providers import EchoGenerator, ExtractiveGenerator
from lens.types import Chunk, ScoredChunk


def sc(text: str, chunk_id: str = "c1", doc_id: str = "d1") -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id, doc_id, text, 0, len(text)), score=1.0)


class TestPromptTemplate:
    def test_numbers_chunks_for_citation(self):
        prompt, used = DEFAULT_TEMPLATE.build("q?", [sc("first"), sc("second", "c2")])
        assert "[1] first" in prompt
        assert "[2] second" in prompt
        assert len(used) == 2

    def test_includes_the_question_and_the_instruction(self):
        prompt, _ = DEFAULT_TEMPLATE.build("what is X?", [sc("context")])
        assert "what is X?" in prompt
        assert "only the context" in prompt

    def test_instructs_the_model_to_decline_rather_than_guess(self):
        """Without this line a model answers from its parameters, which is the
        most common way a RAG system produces a confident falsehood."""
        assert "does not contain the answer" in DEFAULT_TEMPLATE.instruction

    def test_enforces_the_context_budget(self):
        """Overflowing a context window truncates from one end, usually
        dropping either the instructions or the question."""
        template = PromptTemplate(instruction="i", context_budget=120)
        chunks = [sc("x" * 100, f"c{i}") for i in range(10)]
        prompt, used = template.build("q?", chunks)
        assert len(used) == 1, f"budget of 120 should fit one 100-char chunk, got {len(used)}"
        assert len(prompt) < 400

    def test_stops_rather_than_truncating_a_chunk(self):
        """Half a passage can change its meaning, and a partially quoted
        citation is worse than a missing one."""
        template = PromptTemplate(instruction="i", context_budget=50)
        prompt, used = template.build("q?", [sc("y" * 200)])
        assert used == []
        assert "no relevant context" in prompt

    def test_handles_no_retrieved_context(self):
        prompt, used = DEFAULT_TEMPLATE.build("q?", [])
        assert used == []
        assert "no relevant context was retrieved" in prompt


class TestCitations:
    def test_extracts_bracketed_numbers_in_first_use_order(self):
        assert extract_citations("As shown in [2] and [1], and again [2].") == [2, 1]

    def test_returns_nothing_when_there_are_no_citations(self):
        assert extract_citations("A plain answer.") == []

    def test_pipeline_reports_only_the_chunks_the_model_saw(self, hybrid):
        """Listing retrieved-but-unused chunks would overstate grounding."""
        pipeline = RAGPipeline(hybrid, EchoGenerator("see [1] only"), k=5)
        answer = pipeline.answer("reciprocal rank fusion")
        assert len(answer.citations) == 1

    def test_an_out_of_range_citation_is_ignored(self, hybrid):
        pipeline = RAGPipeline(hybrid, EchoGenerator("see [99]"), k=3)
        answer = pipeline.answer("reciprocal rank fusion")
        # Falls back to the chunks actually supplied rather than crashing.
        assert all(c is not None for c in answer.citations)


class TestPipeline:
    def test_answers_from_retrieved_context(self, hybrid):
        pipeline = RAGPipeline(hybrid, ExtractiveGenerator(), k=3)
        answer = pipeline.answer("what is reciprocal rank fusion")
        assert "rank" in answer.text.lower()
        assert answer.citations

    def test_reports_timing_and_usage(self, hybrid):
        pipeline = RAGPipeline(hybrid, ExtractiveGenerator(), k=3)
        answer = pipeline.answer("what does efSearch control")
        assert answer.latency_ms >= 0.0
        assert answer.usage["retrieved"] > 0
        assert answer.usage["context_chunks"] > 0

    def test_build_prompt_exposes_what_the_model_is_given(self, hybrid):
        """The fastest way to debug a wrong answer is to read the prompt."""
        pipeline = RAGPipeline(hybrid, EchoGenerator(), k=2)
        prompt = pipeline.build_prompt("reciprocal rank fusion")
        assert "Question: reciprocal rank fusion" in prompt
        assert "[1]" in prompt

    def test_an_extractive_answer_is_always_grounded(self, hybrid):
        """It copies verbatim, so any factual error is a *retrieval* failure --
        which is exactly the variable the harness isolates."""
        from lens.eval.metrics import answer_contains_citation

        pipeline = RAGPipeline(hybrid, ExtractiveGenerator(), k=4)
        answer = pipeline.answer("how does product quantization save memory")
        contexts = [c.chunk.text for c in answer.citations]
        assert answer_contains_citation(answer.text, contexts) == pytest.approx(1.0)

    def test_name_identifies_the_configuration(self, hybrid):
        pipeline = RAGPipeline(hybrid, EchoGenerator(), k=3)
        assert "hybrid" in pipeline.name and "echo" in pipeline.name

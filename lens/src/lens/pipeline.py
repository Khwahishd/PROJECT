"""The RAG pipeline: retrieve, build a prompt, generate, cite.

The prompt-construction step is the part that gets least attention and causes
most problems, so it is explicit here:

* **A context budget is enforced.** Chunks are added until the budget is spent,
  not until ``k`` is reached. Silently overflowing a model's context window
  truncates from one end, usually dropping either the instructions or the
  question.
* **Chunks are numbered.** The model is asked to cite ``[1]``, ``[2]``, and the
  numbers map back to real chunks with real character offsets, so a citation is
  checkable rather than decorative.
* **The instruction says what to do when the context does not contain the
  answer.** Without that line a model will answer from its parameters, which is
  the single most common way a RAG system produces a confident falsehood.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from .providers.base import Generator
from .types import Answer, ScoredChunk

__all__ = ["DEFAULT_TEMPLATE", "PromptTemplate", "RAGPipeline", "extract_citations"]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """How retrieved context is turned into a prompt."""

    instruction: str
    context_header: str = "Context:"
    question_header: str = "Question:"
    #: Approximate character budget for the context block. Characters rather
    #: than tokens so the pipeline stays tokenizer-agnostic; roughly 4
    #: characters per token is a serviceable rule for English.
    context_budget: int = 6000

    def build(self, question: str, chunks: Sequence[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
        """Renders the prompt and returns the chunks that actually fit."""
        used: list[ScoredChunk] = []
        parts: list[str] = []
        budget = self.context_budget

        for i, sc in enumerate(chunks, start=1):
            block = f"[{i}] {sc.chunk.text}"
            if len(block) > budget:
                # Stop rather than truncate mid-chunk: half a passage can
                # change its meaning, and a partially quoted citation is worse
                # than a missing one.
                break
            parts.append(block)
            used.append(sc)
            budget -= len(block) + 2

        context = "\n\n".join(parts) if parts else "(no relevant context was retrieved)"
        prompt = (
            f"{self.instruction}\n\n"
            f"{self.context_header}\n{context}\n\n"
            f"{self.question_header} {question}"
        )
        return prompt, used


DEFAULT_TEMPLATE = PromptTemplate(
    instruction=(
        "Answer the question using only the context below. "
        "Cite the passages you used with their bracketed numbers, like [1]. "
        "If the context does not contain the answer, say so explicitly rather "
        "than guessing."
    )
)


class RAGPipeline:
    """Retrieval-augmented generation over any retriever and generator."""

    __slots__ = ("_generator", "_k", "_retriever", "_template")

    def __init__(
        self,
        retriever: object,
        generator: Generator,
        *,
        k: int = 5,
        template: PromptTemplate = DEFAULT_TEMPLATE,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._template = template
        self._k = k

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        retriever_name = getattr(self._retriever, "name", "retriever")
        return f"rag[{retriever_name}+{self._generator.name}]"

    def answer(self, question: str, *, k: int | None = None) -> Answer:
        """Retrieves context and generates a cited answer."""
        start = time.perf_counter()
        result = self._retriever.retrieve(question, k=k or self._k)  # type: ignore[attr-defined]
        prompt, used = self._template.build(question, result.chunks)
        generated = self._generator.generate(prompt)

        # Report only the chunks the model was actually shown. Listing
        # retrieved-but-unused chunks as citations would overstate grounding.
        cited_indices = extract_citations(generated.text)
        citations = tuple(used[i - 1] for i in cited_indices if 1 <= i <= len(used)) or tuple(used)

        return Answer(
            question=question,
            text=generated.text,
            citations=citations,
            model=generated.model,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            usage={
                **generated.usage,
                "retrieved": len(result.chunks),
                "context_chunks": len(used),
                "retrieval_ms": int(result.latency_ms),
            },
        )

    def build_prompt(self, question: str, *, k: int | None = None) -> str:
        """Returns the prompt that would be sent, without generating.

        Exposed because the fastest way to debug a wrong answer is to read what
        the model was actually given.
        """
        result = self._retriever.retrieve(question, k=k or self._k)  # type: ignore[attr-defined]
        prompt, _ = self._template.build(question, result.chunks)
        return prompt


_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citations(text: str) -> list[int]:
    """Returns the bracketed citation numbers in ``text``, in first-use order."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen

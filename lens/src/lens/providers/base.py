"""Provider protocols.

Every external model sits behind one of two narrow protocols. The point is not
abstraction for its own sake -- it is that the *default* implementations are
deterministic and local, so the entire test suite, the evaluation harness and
CI run with no API key, no network and no cost, and produce the same numbers on
every machine.

An evaluation harness whose results depend on a remote model's sampling is not
an evaluation harness; it is a weather report.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Embedder", "GenerationResult", "Generator"]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors produced."""
        ...

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs for reproducibility."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds a batch of texts into a ``(len(texts), dim)`` array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query into a ``(dim,)`` array.

        Separate from :meth:`embed` because several real embedding models are
        asymmetric: they expect a different prefix or instruction for queries
        than for documents, and using the document encoding for a query
        measurably degrades retrieval.
        """
        ...


class GenerationResult:
    """A generated completion with its token accounting."""

    __slots__ = ("model", "text", "usage")

    def __init__(self, text: str, model: str, usage: dict[str, int] | None = None) -> None:
        self.text = text
        self.model = model
        self.usage = usage or {}

    def __repr__(self) -> str:
        return f"GenerationResult(model={self.model!r}, text={self.text[:60]!r}…)"


@runtime_checkable
class Generator(Protocol):
    """Produces an answer from a prompt."""

    @property
    def name(self) -> str:
        """Identifier recorded in evaluation runs."""
        ...

    def generate(self, prompt: str, *, max_tokens: int = 512) -> GenerationResult:
        """Generates a completion."""
        ...

"""Rerankers: the no-op (fused retrieval order) and a local cross-encoder.

The cross-encoder scores each (question, chunk) pair jointly, which catches matches that
vector and keyword search rank too low. It runs on CPU in a worker thread, so it never blocks
the event loop. `sentence-transformers` is an optional extra (`uv sync --extra rerank`).
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from app.config import Settings


class HasContent(Protocol):
    @property
    def content(self) -> str: ...


T = TypeVar("T", bound=HasContent)
# Scores (question, passage) pairs; higher means more relevant.
PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


class Reranker(Protocol):
    async def rerank(self, question: str, candidates: Sequence[T], top_k: int) -> list[T]: ...


class NoOpReranker:
    """Keeps the fused retrieval order."""

    async def rerank(self, question: str, candidates: Sequence[T], top_k: int) -> list[T]:
        return list(candidates[:top_k])


class CrossEncoderReranker:
    def __init__(self, score: PairScorer) -> None:
        self._score = score

    async def rerank(self, question: str, candidates: Sequence[T], top_k: int) -> list[T]:
        if len(candidates) <= 1:
            return list(candidates[:top_k])
        pairs = [(question, c.content) for c in candidates]
        scores = await asyncio.to_thread(self._score, pairs)
        # Stable sort: equal scores keep the fused order.
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [candidates[i] for i in order[:top_k]]


def make_reranker(settings: "Settings") -> Reranker:
    if settings.reranker == "none":
        return NoOpReranker()
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "RERANKER=cross-encoder needs the rerank extra: uv sync --extra rerank"
        ) from exc
    model = CrossEncoder(settings.reranker_model, device="cpu")
    return CrossEncoderReranker(lambda pairs: model.predict(pairs, show_progress_bar=False))

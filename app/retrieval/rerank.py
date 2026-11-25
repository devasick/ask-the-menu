"""Reranker interface. MVP ships the no-op; a local cross-encoder slots in during Phase 2."""

from collections.abc import Sequence
from typing import Protocol, TypeVar


class HasContent(Protocol):
    @property
    def content(self) -> str: ...


T = TypeVar("T", bound=HasContent)


class Reranker(Protocol):
    async def rerank(self, question: str, candidates: Sequence[T], top_k: int) -> list[T]: ...


class NoOpReranker:
    """Keeps the fused retrieval order."""

    async def rerank(self, question: str, candidates: Sequence[T], top_k: int) -> list[T]:
        return list(candidates[:top_k])

"""Langfuse tracing. One trace per chat request; every LLM call inside it becomes a generation.

Tracing must never break a request, so every Langfuse call is best effort.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langfuse import Langfuse

from app.config import Settings
from app.llm.aliases import Alias
from app.llm.base import ChatResult, EmbedResult, LLMClient, Message, Usage

log = logging.getLogger(__name__)
_trace: ContextVar[Any | None] = ContextVar("langfuse_trace", default=None)


def make_langfuse(settings: Settings) -> Langfuse | None:
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )


@contextmanager
def request_trace(langfuse: Langfuse | None, **fields: Any) -> Iterator[Any | None]:
    """Opens a trace (or yields None when tracing is off) and makes it current for LLM calls."""
    trace = None
    if langfuse is not None:
        try:
            trace = langfuse.trace(**fields)
        except Exception:  # tracing is best effort
            log.debug("langfuse trace failed", exc_info=True)
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def _usage(usage: Usage) -> dict[str, Any]:
    return {
        "usage_details": {"input": usage.tokens_in, "output": usage.tokens_out},
        "cost_details": {"total": float(usage.cost_usd)},
    }


class TracingLLMClient:
    """Wraps any LLMClient; records each call on the current request trace."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def model_id(self, alias: Alias) -> str:
        return self._inner.model_id(alias)

    async def chat(
        self,
        alias: Alias,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        result = await self._inner.chat(alias, messages, json_schema, stream)
        self._record(f"chat {result.alias}", result.model, messages, result.text, result.usage)
        return result

    async def embed(self, alias: Alias, texts: list[str]) -> EmbedResult:
        result = await self._inner.embed(alias, texts)
        self._record(f"embed {alias}", result.model, {"texts": len(texts)}, None, result.usage)
        return result

    @staticmethod
    def _record(name: str, model: str, input: Any, output: Any, usage: Usage) -> None:
        trace = _trace.get()
        if trace is None:
            return
        try:
            trace.generation(name=name, model=model, input=input, output=output, **_usage(usage))
        except Exception:  # tracing is best effort
            log.debug("langfuse generation failed", exc_info=True)

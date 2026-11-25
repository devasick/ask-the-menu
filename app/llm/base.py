"""The one LLM interface the rest of the app uses. Provider code lives only in providers.py."""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol, TypedDict

from app.config import ModelPrice
from app.llm.aliases import Alias

log = logging.getLogger(__name__)
_MTOK = Decimal(1_000_000)


class Message(TypedDict):
    # A leading "system" message becomes the provider's system prompt.
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal


@dataclass(frozen=True)
class ChatResult:
    text: str
    alias: Alias  # the alias that actually answered (differs from the request after a fallback)
    model: str
    usage: Usage


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str
    usage: Usage


class LLMError(Exception):
    """The provider failed after retries and fallback."""


class LLMClient(Protocol):
    async def chat(
        self,
        alias: Alias,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> ChatResult: ...

    async def embed(self, alias: Alias, texts: list[str]) -> EmbedResult: ...

    def model_id(self, alias: Alias) -> str:
        """The concrete model behind an alias (used to re-embed when the model changes)."""
        ...


def compute_cost(
    prices: dict[str, ModelPrice], model: str, tokens_in: int, tokens_out: int
) -> Decimal:
    price = prices.get(model)
    if price is None:
        # Cost tracking and budgets under-count until the price is configured, so say so loudly.
        log.warning("no price configured for model %s; recording cost 0", model)
        return Decimal(0)
    return (tokens_in * price.input_per_mtok + tokens_out * price.output_per_mtok) / _MTOK

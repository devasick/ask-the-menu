"""Per-restaurant monthly LLM budget. Over budget, chat degrades to search-only mode
(rules parser, SQL route, keyword search, item cards) instead of failing."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QueryLog, Restaurant
from app.llm.aliases import Alias
from app.llm.base import ChatResult, EmbedResult, LLMClient, LLMError, Message


def month_start(now: datetime) -> datetime:
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


async def month_cost(
    session: AsyncSession, restaurant_id: uuid.UUID, now: datetime | None = None
) -> Decimal:
    """Chat cost (query_logs.cost_usd) since the start of the current UTC calendar month."""
    since = month_start(now or datetime.now(UTC))
    total = await session.scalar(
        select(func.coalesce(func.sum(QueryLog.cost_usd), 0)).where(
            QueryLog.restaurant_id == restaurant_id, QueryLog.created_at >= since
        )
    )
    return Decimal(total or 0)


async def over_budget(session: AsyncSession, restaurant: Restaurant) -> bool:
    return await month_cost(session, restaurant.id) >= restaurant.monthly_budget_usd


class SearchOnlyLLM:
    """Stands in for the real client when the budget is spent: every call fails fast, and each
    pipeline stage already falls back to its non-LLM path on LLMError."""

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
        raise LLMError("search-only mode: monthly budget reached")

    async def embed(self, alias: Alias, texts: list[str]) -> EmbedResult:
        raise LLMError("search-only mode: monthly budget reached")

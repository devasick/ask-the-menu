"""Tenant isolation: no route may ever return, or even query, another restaurant's data."""

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models import Chunk, MenuItem
from app.llm.fake import FakeLLMClient
from app.retrieval.query_parser import parse_query
from app.retrieval.rerank import NoOpReranker
from app.retrieval.router import RouteContext, parse_question, route_question

CURRY = uuid.UUID("11111111-1111-4111-8111-111111111111")
PIZZA = uuid.UUID("22222222-2222-4222-8222-222222222222")
TENANT_TABLES = re.compile(
    r"\b(FROM|JOIN)\s+(chunks|menu_items|faqs|answer_cache)\b", re.IGNORECASE
)

QUESTIONS = [
    "Chicken Tikka Pizza",  # exists only at the pizzeria, closely matches curry items
    "Pizza Diavola",
    "Tiramisu mit Mascarpone",
    "Vegane Pizza Verdure",
    "Wann habt ihr geöffnet?",
    "chicken tikka",
    "What's vegan and under €15?",
    "Was ist glutenfrei?",
    "show desserts",
    "Does the chicken korma contain nuts?",
    "ohne Nüsse",
    "Something spicy",
]


@contextmanager
def capture_sql(engine: AsyncEngine) -> Iterator[list[str]]:
    statements: list[str] = []

    def _record(conn: object, cursor: object, statement: str, *args: object) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _record)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _record)


async def _owners(
    maker: async_sessionmaker[AsyncSession], chunk_ids: list[uuid.UUID], item_ids: list[uuid.UUID]
) -> set[uuid.UUID]:
    from sqlalchemy import select

    async with maker() as session:
        owners = set(
            await session.scalars(select(Chunk.restaurant_id).where(Chunk.id.in_(chunk_ids)))
        )
        owners |= set(
            await session.scalars(select(MenuItem.restaurant_id).where(MenuItem.id.in_(item_ids)))
        )
    return owners


@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("tenant", [CURRY, PIZZA])
async def test_every_route_returns_only_own_data(
    seeded: async_sessionmaker[AsyncSession], engine: AsyncEngine, tenant: uuid.UUID, question: str
) -> None:
    llm = FakeLLMClient()
    parsed, _ = await parse_question(llm, question)
    with capture_sql(engine) as statements:
        async with seeded() as session:
            result = await route_question(
                RouteContext(session, llm, NoOpReranker(), tenant, parsed)
            )

    owners = await _owners(
        seeded, [c.chunk_id for c in result.chunks], [i.id for i in result.items]
    )
    assert owners <= {tenant}

    tenant_queries = [s for s in statements if TENANT_TABLES.search(s)]
    assert tenant_queries
    for sql in tenant_queries:
        assert "restaurant_id" in sql.split("WHERE", 1)[-1], sql


async def test_cross_tenant_dish_is_not_retrieved(seeded: async_sessionmaker[AsyncSession]) -> None:
    llm = FakeLLMClient()
    parsed = parse_query("Chicken Tikka Pizza mit Joghurt-Minz-Sauce")
    async with seeded() as session:
        result = await route_question(RouteContext(session, llm, NoOpReranker(), CURRY, parsed))
    names = {c.item.name for c in result.chunks if c.item}
    assert "Chicken Tikka Pizza" not in names
    assert all("pizza" not in c.content.lower() for c in result.chunks)


async def test_unknown_restaurant_gets_nothing(seeded: async_sessionmaker[AsyncSession]) -> None:
    llm = FakeLLMClient()
    parsed = parse_query("chicken")
    async with seeded() as session:
        result = await route_question(
            RouteContext(session, llm, NoOpReranker(), uuid.uuid4(), parsed)
        )
    assert result.chunks == [] and result.items == []

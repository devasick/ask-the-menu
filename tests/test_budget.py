"""Monthly budget: once a restaurant's chat cost reaches monthly_budget_usd, /chat answers in
search-only mode (no LLM calls) instead of failing."""

import json
import re
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import QueryLog
from app.llm.aliases import Alias
from app.llm.base import Message
from app.llm.fake import FakeLLMClient
from app.main import create_app
from app.security.budget import month_cost, month_start

CURRY = uuid.UUID("11111111-1111-4111-8111-111111111111")
KEY = {"X-Widget-Key": "demo-widget-key-curry-house"}


def cite_first_item(alias: Alias, messages: list[Message], schema: Any) -> str:
    ids = re.findall(r'<item id="([^"]+)"', messages[-1]["content"])
    return json.dumps({"answer": "Generated.", "cited_item_ids": ids[:1], "confidence": "high"})


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(responder=cite_first_item)


@pytest.fixture
def client(
    pg_url: str, llm: FakeLLMClient, seeded: async_sessionmaker[AsyncSession]
) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(_env_file=None, database_url=pg_url), llm=llm)) as c:
        yield c


async def _spend(
    sm: async_sessionmaker[AsyncSession], usd: str, at: datetime | None = None
) -> None:
    async with sm() as s, s.begin():
        s.add(
            QueryLog(
                restaurant_id=CURRY,
                question="x",
                route_taken="rag",
                latency_ms=1,
                cost_usd=Decimal(usd),
                **({"created_at": at} if at else {}),
            )
        )


def _ask(client: TestClient, question: str) -> Any:
    body = {"restaurant_id": str(CURRY), "session_id": "s1", "question": question}
    resp = client.post("/chat", json=body, headers=KEY)
    assert resp.status_code == 200
    return resp.json()


def test_month_start_is_first_of_month_utc() -> None:
    assert month_start(datetime(2026, 11, 26, 9, 30, tzinfo=UTC)) == datetime(
        2026, 11, 1, tzinfo=UTC
    )


async def test_month_cost_ignores_last_month(seeded: async_sessionmaker[AsyncSession]) -> None:
    await _spend(seeded, "4.50")
    await _spend(seeded, "99", at=month_start(datetime.now(UTC)) - timedelta(seconds=1))
    async with seeded() as s:
        assert await month_cost(s, CURRY) == Decimal("4.50")


async def test_under_budget_uses_the_model(
    client: TestClient, llm: FakeLLMClient, seeded: async_sessionmaker[AsyncSession]
) -> None:
    await _spend(seeded, "9.99")  # curry house budget is 10.00
    assert _ask(client, "Something light that goes with mango lassi?")["answer"] == "Generated."
    assert [c[0] for c in llm.calls] == ["embed", "chat"]


async def test_over_budget_answers_without_any_llm_call(
    client: TestClient, llm: FakeLLMClient, seeded: async_sessionmaker[AsyncSession]
) -> None:
    await _spend(seeded, "10.00")
    data = _ask(client, "Do you have butter chicken?")
    assert data["route"] == "rag"
    assert data["answer"] == "Here is what I found on the menu:"
    assert "cu-02" in [i["id"] for i in data["items"]]  # keyword search still finds it

    allergen = _ask(client, "Does the chicken korma contain nuts?")
    assert allergen["disclaimer"] and allergen["items"]

    sql = _ask(client, "Vegan curries?")
    assert sql["route"] == "sql" and sql["items"]

    nothing = _ask(client, "Wie wird das Wetter morgen?")
    assert nothing["items"] == [] and "Personal" in nothing["answer"]
    assert llm.calls == []

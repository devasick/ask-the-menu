"""Step 5: query logs on every /chat, GET /admin/stats, Langfuse tracing wrapper."""

import json
import re
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import ModelPrice, Settings
from app.db.models import QueryLog
from app.llm.aliases import Alias
from app.llm.base import Message
from app.llm.fake import FakeLLMClient
from app.llm.tracing import TracingLLMClient, make_langfuse, request_trace
from app.main import create_app

CURRY = "11111111-1111-4111-8111-111111111111"
PIZZA = "22222222-2222-4222-8222-222222222222"
ADMIN = {"X-API-Key": "test-admin-key"}


def cite_first_item(alias: Alias, messages: list[Message], schema: Any) -> str:
    ids = re.findall(r'<item id="([^"]+)"', messages[-1]["content"])
    return json.dumps({"answer": "Try this.", "cited_item_ids": ids[:1], "confidence": "high"})


@pytest.fixture
def client(pg_url: str, seeded: async_sessionmaker[AsyncSession]) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None,
        database_url=pg_url,
        admin_api_key=SecretStr("test-admin-key"),
        # The fake client reports model "fake-<alias>"; price it so cost is non-zero.
        model_prices={"fake-menu-fast": ModelPrice(input_per_mtok=1, output_per_mtok=5)},
    )
    llm = FakeLLMClient(responder=cite_first_item, prices=settings.model_prices)
    with TestClient(create_app(settings, llm=llm)) as c:
        yield c


def _ask(client: TestClient, question: str, restaurant: str = CURRY) -> Any:
    key = "curry-house" if restaurant == CURRY else "pizzeria-finta"
    resp = client.post(
        "/chat",
        json={"restaurant_id": restaurant, "session_id": "s1", "question": question},
        headers={"X-Widget-Key": f"demo-widget-key-{key}"},
    )
    assert resp.status_code == 200
    return resp.json()


async def _logs(sm: async_sessionmaker[AsyncSession]) -> list[QueryLog]:
    async with sm() as s:
        return list((await s.scalars(select(QueryLog).order_by(QueryLog.created_at))).all())


async def test_every_chat_writes_a_query_log(
    client: TestClient, seeded: async_sessionmaker[AsyncSession]
) -> None:
    _ask(client, "What's vegan and under €15?")
    rag = _ask(client, "Something light that goes with mango lassi?")
    sql_log, rag_log = await _logs(seeded)

    assert sql_log.route_taken == "sql"
    assert sql_log.tokens_in == sql_log.tokens_out == 0
    assert sql_log.cost_usd == 0
    assert sql_log.cited_item_ids and sql_log.has_result

    assert rag_log.route_taken == "rag"
    assert rag_log.cited_item_ids == [i["id"] for i in rag["items"]]
    assert rag_log.tokens_in > 0 and rag_log.tokens_out > 0
    assert rag_log.cost_usd > 0
    assert rag_log.latency_ms >= 0


def test_stats_requires_admin_key(client: TestClient) -> None:
    assert client.get("/admin/stats", params={"restaurant_id": CURRY}).status_code == 401
    params = {"restaurant_id": CURRY}
    assert client.get("/admin/stats", params=params, headers={"X-API-Key": "no"}).status_code == 401


def test_stats_aggregates_one_restaurant(client: TestClient) -> None:
    for _ in range(2):
        _ask(client, "What's vegan and under €15?")
    _ask(client, "  what's VEGAN and under €15? ")
    _ask(client, "Something light that goes with mango lassi?")
    _ask(client, "Was ist vegan?", restaurant=PIZZA)  # other tenant: must not count

    resp = client.get("/admin/stats", params={"restaurant_id": CURRY}, headers=ADMIN)
    data = resp.json()
    assert resp.status_code == 200
    assert data["questions"] == 4
    assert data["route_share"] == {"sql": 0.75, "rag": 0.25}
    assert data["top_questions"][0] == {"question": "what's vegan and under €15?", "count": 3}
    assert data["latency_p50_ms"] is not None and data["latency_p95_ms"] >= data["latency_p50_ms"]
    assert Decimal(data["total_cost_usd"]) > 0
    assert Decimal(data["avg_cost_usd"]) == Decimal(data["total_cost_usd"]) / 4
    assert "from" in data


def test_stats_reports_questions_without_result(client: TestClient) -> None:
    _ask(client, "What's vegan and under €2?")  # no items that cheap
    data = client.get("/admin/stats", params={"restaurant_id": CURRY}, headers=ADMIN).json()
    assert data["no_result_count"] == 1
    assert data["no_result_questions"][0]["question"] == "what's vegan and under €2?"


def test_stats_empty_window_and_bad_range(client: TestClient) -> None:
    params = {"restaurant_id": CURRY, "from": "2020-01-01T00:00:00Z", "to": "2020-01-02T00:00:00Z"}
    data = client.get("/admin/stats", params=params, headers=ADMIN).json()
    assert data["questions"] == 0 and data["route_share"] == {}
    assert data["latency_p50_ms"] is None and Decimal(data["avg_cost_usd"]) == 0
    params["from"] = "2020-01-03T00:00:00Z"
    assert client.get("/admin/stats", params=params, headers=ADMIN).status_code == 400


class FakeTrace:
    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []

    def generation(self, **kw: Any) -> None:
        self.generations.append(kw)


class FakeLangfuse:
    def __init__(self) -> None:
        self.trace_obj = FakeTrace()

    def trace(self, **kw: Any) -> FakeTrace:
        return self.trace_obj


async def test_tracing_records_generations_on_current_trace() -> None:
    llm = TracingLLMClient(FakeLLMClient())
    langfuse = FakeLangfuse()
    await llm.chat("menu-fast", [{"role": "user", "content": "outside any trace"}])
    with request_trace(langfuse, name="chat") as trace:
        await llm.chat("menu-fast", [{"role": "user", "content": "hi"}])
        await llm.embed("menu-embed", ["a", "b"])
    assert trace is langfuse.trace_obj
    names = [g["name"] for g in langfuse.trace_obj.generations]
    assert names == ["chat menu-fast", "embed menu-embed"]
    assert langfuse.trace_obj.generations[0]["usage_details"]["input"] > 0


def test_tracing_off_without_keys() -> None:
    assert make_langfuse(Settings(_env_file=None)) is None

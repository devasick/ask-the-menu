"""POST /chat end to end: widget key auth, SQL route, RAG with the allergen guardrail."""

import json
import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.llm.aliases import Alias
from app.llm.base import Message
from app.llm.fake import FakeLLMClient
from app.main import create_app

CURRY = "11111111-1111-4111-8111-111111111111"
PIZZA = "22222222-2222-4222-8222-222222222222"
CURRY_KEY = {"X-Widget-Key": "demo-widget-key-curry-house"}


def cite_first_item(alias: Alias, messages: list[Message], schema: Any) -> str:
    """Pretend model: recommends the first menu item in the prompt."""
    ids = re.findall(r'<item id="([^"]+)"', messages[-1]["content"])
    return json.dumps(
        {"answer": "Our pick for you.", "cited_item_ids": ids[:1], "confidence": "high"}
    )


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(responder=cite_first_item)


@pytest.fixture
def client(
    pg_url: str, llm: FakeLLMClient, seeded: async_sessionmaker[AsyncSession]
) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(_env_file=None, database_url=pg_url), llm=llm)) as c:
        yield c


def _ask(
    client: TestClient,
    question: str,
    headers: dict[str, str] | None = None,
    restaurant: str = CURRY,
) -> Any:
    body = {"restaurant_id": restaurant, "session_id": "s1", "question": question}
    return client.post("/chat", json=body, headers=CURRY_KEY if headers is None else headers)


def test_rejects_missing_wrong_or_foreign_widget_key(client: TestClient) -> None:
    assert _ask(client, "hi", headers={}).status_code == 401
    assert _ask(client, "hi", headers={"X-Widget-Key": "x" * 20}).status_code == 401
    assert _ask(client, "hi", restaurant=PIZZA).status_code == 401  # curry key, pizzeria id
    other = {"X-Widget-Key": "demo-widget-key-pizzeria-finta"}
    assert _ask(client, "hi", headers=other, restaurant=PIZZA).status_code == 200


def test_validates_question_length(client: TestClient) -> None:
    assert _ask(client, "x" * 501).status_code == 422


def test_pure_filter_is_answered_from_sql_without_llm(
    client: TestClient, llm: FakeLLMClient
) -> None:
    resp = _ask(client, "What's vegan and under €15?")
    data = resp.json()
    assert resp.status_code == 200
    assert data["route"] == "sql"
    assert data["items"] and all("vegan" in i["dietary_tags"] for i in data["items"])
    assert all(float(i["price"]) < 15 for i in data["items"])
    assert llm.calls == []


def test_rag_answer_cites_retrieved_items(client: TestClient) -> None:
    data = _ask(client, "Something light that goes with mango lassi?").json()
    assert data["route"] == "rag"
    assert data["answer"] == "Our pick for you."
    assert len(data["items"]) == 1
    assert data["disclaimer"] is None


def test_allergen_question_uses_stored_allergens(client: TestClient) -> None:
    data = _ask(client, "Does the chicken korma contain nuts?").json()
    assert data["route"] == "rag"
    assert data["items"][0]["id"] == "cu-03"
    assert data["answer"] == "According to our menu, Chicken Korma contains: milk, nuts."
    assert data["disclaimer"]


def test_unknown_allergens_tell_the_guest_to_ask_staff(client: TestClient) -> None:
    data = _ask(client, "Enthält Kheer Nüsse?").json()
    assert data["items"][0]["id"] == "de-03"
    assert data["items"][0]["allergens"] is None
    assert "keine Allergenangaben" in data["answer"]
    assert data["disclaimer"].startswith("Die Allergenangaben")


def test_allergen_exclusion_in_german_skips_unknown_items(client: TestClient) -> None:
    data = _ask(client, "Was ist ohne Nüsse?").json()
    assert data["route"] == "sql"
    assert data["answer"].startswith("Ich habe")
    assert all(i["allergens"] and "nuts" not in i["allergens"] for i in data["items"])
    assert data["disclaimer"]

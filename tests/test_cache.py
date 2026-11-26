"""Routes that answer without generation: FAQ match, exact cache, semantic cache, and cache
invalidation on ingest."""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.db.models import AnswerCache
from app.ingest.pipeline import ingest_menu
from app.llm.aliases import Alias
from app.llm.base import Message
from app.llm.fake import FakeLLMClient
from app.main import create_app
from app.retrieval.answer_cache import question_key
from app.seed import load_payload

ROOT = Path(__file__).resolve().parent.parent
CURRY = "11111111-1111-4111-8111-111111111111"
KEY = {"X-Widget-Key": "demo-widget-key-curry-house"}
Maker = async_sessionmaker[AsyncSession]


def confident(alias: Alias, messages: list[Message], schema: Any) -> str:
    if schema is None or "<menu>" not in messages[-1]["content"]:
        return "{}"
    ids = re.findall(r'<item id="([^"]+)"', messages[-1]["content"])
    return json.dumps({"answer": "Try this one.", "cited_item_ids": ids[:1], "confidence": "high"})


def unsure(alias: Alias, messages: list[Message], schema: Any) -> str:
    if schema is None:
        return "{}"
    return json.dumps({"answer": "Not sure.", "cited_item_ids": [], "confidence": "low"})


def generations(llm: FakeLLMClient) -> int:
    return sum(kind == "chat" for kind, _, _ in llm.calls)


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(responder=confident)


@pytest.fixture
def client(pg_url: str, llm: FakeLLMClient, seeded: Maker) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(_env_file=None, database_url=pg_url), llm=llm)) as c:
        yield c


def ask(client: TestClient, question: str, language: str | None = None) -> Any:
    body = {"restaurant_id": CURRY, "session_id": "s1", "question": question, "language": language}
    resp = client.post("/chat", json=body, headers=KEY)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_question_key_ignores_case_spacing_and_end_punctuation() -> None:
    assert question_key("  Something MILD   with lassi?! ") == "something mild with lassi"
    assert question_key("ohne Nüsse") != question_key("mit Nüssen")


def test_faq_question_gets_the_restaurants_answer(client: TestClient, llm: FakeLLMClient) -> None:
    data = ask(client, "What are your opening hours?")
    assert data["route"] == "faq"
    assert data["answer"].startswith("We are open Tuesday to Sunday")
    assert data["items"] == [] and data["sources"][0]["type"] == "faq"
    assert generations(llm) == 0


def test_faq_needs_a_close_match(client: TestClient) -> None:
    assert ask(client, "Do you deliver to my address?")["route"] == "rag"


def test_faq_is_not_answered_in_another_language(client: TestClient) -> None:
    # Same words, so the vectors match, but the guest wants German and the FAQ is English.
    assert ask(client, "What are your opening hours?", language="de")["route"] != "faq"


def test_repeat_question_comes_from_the_exact_cache(client: TestClient, llm: FakeLLMClient) -> None:
    first = ask(client, "Something light that goes with mango lassi?")
    assert first["route"] == "rag" and first["items"]
    calls = generations(llm)
    again = ask(client, "something light that goes with MANGO LASSI")
    assert again["route"] == "exact_cache"
    assert generations(llm) == calls
    assert {k: again[k] for k in ("answer", "items")} == {k: first[k] for k in ("answer", "items")}


def test_reworded_question_comes_from_the_semantic_cache(client: TestClient) -> None:
    ask(client, "Something light that goes with mango lassi?")
    # Same words in another order: identical bag-of-words vector offline.
    assert ask(client, "With mango lassi, something light that goes?")["route"] == "semantic_cache"


def test_allergen_and_filter_questions_never_match_semantically(client: TestClient) -> None:
    ask(client, "Which curry goes well without nuts?")
    assert ask(client, "Which curry goes well with nuts?")["route"] == "rag"
    assert ask(client, "which curry goes well without nuts")["route"] == "exact_cache"


def test_low_confidence_answers_are_not_cached(pg_url: str, seeded: Maker) -> None:
    llm = FakeLLMClient(responder=unsure)
    with TestClient(create_app(Settings(_env_file=None, database_url=pg_url), llm=llm)) as c:
        assert ask(c, "Something light that goes with mango lassi?")["route"] == "rag"
        assert ask(c, "Something light that goes with mango lassi?")["route"] == "rag"


async def test_menu_change_invalidates_the_cache(client: TestClient, seeded: Maker) -> None:
    ask(client, "Something light that goes with mango lassi?")
    async with seeded() as s:
        assert await s.scalar(select(func.count()).select_from(AnswerCache)) == 1

    payload = load_payload(ROOT / "seed/curry_house.json")
    item = payload.items[0].model_copy(update={"price_cents": payload.items[0].price_cents + 50})
    changed = payload.model_copy(update={"items": [item, *payload.items[1:]]})
    async with seeded() as s, s.begin():
        await ingest_menu(s, FakeLLMClient(), changed)
    async with seeded() as s:
        assert await s.scalar(select(func.count()).select_from(AnswerCache)) == 0

    assert ask(client, "Something light that goes with mango lassi?")["route"] == "rag"

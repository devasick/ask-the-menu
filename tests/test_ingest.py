"""POST /ingest end to end against Postgres with the fake LLM client."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.models import EMBEDDING_DIM
from app.llm.fake import FakeLLMClient
from app.main import create_app

SEED = Path(__file__).resolve().parent.parent / "seed"
KEY = "test-ingest-key"
HEADERS = {"X-API-Key": KEY}


def _load(name: str) -> dict[str, Any]:
    import json

    data: dict[str, Any] = json.loads((SEED / name).read_text(encoding="utf-8"))
    return data


CURRY = _load("curry_house.json")
PIZZA = _load("pizzeria_finta.json")
CURRY_ID = CURRY["restaurant"]["id"]
PIZZA_ID = PIZZA["restaurant"]["id"]


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def client(pg_url: str, llm: FakeLLMClient, sessionmaker: object) -> Iterator[TestClient]:
    # `sessionmaker` is requested only for its teardown (truncate after each test).
    settings = Settings(_env_file=None, database_url=pg_url, ingest_api_key=SecretStr(KEY))
    with TestClient(create_app(settings, llm=llm)) as c:
        yield c


def _ingest(client: TestClient, payload: dict[str, Any], rid: str | None = None) -> Any:
    rid = rid or payload["restaurant"]["id"]
    resp = client.post(f"/ingest/{rid}", json=payload, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _embed_calls(llm: FakeLLMClient) -> list[list[str]]:
    return [texts for kind, _, texts in llm.calls if kind == "embed"]


def test_rejects_missing_or_wrong_key(client: TestClient) -> None:
    assert client.post(f"/ingest/{CURRY_ID}", json=CURRY).status_code == 401
    bad = {"X-API-Key": "nope"}
    assert client.post(f"/ingest/{CURRY_ID}", json=CURRY, headers=bad).status_code == 401


def test_rejects_restaurant_id_mismatch(client: TestClient) -> None:
    resp = client.post(f"/ingest/{PIZZA_ID}", json=CURRY, headers=HEADERS)
    assert resp.status_code == 400


def test_rejects_invalid_payload(client: TestClient) -> None:
    bad = {**CURRY, "items": [{**CURRY["items"][0], "allergens": ["peanut"]}]}
    assert client.post(f"/ingest/{CURRY_ID}", json=bad, headers=HEADERS).status_code == 422


async def test_first_ingest_embeds_everything(
    client: TestClient, llm: FakeLLMClient, engine: AsyncEngine
) -> None:
    result = _ingest(client, CURRY)
    expected_chunks = len(CURRY["items"]) + len(CURRY["faqs"]) + 1  # + info chunk
    assert result["menu_version"] == 1
    assert result["chunks_embedded"] == expected_chunks
    # + each FAQ question on its own, for the FAQ route
    assert sum(len(t) for t in _embed_calls(llm)) == expected_chunks + len(CURRY["faqs"])

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT source_type, count(*), min(vector_dims(embedding)), "
                    "bool_and(tsv IS NOT NULL) FROM chunks WHERE restaurant_id = :r GROUP BY 1"
                ),
                {"r": CURRY_ID},
            )
        ).all()
    by_type = {r[0]: r[1:] for r in rows}
    assert by_type["menu_item"][0] == len(CURRY["items"])
    assert by_type["faq"][0] == len(CURRY["faqs"])
    assert by_type["info"][0] == 1
    assert all(dims == EMBEDDING_DIM and has_tsv for _, dims, has_tsv in by_type.values())


def test_same_payload_twice_changes_nothing(client: TestClient, llm: FakeLLMClient) -> None:
    _ingest(client, CURRY)
    calls_before = len(_embed_calls(llm))
    again = _ingest(client, CURRY)
    assert again["menu_version"] == 1
    assert again["chunks_embedded"] == 0
    assert again["items_changed"] == 0
    assert len(_embed_calls(llm)) == calls_before  # no embedding API call at all


def test_change_reembeds_only_that_item_and_bumps_version(
    client: TestClient, llm: FakeLLMClient
) -> None:
    _ingest(client, CURRY)
    data = {**CURRY, "items": [dict(i) for i in CURRY["items"]]}
    korma = next(i for i in data["items"] if i["external_id"] == "cu-03")
    korma["price_cents"] = 1600

    result = _ingest(client, data)
    assert result["menu_version"] == 2
    assert result["items_changed"] == 1
    assert result["chunks_embedded"] == 1
    assert "Price: 16.00 EUR" in _embed_calls(llm)[-1][0]


def test_availability_change_bumps_version_without_reembedding(client: TestClient) -> None:
    _ingest(client, CURRY)
    data = {**CURRY, "items": [dict(i) for i in CURRY["items"]]}
    data["items"][0]["is_available"] = False
    result = _ingest(client, data)
    assert result["items_changed"] == 1
    assert result["chunks_embedded"] == 0
    assert result["menu_version"] == 2  # caches must still be invalidated


async def test_removed_item_and_faq_lose_their_chunks(
    client: TestClient, engine: AsyncEngine
) -> None:
    _ingest(client, CURRY)
    data = {**CURRY, "items": CURRY["items"][1:], "faqs": CURRY["faqs"][1:]}
    result = _ingest(client, data)
    assert result["items_deleted"] == 1
    assert result["faqs_deleted"] == 1
    async with engine.connect() as conn:
        count = await conn.scalar(
            text("SELECT count(*) FROM chunks WHERE restaurant_id = :r"), {"r": CURRY_ID}
        )
    assert count == len(data["items"]) + len(data["faqs"]) + 1


async def test_ingest_never_touches_another_restaurant(
    client: TestClient, engine: AsyncEngine
) -> None:
    _ingest(client, CURRY)
    _ingest(client, PIZZA)
    empty = {**PIZZA, "items": [], "faqs": [], "restaurant": {**PIZZA["restaurant"], "info": ""}}
    result = _ingest(client, empty)
    assert result["items_deleted"] == len(PIZZA["items"])

    async with engine.connect() as conn:
        counts: dict[str, int] = dict(
            (
                await conn.execute(
                    text("SELECT restaurant_id::text, count(*) FROM chunks GROUP BY 1")
                )
            ).all()
        )
        curry_version = await conn.scalar(
            text("SELECT menu_version FROM restaurants WHERE id = :r"), {"r": CURRY_ID}
        )
    assert PIZZA_ID not in counts
    assert counts[CURRY_ID] == len(CURRY["items"]) + len(CURRY["faqs"]) + 1
    assert curry_version == 1


def test_slug_taken_by_another_restaurant_is_a_conflict(client: TestClient) -> None:
    _ingest(client, CURRY)
    other = {**PIZZA, "restaurant": {**PIZZA["restaurant"], "slug": CURRY["restaurant"]["slug"]}}
    resp = client.post(f"/ingest/{PIZZA_ID}", json=other, headers=HEADERS)
    assert resp.status_code == 409

"""SQL route, hybrid search and the route chain against the ingested seed menus."""

import json
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.base import Message
from app.llm.fake import FakeLLMClient, fake_embedding
from app.retrieval.hybrid import hybrid_search
from app.retrieval.query_parser import parse_query
from app.retrieval.rerank import NoOpReranker
from app.retrieval.router import RouteContext, RouteResult, parse_question, route_question
from app.retrieval.sql_search import search_items

CURRY = uuid.UUID("11111111-1111-4111-8111-111111111111")
Maker = async_sessionmaker[AsyncSession]


async def _route(maker: Maker, question: str, llm: FakeLLMClient | None = None) -> RouteResult:
    llm = llm or FakeLLMClient()
    parsed, _ = await parse_question(llm, question)
    async with maker() as session:
        ctx = RouteContext(session, llm, NoOpReranker(), CURRY, parsed)
        return await route_question(ctx)


async def test_pure_filter_goes_to_sql_without_any_llm_call(seeded: Maker) -> None:
    llm = FakeLLMClient()
    result = await _route(seeded, "What's vegan and under €15?", llm)
    assert result.route == "sql"
    assert llm.calls == []  # no embedding, no generation
    assert result.items
    assert all("vegan" in i.dietary_tags and i.price_cents < 1500 for i in result.items)
    assert all(i.is_available for i in result.items)
    assert result.answer and "vegan" in result.answer


async def test_german_filter_answer_is_german(seeded: Maker) -> None:
    result = await _route(seeded, "Zeig mir vegetarische Gerichte unter 12 €")
    assert result.route == "sql"
    assert result.answer is not None and result.answer.startswith("Ich habe")


async def test_allergen_exclusion_drops_unknown_allergen_items(seeded: Maker) -> None:
    async with seeded() as session:
        items = await search_items(session, CURRY, parse_query("ohne Nüsse").filters)
    ids = {i.external_id for i in items}
    assert items and all(i.allergens is not None and "nuts" not in i.allergens for i in items)
    assert "cu-03" not in ids  # korma has nuts
    assert not ids & {"cs-07", "cu-12", "cu-14", "de-03"}  # allergens unknown


async def test_sql_route_hides_sold_out_items(seeded: Maker) -> None:
    async with seeded() as session:
        items = await search_items(session, CURRY, parse_query("show curries").filters, 50)
    assert items and "cu-06" not in {i.external_id for i in items}  # Goan Fish Curry, sold out


async def test_empty_filter_result_says_so(seeded: Maker) -> None:
    result = await _route(seeded, "vegan desserts under €2")
    assert result.route == "sql"
    assert result.items == []
    assert result.answer is not None and "nothing" in result.answer


async def test_hybrid_finds_the_dish_asked_about(seeded: Maker) -> None:
    result = await _route(seeded, "Does the chicken korma contain nuts?")
    assert result.route == "rag"
    assert result.chunks[0].item is not None
    assert result.chunks[0].item.external_id == "cu-03"
    assert len(result.chunks) <= 5


async def test_hybrid_retrieves_faqs(seeded: Maker) -> None:
    # Not close enough to "Do you deliver?" for the FAQ route, so retrieval answers.
    result = await _route(seeded, "Do you deliver to my address?")
    assert result.route == "rag"
    assert any(c.source_type == "faq" and "deliver" in c.content for c in result.chunks)


async def test_hybrid_applies_filters_as_where(seeded: Maker) -> None:
    result = await _route(seeded, "Something spicy that goes with mango lassi?")
    assert result.route == "rag"
    assert result.chunks
    assert all(c.item is not None and "spicy" in c.item.dietary_tags for c in result.chunks)


async def test_rag_can_see_sold_out_items_to_say_so(seeded: Maker) -> None:
    parsed = parse_query("goan fish curry with coconut")
    async with seeded() as session:
        vector = fake_embedding(parsed.question)
        chunks = await hybrid_search(session, NoOpReranker(), CURRY, parsed, vector)
    goan = next(c for c in chunks if c.item and c.item.external_id == "cu-06")
    assert goan.item is not None and goan.item.is_available is False


async def test_llm_fallback_parser_only_runs_when_rules_fail(seeded: Maker) -> None:
    def responder(alias: str, messages: list[Message], schema: dict[str, Any] | None) -> str:
        assert schema is not None and "<question>" in messages[-1]["content"]
        return json.dumps({"max_price_eur": 8, "dietary": ["vegan"], "exclude_allergens": []})

    llm = FakeLLMClient(responder=responder)
    parsed, _ = await parse_question(llm, "Something cheap?")
    assert parsed.filters.max_price_cents == 800
    assert parsed.filters.dietary == {"vegan"}
    assert [c[0] for c in llm.calls] == ["chat"]

    llm2 = FakeLLMClient(responder=responder)
    await parse_question(llm2, "vegan under €15")
    assert llm2.calls == []


async def test_llm_fallback_ignores_garbage(seeded: Maker) -> None:
    llm = FakeLLMClient(responder=lambda *_: "not json")
    parsed, _ = await parse_question(llm, "Something cheap?")
    assert parsed.filters.max_price_cents is None

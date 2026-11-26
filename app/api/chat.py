import asyncio
import hmac
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.ingest import get_llm
from app.config import Settings
from app.db.models import QueryLog, Restaurant
from app.db.session import get_session
from app.generation.generate import ChatAnswer, add_usage, answer_question, remember_answer
from app.generation.stream import OnDelta
from app.ingest.schemas import has_allergen_data
from app.llm.base import LLMClient, Usage
from app.llm.tracing import request_trace
from app.retrieval.rerank import Reranker
from app.retrieval.router import RouteContext, parse_question, route_question
from app.retrieval.sql_search import ItemHit
from app.security.auth import hash_widget_key
from app.security.budget import SearchOnlyLLM, over_budget
from app.security.ratelimit import RateLimiter

log = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    restaurant_id: uuid.UUID
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=500)
    language: Literal["en", "de"] | None = None


class ItemCard(BaseModel):
    id: str  # MenuNow's item ID (external_id)
    name: str
    price: Decimal
    currency: str
    allergens: list[str] | None  # None = unknown, ask staff
    dietary_tags: list[str]
    available: bool

    @classmethod
    def of(cls, item: ItemHit) -> "ItemCard":
        return cls(
            id=item.external_id,
            name=item.name,
            price=Decimal(item.price_cents) / 100,
            currency=item.currency,
            allergens=item.allergens if has_allergen_data(item.allergens) else None,
            dietary_tags=item.dietary_tags,
            available=item.is_available,
        )


class SourceOut(BaseModel):
    type: str
    id: str


class ChatResponse(BaseModel):
    answer: str
    items: list[ItemCard]
    route: str
    sources: list[SourceOut]
    disclaimer: str | None = None

    @classmethod
    def of(cls, a: ChatAnswer) -> "ChatResponse":
        return cls(
            answer=a.answer,
            items=[ItemCard.of(i) for i in a.items],
            route=a.route,
            sources=[SourceOut(type=s.type, id=s.id) for s in a.sources],
            disclaimer=a.disclaimer,
        )


async def _authorized_restaurant(
    session: AsyncSession, restaurant_id: uuid.UUID, widget_key: str | None
) -> Restaurant:
    restaurant = await session.get(Restaurant, restaurant_id)
    expected = restaurant.widget_key_hash if restaurant else None
    # Same 401 for unknown restaurant, missing key and wrong key.
    if (
        restaurant is None
        or expected is None
        or widget_key is None
        or not hmac.compare_digest(hash_widget_key(widget_key), expected)
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid widget key")
    return restaurant


async def _admit(
    request: Request,
    session: AsyncSession,
    body: ChatRequest,
    widget_key: str | None,
    llm: LLMClient,
) -> tuple[Restaurant, LLMClient]:
    """Auth, rate limit and budget, before any work. Returns the LLM client to use: the real
    one, or a search-only stand-in once the restaurant's monthly budget is spent."""
    restaurant = await _authorized_restaurant(session, body.restaurant_id, widget_key)
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is not None:
        ip = request.client.host if request.client else None
        wait = await limiter.retry_after(restaurant.id, body.session_id, ip)
        if wait is not None:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "too many questions, please try again later",
                headers={"Retry-After": str(wait)},
            )
    if await over_budget(session, restaurant):
        llm = SearchOnlyLLM(llm)
    return restaurant, llm


async def _answer(
    request: Request,
    session: AsyncSession,
    llm: LLMClient,
    restaurant: Restaurant,
    body: ChatRequest,
    on_delta: OnDelta | None = None,
) -> ChatResponse:
    """The whole pipeline for one question, traced and logged."""
    started = time.perf_counter()
    reranker: Reranker = request.app.state.reranker
    settings: Settings = request.app.state.settings
    trace_fields = {
        "name": "chat",
        "session_id": body.session_id,
        "input": body.question,
        "metadata": {"restaurant_id": str(restaurant.id), "stream": on_delta is not None},
    }
    with request_trace(request.app.state.langfuse, **trace_fields) as trace:
        parsed, parse_usage = await parse_question(llm, body.question, body.language)
        ctx = RouteContext(
            session,
            llm,
            reranker,
            restaurant.id,
            parsed,
            menu_version=restaurant.menu_version,
            faq_min_similarity=settings.faq_min_similarity,
            semantic_cache_min_similarity=settings.semantic_cache_min_similarity,
        )
        result = await route_question(ctx)
        answer = await answer_question(llm, settings, restaurant.name, result, on_delta)
        await remember_answer(ctx, answer)  # committed with the query log
        response = ChatResponse.of(answer)
        if trace is not None:
            try:
                trace.update(output=response.answer, metadata={"route": answer.route})
            except Exception:  # tracing is best effort
                log.debug("langfuse trace update failed", exc_info=True)
    latency_ms = round((time.perf_counter() - started) * 1000)
    await _log_query(session, restaurant.id, body.question, answer, parse_usage, latency_ms)
    return response


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    x_widget_key: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    restaurant, llm = await _admit(request, session, body, x_widget_key, llm)
    return await _answer(request, session, llm, restaurant, body)


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", response_class=StreamingResponse)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    x_widget_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Server-sent events: `delta` {"text"} while the answer is written, then `done` with the
    same body as POST /chat (authoritative: guardrails may change the text), or `error`.
    Auth, validation and rate limit errors are plain HTTP errors before the stream starts."""
    sessionmaker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with sessionmaker() as session:
        restaurant, llm = await _admit(request, session, body, x_widget_key, llm)

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def on_delta(text: str) -> None:
            await queue.put(_sse("delta", {"text": text}))

        async def work() -> None:
            try:
                async with sessionmaker() as session:
                    response = await _answer(request, session, llm, restaurant, body, on_delta)
                await queue.put(_sse("done", response.model_dump(mode="json")))
            except Exception:
                log.exception("chat stream failed")
                await queue.put(_sse("error", {"detail": "something went wrong"}))
            finally:
                await queue.put(None)

        task = asyncio.create_task(work())
        try:
            while (item := await queue.get()) is not None:
                yield item
        finally:
            task.cancel()  # guest disconnected: stop generating (no-op when already done)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _log_query(
    session: AsyncSession,
    restaurant_id: uuid.UUID,
    question: str,
    answer: ChatAnswer,
    parse_usage: Usage,
    latency_ms: int,
) -> None:
    """Best effort: a failed analytics write must not fail the guest's answer."""
    usage = add_usage(parse_usage, answer.usage)
    session.add(
        QueryLog(
            restaurant_id=restaurant_id,
            question=question,
            route_taken=answer.route,
            cited_item_ids=[i.external_id for i in answer.items],
            has_result=bool(answer.items) or answer.confidence == "high",
            latency_ms=latency_ms,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=usage.cost_usd,
        )
    )
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        log.exception("query log write failed")

import hmac
import uuid
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.ingest import get_llm
from app.config import Settings
from app.db.models import Restaurant
from app.db.session import get_session
from app.generation.generate import ChatAnswer, answer_question
from app.ingest.schemas import has_allergen_data
from app.llm.base import LLMClient
from app.retrieval.rerank import Reranker
from app.retrieval.router import RouteContext, parse_question, route_question
from app.retrieval.sql_search import ItemHit
from app.security.auth import get_settings, hash_widget_key

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


def get_reranker(request: Request) -> Reranker:
    reranker: Reranker = request.app.state.reranker
    return reranker


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


@router.post("/chat")
async def chat(
    body: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    llm: Annotated[LLMClient, Depends(get_llm)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_widget_key: Annotated[str | None, Header()] = None,
) -> ChatResponse:
    restaurant = await _authorized_restaurant(session, body.restaurant_id, x_widget_key)
    # parse_usage is recorded with the query log (step 5).
    parsed, _parse_usage = await parse_question(llm, body.question, body.language)
    result = await route_question(RouteContext(session, llm, reranker, restaurant.id, parsed))
    answer = await answer_question(llm, settings, restaurant.name, result)
    return ChatResponse.of(answer)

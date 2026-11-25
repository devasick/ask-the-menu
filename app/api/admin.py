"""Admin analytics. Every query is scoped to one restaurant_id and a time window."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import QueryLog
from app.db.session import get_session
from app.security.auth import require_admin_key

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_key)])

DEFAULT_WINDOW = timedelta(days=30)
TOP_N = 10


class QuestionCount(BaseModel):
    question: str
    count: int


class Stats(BaseModel):
    restaurant_id: uuid.UUID
    from_: datetime = Field(serialization_alias="from")
    to: datetime
    questions: int
    route_share: dict[str, float]
    top_questions: list[QuestionCount]
    no_result_questions: list[QuestionCount]
    no_result_count: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    tokens_in: int
    tokens_out: int
    total_cost_usd: Decimal
    avg_cost_usd: Decimal


async def _top_questions(
    session: AsyncSession, where: list[ColumnElement[bool]]
) -> list[QuestionCount]:
    # Group case and whitespace variants of the same question together.
    q = func.lower(func.btrim(QueryLog.question))
    rows = await session.execute(
        select(q, func.count())
        .where(*where)
        .group_by(q)
        .order_by(func.count().desc(), q)
        .limit(TOP_N)
    )
    return [QuestionCount(question=text, count=n) for text, n in rows]


@router.get("/stats")
async def stats(
    session: Annotated[AsyncSession, Depends(get_session)],
    restaurant_id: uuid.UUID,
    from_: Annotated[datetime | None, Query(alias="from")] = None,
    to: datetime | None = None,
) -> Stats:
    to = to or datetime.now(UTC)
    from_ = from_ or to - DEFAULT_WINDOW
    if from_ >= to:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "'from' must be before 'to'")
    where = [
        QueryLog.restaurant_id == restaurant_id,
        QueryLog.created_at >= from_,
        QueryLog.created_at < to,
    ]
    totals = (
        await session.execute(
            select(
                func.count(),
                func.percentile_cont(0.5).within_group(QueryLog.latency_ms),
                func.percentile_cont(0.95).within_group(QueryLog.latency_ms),
                func.coalesce(func.sum(QueryLog.tokens_in), 0),
                func.coalesce(func.sum(QueryLog.tokens_out), 0),
                func.coalesce(func.sum(QueryLog.cost_usd), 0),
                func.count().filter(QueryLog.has_result.is_(False)),
            ).where(*where)
        )
    ).one()
    count, p50, p95, tokens_in, tokens_out, cost, no_result = totals
    routes = await session.execute(
        select(QueryLog.route_taken, func.count()).where(*where).group_by(QueryLog.route_taken)
    )
    return Stats(
        restaurant_id=restaurant_id,
        from_=from_,
        to=to,
        questions=count,
        route_share={route: n / count for route, n in routes} if count else {},
        top_questions=await _top_questions(session, where),
        no_result_questions=await _top_questions(session, [*where, ~QueryLog.has_result]),
        no_result_count=no_result,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_cost_usd=Decimal(cost),
        avg_cost_usd=Decimal(cost) / count if count else Decimal(0),
    )

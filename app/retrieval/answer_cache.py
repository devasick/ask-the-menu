"""Answer cache in Postgres: exact (same normalized question) and semantic (pgvector).

Entries are keyed by the restaurant's menu_version, so an ingest that changes anything makes
older answers unreachable; ingest also deletes them. Only confident generated answers are
stored. Questions with filters or allergens get no embedding, so they only ever match exactly:
"dishes with nuts" and "dishes without nuts" embed close together but need opposite answers.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AnswerCache, MenuItem
from app.retrieval.query_parser import normalize_question
from app.retrieval.sql_search import ITEM_COLUMNS, ItemHit

_TRAILING = re.compile(r"[\s?!.¿¡]+$")


def question_key(question: str) -> str:
    """Case, whitespace and trailing punctuation don't change the answer."""
    return _TRAILING.sub("", normalize_question(question))


@dataclass(frozen=True)
class CacheKey:
    restaurant_id: uuid.UUID
    menu_version: int
    language: str


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    items: list[ItemHit]


async def _hydrate(
    session: AsyncSession, key: CacheKey, answer: str, ids: list[str]
) -> CachedAnswer:
    """Item cards are re-read: same menu_version means the same data."""
    rows = await session.execute(
        select(*ITEM_COLUMNS).where(
            MenuItem.restaurant_id == key.restaurant_id, MenuItem.external_id.in_(ids)
        )
    )
    by_id = {r.external_id: ItemHit(*r) for r in rows}
    return CachedAnswer(answer, [by_id[i] for i in ids if i in by_id])


async def exact_match(session: AsyncSession, key: CacheKey, question: str) -> CachedAnswer | None:
    row = (
        await session.execute(
            select(AnswerCache.answer, AnswerCache.item_ids).where(
                AnswerCache.restaurant_id == key.restaurant_id,
                AnswerCache.menu_version == key.menu_version,
                AnswerCache.language == key.language,
                AnswerCache.question_key == question_key(question),
            )
        )
    ).first()
    return None if row is None else await _hydrate(session, key, row.answer, row.item_ids)


async def semantic_match(
    session: AsyncSession, key: CacheKey, vector: list[float], min_similarity: float
) -> CachedAnswer | None:
    # shortcut: exact scan over one restaurant's entries for this menu version, no ANN index
    # (a filtered HNSW scan can miss rows); add a partial index if one menu gets >100k entries.
    distance = AnswerCache.embedding.cosine_distance(vector)
    row = (
        await session.execute(
            select(AnswerCache.answer, AnswerCache.item_ids, distance.label("distance"))
            .where(
                AnswerCache.restaurant_id == key.restaurant_id,
                AnswerCache.menu_version == key.menu_version,
                AnswerCache.language == key.language,
                AnswerCache.embedding.isnot(None),
            )
            .order_by(distance)
            .limit(1)
        )
    ).first()
    if row is None or 1 - row.distance < min_similarity:
        return None
    return await _hydrate(session, key, row.answer, row.item_ids)


async def store(
    session: AsyncSession,
    key: CacheKey,
    question: str,
    vector: list[float] | None,
    answer: str,
    item_ids: list[str],
) -> None:
    """Adds to the caller's transaction. A concurrent identical question keeps the first."""
    await session.execute(
        insert(AnswerCache)
        .values(
            id=uuid.uuid4(),
            restaurant_id=key.restaurant_id,
            menu_version=key.menu_version,
            language=key.language,
            question_key=question_key(question),
            embedding=vector,
            answer=answer,
            item_ids=item_ids,
        )
        .on_conflict_do_nothing(constraint="uq_answer_cache_key")
    )

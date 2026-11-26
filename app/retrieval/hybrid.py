"""Hybrid retrieval: pgvector cosine + Postgres full text, fused with Reciprocal Rank Fusion.

Both legs filter by restaurant_id and apply the structured filters as SQL WHERE clauses, so a
filtered-out or foreign item can never reach the reranker or the prompt.
"""

import logging
import re
import uuid
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, MenuItem
from app.retrieval.query_parser import FILLER, ParsedQuery, normalize_question
from app.retrieval.rerank import Reranker
from app.retrieval.sql_search import ITEM_COLUMNS, ItemHit, item_filter_clauses

log = logging.getLogger(__name__)
K = TypeVar("K", bound=Hashable)
RRF_K = 60
CANDIDATES = 20
TOP_K = 5
_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    source_type: str
    content: str
    score: float
    item: ItemHit | None  # set for menu_item chunks


def rrf_merge[K: Hashable](*rankings: Sequence[K], k: int = RRF_K) -> list[tuple[K, float]]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1. Stable on ties."""
    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def build_tsquery(terms: Sequence[str]) -> str | None:
    """OR-query of sanitized tokens; prefix match on longer ones ("korma" -> korma:*)."""
    tokens = sorted({t for term in terms for t in _TOKEN.findall(term.lower()) if len(t) > 1})
    if not tokens:
        return None
    return " | ".join(f"{t}:*" if len(t) >= 4 else t for t in tokens)


def _scoped(restaurant_id: uuid.UUID, parsed: ParsedQuery) -> Select[Any]:
    stmt = select(Chunk.id).outerjoin(
        MenuItem,
        and_(MenuItem.id == Chunk.menu_item_id, MenuItem.restaurant_id == restaurant_id),
    )
    stmt = stmt.where(Chunk.restaurant_id == restaurant_id)
    if parsed.filters.any_item_filter():
        stmt = stmt.where(Chunk.source_type == "menu_item", *item_filter_clauses(parsed.filters))
    return stmt


async def hybrid_search(
    session: AsyncSession,
    reranker: Reranker,
    restaurant_id: uuid.UUID,
    parsed: ParsedQuery,
    vector: list[float] | None,
    candidates: int = CANDIDATES,
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    """`vector` is the question's embedding; without one only the keyword leg runs."""
    vector_ids: list[uuid.UUID] = []
    if vector is not None:
        vector_ids = list(
            await session.scalars(
                _scoped(restaurant_id, parsed)
                .where(Chunk.embedding.isnot(None))
                .order_by(Chunk.embedding.cosine_distance(vector))
                .limit(candidates)
            )
        )

    keyword_ids: list[uuid.UUID] = []
    terms = parsed.semantic_terms or tuple(
        w for w in _TOKEN.findall(normalize_question(parsed.question)) if w not in FILLER
    )
    tsquery = build_tsquery(terms)
    if tsquery:
        q = func.to_tsquery("simple", tsquery)
        keyword_ids = list(
            await session.scalars(
                _scoped(restaurant_id, parsed)
                .where(Chunk.tsv.op("@@")(q))
                .order_by(func.ts_rank_cd(Chunk.tsv, q).desc(), Chunk.id)
                .limit(candidates)
            )
        )

    fused = rrf_merge(vector_ids, keyword_ids)[:candidates]
    chunks = await _load(session, restaurant_id, [cid for cid, _ in fused])
    ranked = [
        RetrievedChunk(c.chunk_id, c.source_type, c.content, score, c.item)
        for cid, score in fused
        if (c := chunks.get(cid))
    ]
    return await reranker.rerank(parsed.question, ranked, top_k)


async def _load(
    session: AsyncSession, restaurant_id: uuid.UUID, ids: list[uuid.UUID]
) -> dict[uuid.UUID, RetrievedChunk]:
    if not ids:
        return {}
    rows = await session.execute(
        select(Chunk.id, Chunk.source_type, Chunk.content, *ITEM_COLUMNS)
        .outerjoin(
            MenuItem,
            and_(MenuItem.id == Chunk.menu_item_id, MenuItem.restaurant_id == restaurant_id),
        )
        .where(Chunk.restaurant_id == restaurant_id, Chunk.id.in_(ids))
    )
    out: dict[uuid.UUID, RetrievedChunk] = {}
    for chunk_id, source_type, content, *item in rows:
        hit = ItemHit(*item) if item[0] is not None else None
        out[chunk_id] = RetrievedChunk(chunk_id, source_type, content, 0.0, hit)
    return out

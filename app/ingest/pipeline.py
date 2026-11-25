"""Menu ingestion: make the DB match a MenuNow payload for exactly one restaurant.

1. Upsert restaurant, items (by external_id) and FAQs (by question); delete what's gone.
2. Rebuild the desired chunk set and diff it against stored chunks by content_hash.
3. Embed only new/changed chunks, delete stale ones.
4. Bump menu_version if anything changed. Same payload twice = no writes, no embedding calls.

Everything runs in the caller's transaction (embedding calls included, which is fine at menu
scale: ~50 chunks per restaurant).
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import delete, func, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Faq, MenuItem, Restaurant
from app.ingest.chunk import ChunkSpec, content_hash, faq_chunk, info_chunk, item_chunk
from app.ingest.embed import embed_texts
from app.ingest.schemas import MenuPayload
from app.llm.base import LLMClient


@dataclass(frozen=True)
class SyncResult:
    items_changed: int
    items_deleted: int
    faqs_changed: int
    faqs_deleted: int

    @property
    def changed(self) -> bool:
        return any((self.items_changed, self.items_deleted, self.faqs_changed, self.faqs_deleted))


@dataclass(frozen=True)
class IngestResult:
    restaurant_id: uuid.UUID
    menu_version: int
    items_changed: int
    items_deleted: int
    faqs_changed: int
    faqs_deleted: int
    chunks_embedded: int
    chunks_deleted: int
    embed_tokens: int
    embed_cost_usd: Decimal


async def ingest_menu(session: AsyncSession, llm: LLMClient, payload: MenuPayload) -> IngestResult:
    rid = payload.restaurant.id
    synced = await sync_menu(session, payload)
    desired = await _desired_chunks(session, payload)
    embedded, deleted, tokens, cost = await _sync_chunks(session, llm, rid, desired)

    if synced.changed or embedded or deleted:
        version = await session.scalar(
            update(Restaurant)
            .where(Restaurant.id == rid)
            .values(menu_version=Restaurant.menu_version + 1)
            .returning(Restaurant.menu_version)
        )
    else:
        version = await session.scalar(select(Restaurant.menu_version).where(Restaurant.id == rid))

    return IngestResult(
        restaurant_id=rid,
        menu_version=version or 0,
        items_changed=synced.items_changed,
        items_deleted=synced.items_deleted,
        faqs_changed=synced.faqs_changed,
        faqs_deleted=synced.faqs_deleted,
        chunks_embedded=embedded,
        chunks_deleted=deleted,
        embed_tokens=tokens,
        embed_cost_usd=cost,
    )


async def sync_menu(session: AsyncSession, payload: MenuPayload) -> SyncResult:
    """Rows only. Counts include inserts and real updates, never no-op rewrites."""
    r = payload.restaurant
    profile = {
        "slug": r.slug,
        "name": r.name,
        "default_language": r.default_language,
        "monthly_budget_usd": r.monthly_budget_usd,
    }
    await session.execute(
        insert(Restaurant)
        .values(id=r.id, menu_version=0, **profile)
        .on_conflict_do_update(index_elements=[Restaurant.id], set_=profile)
    )

    items_changed = 0
    if payload.items:
        stmt = insert(MenuItem).values(
            [{"restaurant_id": r.id, **i.model_dump()} for i in payload.items]
        )
        changed = {
            col.name: stmt.excluded[col.name]
            for col in MenuItem.__table__.columns
            if col.name not in {"id", "restaurant_id", "external_id", "updated_at"}
        }
        current = tuple_(*(MenuItem.__table__.c[c] for c in changed))
        res = await session.execute(
            stmt.on_conflict_do_update(
                constraint="uq_menu_items_restaurant_external",
                set_={**changed, "updated_at": func.now()},
                # Skip identical rows so updated_at and the change count stay meaningful.
                where=current.is_distinct_from(tuple_(*changed.values())),
            ).returning(MenuItem.id)
        )
        items_changed = len(res.all())
    items_deleted = await session.execute(
        delete(MenuItem)
        .where(
            MenuItem.restaurant_id == r.id,
            MenuItem.external_id.not_in([i.external_id for i in payload.items]),
        )
        .returning(MenuItem.id)
    )

    faqs_changed = 0
    if payload.faqs:
        rows = [{"restaurant_id": r.id, **f.model_dump()} for f in payload.faqs]
        fstmt = insert(Faq).values(rows)
        res = await session.execute(
            fstmt.on_conflict_do_update(
                constraint="uq_faqs_restaurant_question",
                set_={"answer": fstmt.excluded.answer},
                where=Faq.answer.is_distinct_from(fstmt.excluded.answer),
            ).returning(Faq.id)
        )
        faqs_changed = len(res.all())
    faqs_deleted = await session.execute(
        delete(Faq)
        .where(Faq.restaurant_id == r.id, Faq.question.not_in([f.question for f in payload.faqs]))
        .returning(Faq.id)
    )

    return SyncResult(
        items_changed=items_changed,
        items_deleted=len(items_deleted.all()),
        faqs_changed=faqs_changed,
        faqs_deleted=len(faqs_deleted.all()),
    )


async def _desired_chunks(session: AsyncSession, payload: MenuPayload) -> list[ChunkSpec]:
    rid = payload.restaurant.id
    items = await session.scalars(select(MenuItem).where(MenuItem.restaurant_id == rid))
    faqs = await session.scalars(select(Faq).where(Faq.restaurant_id == rid))
    specs = [
        item_chunk(
            menu_item_id=i.id,
            external_id=i.external_id,
            name=i.name,
            description=i.description,
            category=i.category,
            price_cents=i.price_cents,
            currency=i.currency,
            allergens=i.allergens,
            dietary_tags=i.dietary_tags,
        )
        for i in items
    ]
    specs += [faq_chunk(faq_id=f.id, question=f.question, answer=f.answer) for f in faqs]
    info = info_chunk(name=payload.restaurant.name, info=payload.restaurant.info)
    return [*specs, info] if info else specs


async def _sync_chunks(
    session: AsyncSession, llm: LLMClient, rid: uuid.UUID, desired: list[ChunkSpec]
) -> tuple[int, int, int, Decimal]:
    """Returns (embedded, deleted, embed tokens, embed cost)."""
    model = llm.model_id("menu-embed")
    stored = {
        (c.source_type, c.menu_item_id or c.faq_id): c
        for c in await session.scalars(select(Chunk).where(Chunk.restaurant_id == rid))
    }
    wanted = {spec.key: (spec, content_hash(spec.content, model)) for spec in desired}

    stale = [c.id for key, c in stored.items() if key not in wanted]
    if stale:
        await session.execute(delete(Chunk).where(Chunk.restaurant_id == rid, Chunk.id.in_(stale)))

    todo = [
        (spec, h)
        for key, (spec, h) in wanted.items()
        if key not in stored or stored[key].content_hash != h
    ]
    if not todo:
        return 0, len(stale), 0, Decimal(0)

    result = await embed_texts(llm, [spec.content for spec, _ in todo])
    for (spec, h), vector in zip(todo, result.vectors, strict=True):
        row = stored.get(spec.key)
        if row is None:
            row = Chunk(
                restaurant_id=rid,
                source_type=spec.source_type,
                menu_item_id=spec.menu_item_id,
                faq_id=spec.faq_id,
            )
            session.add(row)
        row.content = spec.content
        row.metadata_ = spec.metadata
        row.content_hash = h
        row.embedding = vector
    await session.flush()
    return len(todo), len(stale), result.usage.tokens_in, result.usage.cost_usd

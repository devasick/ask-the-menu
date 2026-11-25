"""Menu synchronisation: make the DB match a MenuNow payload for one restaurant.

Step 1 syncs restaurants, menu items and FAQs. Chunking, embedding diffs and the
`menu_version` bump are added on top of this in step 2.
"""

from dataclasses import dataclass

from sqlalchemy import delete, func, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Faq, MenuItem, Restaurant
from app.ingest.schemas import MenuPayload


@dataclass(frozen=True)
class SyncResult:
    items_upserted: int
    items_deleted: int
    faqs_upserted: int
    faqs_deleted: int


async def sync_menu(session: AsyncSession, payload: MenuPayload) -> SyncResult:
    """Upsert the restaurant, its items (by external_id) and FAQs (by question); delete the rest.

    Runs in the caller's transaction; the caller commits.
    """
    r = payload.restaurant
    await session.execute(
        insert(Restaurant)
        .values(
            id=r.id,
            slug=r.slug,
            name=r.name,
            default_language=r.default_language,
            monthly_budget_usd=r.monthly_budget_usd,
            menu_version=0,
        )
        .on_conflict_do_update(
            index_elements=[Restaurant.id],
            set_={
                "slug": r.slug,
                "name": r.name,
                "default_language": r.default_language,
                "monthly_budget_usd": r.monthly_budget_usd,
            },
        )
    )

    external_ids = [i.external_id for i in payload.items]
    if payload.items:
        stmt = insert(MenuItem).values(
            [{"restaurant_id": r.id, **i.model_dump()} for i in payload.items]
        )
        changed = {
            col.name: stmt.excluded[col.name]
            for col in MenuItem.__table__.columns
            if col.name not in {"id", "restaurant_id", "external_id", "updated_at"}
        }
        await session.execute(
            stmt.on_conflict_do_update(
                constraint="uq_menu_items_restaurant_external",
                set_={**changed, "updated_at": func.now()},
                # Only touch rows whose content actually changed, so updated_at stays meaningful.
                where=tuple_(*(MenuItem.__table__.c[c] for c in changed)).is_distinct_from(
                    tuple_(*changed.values())
                ),
            )
        )
    items_deleted = await session.execute(
        delete(MenuItem)
        .where(MenuItem.restaurant_id == r.id, MenuItem.external_id.not_in(external_ids))
        .returning(MenuItem.id)
    )

    questions = [f.question for f in payload.faqs]
    if payload.faqs:
        fstmt = insert(Faq).values(
            [{"restaurant_id": r.id, **f.model_dump()} for f in payload.faqs]
        )
        await session.execute(
            fstmt.on_conflict_do_update(
                constraint="uq_faqs_restaurant_question",
                set_={"answer": fstmt.excluded.answer},
                where=Faq.answer.is_distinct_from(fstmt.excluded.answer),
            )
        )
    faqs_deleted = await session.execute(
        delete(Faq)
        .where(Faq.restaurant_id == r.id, Faq.question.not_in(questions))
        .returning(Faq.id)
    )

    return SyncResult(
        items_upserted=len(payload.items),
        items_deleted=len(items_deleted.all()),
        faqs_upserted=len(payload.faqs),
        faqs_deleted=len(faqs_deleted.all()),
    )

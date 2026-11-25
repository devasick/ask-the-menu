"""Structured item search. Every query is scoped to one restaurant_id."""

import uuid
from dataclasses import dataclass

from sqlalchemy import ColumnElement, and_, func, not_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MenuItem
from app.retrieval.query_parser import QueryFilters


@dataclass(frozen=True)
class ItemHit:
    id: uuid.UUID
    external_id: str
    name: str
    category: str
    price_cents: int
    currency: str
    allergens: list[str] | None
    dietary_tags: list[str]
    is_available: bool


ITEM_COLUMNS = (
    MenuItem.id,
    MenuItem.external_id,
    MenuItem.name,
    MenuItem.category,
    MenuItem.price_cents,
    MenuItem.currency,
    MenuItem.allergens,
    MenuItem.dietary_tags,
    MenuItem.is_available,
)


def item_filter_clauses(filters: QueryFilters) -> list[ColumnElement[bool]]:
    clauses: list[ColumnElement[bool]] = []
    if filters.max_price_cents is not None:
        clauses.append(MenuItem.price_cents <= filters.max_price_cents)
    if filters.dietary:
        clauses.append(MenuItem.dietary_tags.contains(sorted(filters.dietary)))
    if filters.exclude_dietary:
        clauses.append(not_(MenuItem.dietary_tags.overlap(sorted(filters.exclude_dietary))))
    if filters.exclude_allergens:
        # Unknown (NULL or empty) allergen data is never "free of" anything.
        clauses.append(
            and_(
                func.cardinality(MenuItem.allergens) > 0,
                not_(MenuItem.allergens.overlap(sorted(filters.exclude_allergens))),
            )
        )
    if filters.category_stems:
        clauses.append(
            or_(
                *(
                    func.lower(MenuItem.category).startswith(s)
                    for s in sorted(filters.category_stems)
                )
            )
        )
    return clauses


async def search_items(
    session: AsyncSession, restaurant_id: uuid.UUID, filters: QueryFilters, limit: int = 20
) -> list[ItemHit]:
    """Available items matching every filter, cheapest first within each category."""
    rows = await session.execute(
        select(*ITEM_COLUMNS)
        .where(
            MenuItem.restaurant_id == restaurant_id,
            MenuItem.is_available.is_(True),
            *item_filter_clauses(filters),
        )
        .order_by(MenuItem.category, MenuItem.price_cents, MenuItem.name)
        .limit(limit)
    )
    return [ItemHit(*row) for row in rows]

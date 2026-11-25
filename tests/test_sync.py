"""Menu sync against a real Postgres: idempotency, diffing and tenant scoping."""

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Faq, MenuItem
from app.ingest.pipeline import SyncResult, sync_menu
from app.ingest.schemas import MenuPayload
from app.seed import load_payload

SEED = Path(__file__).resolve().parent.parent / "seed"
CURRY = load_payload(SEED / "curry_house.json")
PIZZA = load_payload(SEED / "pizzeria_finta.json")

Maker = async_sessionmaker[AsyncSession]


async def _sync(maker: Maker, payload: MenuPayload) -> SyncResult:
    async with maker() as session, session.begin():
        return await sync_menu(session, payload)


async def _items(maker: Maker, payload: MenuPayload) -> dict[str, MenuItem]:
    async with maker() as session:
        rows = await session.scalars(
            select(MenuItem).where(MenuItem.restaurant_id == payload.restaurant.id)
        )
        return {i.external_id: i for i in rows}


async def test_seed_loads_items_and_faqs(sessionmaker: Maker) -> None:
    await _sync(sessionmaker, CURRY)
    items = await _items(sessionmaker, CURRY)
    assert len(items) == len(CURRY.items)
    korma = items["cu-03"]
    assert korma.allergens == ["milk", "nuts"]
    assert korma.price_cents == 1550
    # Unknown allergens are stored as NULL, "none" as an empty array.
    assert items["cs-07"].allergens is None
    assert items["cs-02"].allergens == []
    async with sessionmaker() as session:
        assert await session.scalar(select(func.count()).select_from(Faq)) == len(CURRY.faqs)


async def test_sync_is_idempotent(sessionmaker: Maker) -> None:
    await _sync(sessionmaker, CURRY)
    before = {k: (v.id, v.updated_at) for k, v in (await _items(sessionmaker, CURRY)).items()}
    result = await _sync(sessionmaker, CURRY)
    after = {k: (v.id, v.updated_at) for k, v in (await _items(sessionmaker, CURRY)).items()}
    assert after == before
    assert not result.changed


async def test_sync_updates_changed_and_deletes_removed_items(sessionmaker: Maker) -> None:
    await _sync(sessionmaker, CURRY)
    before = await _items(sessionmaker, CURRY)

    data = CURRY.model_dump(mode="json")
    data["items"] = [i for i in data["items"] if i["external_id"] != "dr-05"]
    korma = next(i for i in data["items"] if i["external_id"] == "cu-03")
    korma["price_cents"] = 1600
    data["faqs"] = data["faqs"][1:]
    result = await _sync(sessionmaker, MenuPayload.model_validate(data))

    after = await _items(sessionmaker, CURRY)
    assert "dr-05" not in after
    assert result.items_deleted == 1
    assert result.faqs_deleted == 1
    assert after["cu-03"].price_cents == 1600
    assert after["cu-03"].id == before["cu-03"].id  # upsert, not re-insert
    assert after["cu-03"].updated_at > before["cu-03"].updated_at
    assert after["cu-02"].updated_at == before["cu-02"].updated_at


async def test_sync_never_touches_another_restaurant(sessionmaker: Maker) -> None:
    await _sync(sessionmaker, CURRY)
    await _sync(sessionmaker, PIZZA)
    # An empty menu for the pizzeria deletes only the pizzeria's items.
    empty = PIZZA.model_copy(update={"items": [], "faqs": []})
    result = await _sync(sessionmaker, empty)
    assert result.items_deleted == len(PIZZA.items)
    assert await _items(sessionmaker, PIZZA) == {}
    assert len(await _items(sessionmaker, CURRY)) == len(CURRY.items)

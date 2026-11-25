"""The seed files are the ground truth for evals, so their shape is pinned by tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.ingest.schemas import MenuItemIn, MenuPayload
from app.seed import load_payload

SEED = Path(__file__).resolve().parent.parent / "seed"


@pytest.fixture(scope="module")
def curry() -> MenuPayload:
    return load_payload(SEED / "curry_house.json")


@pytest.fixture(scope="module")
def pizza() -> MenuPayload:
    return load_payload(SEED / "pizzeria_finta.json")


def test_curry_house_shape(curry: MenuPayload) -> None:
    assert 35 <= len(curry.items) <= 45
    assert len(curry.faqs) == 8
    categories = {i.category for i in curry.items}
    assert {"Starters", "Curries", "Biryanis", "Breads", "Desserts", "Drinks"} <= categories
    assert all(i.currency == "EUR" for i in curry.items)


def test_curry_house_has_items_with_unknown_allergens(curry: MenuPayload) -> None:
    unknown = [i.external_id for i in curry.items if i.allergens is None]
    assert len(unknown) >= 3
    # "no allergens" ([]) must also exist and stay distinct from "unknown" (None).
    assert any(i.allergens == [] for i in curry.items)


def test_second_tenant_is_separate(curry: MenuPayload, pizza: MenuPayload) -> None:
    assert curry.restaurant.id != pizza.restaurant.id
    assert len(pizza.items) == 5
    assert not {i.external_id for i in curry.items} & {i.external_id for i in pizza.items}


def _item(**overrides: object) -> dict[str, object]:
    return {
        "external_id": "x",
        "name": "X",
        "category": "C",
        "price_cents": 100,
        "allergens": [],
    } | overrides


def test_unknown_allergen_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown allergens"):
        MenuItemIn.model_validate(_item(allergens=["peanut"]))


def test_missing_allergens_stay_none_and_are_required() -> None:
    assert MenuItemIn.model_validate(_item(allergens=None)).allergens is None
    with pytest.raises(ValidationError):
        MenuItemIn.model_validate({k: v for k, v in _item().items() if k != "allergens"})


def test_duplicate_external_ids_are_rejected(curry: MenuPayload) -> None:
    data = curry.model_dump(mode="json")
    data["items"].append(data["items"][0])
    with pytest.raises(ValidationError, match="duplicate external_id"):
        MenuPayload.model_validate(data)

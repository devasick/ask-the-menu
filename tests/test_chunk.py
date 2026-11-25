import uuid

from app.ingest.chunk import content_hash, faq_chunk, info_chunk, item_chunk
from app.ingest.normalize import normalize_text

ITEM = {
    "menu_item_id": uuid.UUID(int=1),
    "external_id": "cu-03",
    "name": "Chicken Korma",
    "description": "Mild curry  with\ncashews.",
    "category": "Curries",
    "price_cents": 1550,
    "currency": "EUR",
    "allergens": ["milk", "nuts"],
    "dietary_tags": ["gluten_free", "halal"],
}


def test_item_chunk_template_is_deterministic_and_readable() -> None:
    chunk = item_chunk(**ITEM)  # type: ignore[arg-type]
    assert chunk.content == (
        "Chicken Korma\n"
        "Category: Curries\n"
        "Mild curry with cashews.\n"
        "Price: 15.50 EUR\n"
        "Allergens: milk, nuts\n"
        "Dietary: gluten_free, halal"
    )
    assert chunk.metadata["allergens"] == ["milk", "nuts"]
    assert chunk.key == ("menu_item", uuid.UUID(int=1))


def test_unknown_and_no_allergens_are_worded_differently() -> None:
    unknown = item_chunk(**{**ITEM, "allergens": None})  # type: ignore[arg-type]
    none = item_chunk(**{**ITEM, "allergens": []})  # type: ignore[arg-type]
    assert "Allergens: unknown, ask staff" in unknown.content
    assert "Allergens: none of the 14 EU allergens" in none.content
    assert unknown.metadata["allergens"] is None


def test_price_formatting_keeps_cents() -> None:
    assert "Price: 3.05 EUR" in item_chunk(**{**ITEM, "price_cents": 305}).content  # type: ignore[arg-type]


def test_hash_tracks_content_and_embedding_model() -> None:
    base = content_hash("a", "m1")
    assert base == content_hash("a", "m1")
    assert base != content_hash("b", "m1")
    assert base != content_hash("a", "m2")


def test_faq_and_info_chunks() -> None:
    faq = faq_chunk(faq_id=uuid.UUID(int=2), question="Do you deliver?", answer="Yes.")
    assert faq.content == "Q: Do you deliver?\nA: Yes."
    assert info_chunk(name="X", info="  ") is None
    assert info_chunk(name="X", info="Berlin") is not None


def test_normalize_text_nfc() -> None:
    decomposed = "Gemüse"
    assert normalize_text(decomposed) == "Gemüse"

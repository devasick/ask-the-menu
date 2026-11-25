"""One chunk per menu item, one per FAQ, one for restaurant info. Deterministic templates."""

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.ingest.normalize import normalize_text

SourceType = Literal["menu_item", "faq", "info"]


@dataclass(frozen=True)
class ChunkSpec:
    source_type: SourceType
    content: str
    menu_item_id: uuid.UUID | None = None
    faq_id: uuid.UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, uuid.UUID | None]:
        """Identity of the chunk within a restaurant, used for diffing."""
        return (self.source_type, self.menu_item_id or self.faq_id)


def format_price(price_cents: int, currency: str) -> str:
    return f"{price_cents // 100}.{price_cents % 100:02d} {currency}"


def allergen_line(allergens: list[str] | None) -> str:
    if allergens is None:
        return "Allergens: unknown, ask staff"
    if not allergens:
        return "Allergens: none of the 14 EU allergens"
    return "Allergens: " + ", ".join(allergens)


def item_chunk(
    *,
    menu_item_id: uuid.UUID,
    external_id: str,
    name: str,
    description: str,
    category: str,
    price_cents: int,
    currency: str,
    allergens: list[str] | None,
    dietary_tags: list[str],
) -> ChunkSpec:
    # Availability is deliberately not in the text: toggling "sold out" must not re-embed.
    lines = [
        normalize_text(name),
        f"Category: {normalize_text(category)}",
        normalize_text(description),
        f"Price: {format_price(price_cents, currency)}",
        allergen_line(allergens),
        "Dietary: " + (", ".join(dietary_tags) if dietary_tags else "none"),
    ]
    return ChunkSpec(
        source_type="menu_item",
        content="\n".join(line for line in lines if line),
        menu_item_id=menu_item_id,
        metadata={
            "external_id": external_id,
            "name": name,
            "category": category,
            "price_cents": price_cents,
            "currency": currency,
            "allergens": allergens,
            "dietary_tags": dietary_tags,
        },
    )


def faq_chunk(*, faq_id: uuid.UUID, question: str, answer: str) -> ChunkSpec:
    content = f"Q: {normalize_text(question)}\nA: {normalize_text(answer)}"
    return ChunkSpec(source_type="faq", content=content, faq_id=faq_id)


def info_chunk(*, name: str, info: str) -> ChunkSpec | None:
    if not info.strip():
        return None
    return ChunkSpec(source_type="info", content=f"{normalize_text(name)}\n{normalize_text(info)}")


def content_hash(content: str, embed_model: str) -> str:
    """Changes when the text or the embedding model changes; either needs a re-embed."""
    return hashlib.sha256(f"{embed_model}\n{content}".encode()).hexdigest()

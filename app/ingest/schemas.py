"""The menu payload MenuNow sends to `POST /ingest/{restaurant_id}` (same shape as seed/*.json)."""

import uuid
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The 14 allergen groups of EU Regulation 1169/2011, Annex II.
EU_ALLERGENS = frozenset(
    {
        "gluten",
        "crustaceans",
        "eggs",
        "fish",
        "peanuts",
        "soybeans",
        "milk",
        "nuts",
        "celery",
        "mustard",
        "sesame",
        "sulphites",
        "lupin",
        "molluscs",
    }
)
DIETARY_TAGS = frozenset({"vegan", "vegetarian", "gluten_free", "halal", "spicy"})

Language = Literal["en", "de"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RestaurantInfo(_Strict):
    id: uuid.UUID
    slug: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9-]+$")
    name: str = Field(min_length=1, max_length=200)
    default_language: Language = "de"
    monthly_budget_usd: Decimal = Field(default=Decimal("10.00"), ge=0)
    # Free text (address, cuisine, story). Becomes the `info` chunk at ingestion.
    info: str = ""


class MenuItemIn(_Strict):
    external_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    category: str = Field(min_length=1, max_length=80)
    price_cents: int = Field(ge=0)
    currency: str = Field(default="EUR", pattern=r"^[A-Z]{3}$")
    # None = unknown (never guess). [] = contains none of the 14 EU allergens.
    allergens: list[str] | None
    dietary_tags: list[str] = Field(default_factory=list)
    is_available: bool = True
    language: Language = "en"

    @field_validator("allergens")
    @classmethod
    def _known_allergens(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unknown = set(value) - EU_ALLERGENS
        if unknown:
            raise ValueError(f"unknown allergens: {sorted(unknown)}")
        return sorted(set(value))

    @field_validator("dietary_tags")
    @classmethod
    def _known_tags(cls, value: list[str]) -> list[str]:
        unknown = set(value) - DIETARY_TAGS
        if unknown:
            raise ValueError(f"unknown dietary tags: {sorted(unknown)}")
        return sorted(set(value))


class FaqIn(_Strict):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class MenuPayload(_Strict):
    restaurant: RestaurantInfo
    items: list[MenuItemIn]
    faqs: list[FaqIn] = Field(default_factory=list)

    @field_validator("items")
    @classmethod
    def _unique_external_ids(cls, value: list[MenuItemIn]) -> list[MenuItemIn]:
        ids = [i.external_id for i in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate external_id in items")
        return value

    @field_validator("faqs")
    @classmethod
    def _unique_questions(cls, value: list[FaqIn]) -> list[FaqIn]:
        qs = [f.question for f in value]
        if len(qs) != len(set(qs)):
            raise ValueError("duplicate question in faqs")
        return value

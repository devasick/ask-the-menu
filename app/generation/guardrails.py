"""Guardrails enforced in code, independent of what the model writes."""

import logging
import re

from app.ingest.schemas import has_allergen_data
from app.retrieval.query_parser import Language
from app.retrieval.sql_search import ItemHit

log = logging.getLogger(__name__)

_ALLERGEN_NAMES_DE = {
    "gluten": "Gluten",
    "crustaceans": "Krebstiere",
    "eggs": "Eier",
    "fish": "Fisch",
    "peanuts": "Erdnüsse",
    "soybeans": "Soja",
    "milk": "Milch",
    "nuts": "Schalenfrüchte",
    "celery": "Sellerie",
    "mustard": "Senf",
    "sesame": "Sesam",
    "sulphites": "Sulfite",
    "lupin": "Lupinen",
    "molluscs": "Weichtiere",
}


def validate_citations(cited: list[str], allowed: list[str]) -> list[str]:
    """Keep only IDs that were actually retrieved, in order, without duplicates."""
    allowed_set = set(allowed)
    kept = list(dict.fromkeys(c for c in cited if c in allowed_set))
    dropped = [c for c in cited if c not in allowed_set]
    if dropped:
        log.warning("dropped %d citation(s) not in the retrieved set: %s", len(dropped), dropped)
    return kept


def allergen_answer(items: list[ItemHit], lang: Language) -> str:
    """Allergen facts built only from the stored allergens field, never from model text."""
    lines = []
    for item in items:
        if has_allergen_data(item.allergens):
            names = (
                [_ALLERGEN_NAMES_DE[a] for a in item.allergens] if lang == "de" else item.allergens
            )
            joined = ", ".join(names)
            lines.append(
                f"{item.name} enthält laut Speisekarte: {joined}."
                if lang == "de"
                else f"According to our menu, {item.name} contains: {joined}."
            )
        else:
            lines.append(
                f"Für {item.name} liegen keine Allergenangaben vor, bitte frag unser Personal."
                if lang == "de"
                else f"Allergen information for {item.name} is not available, please ask our staff."
            )
    if not lines:
        return (
            "Dazu habe ich keine Allergenangaben, bitte frag unser Personal."
            if lang == "de"
            else "I don't have allergen information for that, please ask our staff."
        )
    return " ".join(lines)


def is_multi_part(question: str, min_questions: int, min_words: int) -> bool:
    """Heuristic for escalating to `menu-smart`: several questions or a long question."""
    return question.count("?") >= min_questions or len(re.findall(r"\w+", question)) >= min_words

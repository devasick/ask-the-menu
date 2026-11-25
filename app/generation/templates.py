"""Templated answers for routes that need no generation call (EN/DE)."""

from app.retrieval.query_parser import Language, QueryFilters

ALLERGEN_DISCLAIMER = {
    "en": "Allergen information comes from the restaurant's menu data. If you have an allergy, "
    "please confirm with staff before ordering.",
    "de": "Die Allergenangaben stammen aus den Menüdaten des Restaurants. Bei Allergien bitte "
    "vor der Bestellung beim Personal nachfragen.",
}

_DIETARY = {
    "en": {
        "vegan": "vegan",
        "vegetarian": "vegetarian",
        "gluten_free": "gluten-free",
        "halal": "halal",
        "spicy": "spicy",
    },
    "de": {
        "vegan": "vegan",
        "vegetarian": "vegetarisch",
        "gluten_free": "glutenfrei",
        "halal": "halal",
        "spicy": "scharf",
    },
}
_ALLERGEN_DE = {
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


def _price(cents: int, lang: Language) -> str:
    text = f"{cents // 100}.{cents % 100:02d}"
    return text.replace(".", ",") + " €" if lang == "de" else "€" + text


def describe_filters(filters: QueryFilters, lang: Language) -> str:
    parts = [_DIETARY[lang][t] for t in sorted(filters.dietary)]
    if "spicy" in filters.exclude_dietary:
        parts.append("nicht scharf" if lang == "de" else "not spicy")
    if filters.category_stems:
        parts += sorted(filters.category_stems)
    if filters.max_price_cents is not None:
        word = "bis" if lang == "de" else "up to"
        parts.append(f"{word} {_price(filters.max_price_cents, lang)}")
    if filters.exclude_allergens:
        names = sorted(_ALLERGEN_DE[a] if lang == "de" else a for a in filters.exclude_allergens)
        parts.append(("ohne " if lang == "de" else "without ") + ", ".join(names))
    return ", ".join(parts)


def filter_answer(filters: QueryFilters, count: int, lang: Language) -> str:
    summary = describe_filters(filters, lang)
    if lang == "de":
        if count == 0:
            return f"Leider passt nichts auf unserer Karte zu: {summary}."
        noun = "passendes Gericht" if count == 1 else "passende Gerichte"
        return f"Ich habe {count} {noun} gefunden ({summary})."
    if count == 0:
        return f"Sorry, nothing on our menu matches: {summary}."
    noun = "matching dish" if count == 1 else "matching dishes"
    return f"I found {count} {noun} ({summary})."

import pytest

from app.retrieval.query_parser import detect_language, parse_query


@pytest.mark.parametrize(
    ("question", "max_price", "dietary", "excluded", "pure"),
    [
        ("What's vegan and under €15?", 1499, {"vegan"}, set(), True),
        ("vegan dishes under 15 euros", 1499, {"vegan"}, set(), True),
        ("Zeig mir vegetarische Gerichte unter 12 €", 1199, {"vegetarian"}, set(), True),
        ("Was ist glutenfrei?", None, {"gluten_free"}, {"gluten"}, True),
        ("gluten-free options up to €10", 1000, {"gluten_free"}, {"gluten"}, True),
        ("Gerichte bis 9,50 €", 950, set(), set(), True),
        ("ohne Nüsse bitte", None, set(), {"nuts"}, True),
        ("nut-free curries", None, set(), {"nuts"}, True),
        ("Something without dairy or eggs", None, set(), {"milk", "eggs"}, True),
        ("I'm allergic to nuts and milk, what can I eat?", None, set(), {"nuts", "milk"}, True),
        ("Ich habe eine Allergie gegen Erdnüsse", None, set(), {"peanuts"}, True),
        ("anything without shellfish?", None, set(), {"crustaceans", "molluscs"}, True),
        ("halal dishes", None, {"halal"}, set(), True),
    ],
)
def test_filters(
    question: str, max_price: int | None, dietary: set[str], excluded: set[str], pure: bool
) -> None:
    parsed = parse_query(question)
    assert parsed.filters.max_price_cents == max_price
    assert parsed.filters.dietary == dietary
    assert parsed.filters.exclude_allergens == excluded
    assert parsed.is_pure_filter is pure


def test_question_about_a_dish_mentions_but_does_not_exclude() -> None:
    parsed = parse_query("Does the chicken korma contain nuts?")
    assert parsed.mentions_allergen
    assert parsed.filters.exclude_allergens == frozenset()
    assert not parsed.is_pure_filter
    assert {"chicken", "korma"} <= set(parsed.semantic_terms)


def test_semantic_question_keeps_filters_but_is_not_pure() -> None:
    parsed = parse_query("Something light and spicy that goes with mango lassi?")
    assert parsed.filters.dietary == {"spicy"}
    assert not parsed.is_pure_filter
    assert "lassi" in parsed.semantic_terms


def test_categories_and_mild() -> None:
    assert parse_query("show desserts").filters.category_stems == {"dessert"}
    assert parse_query("Welche Getränke habt ihr?").filters.category_stems == {"drink"}
    mild = parse_query("mild curries")
    assert mild.filters.exclude_dietary == {"spicy"}
    assert "spicy" not in mild.filters.dietary
    assert parse_query("Was ist nicht scharf?").filters.exclude_dietary == {"spicy"}


def test_unparsable_price_flags_rules_incomplete() -> None:
    assert parse_query("Something cheap?").rules_incomplete
    assert not parse_query("vegan under €15").rules_incomplete


def test_no_filters_is_never_pure() -> None:
    assert not parse_query("Do you deliver?").is_pure_filter
    assert not parse_query("hello").is_pure_filter


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Was ist glutenfrei?", "de"),
        ("Enthält das Korma Nüsse?", "de"),
        ("Welche Gerichte sind vegan?", "de"),
        ("What's vegan and under €15?", "en"),
        ("Does the chicken korma contain nuts?", "en"),
        ("vegan", "en"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text) == lang


@pytest.mark.parametrize(
    "question",
    [
        "I have a nut allergy. Which starters can I order?",
        "Ich habe eine Nussallergie.",
        "Nuss-Allergie, was geht?",
        "milk intolerance",
    ],
)
def test_allergy_phrases_are_exclusions(question: str) -> None:
    parsed = parse_query(question)
    assert parsed.filters.exclude_allergens & {"nuts", "milk"}
    assert parsed.mentions_allergen

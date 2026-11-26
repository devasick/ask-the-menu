"""Rules-first query parser (English + German): structured filters out of a diner's question.

    "vegan dishes under €15"  -> dietary={vegan}, max_price=1499, pure filter
    "ohne Nüsse"              -> exclude_allergens={nuts}
    "Does the korma contain nuts?" -> mentions_allergen, but NOT an exclusion (asks about a dish)

Anything left over after removing filter phrases and filler words is the "semantic" part;
a question with filters and no semantic part can be answered from SQL alone.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from app.security.injection import detect_injection

Language = Literal["en", "de"]

# --- vocabularies -------------------------------------------------------------------------

# Words (lowercase, NFC) -> EU allergen code. Plurals/inflections listed explicitly.
ALLERGEN_TERMS: dict[str, str] = {
    **dict.fromkeys(
        [
            "nut",
            "nuts",
            "tree nuts",
            "cashew",
            "cashews",
            "almond",
            "almonds",
            "pistachio",
            "pistachios",
            "walnut",
            "walnuts",
            "hazelnut",
            "hazelnuts",
            "nuss",
            "nüsse",
            "nüssen",
            "cashewkerne",
            "mandel",
            "mandeln",
            "pistazie",
            "pistazien",
            "walnüsse",
            "haselnüsse",
        ],
        "nuts",
    ),
    **dict.fromkeys(["peanut", "peanuts", "erdnuss", "erdnüsse", "erdnüssen"], "peanuts"),
    **dict.fromkeys(
        ["milk", "dairy", "lactose", "milch", "milchprodukte", "laktose", "lactose-free"], "milk"
    ),
    **dict.fromkeys(["gluten", "wheat", "weizen"], "gluten"),
    **dict.fromkeys(["egg", "eggs", "ei", "eier", "eiern"], "eggs"),
    **dict.fromkeys(["fish", "fisch"], "fish"),
    **dict.fromkeys(
        [
            "crustacean",
            "crustaceans",
            "prawn",
            "prawns",
            "shrimp",
            "shrimps",
            "krebstiere",
            "garnele",
            "garnelen",
        ],
        "crustaceans",
    ),
    **dict.fromkeys(["mollusc", "molluscs", "mollusk", "mollusks", "weichtiere"], "molluscs"),
    **dict.fromkeys(["soy", "soya", "soja"], "soybeans"),
    **dict.fromkeys(["celery", "sellerie"], "celery"),
    **dict.fromkeys(["mustard", "senf"], "mustard"),
    **dict.fromkeys(["sesame", "sesam"], "sesame"),
    **dict.fromkeys(["sulphite", "sulphites", "sulfite", "sulfites", "sulfit"], "sulphites"),
    **dict.fromkeys(["lupin", "lupine", "lupinen"], "lupin"),
}
# "shellfish" covers two EU groups.
MULTI_ALLERGEN_TERMS: dict[str, frozenset[str]] = {
    "shellfish": frozenset({"crustaceans", "molluscs"}),
    "meeresfrüchte": frozenset({"crustaceans", "molluscs"}),
}

DIETARY_TERMS: dict[str, str] = {
    **dict.fromkeys(
        ["vegan", "vegane", "veganes", "veganer", "veganen", "plant-based", "pflanzlich"], "vegan"
    ),
    **dict.fromkeys(
        [
            "vegetarian",
            "veggie",
            "vegetarisch",
            "vegetarische",
            "vegetarisches",
            "vegetarischen",
            "vegetarier",
        ],
        "vegetarian",
    ),
    **dict.fromkeys(
        [
            "gluten-free",
            "gluten free",
            "glutenfree",
            "glutenfrei",
            "glutenfreie",
            "glutenfreies",
            "glutenfreien",
        ],
        "gluten_free",
    ),
    **dict.fromkeys(["halal"], "halal"),
    **dict.fromkeys(["spicy", "scharf", "scharfe", "scharfes", "scharfen"], "spicy"),
}
NOT_SPICY = re.compile(r"\b(not spicy|non-spicy|mild|milde|mildes|milden|nicht scharf)\b")

# Generic category words (EN/DE) -> English stem matched against the restaurant's categories.
CATEGORY_TERMS: dict[str, str] = {
    **dict.fromkeys(
        ["starter", "starters", "appetizer", "appetizers", "vorspeise", "vorspeisen"], "starter"
    ),
    **dict.fromkeys(["curry", "curries", "currys"], "curr"),
    **dict.fromkeys(["biryani", "biryanis"], "biryani"),
    **dict.fromkeys(["bread", "breads", "naan", "naans", "brot", "brote"], "bread"),
    **dict.fromkeys(["side", "sides", "beilage", "beilagen"], "side"),
    **dict.fromkeys(
        ["dessert", "desserts", "nachtisch", "nachspeise", "nachspeisen", "sweets"], "dessert"
    ),
    **dict.fromkeys(["drink", "drinks", "beverage", "beverages", "getränk", "getränke"], "drink"),
    **dict.fromkeys(["pizza", "pizzas", "pizzen"], "pizza"),
}

# A negation/avoidance cue before an allergen turns a mention into an exclusion.
EXCLUDE_CUE = re.compile(
    r"\b(without|no|free of|free from|allergic to|allergy to|intolerant to|avoid|"
    r"ohne|kein|keine|keinen|frei von|allergisch gegen|allergie gegen|unverträglichkeit)\b"
)
ALLERGY_WORDS = re.compile(
    r"\b(allerg\w*|intoleran\w*|unverträglich\w*|contain\w*|enthält|enthalten|"
    r"ingredient\w*|zutat\w*|traces|spuren)\b"
)

_NUM = r"(\d+(?:[.,]\d{1,2})?)"
_CUR = r"(?:€|euros|euro|eur)?"
# (pattern, inclusive). "under 15" excludes 15.00, "up to 15" includes it.
PRICE_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (
        re.compile(
            rf"\b(?:under|below|less than|cheaper than|unter|weniger als|günstiger als|"
            rf"billiger als)\s*{_CUR}\s*{_NUM}\s*{_CUR}"
        ),
        False,
    ),
    (
        re.compile(
            rf"\b(?:up to|max(?:imum)?|at most|no more than|bis|höchstens|maximal|"
            rf"bis zu)\s*{_CUR}\s*{_NUM}\s*{_CUR}"
        ),
        True,
    ),
    (re.compile(rf"{_CUR}\s*{_NUM}\s*{_CUR}\s*(?:or less|or under|oder weniger)\b"), True),
]
# Price words with no parsable amount: the rules failed, so the LLM parser may help.
PRICE_CUE = re.compile(
    r"(€|\beur\b|\beuro|\bcheap\w*|\bbudget|\bgünstig\w*|\bbillig\w*|"
    r"\bpreis\w*|\bprice\w*|\bunder\b|\bunter\b)"
)

FILLER = frozenset(
    """
    a an the and or but of for with to in on at is are be do does did you your we our i me my
    can could would should please any some something anything have has there what whats what's
    which who show list give tell see all also only just dishes dish options option food foods
    items item menu meals meal eat order get want like would'd i'd im i'm it its it's this that
    these those from how much many more other etc thing things kind kinds available here eur
    euro euros price prices der die das ein eine einen einem einer und oder aber von für mit zu
    im in am an auf ist sind sein habt ihr hast du haben gibt es was welche welcher welches wer
    zeig zeige zeigt mir mich bitte etwas gerichte gericht speisen speise essen alle auch nur
    noch karte möchte will kann können euch bei vom zum zur ich habe hab wir uns eure euer
    """.split()
)

GERMAN_HINTS = frozenset(
    """
    ist was welche welcher welches gibt habt ihr und ohne unter bis mit für nicht keine kein
    gerichte gericht speisen etwas zeig zeige bitte enthält vegetarisch glutenfrei scharf
    nachtisch getränke vorspeisen haben sie der die das ein eine wann wie mir mich dir deine
    deinen alle auch noch
    """.split()
)

_WORD = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class QueryFilters:
    max_price_cents: int | None = None
    dietary: frozenset[str] = frozenset()
    exclude_dietary: frozenset[str] = frozenset()
    exclude_allergens: frozenset[str] = frozenset()
    category_stems: frozenset[str] = frozenset()

    def any_item_filter(self) -> bool:
        return bool(
            self.max_price_cents is not None
            or self.dietary
            or self.exclude_dietary
            or self.exclude_allergens
            or self.category_stems
        )


@dataclass(frozen=True)
class ParsedQuery:
    question: str
    language: Language
    filters: QueryFilters
    # Allergens or intolerances are mentioned (the allergen guardrail applies).
    mentions_allergen: bool
    # Content words left after removing filter phrases and filler.
    semantic_terms: tuple[str, ...] = field(default_factory=tuple)
    # Price-like words we could not parse: the LLM parser may be worth a try.
    rules_incomplete: bool = False
    # Kind of prompt injection detected (app/security/injection.py), if any.
    injection: str | None = None

    @property
    def is_pure_filter(self) -> bool:
        return self.filters.any_item_filter() and not self.semantic_terms


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()


def detect_language(text: str) -> Language:
    norm = normalize_question(text)
    if re.search(r"[äöüß]", norm):
        return "de"
    words = _WORD.findall(norm)
    german = sum(w in GERMAN_HINTS for w in words)
    return "de" if german and german * 3 >= len(words) else "en"


def parse_query(question: str, language: Language | None = None) -> ParsedQuery:
    text = normalize_question(question)
    consumed: list[tuple[int, int]] = []

    max_price, price_spans = _parse_price(text)
    consumed += price_spans

    dietary, spans = _match_terms(text, DIETARY_TERMS)
    consumed += spans
    exclude_dietary: set[str] = set()
    for m in NOT_SPICY.finditer(text):
        exclude_dietary.add("spicy")
        consumed.append(m.span())
    if "spicy" in exclude_dietary:
        dietary.discard("spicy")

    categories, spans = _match_terms(text, CATEGORY_TERMS)
    consumed += spans

    excluded, mentioned, spans = _parse_allergens(text)
    consumed += spans
    if "gluten_free" in dietary:
        excluded.add("gluten")  # restaurant tag AND allergen data must both agree

    filters = QueryFilters(
        max_price_cents=max_price,
        dietary=frozenset(dietary),
        exclude_dietary=frozenset(exclude_dietary),
        exclude_allergens=frozenset(excluded),
        category_stems=frozenset(categories),
    )
    semantic = tuple(
        w
        for w in _WORD.findall(_blank(text, consumed))
        if w.strip("'-") and w not in FILLER and not w.isdigit()
    )
    return ParsedQuery(
        question=question,
        language=language or detect_language(question),
        filters=filters,
        mentions_allergen=bool(mentioned or excluded or ALLERGY_WORDS.search(text)),
        semantic_terms=semantic,
        rules_incomplete=max_price is None and bool(PRICE_CUE.search(text)),
        injection=detect_injection(question),
    )


# --- helpers ------------------------------------------------------------------------------


def _to_cents(raw: str) -> int:
    euros, _, cents = raw.replace(",", ".").partition(".")
    return int(euros) * 100 + int((cents + "00")[:2])


def _parse_price(text: str) -> tuple[int | None, list[tuple[int, int]]]:
    best: int | None = None
    spans: list[tuple[int, int]] = []
    for pattern, inclusive in PRICE_PATTERNS:
        for m in pattern.finditer(text):
            cents = _to_cents(m.group(1)) - (0 if inclusive else 1)
            best = cents if best is None else min(best, cents)
            spans.append(m.span())
    return best, spans


def _match_terms(text: str, vocab: dict[str, str]) -> tuple[set[str], list[tuple[int, int]]]:
    found: set[str] = set()
    spans: list[tuple[int, int]] = []
    # Longest terms first so "gluten free" wins over "gluten".
    for term in sorted(vocab, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            found.add(vocab[term])
            spans.append(m.span())
    return found, spans


def _parse_allergens(text: str) -> tuple[set[str], set[str], list[tuple[int, int]]]:
    """Returns (excluded codes, mentioned codes, spans consumed by exclusions)."""
    excluded: set[str] = set()
    mentioned: set[str] = set()
    spans: list[tuple[int, int]] = []
    vocab: dict[str, frozenset[str]] = {
        **{k: frozenset({v}) for k, v in ALLERGEN_TERMS.items()},
        **MULTI_ALLERGEN_TERMS,
    }
    for term in sorted(vocab, key=len, reverse=True):
        # "nut-free", "nussfrei", "lactose-free", "nut allergy", "Nussallergie"
        free = (
            rf"(?<![\w-]){re.escape(term)}(?:-free|-frei|frei|freie|freies|freien|"
            rf"-?allergie|-?unverträglichkeit|\s+(?:allergy|allergies|intolerance))(?![\w-])"
        )
        for m in re.finditer(free, text):
            excluded |= vocab[term]
            mentioned |= vocab[term]
            spans.append(m.span())
        for m in re.finditer(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            mentioned |= vocab[term]
            # An exclusion cue in the same clause, before the allergen word.
            clause = re.split(r"[.;?!]| but | aber ", text[: m.start()])[-1]
            cue = EXCLUDE_CUE.search(clause)
            if cue:
                excluded |= vocab[term]
                spans.append((m.start() - len(clause) + cue.start(), m.end()))
    return excluded, mentioned, spans


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(max(start, 0), min(end, len(chars))):
            chars[i] = " "
    return "".join(chars)

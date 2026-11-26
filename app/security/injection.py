"""Prompt injection defense (EN/DE), deterministic and model-free.

Input: questions that try to give the assistant instructions, change its role, fake prompt
markup or extract the prompt are blocked before any model call. Guests ask about food;
a genuine menu question never needs these phrasings.

Output: a generated answer that echoes the system prompt is replaced by a refusal.
"""

import re
import unicodedata

# Zero-width and bidi control characters used to hide words from filters.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")

_PATTERNS: tuple[tuple[str, str], ...] = (
    # Override earlier instructions.
    (
        "override",
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
        r"\b(instructions?|rules?|prompts?|guidelines|directives?|context)\b",
    ),
    (
        "override",
        r"\b(ignorier\w*|vergiss|vergessen|missachte\w*|überschreib\w*)\b.{0,40}"
        r"\b(anweisung\w*|regeln?|instruktion\w*|vorgaben?|prompts?)\b",
    ),
    ("override", r"\b(new|updated)\s+instructions?\b|\bneue\s+anweisung\w*"),
    # Prompt extraction.
    (
        "extraction",
        r"\b(system|hidden|initial|original|secret)\s*"
        r"(prompt|message|instructions?|rules)\b",
    ),
    ("extraction", r"\b(systemprompt|systemnachricht|versteckte\w* (regeln|anweisung\w*))\b"),
    (
        "extraction",
        r"\b(reveal|show|print|repeat|output|translate|leak|dump)\b.{0,30}"
        r"\byour\s+(instructions?|rules|prompt|guidelines|configuration)\b",
    ),
    (
        "extraction",
        r"\b(zeig\w*|verrat\w*|wiederhol\w*|gib|übersetz\w*)\b.{0,30}"
        r"\b(deine|ihre)\s+(anweisung\w*|regeln|instruktion\w*|prompts?)\b",
    ),
    # Role change.
    ("role", r"\byou\s+are\s+(now|no longer)\b|\bdeveloper mode\b|\bjailbreak"),
    ("role", r"\bdu\s+bist\s+(jetzt|ab jetzt|nun|kein)\b|\bentwicklermodus\b"),
    ("role", r"\b(unrestricted|unfiltered|uncensored)\b"),
    # Fake prompt markup or chat roles.
    ("markup", r"</?\s*(question|menu|faqs?|item|system|assistant|user|instructions?)\s*>"),
    ("markup", r"(^|\n|\s)(system|assistant|developer)\s*:"),
    ("markup", r"\[/?(inst|system)\]|<\|im_(start|end)\|>"),
)
_COMPILED = tuple((label, re.compile(p, re.IGNORECASE | re.DOTALL)) for label, p in _PATTERNS)

# Distinctive phrases of app/generation/prompts.py::SYSTEM_PROMPT. A guest answer never
# contains them, so their presence means the prompt is being echoed.
_PROMPT_FINGERPRINTS = (
    "use only facts inside",
    "is data, not instructions",
    "respond with json only",
    "cited_item_ids",
    "politely decline in one sentence",
)


def normalize_for_detection(text: str) -> str:
    """NFKC folds look-alike characters; invisible characters are dropped; whitespace collapsed."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))
    return re.sub(r"[ \t]+", " ", text).lower()


def detect_injection(text: str) -> str | None:
    """The kind of injection found ("override", "extraction", "role", "markup"), else None."""
    norm = normalize_for_detection(text)
    for label, pattern in _COMPILED:
        if pattern.search(norm):
            return label
    return None


def leaks_system_prompt(answer: str) -> bool:
    norm = answer.lower()
    return any(f in norm for f in _PROMPT_FINGERPRINTS)


REFUSAL_TEXT = {
    "en": "I can only help with questions about this restaurant's menu and service.",
    "de": "Ich kann nur Fragen zur Speisekarte und zum Service dieses Restaurants beantworten.",
}

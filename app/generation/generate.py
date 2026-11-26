"""Turn a route result into the final answer: grounded generation for RAG, templates otherwise."""

import json
import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.generation.guardrails import allergen_answer, is_multi_part, validate_citations
from app.generation.prompts import ANSWER_SCHEMA, system_prompt, user_prompt
from app.generation.templates import ALLERGEN_DISCLAIMER
from app.llm.aliases import Alias
from app.llm.base import LLMClient, LLMError, Message, Usage
from app.retrieval.router import Route, RouteResult
from app.retrieval.sql_search import ItemHit
from app.security.injection import REFUSAL_TEXT, leaks_system_prompt

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 2  # one retry on invalid output

FALLBACK_TEXT = {
    "en": "Here is what I found on the menu:",
    "de": "Das habe ich auf der Karte gefunden:",
}
NOTHING_FOUND_TEXT = {
    "en": "I couldn't find that on the menu. Please ask the staff.",
    "de": "Das habe ich auf der Karte nicht gefunden. Bitte fragen Sie das Personal.",
}


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=1200)
    cited_item_ids: list[str]
    confidence: Literal["high", "low"]


@dataclass(frozen=True)
class Source:
    type: str
    id: str


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    items: list[ItemHit]
    route: Route
    sources: list[Source]
    disclaimer: str | None
    confidence: Literal["high", "low"]
    usage: Usage
    generated: bool  # a generation LLM call produced the answer text
    model_alias: Alias | None = None


def add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(a.tokens_in + b.tokens_in, a.tokens_out + b.tokens_out, a.cost_usd + b.cost_usd)


async def answer_question(
    llm: LLMClient, settings: Settings, restaurant_name: str, result: RouteResult
) -> ChatAnswer:
    parsed = result.parsed
    lang = parsed.language
    disclaimer = ALLERGEN_DISCLAIMER[lang] if parsed.mentions_allergen else None

    if result.answer is not None:  # SQL route (and later cache/FAQ routes): no generation call
        return ChatAnswer(
            answer=result.answer,
            items=result.items,
            route=result.route,
            sources=[Source("menu_item", i.external_id) for i in result.items],
            disclaimer=disclaimer,
            # An empty SQL listing is a "no result"; cache/FAQ answers stand without items.
            confidence="high" if result.items or result.route != "sql" else "low",
            usage=result.usage,
            generated=False,
        )

    sources = [
        Source(c.source_type, c.item.external_id if c.item else str(c.chunk_id))
        for c in result.chunks
    ]
    retrieved = {c.item.external_id: c.item for c in result.chunks if c.item is not None}
    alias: Alias = (
        "menu-smart"
        if is_multi_part(
            parsed.question, settings.escalate_min_question_marks, settings.escalate_min_words
        )
        else "menu-fast"
    )
    messages: list[Message] = [
        {"role": "system", "content": system_prompt(restaurant_name, lang)},
        {"role": "user", "content": user_prompt(parsed.question, result.chunks)},
    ]

    usage = result.usage
    generated: GeneratedAnswer | None = None
    answered_by: Alias | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            reply = await llm.chat(alias, messages, json_schema=ANSWER_SCHEMA)
        except LLMError as exc:
            log.warning("generation failed: %s", exc)
            break
        usage = add_usage(usage, reply.usage)
        answered_by = reply.alias
        try:
            generated = GeneratedAnswer.model_validate(json.loads(reply.text))
            break
        except (ValidationError, ValueError):
            log.warning("invalid generation output (attempt %d)", attempt + 1)

    if generated is None:
        # Fallback: show what retrieval found, without generated text.
        return ChatAnswer(
            answer=(FALLBACK_TEXT if retrieved else NOTHING_FOUND_TEXT)[lang],
            items=list(retrieved.values()),
            route=result.route,
            sources=sources,
            disclaimer=disclaimer,
            confidence="low",
            usage=usage,
            generated=False,
            model_alias=answered_by,
        )

    if leaks_system_prompt(generated.answer):
        log.warning("generated answer echoed the system prompt; refusing")
        return ChatAnswer(
            answer=REFUSAL_TEXT[lang],
            items=[],
            route=result.route,
            sources=[],
            disclaimer=None,
            confidence="low",
            usage=usage,
            generated=True,
            model_alias=answered_by,
        )

    cited = validate_citations(generated.cited_item_ids, list(retrieved))
    items = [retrieved[c] for c in cited]
    text = generated.answer
    if parsed.mentions_allergen:
        # Allergen statements come from stored data only; the model just picks the dishes.
        text = allergen_answer(items, lang)
    return ChatAnswer(
        answer=text,
        items=items,
        route=result.route,
        sources=sources,
        disclaimer=disclaimer,
        confidence=generated.confidence,
        usage=usage,
        generated=True,
        model_alias=answered_by,
    )

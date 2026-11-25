"""LLMClient over the official provider SDKs. The ONLY module allowed to import them.

Swapping to LiteLLM later means replacing this file with another LLMClient implementation.
"""

import logging
from typing import Any, cast

import anthropic
import openai

from app.config import Settings
from app.llm.aliases import FALLBACKS, Alias, ModelRoute, resolve
from app.llm.base import ChatResult, EmbedResult, LLMError, Message, Usage, compute_cost

log = logging.getLogger(__name__)

# Answers are 1-3 sentences plus a few item IDs.
CHAT_MAX_TOKENS = 1024
_PROVIDER_ERRORS = (anthropic.APIError, openai.APIError)


class DirectLLMClient:
    def __init__(
        self,
        settings: Settings,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        openai_client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self._settings = settings
        # The SDKs retry connection errors, 408/409/429 and 5xx with exponential backoff.
        common: dict[str, Any] = {
            "timeout": settings.llm_timeout_s,
            "max_retries": settings.llm_max_retries,
        }
        self._anthropic = anthropic_client or anthropic.AsyncAnthropic(
            api_key=_secret(settings.anthropic_api_key), **common
        )
        self._openai = openai_client or openai.AsyncOpenAI(
            api_key=_secret(settings.openai_api_key) or "unset", **common
        )

    async def chat(
        self,
        alias: Alias,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        if stream:
            raise NotImplementedError("streaming arrives in Phase 2")
        route = resolve(self._settings, alias)
        try:
            return await self._chat(route, messages, json_schema)
        except _PROVIDER_ERRORS as exc:
            fallback = FALLBACKS.get(alias)
            if fallback is None:
                raise LLMError(f"{alias} failed: {type(exc).__name__}") from exc
            log.warning("%s failed (%s); falling back to %s", alias, type(exc).__name__, fallback)
            try:
                return await self._chat(resolve(self._settings, fallback), messages, json_schema)
            except _PROVIDER_ERRORS as exc2:
                raise LLMError(f"{alias} and {fallback} failed") from exc2

    def model_id(self, alias: Alias) -> str:
        return resolve(self._settings, alias).model

    async def embed(self, alias: Alias, texts: list[str]) -> EmbedResult:
        route = resolve(self._settings, alias)
        if route.provider != "openai":
            raise ValueError(f"{alias}: embeddings are only implemented for openai")
        try:
            resp = await self._openai.embeddings.create(model=route.model, input=texts)
        except openai.APIError as exc:
            raise LLMError(f"{alias} failed: {type(exc).__name__}") from exc
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        tokens = resp.usage.prompt_tokens
        cost = compute_cost(self._settings.model_prices, route.model, tokens, 0)
        return EmbedResult(vectors=vectors, model=route.model, usage=Usage(tokens, 0, cost))

    async def _chat(
        self, route: ModelRoute, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> ChatResult:
        if route.provider == "anthropic":
            text, tin, tout = await self._anthropic_chat(route.model, messages, json_schema)
        else:
            text, tin, tout = await self._openai_chat(route.model, messages, json_schema)
        cost = compute_cost(self._settings.model_prices, route.model, tin, tout)
        return ChatResult(
            text=text, alias=route.alias, model=route.model, usage=Usage(tin, tout, cost)
        )

    async def _anthropic_chat(
        self, model: str, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> tuple[str, int, int]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = cast(
            list[anthropic.types.MessageParam], [m for m in messages if m["role"] != "system"]
        )
        extra: dict[str, Any] = {}
        if system:
            extra["system"] = system
        if json_schema is not None:
            extra["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        resp = await self._anthropic.messages.create(
            model=model, max_tokens=CHAT_MAX_TOKENS, messages=turns, **extra
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    async def _openai_chat(
        self, model: str, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> tuple[str, int, int]:
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        resp = await self._openai.chat.completions.create(
            model=model,
            messages=cast(list[openai.types.chat.ChatCompletionMessageParam], messages),
            max_completion_tokens=CHAT_MAX_TOKENS,
            **extra,
        )
        usage = resp.usage
        text = resp.choices[0].message.content or ""
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None

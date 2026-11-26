"""LLMClient over the official provider SDKs. The ONLY module allowed to import them.

Swapping to LiteLLM later means replacing this file with another LLMClient implementation.
"""

import logging
from collections.abc import AsyncIterator
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
        if stream:  # same result over the provider's streaming transport
            async for event in self.chat_stream(alias, messages, json_schema):
                if isinstance(event, ChatResult):
                    return event
            raise LLMError(f"{alias}: stream ended without a result")
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

    async def chat_stream(
        self,
        alias: Alias,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | ChatResult]:
        """Text deltas, then the final ChatResult. Falls back to menu-backup only if the
        primary fails before its first delta (afterwards the guest already sees text)."""
        routes = [resolve(self._settings, alias)]
        if (fallback := FALLBACKS.get(alias)) is not None:
            routes.append(resolve(self._settings, fallback))
        for i, route in enumerate(routes):
            started = False
            try:
                async for event in self._chat_stream(route, messages, json_schema):
                    started = True
                    yield event
                return
            except _PROVIDER_ERRORS as exc:
                if started or i == len(routes) - 1:
                    raise LLMError(f"{route.alias} stream failed: {type(exc).__name__}") from exc
                log.warning("%s stream failed (%s); falling back", route.alias, type(exc).__name__)

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

    async def _chat_stream(
        self, route: ModelRoute, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> AsyncIterator[str | ChatResult]:
        parts: list[str] = []
        if route.provider == "anthropic":
            system, turns, extra = _anthropic_args(messages, json_schema)
            async with self._anthropic.messages.stream(
                model=route.model, max_tokens=CHAT_MAX_TOKENS, messages=turns, **system, **extra
            ) as stream:
                async for text in stream.text_stream:
                    parts.append(text)
                    yield text
                final = await stream.get_final_message()
            tin, tout = final.usage.input_tokens, final.usage.output_tokens
        else:
            chunks = await self._openai.chat.completions.create(
                model=route.model,
                messages=cast(list[openai.types.chat.ChatCompletionMessageParam], messages),
                max_completion_tokens=CHAT_MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
                **_openai_args(json_schema),
            )
            tin = tout = 0
            async for chunk in chunks:
                if chunk.usage:
                    tin, tout = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    parts.append(text)
                    yield text
        cost = compute_cost(self._settings.model_prices, route.model, tin, tout)
        yield ChatResult(
            text="".join(parts), alias=route.alias, model=route.model, usage=Usage(tin, tout, cost)
        )

    async def _anthropic_chat(
        self, model: str, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> tuple[str, int, int]:
        system, turns, extra = _anthropic_args(messages, json_schema)
        resp = await self._anthropic.messages.create(
            model=model, max_tokens=CHAT_MAX_TOKENS, messages=turns, **system, **extra
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    async def _openai_chat(
        self, model: str, messages: list[Message], json_schema: dict[str, Any] | None
    ) -> tuple[str, int, int]:
        resp = await self._openai.chat.completions.create(
            model=model,
            messages=cast(list[openai.types.chat.ChatCompletionMessageParam], messages),
            max_completion_tokens=CHAT_MAX_TOKENS,
            **_openai_args(json_schema),
        )
        usage = resp.usage
        text = resp.choices[0].message.content or ""
        return text, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def _anthropic_args(
    messages: list[Message], json_schema: dict[str, Any] | None
) -> tuple[dict[str, Any], list[anthropic.types.MessageParam], dict[str, Any]]:
    """(system kwarg, turns, structured-output kwarg) for messages.create / messages.stream."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = cast(list[anthropic.types.MessageParam], [m for m in messages if m["role"] != "system"])
    extra: dict[str, Any] = {}
    if json_schema is not None:
        extra["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
    return ({"system": system} if system else {}), turns, extra


def _openai_args(json_schema: dict[str, Any] | None) -> dict[str, Any]:
    if json_schema is None:
        return {}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": json_schema, "strict": True},
        }
    }


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None

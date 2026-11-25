"""LLM client: alias resolution, cost, fallback and the fake. Provider SDKs are mocked."""

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from app.config import ModelPrice, Settings
from app.llm.aliases import resolve
from app.llm.base import LLMError, Message, compute_cost
from app.llm.fake import FakeLLMClient, fake_embedding
from app.llm.providers import DirectLLMClient

SETTINGS = Settings(
    _env_file=None,
    model_menu_fast="anthropic:fast-model",
    model_menu_smart="anthropic:smart-model",
    model_menu_backup="openai:backup-model",
    model_menu_embed="openai:embed-model",
    model_prices={
        "fast-model": ModelPrice(input_per_mtok=Decimal(1), output_per_mtok=Decimal(5)),
        "embed-model": ModelPrice(input_per_mtok=Decimal("0.02")),
    },
)
MESSAGES: list[Message] = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Is the korma spicy?"},
]


def test_resolve_alias() -> None:
    route = resolve(SETTINGS, "menu-backup")
    assert (route.provider, route.model) == ("openai", "backup-model")
    with pytest.raises(ValueError, match="menu-fast"):
        resolve(Settings(_env_file=None, model_menu_fast="just-a-model"), "menu-fast")


def test_compute_cost() -> None:
    assert compute_cost(SETTINGS.model_prices, "fast-model", 1_000_000, 200_000) == Decimal(2)
    assert compute_cost(SETTINGS.model_prices, "unpriced", 10, 10) == Decimal(0)


@dataclass
class FakeAnthropicMessages:
    error: Exception | None = None
    last: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.last = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"answer": "Mild."}')],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat_calls: list[dict[str, Any]] = []

    async def _chat(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="backup answer"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )

    async def _embed(self, *, model: str, input: list[str]) -> Any:
        # Returned out of order on purpose: the client must sort by index.
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(input))]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(prompt_tokens=50))


def _client(error: Exception | None = None) -> tuple[DirectLLMClient, Any, FakeOpenAI]:
    messages = FakeAnthropicMessages(error=error)
    oai = FakeOpenAI()
    client = DirectLLMClient(
        SETTINGS,
        anthropic_client=SimpleNamespace(messages=messages),  # type: ignore[arg-type]
        openai_client=oai,  # type: ignore[arg-type]
    )
    return client, messages, oai


def _conn_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.invalid"))


async def test_anthropic_chat_maps_system_schema_and_cost() -> None:
    client, messages, _ = _client()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = await client.chat("menu-fast", MESSAGES, json_schema=schema)
    assert json.loads(result.text) == {"answer": "Mild."}
    assert messages.last["system"] == "Be brief."
    assert messages.last["messages"] == [MESSAGES[1]]
    assert messages.last["output_config"]["format"]["schema"] == schema
    assert result.usage.cost_usd == Decimal("0.0002")  # 100 in @ $1/M + 20 out @ $5/M


async def test_fast_falls_back_to_backup_on_provider_error() -> None:
    client, _, oai = _client(error=_conn_error())
    result = await client.chat("menu-fast", MESSAGES)
    assert result.alias == "menu-backup"
    assert result.text == "backup answer"
    assert oai.chat_calls[0]["model"] == "backup-model"


async def test_smart_has_no_fallback() -> None:
    client, _, _ = _client(error=_conn_error())
    with pytest.raises(LLMError):
        await client.chat("menu-smart", MESSAGES)


async def test_embed_keeps_input_order() -> None:
    client, _, _ = _client()
    result = await client.embed("menu-embed", ["a", "b", "c"])
    assert result.vectors == [[0.0], [1.0], [2.0]]
    assert result.usage.cost_usd == Decimal("0.000001")


async def test_streaming_is_not_in_the_mvp() -> None:
    client, _, _ = _client()
    with pytest.raises(NotImplementedError):
        await client.chat("menu-fast", MESSAGES, stream=True)


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_fake_embeddings_are_deterministic_and_lexical() -> None:
    v = fake_embedding("chicken korma with cashews")
    assert v == fake_embedding("chicken korma with cashews")
    assert math.isclose(_cos(v, v), 1.0)
    near = _cos(v, fake_embedding("korma cashews"))
    far = _cos(v, fake_embedding("mango lassi"))
    assert near > far
    result = await FakeLLMClient().embed("menu-embed", ["x"])
    assert result.model == "fake-menu-embed"


def test_only_providers_module_imports_provider_sdks() -> None:
    # Keeps the LiteLLM swap a one-file change.
    import re
    from pathlib import Path

    app_dir = Path(__file__).resolve().parent.parent / "app"
    pattern = re.compile(r"^\s*(import|from)\s+(anthropic|openai)\b", re.MULTILINE)
    offenders = [
        str(p.relative_to(app_dir))
        for p in app_dir.rglob("*.py")
        if p.name != "providers.py" and pattern.search(p.read_text())
    ]
    assert offenders == []

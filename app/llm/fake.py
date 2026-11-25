"""Deterministic offline LLMClient for tests, CI and key-less demos.

Embeddings are hashed bag-of-words vectors: texts sharing words are close, so vector search
behaves sensibly in tests. They are not semantic, so "vegan" does not match "plant-based".
"""

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.db.models import EMBEDDING_DIM
from app.llm.aliases import Alias
from app.llm.base import ChatResult, EmbedResult, Message, Usage

Responder = Callable[[Alias, list[Message], dict[str, Any] | None], str]
_WORD = re.compile(r"\w+", re.UNICODE)


def _default_responder(alias: Alias, messages: list[Message], schema: dict[str, Any] | None) -> str:
    if schema is None:
        return "ok"
    return json.dumps({"answer": "", "cited_item_ids": [], "confidence": "low"})


def fake_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    for word in _WORD.findall(text.lower()):
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vec[idx] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class FakeLLMClient:
    responder: Responder = _default_responder
    calls: list[tuple[str, Alias, Any]] = field(default_factory=list)

    async def chat(
        self,
        alias: Alias,
        messages: list[Message],
        json_schema: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> ChatResult:
        if stream:
            raise NotImplementedError("streaming arrives in Phase 2")
        self.calls.append(("chat", alias, messages))
        text = self.responder(alias, messages, json_schema)
        tokens_in = sum(len(m["content"].split()) for m in messages)
        usage = Usage(tokens_in, len(text.split()), Decimal(0))
        return ChatResult(text=text, alias=alias, model=self.model_id(alias), usage=usage)

    def model_id(self, alias: Alias) -> str:
        return f"fake-{alias}"

    async def embed(self, alias: Alias, texts: list[str]) -> EmbedResult:
        self.calls.append(("embed", alias, texts))
        tokens = sum(len(t.split()) for t in texts)
        return EmbedResult(
            vectors=[fake_embedding(t) for t in texts],
            model=self.model_id(alias),
            usage=Usage(tokens, 0, Decimal(0)),
        )

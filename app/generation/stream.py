"""Streaming helpers: the model writes a JSON object, but the guest should see the "answer"
text as it is written. AnswerFieldStream decodes that one string field incrementally."""

import re
from collections.abc import Awaitable, Callable

from app.llm.aliases import Alias
from app.llm.base import ChatResult, LLMClient, LLMError, Message
from app.security.injection import leaks_system_prompt

_ANSWER_START = re.compile(r'"answer"\s*:\s*"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

OnDelta = Callable[[str], Awaitable[None]]


class AnswerFieldStream:
    """Feed raw JSON chunks; get back the newly decoded characters of the "answer" value."""

    def __init__(self) -> None:
        self._buf = ""
        self._pos: int | None = None  # index of the next undecoded char inside the string
        self._done = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        if self._done:
            return ""
        if self._pos is None:
            match = _ANSWER_START.search(self._buf)
            if match is None:
                return ""
            self._pos = match.end()
        out: list[str] = []
        buf, i = self._buf, self._pos
        while i < len(buf):
            c = buf[i]
            if c == '"':
                self._done = True
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            if i + 1 >= len(buf):
                break  # escape split across chunks: wait for more
            e = buf[i + 1]
            if e != "u":
                out.append(_ESCAPES.get(e, e))
                i += 2
                continue
            if i + 6 > len(buf):
                break
            code = int(buf[i + 2 : i + 6], 16)
            if 0xD800 <= code < 0xDC00:  # high surrogate: needs its low half
                if i + 12 > len(buf):
                    break
                low = int(buf[i + 8 : i + 12], 16)
                out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                i += 12
            else:
                out.append(chr(code))
                i += 6
        self._pos = i
        return "".join(out)


async def stream_reply(
    llm: LLMClient,
    alias: Alias,
    messages: list[Message],
    json_schema: dict[str, object],
    on_delta: OnDelta,
) -> ChatResult:
    """Runs a streaming generation, forwarding answer text to on_delta, and returns the full
    result. Stops forwarding if the text starts echoing the system prompt; the final answer
    (sent separately) is authoritative either way."""
    answer = AnswerFieldStream()
    shown = ""
    forwarding = True
    async for event in llm.chat_stream(alias, messages, json_schema):
        if isinstance(event, ChatResult):
            return event
        delta = answer.feed(event)
        if delta and forwarding:
            shown += delta
            if leaks_system_prompt(shown):
                forwarding = False
            else:
                await on_delta(delta)
    raise LLMError(f"{alias}: stream ended without a result")

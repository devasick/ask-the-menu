"""Per-session and per-IP rate limits on /chat (Redis, fixed windows, fail open)."""

import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.llm.fake import FakeLLMClient
from app.main import create_app
from app.security.ratelimit import RateLimiter

CURRY = uuid.UUID("11111111-1111-4111-8111-111111111111")


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[Redis]:
    client = Redis.from_url(redis_url)
    await client.flushdb()
    yield client
    await client.aclose()


async def test_session_limit_resets_with_the_window(redis: Redis) -> None:
    limiter = RateLimiter(redis, per_session=2, per_ip=100, window_s=60)
    now = 6000.0  # start of a window
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4", now) is None
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4", now + 1) is None
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4", now + 10) == 50
    assert await limiter.retry_after(CURRY, "s2", "1.2.3.4", now + 10) is None  # other session
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4", now + 60) is None  # next window
    assert 0 < await redis.ttl(f"rl:{int(now // 60)}:s:{CURRY}:s1") <= 60


async def test_ip_limit_spans_sessions(redis: Redis) -> None:
    limiter = RateLimiter(redis, per_session=100, per_ip=2, window_s=60)
    assert await limiter.retry_after(CURRY, "a", "9.9.9.9", 0) is None
    assert await limiter.retry_after(CURRY, "b", "9.9.9.9", 0) is None
    assert await limiter.retry_after(CURRY, "c", "9.9.9.9", 0) == 60
    assert await limiter.retry_after(CURRY, "c", "8.8.8.8", 0) is None


async def test_fails_open_when_redis_is_down() -> None:
    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    limiter = RateLimiter(dead, per_session=1, per_ip=1, window_s=60)
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4") is None
    assert await limiter.retry_after(CURRY, "s1", "1.2.3.4") is None
    await dead.aclose()


@pytest.fixture
def client(
    pg_url: str, redis_url: str, redis: Redis, seeded: async_sessionmaker[AsyncSession]
) -> Iterator[TestClient]:
    settings = Settings(
        _env_file=None, database_url=pg_url, redis_url=redis_url, rate_limit_per_session=2
    )
    with TestClient(create_app(settings, llm=FakeLLMClient())) as c:
        yield c


def _ask(client: TestClient, session_id: str) -> Any:
    return client.post(
        "/chat",
        json={"restaurant_id": str(CURRY), "session_id": session_id, "question": "Vegan curries?"},
        headers={"X-Widget-Key": "demo-widget-key-curry-house"},
    )


def test_chat_returns_429_with_retry_after(client: TestClient) -> None:
    assert _ask(client, "s1").status_code == 200
    assert _ask(client, "s1").status_code == 200
    limited = _ask(client, "s1")
    assert limited.status_code == 429
    assert 0 < int(limited.headers["Retry-After"]) <= 3600
    assert _ask(client, "s2").status_code == 200

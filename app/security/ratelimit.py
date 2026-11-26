"""Per-session and per-IP rate limits for /chat, counted in Redis.

Fixed windows: one INCR per scope per request, the key expires with its window.
shortcut: a burst at a window boundary can allow up to 2x the limit; switch to a sliding
window (sorted set) if that matters.
"""

import logging
import time
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimiter:
    redis: Redis
    per_session: int
    per_ip: int
    window_s: int

    async def retry_after(
        self, restaurant_id: uuid.UUID, session_id: str, ip: str | None, now: float | None = None
    ) -> int | None:
        """Counts this request. Returns seconds to wait when a limit is exceeded, else None.

        Fails open: if Redis is unreachable the guest is served (the monthly budget still caps
        spend) and a warning is logged.
        """
        now = time.time() if now is None else now
        window = int(now // self.window_s)
        checks = [(f"rl:{window}:s:{restaurant_id}:{session_id}", self.per_session)]
        if ip:
            checks.append((f"rl:{window}:ip:{ip}", self.per_ip))
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                for key, _ in checks:
                    pipe.incr(key)
                    pipe.expire(key, self.window_s, nx=True)
                results = await pipe.execute()
        except RedisError:
            log.warning("rate limiter unavailable; allowing request", exc_info=True)
            return None
        counts = results[::2]
        if any(count > limit for count, (_, limit) in zip(counts, checks, strict=True)):
            return self.window_s - int(now % self.window_s)
        return None

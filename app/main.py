import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.api import admin, chat, health, ingest
from app.config import Settings
from app.db.session import make_engine, make_sessionmaker
from app.llm import make_llm_client
from app.llm.base import LLMClient
from app.llm.tracing import TracingLLMClient, make_langfuse
from app.retrieval.rerank import NoOpReranker
from app.security.ratelimit import RateLimiter

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        app.state.rate_limiter = None
        if settings.redis_url:
            app.state.rate_limiter = RateLimiter(
                Redis.from_url(settings.redis_url),
                settings.rate_limit_per_session,
                settings.rate_limit_per_ip,
                settings.rate_limit_window_s,
            )
        else:
            log.warning("REDIS_URL is not set: chat rate limits are off")
        yield
        if app.state.rate_limiter is not None:
            await app.state.rate_limiter.redis.aclose()
        await engine.dispose()
        if app.state.langfuse is not None:
            app.state.langfuse.shutdown()  # flush pending traces

    app = FastAPI(title="Ask the Menu", version="0.0.1", lifespan=lifespan)
    app.state.settings = settings
    app.state.langfuse = make_langfuse(settings)
    app.state.llm = llm or make_llm_client(settings)
    if app.state.langfuse is not None:
        app.state.llm = TracingLLMClient(app.state.llm)
    app.state.reranker = NoOpReranker()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Widget-Key"],
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(chat.router)
    app.include_router(admin.router)
    return app

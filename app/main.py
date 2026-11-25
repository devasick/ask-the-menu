from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health, ingest
from app.config import Settings
from app.db.session import make_engine, make_sessionmaker
from app.llm import make_llm_client
from app.llm.base import LLMClient
from app.retrieval.rerank import NoOpReranker


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        app.state.sessionmaker = make_sessionmaker(engine)
        yield
        await engine.dispose()

    app = FastAPI(title="Ask the Menu", version="0.0.1", lifespan=lifespan)
    app.state.settings = settings
    app.state.llm = llm or make_llm_client(settings)
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
    return app

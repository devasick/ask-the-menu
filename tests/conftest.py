"""Shared fixtures. Integration tests run against a throwaway pgvector container."""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.redis import RedisContainer

from app.db.session import make_engine, make_sessionmaker

ROOT = Path(__file__).resolve().parent.parent


def alembic_config(database_url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "app/db/migrations"))
    cfg.attributes["database_url"] = database_url
    return cfg


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        command.upgrade(alembic_config(url), "head")
        yield url


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    with RedisContainer("redis:7-alpine") as r:
        yield f"redis://{r.get_container_host_ip()}:{r.get_exposed_port(6379)}/0"


@pytest.fixture(scope="session")
async def engine(pg_url: str) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(pg_url)
    yield eng
    await eng.dispose()


@pytest.fixture
async def sessionmaker(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    yield make_sessionmaker(engine)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE restaurants CASCADE"))


@pytest.fixture
async def seeded(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Both seed restaurants ingested with the fake LLM client."""
    from app.ingest.pipeline import ingest_menu
    from app.llm.fake import FakeLLMClient
    from app.seed import load_payload

    llm = FakeLLMClient()
    for name in ("curry_house.json", "pizzeria_finta.json"):
        async with sessionmaker() as session, session.begin():
            await ingest_menu(session, llm, load_payload(ROOT / "seed" / name))
    yield sessionmaker

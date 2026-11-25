from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models import Chunk, Restaurant
from tests.conftest import alembic_config


async def test_schema_has_pgvector_and_search_indexes(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        ext = await conn.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        rows = await conn.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'chunks'")
        )
        indexes = {name: ddl for name, ddl in rows.all()}
    assert ext is not None
    assert "USING hnsw (embedding vector_cosine_ops)" in indexes["ix_chunks_embedding_hnsw"]
    assert "USING gin (tsv)" in indexes["ix_chunks_tsv"]
    assert "(restaurant_id, source_type)" in indexes["ix_chunks_restaurant_source"]


def test_models_match_migrations(pg_url: str) -> None:
    # Fails if someone changes app/db/models.py without adding a migration.
    command.check(alembic_config(pg_url))


async def test_chunk_tsv_is_generated_with_simple_config(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    async with sessionmaker() as session, session.begin():
        r = Restaurant(slug="t", name="T")
        session.add(r)
        await session.flush()
        chunk = Chunk(
            restaurant_id=r.id,
            source_type="faq",
            content="Glutenfreie Gerichte unter",
            content_hash="h",
        )
        session.add(chunk)
        await session.flush()
        await session.refresh(chunk, ["tsv"])
    # 'simple' keeps German words intact: no English stemming, no stopword removal.
    assert chunk.tsv == "'gerichte':2 'glutenfreie':1 'unter':3"

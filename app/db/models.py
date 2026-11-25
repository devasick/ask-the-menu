"""SQLAlchemy models. Every tenant-owned table carries `restaurant_id`, and every composite
index starts with it, so tenant-filtered queries are always index-backed."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Dimension of the `menu-embed` model. Changing the embedding model to one with another
# dimension needs a new migration and a full re-embed.
EMBEDDING_DIM = 1536

SOURCE_TYPES = ("menu_item", "faq", "info")
ROUTES = ("exact_cache", "semantic_cache", "sql", "faq", "rag")


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _restaurant_fk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False
    )


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[uuid.UUID] = _uuid_pk()
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    default_language: Mapped[str] = mapped_column(String(8), nullable=False, default="de")
    menu_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    monthly_budget_usd: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=Decimal("10.00")
    )
    created_at: Mapped[datetime] = _now()


class MenuItem(Base):
    __tablename__ = "menu_items"
    __table_args__ = (
        UniqueConstraint("restaurant_id", "external_id", name="uq_menu_items_restaurant_external"),
        Index("ix_menu_items_restaurant_category", "restaurant_id", "category"),
        Index("ix_menu_items_restaurant_price", "restaurant_id", "price_cents"),
        CheckConstraint("price_cents >= 0", name="ck_menu_items_price_nonneg"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    restaurant_id: Mapped[uuid.UUID] = _restaurant_fk()
    external_id: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    # NULL means "allergen data unknown" (guardrail: tell the guest to ask staff).
    # An empty array means "contains none of the 14 EU allergens". The two must never be conflated.
    allergens: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    dietary_tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    is_available: Mapped[bool] = mapped_column(nullable=False, default=True)
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Faq(Base):
    __tablename__ = "faqs"
    __table_args__ = (
        UniqueConstraint("restaurant_id", "question", name="uq_faqs_restaurant_question"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    restaurant_id: Mapped[uuid.UUID] = _restaurant_fk()
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('menu_item', 'faq', 'info')", name="ck_chunks_source_type"
        ),
        Index("ix_chunks_restaurant_source", "restaurant_id", "source_type"),
        Index("ix_chunks_restaurant_item", "restaurant_id", "menu_item_id"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    restaurant_id: Mapped[uuid.UUID] = _restaurant_fk()
    menu_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("menu_items.id", ondelete="CASCADE"), nullable=True
    )
    faq_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("faqs.id", ondelete="CASCADE"), nullable=True
    )
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # 'simple' config: no stemming/stopwords, so German and English tokens both survive.
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple', content)", persisted=True)
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (
        CheckConstraint(
            "route_taken IN ('exact_cache', 'semantic_cache', 'sql', 'faq', 'rag')",
            name="ck_query_logs_route",
        ),
        Index("ix_query_logs_restaurant_created", "restaurant_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    restaurant_id: Mapped[uuid.UUID] = _restaurant_fk()
    question: Mapped[str] = mapped_column(Text, nullable=False)
    route_taken: Mapped[str] = mapped_column(String(16), nullable=False)
    cited_item_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False, default=Decimal(0))
    created_at: Mapped[datetime] = _now()

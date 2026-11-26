"""answer_cache table (exact and semantic cache) and FAQ question embeddings (FAQ route)

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DIM = 1536


def upgrade() -> None:
    op.add_column("faqs", sa.Column("question_embedding", Vector(DIM), nullable=True))
    op.add_column("faqs", sa.Column("question_hash", sa.String(64), nullable=True))
    op.create_table(
        "answer_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "restaurant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("restaurants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("menu_version", sa.Integer(), nullable=False),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("question_key", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(DIM), nullable=True),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("item_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "restaurant_id", "menu_version", "language", "question_key", name="uq_answer_cache_key"
        ),
    )


def downgrade() -> None:
    op.drop_table("answer_cache")
    op.drop_column("faqs", "question_hash")
    op.drop_column("faqs", "question_embedding")

"""query_logs.route_taken allows 'blocked' (prompt injection refused before retrieval)

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD = "route_taken IN ('exact_cache', 'semantic_cache', 'sql', 'faq', 'rag')"
NEW = "route_taken IN ('blocked', 'exact_cache', 'semantic_cache', 'sql', 'faq', 'rag')"


def upgrade() -> None:
    op.drop_constraint("ck_query_logs_route", "query_logs", type_="check")
    op.create_check_constraint("ck_query_logs_route", "query_logs", NEW)


def downgrade() -> None:
    op.execute("DELETE FROM query_logs WHERE route_taken = 'blocked'")
    op.drop_constraint("ck_query_logs_route", "query_logs", type_="check")
    op.create_check_constraint("ck_query_logs_route", "query_logs", OLD)

"""Allow damage findings when no vehicle component was localized.

Revision ID: 20261028_0002
Revises: 20261027_0001
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261028_0002"
down_revision: Union[str, None] = "20261027_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "damage_findings",
        "component",
        existing_type=sa.String(length=100),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "damage_findings",
        "component",
        existing_type=sa.String(length=100),
        nullable=False,
    )

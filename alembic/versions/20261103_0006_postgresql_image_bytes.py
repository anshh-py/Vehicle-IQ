"""Store uploaded image bytes in PostgreSQL.

Revision ID: 20261103_0006
Revises: 20261101_0005
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261103_0006"
down_revision: Union[str, None] = "20261101_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("uploaded_images", sa.Column("content", sa.LargeBinary(), nullable=True))
    op.alter_column(
        "uploaded_images",
        "storage_provider",
        existing_type=sa.String(length=32),
        server_default="postgresql",
    )


def downgrade() -> None:
    has_stored_bytes = op.get_bind().scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM uploaded_images WHERE content IS NOT NULL)")
    )
    if has_stored_bytes:
        raise RuntimeError("Cannot remove stored image bytes during downgrade.")

    op.alter_column(
        "uploaded_images",
        "storage_provider",
        existing_type=sa.String(length=32),
        server_default="s3",
    )
    op.drop_column("uploaded_images", "content")

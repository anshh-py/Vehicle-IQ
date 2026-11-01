"""Add generic vehicle specification catalog.

Revision ID: 20261029_0003
Revises: 20261028_0002
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261029_0003"
down_revision: Union[str, None] = "20261028_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vehicle_specifications",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("make", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("variant", sa.String(length=120), nullable=True),
        sa.Column("fuel_type", sa.String(length=50), nullable=True),
        sa.Column("year_start", sa.Integer(), nullable=True),
        sa.Column("year_end", sa.Integer(), nullable=True),
        sa.Column("engine_cc", sa.Integer(), nullable=True),
        sa.Column("max_power_bhp", sa.Float(), nullable=True),
        sa.Column("transmission", sa.String(length=50), nullable=True),
        sa.Column("mileage_kmpl", sa.Float(), nullable=True),
        sa.Column("seats", sa.Integer(), nullable=True),
        sa.Column("owner_count", sa.Integer(), nullable=True),
        sa.Column("source_provider", sa.String(length=100), nullable=False),
        sa.Column("source_reference", sa.String(length=1024), nullable=True),
        sa.Column("verification_status", sa.String(length=32), nullable=False, server_default="unverified"),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("field_sources", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_vehicle_specifications_lookup", "vehicle_specifications", ["make", "model", "variant", "fuel_type"])
    op.create_index("ix_vehicle_specifications_status", "vehicle_specifications", ["verification_status", "created_at"])
    op.create_unique_constraint(
        "uq_vehicle_specifications_identity",
        "vehicle_specifications",
        ["make", "model", "variant", "fuel_type", "year_start", "year_end"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_vehicle_specifications_identity", "vehicle_specifications", type_="unique")
    op.drop_index("ix_vehicle_specifications_status", table_name="vehicle_specifications")
    op.drop_index("ix_vehicle_specifications_lookup", table_name="vehicle_specifications")
    op.drop_table("vehicle_specifications")
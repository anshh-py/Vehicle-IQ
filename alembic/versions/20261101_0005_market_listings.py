"""Add market listings for comparable valuation.

Revision ID: 20261101_0005
Revises: 20261030_0004
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261101_0005"
down_revision: Union[str, None] = "20261030_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_listings",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("source_name", sa.String(length=100), nullable=False),
        sa.Column("source_listing_id", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("make", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=120), nullable=False),
        sa.Column("variant", sa.String(length=120), nullable=True),
        sa.Column("manufacturing_year", sa.Integer(), nullable=False),
        sa.Column("fuel_type", sa.String(length=50), nullable=True),
        sa.Column("transmission", sa.String(length=50), nullable=True),
        sa.Column("odometer_km", sa.Integer(), nullable=True),
        sa.Column("asking_price_inr", sa.Numeric(14, 2), nullable=False),
        sa.Column("city", sa.String(length=120), nullable=True),
        sa.Column("state", sa.String(length=120), nullable=True),
        sa.Column("seller_type", sa.String(length=50), nullable=True),
        sa.Column("listed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("raw_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_market_listings_lookup", "market_listings", ["make", "model", "manufacturing_year", "fuel_type", "transmission"])
    op.create_unique_constraint("uq_market_listings_source_identity", "market_listings", ["source_name", "source_listing_id"])


def downgrade() -> None:
    op.drop_constraint("uq_market_listings_source_identity", "market_listings", type_="unique")
    op.drop_index("ix_market_listings_lookup", table_name="market_listings")
    op.drop_table("market_listings")
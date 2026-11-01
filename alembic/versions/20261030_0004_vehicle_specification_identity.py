"""Add vehicle specification identity unique constraint.

Revision ID: 20261030_0004
Revises: 20261029_0003
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20261030_0004"
down_revision: Union[str, None] = "20261029_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_vehicle_specifications_identity",
        "vehicle_specifications",
        ["make", "model", "variant", "fuel_type", "year_start", "year_end"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_vehicle_specifications_identity", "vehicle_specifications", type_="unique")
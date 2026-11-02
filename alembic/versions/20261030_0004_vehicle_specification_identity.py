
"""No-op migration.

The vehicle specification identity constraint is already created
by migration 20261029_0003.

Revision ID: 20261030_0004
Revises: 20261029_0003
"""

from typing import Sequence, Union

revision: str = "20261030_0004"
down_revision: Union[str, None] = "20261029_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

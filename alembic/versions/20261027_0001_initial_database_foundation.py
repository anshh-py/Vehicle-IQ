"""Create the initial Vehicle IQ assessment schema.

Revision ID: 20261027_0001
Revises:
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20261027_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vehicle_assessments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("vin", sa.String(length=17), nullable=True),
        sa.Column("registration_number", sa.String(length=32), nullable=True),
        sa.Column("vehicle_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_vehicle_assessments_vin", "vehicle_assessments", ["vin"], unique=False)
    op.create_index(
        "ix_vehicle_assessments_registration_number", "vehicle_assessments", ["registration_number"], unique=False
    )
    op.create_index(
        "ix_vehicle_assessments_status_created_at", "vehicle_assessments", ["status", "created_at"], unique=False
    )

    op.create_table(
        "uploaded_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("storage_provider", sa.String(length=32), server_default="s3", nullable=False),
        sa.Column("storage_bucket", sa.String(length=255), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("content_type", sa.String(length=127), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("image_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_provider", "storage_bucket", "object_key", name="uq_uploaded_images_object_ref"),
    )
    op.create_index("ix_uploaded_images_assessment_id", "uploaded_images", ["assessment_id"], unique=False)

    op.create_table(
        "ocr_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("result_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="completed", nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("result_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.ForeignKeyConstraint(["image_id"], ["uploaded_images.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ocr_results_assessment_id", "ocr_results", ["assessment_id"], unique=False)
    op.create_index("ix_ocr_results_image_id", "ocr_results", ["image_id"], unique=False)

    op.create_table(
        "damage_findings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("component", sa.String(length=100), nullable=False),
        sa.Column("damage_type", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("bounding_box", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.ForeignKeyConstraint(["image_id"], ["uploaded_images.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_damage_findings_assessment_id", "damage_findings", ["assessment_id"], unique=False)
    op.create_index("ix_damage_findings_image_id", "damage_findings", ["image_id"], unique=False)

    op.create_table(
        "price_estimates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), server_default="INR", nullable=False),
        sa.Column("lower_bound", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("upper_bound", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("method", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=True),
        sa.Column("estimate_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_price_estimates_assessment_id", "price_estimates", ["assessment_id"], unique=False)

    op.create_table(
        "evidence_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("image_id", sa.Uuid(), nullable=True),
        sa.Column("evidence_type", sa.String(length=64), nullable=False),
        sa.Column("source_reference", sa.String(length=1024), nullable=True),
        sa.Column("evidence_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.ForeignKeyConstraint(["image_id"], ["uploaded_images.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_records_assessment_id", "evidence_records", ["assessment_id"], unique=False)
    op.create_index("ix_evidence_records_image_id", "evidence_records", ["image_id"], unique=False)

    op.create_table(
        "assessment_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("assessment_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="draft", nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("report_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["assessment_id"], ["vehicle_assessments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assessment_id"),
    )


def downgrade() -> None:
    raise RuntimeError("Automatic downgrade is disabled to protect application data.")

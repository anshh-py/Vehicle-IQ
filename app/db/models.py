from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from decimal import InvalidOperation
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")


class VehicleAssessment(Base):
    __tablename__ = "vehicle_assessments"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    vin: Mapped[str | None] = mapped_column(String(17), index=True)
    registration_number: Mapped[str | None] = mapped_column(String(32), index=True)
    vehicle_details: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    images: Mapped[list[UploadedImage]] = relationship(back_populates="assessment")
    lookup_results: Mapped[list[OCRResult]] = relationship(back_populates="assessment")
    damage_findings: Mapped[list[DamageFinding]] = relationship(back_populates="assessment")
    price_estimates: Mapped[list[PriceEstimate]] = relationship(back_populates="assessment")
    evidence_records: Mapped[list[EvidenceRecord]] = relationship(back_populates="assessment")
    report: Mapped[AssessmentReport | None] = relationship(back_populates="assessment", uselist=False)

    @property
    def asking_price_inr(self) -> Decimal | None:
        """Seller-provided asking price stored in the existing vehicle_details JSON."""
        raw_value = (self.vehicle_details or {}).get("asking_price_inr")
        if raw_value is None:
            return None
        try:
            return Decimal(str(raw_value))
        except (InvalidOperation, ValueError):
            return None

    @asking_price_inr.setter
    def asking_price_inr(self, value: Decimal | None) -> None:
        details = dict(self.vehicle_details or {})
        if value is None:
            details.pop("asking_price_inr", None)
        else:
            details["asking_price_inr"] = format(value, "f")
        self.vehicle_details = details or None


class UploadedImage(Base):
    __tablename__ = "uploaded_images"
    __table_args__ = (
        UniqueConstraint("storage_provider", "storage_bucket", "object_key", name="uq_uploaded_images_object_ref"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, index=True
    )
    storage_provider: Mapped[str] = mapped_column(String(32), nullable=False, default="s3", server_default="s3")
    storage_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(127))
    byte_size: Mapped[int | None] = mapped_column()
    sha256: Mapped[str | None] = mapped_column(String(64))
    image_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="images")
    lookup_results: Mapped[list[OCRResult]] = relationship(back_populates="image")
    damage_findings: Mapped[list[DamageFinding]] = relationship(back_populates="image")


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, index=True
    )
    image_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("uploaded_images.id"), index=True)
    result_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed", server_default="completed")
    provider: Mapped[str | None] = mapped_column(String(100))
    raw_text: Mapped[str | None] = mapped_column(Text)
    result_data: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="lookup_results")
    image: Mapped[UploadedImage | None] = relationship(back_populates="lookup_results")


class DamageFinding(Base):
    __tablename__ = "damage_findings"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, index=True
    )
    image_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("uploaded_images.id"), index=True)
    component: Mapped[str | None] = mapped_column(String(100), nullable=True)
    damage_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(Text)
    bounding_box: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    model_name: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="damage_findings")
    image: Mapped[UploadedImage | None] = relationship(back_populates="damage_findings")


class PriceEstimate(Base):
    __tablename__ = "price_estimates"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, index=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR", server_default="INR")
    lower_bound: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    upper_bound: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    method: Mapped[str | None] = mapped_column(String(100))
    model_name: Mapped[str | None] = mapped_column(String(100))
    estimate_data: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="price_estimates")


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, index=True
    )
    image_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("uploaded_images.id"), index=True)
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_reference: Mapped[str | None] = mapped_column(String(1024))
    evidence_data: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="evidence_records")


class AssessmentReport(Base):
    __tablename__ = "assessment_reports"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    assessment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("vehicle_assessments.id"), nullable=False, unique=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft", server_default="draft")
    summary: Mapped[str | None] = mapped_column(Text)
    report_data: Mapped[dict[str, Any] | None] = mapped_column(JSON_DOCUMENT)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    assessment: Mapped[VehicleAssessment] = relationship(back_populates="report")


Index("ix_vehicle_assessments_status_created_at", VehicleAssessment.status, VehicleAssessment.created_at)

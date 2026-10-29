from datetime import date as Date
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


ServiceStatus = Literal["available", "unavailable", "partial", "failed"]


class ChallanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = Field(default=None, max_length=80)
    violation_description: str | None = Field(default=None, max_length=500)
    amount: Decimal | None = Field(default=None, ge=0, max_digits=12, decimal_places=2)
    currency: str | None = Field(default=None, max_length=8)
    date: Date | None = None
    payment_status: str | None = Field(default=None, max_length=80)


class ChallanDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = Field(default=None, max_length=80)
    records: list[ChallanRecord] = Field(default_factory=list)


class FastagDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = Field(default=None, max_length=80)
    issuing_bank: str | None = Field(default=None, max_length=120)
    issue_date: Date | None = None
    vehicle_class: str | None = Field(default=None, max_length=100)


class ServiceLookupResult(BaseModel):
    status: ServiceStatus
    provider: str | None = None
    details: ChallanDetails | FastagDetails | None = None
    error_code: str | None = None


class ChallanServiceResult(BaseModel):
    status: ServiceStatus
    provider: str | None = None
    details: ChallanDetails | None = None
    error_code: str | None = None


class FastagServiceResult(BaseModel):
    status: ServiceStatus
    provider: str | None = None
    details: FastagDetails | None = None
    error_code: str | None = None


class VehicleServicesLookupResponse(BaseModel):
    assessment_id: UUID
    image_id: UUID | None
    registration_number: str
    status: ServiceStatus
    challan: ChallanServiceResult
    fastag: FastagServiceResult

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import VehicleSpecification
from app.vehicle_specs.repository import VehicleSpecificationRepository
from app.vehicle_specs.sources import is_approved_source
from app.vehicle_specs.schemas import ResolvedVehicleSpecification, VehicleSpecificationResolution

PRICING_FIELDS = ("engine_cc", "max_power_bhp", "transmission", "mileage_kmpl", "seats", "owner_count")


class VehicleSpecificationResolver:
    def __init__(self, db: Session):
        self.db = db
        self.repository = VehicleSpecificationRepository(db)

    def resolve(
        self,
        *,
        make: str,
        model: str,
        variant: str | None = None,
        fuel_type: str | None = None,
        manufacturing_year: int | None = None,
    ) -> VehicleSpecificationResolution:
        requested = {
            "make": make,
            "model": model,
            "variant": variant,
            "fuel_type": fuel_type,
            "manufacturing_year": manufacturing_year,
        }
        specification = self.repository.find_best(
            make=make,
            model=model,
            variant=variant,
            fuel_type=fuel_type,
            manufacturing_year=manufacturing_year,
        )
        if specification is None:
            return VehicleSpecificationResolution(requested=requested, missing_fields=list(PRICING_FIELDS))
        resolved = ResolvedVehicleSpecification(
            make=specification.make,
            model=specification.model,
            variant=specification.variant,
            fuel_type=specification.fuel_type,
            manufacturing_year=manufacturing_year or specification.year_start,
            engine_cc=specification.engine_cc,
            max_power_bhp=specification.max_power_bhp,
            transmission=specification.transmission,
            mileage_kmpl=specification.mileage_kmpl,
            seats=specification.seats,
            owner_count=specification.owner_count,
            source_provider=specification.source_provider,
            source_reference=specification.source_reference,
            verification_status=specification.verification_status,
            verified_at=specification.verified_at,
            confidence=specification.confidence,
            field_sources=specification.field_sources or {},
            catalog_id=specification.id,
        )
        missing = [field for field in PRICING_FIELDS if getattr(resolved, field) is None]
        provenance = {
            field: (resolved.field_sources.get(field) or resolved.source_provider)
            for field in PRICING_FIELDS
            if getattr(resolved, field) is not None
        }

        if resolved.verification_status != "verified" or not is_approved_source(resolved.source_provider):
            for field in ("engine_cc", "max_power_bhp", "transmission"):
                setattr(resolved, field, None)
                resolved.field_sources.pop(field, None)
                provenance.pop(field, None)
            missing = [field for field in PRICING_FIELDS if getattr(resolved, field) is None]
        return VehicleSpecificationResolution(
            requested=requested,
            resolved=resolved,
            missing_fields=missing,
            provenance=provenance,
        )

    def upsert(self, specification: VehicleSpecification) -> VehicleSpecification:
        if specification.verified_at is None and specification.verification_status == "verified":
            specification.verified_at = datetime.now(timezone.utc)
        return self.repository.add(specification)
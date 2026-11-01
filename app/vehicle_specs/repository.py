from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import VehicleSpecification


class VehicleSpecificationRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_best(
        self,
        *,
        make: str,
        model: str,
        variant: str | None = None,
        fuel_type: str | None = None,
        manufacturing_year: int | None = None,
    ) -> VehicleSpecification | None:
        stmt = select(VehicleSpecification).where(
            VehicleSpecification.make.ilike(make.strip()),
            VehicleSpecification.model.ilike(model.strip()),
        )
        if variant:
            stmt = stmt.where(VehicleSpecification.variant.ilike(variant.strip()))
        if fuel_type:
            stmt = stmt.where(VehicleSpecification.fuel_type.ilike(fuel_type.strip()))
        candidates = list(self.db.scalars(stmt))
        eligible = []
        for candidate in candidates:
            if manufacturing_year is not None:
                if candidate.year_start is not None and manufacturing_year < candidate.year_start:
                    continue
                if candidate.year_end is not None and manufacturing_year > candidate.year_end:
                    continue
            eligible.append(candidate)
        if not eligible:
            return None
        def sort_key(candidate: VehicleSpecification):
            exact_year = 0
            if manufacturing_year is not None and candidate.year_start == candidate.year_end == manufacturing_year:
                exact_year = 1
            confidence = candidate.confidence if candidate.confidence is not None else 0.0
            verified = 1 if candidate.verification_status == "verified" else 0
            return (verified, exact_year, confidence, candidate.created_at.timestamp() if candidate.created_at else 0.0)
        return max(eligible, key=sort_key)

    def add(self, specification: VehicleSpecification) -> VehicleSpecification:
        self.db.add(specification)
        self.db.flush()
        return specification
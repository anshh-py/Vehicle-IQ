from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MarketListing as MarketListingModel
from app.pricing.market import MarketListing, MarketListingProvider, MarketVehicleCriteria


class MarketListingRepository:
    def __init__(self, db: Session):
        self.db = db

    def _as_utc(self, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def list_for_criteria(self, criteria: MarketVehicleCriteria, *, now: datetime | None = None) -> list[MarketListing]:
        now = now or datetime.now(timezone.utc)
        rows = self.db.scalars(select(MarketListingModel)).all()
        listings: list[MarketListing] = []
        for row in rows:
            listed_at = self._as_utc(row.listed_at) or self._as_utc(row.collected_at) or now
            retrieved_at = self._as_utc(row.collected_at) or now
            listings.append(
                MarketListing(
                    source_name=row.source_name,
                    listing_id=row.source_listing_id,
                    make=row.make,
                    model=row.model,
                    variant=row.variant,
                    year=row.manufacturing_year,
                    fuel_type=row.fuel_type,
                    transmission=row.transmission,
                    mileage_km=row.odometer_km,
                    location=row.city or row.state,
                    advertised_price_inr=row.asking_price_inr,
                    source_url=row.source_url,
                    listed_at=listed_at,
                    retrieved_at=retrieved_at,
                )
            )
        return listings

    def find_duplicate(self, *, source_name: str, source_listing_id: str | None, source_url: str) -> MarketListingModel | None:
        if source_listing_id:
            return self.db.scalar(
                select(MarketListingModel).where(
                    MarketListingModel.source_name == source_name,
                    MarketListingModel.source_listing_id == source_listing_id,
                )
            )
        return self.db.scalar(select(MarketListingModel).where(MarketListingModel.source_url == source_url))

    def add(self, listing: MarketListingModel) -> MarketListingModel:
        self.db.add(listing)
        self.db.flush()
        return listing


class DatabaseMarketListingProvider:
    provider_name = "database_market_listings"

    def __init__(self, db: Session):
        self.db = db

    def fetch_listings(self, criteria: MarketVehicleCriteria) -> Iterable[MarketListing]:
        repository = MarketListingRepository(self.db)
        return repository.list_for_criteria(criteria)
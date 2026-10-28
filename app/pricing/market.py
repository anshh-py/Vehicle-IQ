"""Source-independent contracts and conservative comparable-listing valuation.

No listing source is currently configured. A future provider must have documented
permission for listing reuse and must return listing dates as well as retrieval times.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


MAX_LISTING_AGE = timedelta(days=30)
MIN_COMPARABLE_COUNT = 5
YEAR_TOLERANCE = 1
MILEAGE_TOLERANCE_FRACTION = Decimal("0.25")
MILEAGE_TOLERANCE_FLOOR_KM = 20_000


class MarketProviderError(Exception):
    """A configured market-listing source failed safely."""


class MarketVehicleCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    make: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=120)
    variant: str | None = None
    year: int | None = Field(default=None, ge=1886, le=2100)
    fuel_type: str | None = None
    transmission: str | None = None
    mileage_km: int | None = Field(default=None, ge=0, le=2_000_000)
    location: str | None = None


class MarketListing(BaseModel):
    """Normalized advertised listing; asking price is not a transaction price."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1, max_length=100)
    listing_id: str | None = Field(default=None, max_length=255)
    make: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=120)
    variant: str | None = None
    year: int = Field(ge=1886, le=2100)
    fuel_type: str | None = None
    transmission: str | None = None
    mileage_km: int | None = Field(default=None, ge=0, le=2_000_000)
    location: str | None = None
    advertised_price_inr: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    source_url: HttpUrl
    listed_at: datetime
    retrieved_at: datetime

    @field_validator("source_name", "make", "model", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("listing_id", "variant", "fuel_type", "transmission", "location", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> Any:
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("source_url")
    @classmethod
    def require_https_source(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("listing source URL must use HTTPS")
        if value.username or value.password:
            raise ValueError("listing source URL cannot contain credentials")
        return value

    @field_validator("listed_at", "retrieved_at")
    @classmethod
    def require_timezone_aware_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("listing timestamps must include a timezone")
        return value.astimezone(timezone.utc)


class MarketPriceRange(BaseModel):
    lower_inr: Decimal
    median_inr: Decimal
    upper_inr: Decimal
    method: Literal["inlier_interquartile_range"] = "inlier_interquartile_range"


class MarketComparableResult(BaseModel):
    status: Literal["available", "low_evidence", "unavailable"]
    valuation_date: datetime
    evidence_count: int
    minimum_evidence_count: int = MIN_COMPARABLE_COUNT
    freshness_threshold_days: int = 30
    data_freshness_days: int | None = None
    maximum_listing_age_days: int | None = None
    price_range: MarketPriceRange | None = None
    listings: list[MarketListing] = Field(default_factory=list)
    rejected_counts: dict[str, int] = Field(default_factory=dict)


class MarketListingProvider(Protocol):
    provider_name: str

    def fetch_listings(self, criteria: MarketVehicleCriteria) -> Iterable[MarketListing]:
        """Fetch permitted, documented India listing data with bounded timeouts."""


def get_market_listing_provider() -> MarketListingProvider | None:
    """No authorized comparable-listing source is configured yet."""
    return None


def _key(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.casefold().split()) or None


def _listing_url_key(url: HttpUrl) -> str:
    parts = urlsplit(str(url))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _quantile(sorted_values: list[Decimal], fraction: Decimal) -> Decimal:
    position = Decimal(len(sorted_values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    value = sorted_values[lower_index] * (Decimal(1) - weight) + sorted_values[upper_index] * weight
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _matches(listing: MarketListing, criteria: MarketVehicleCriteria) -> str | None:
    if _key(listing.make) != _key(criteria.make):
        return "make_mismatch"
    if _key(listing.model) != _key(criteria.model):
        return "model_mismatch"
    if listing.variant and criteria.variant and _key(listing.variant) != _key(criteria.variant):
        return "variant_mismatch"
    if criteria.year is not None and abs(listing.year - criteria.year) > YEAR_TOLERANCE:
        return "year_mismatch"
    if listing.fuel_type and criteria.fuel_type and _key(listing.fuel_type) != _key(criteria.fuel_type):
        return "fuel_mismatch"
    if listing.transmission and criteria.transmission and _key(listing.transmission) != _key(criteria.transmission):
        return "transmission_mismatch"
    if listing.mileage_km is not None and criteria.mileage_km is not None:
        allowed_delta = max(
            Decimal(MILEAGE_TOLERANCE_FLOOR_KM),
            Decimal(criteria.mileage_km) * MILEAGE_TOLERANCE_FRACTION,
        )
        if abs(listing.mileage_km - criteria.mileage_km) > allowed_delta:
            return "mileage_mismatch"
    if listing.location and criteria.location and _key(listing.location) != _key(criteria.location):
        return "location_mismatch"
    return None


def calculate_market_comparables(
    raw_listings: Iterable[MarketListing | dict[str, Any]],
    criteria: MarketVehicleCriteria,
    *,
    valuation_date: datetime | None = None,
) -> MarketComparableResult:
    """Filter listings no older than 30 days, remove Tukey outliers, require five inliers.

    Vehicle matching allows a one-year model-year difference and mileage within
    max(20,000 km, 25% of target mileage). Outliers are removed with 1.5*IQR
    Tukey fences. The returned quartile range describes advertised prices only.
    """
    now = valuation_date or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("valuation_date must include a timezone")
    now = now.astimezone(timezone.utc)
    rejected: dict[str, int] = {}
    candidates: list[MarketListing] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for raw in raw_listings:
        try:
            listing = raw if isinstance(raw, MarketListing) else MarketListing.model_validate(raw)
        except Exception:
            reject("malformed_listing")
            continue
        age = now - listing.listed_at
        if age < timedelta(0):
            reject("future_listing_date")
            continue
        if age > MAX_LISTING_AGE:
            reject("stale_listing")
            continue
        retrieval_age = now - listing.retrieved_at
        if retrieval_age < timedelta(0):
            reject("future_retrieval_timestamp")
            continue
        if retrieval_age > MAX_LISTING_AGE:
            reject("stale_retrieval_timestamp")
            continue
        mismatch = _matches(listing, criteria)
        if mismatch:
            reject(mismatch)
            continue
        identity = (listing.source_name.casefold(), listing.listing_id.casefold()) if listing.listing_id else None
        url_key = _listing_url_key(listing.source_url)
        if (identity and identity in seen_ids) or url_key in seen_urls:
            reject("duplicate_listing")
            continue
        if identity:
            seen_ids.add(identity)
        seen_urls.add(url_key)
        candidates.append(listing)

    sorted_prices = sorted(listing.advertised_price_inr for listing in candidates)
    inliers = candidates
    if len(sorted_prices) >= 4:
        q1 = _quantile(sorted_prices, Decimal("0.25"))
        q3 = _quantile(sorted_prices, Decimal("0.75"))
        spread = q3 - q1
        lower_fence = q1 - Decimal("1.5") * spread
        upper_fence = q3 + Decimal("1.5") * spread
        inliers = [
            listing
            for listing in candidates
            if lower_fence <= listing.advertised_price_inr <= upper_fence
        ]
        rejected["price_outlier"] = len(candidates) - len(inliers)
        if rejected["price_outlier"] == 0:
            rejected.pop("price_outlier")

    if len(inliers) < MIN_COMPARABLE_COUNT:
        data_freshness_days = max(
            ((now - item.retrieved_at).days for item in inliers), default=None
        )
        maximum_listing_age_days = max(
            ((now - item.listed_at).days for item in inliers), default=None
        )
        return MarketComparableResult(
            status="low_evidence",
            valuation_date=now,
            evidence_count=len(inliers),
            data_freshness_days=data_freshness_days,
            maximum_listing_age_days=maximum_listing_age_days,
            listings=inliers,
            rejected_counts=rejected,
        )

    inlier_prices = sorted(listing.advertised_price_inr for listing in inliers)
    retrieval_ages = [now - listing.retrieved_at for listing in inliers]
    listing_ages = [now - listing.listed_at for listing in inliers]
    return MarketComparableResult(
        status="available",
        valuation_date=now,
        evidence_count=len(inliers),
        data_freshness_days=max(age.days for age in retrieval_ages),
        maximum_listing_age_days=max(age.days for age in listing_ages),
        price_range=MarketPriceRange(
            lower_inr=_quantile(inlier_prices, Decimal("0.25")),
            median_inr=_quantile(inlier_prices, Decimal("0.5")),
            upper_inr=_quantile(inlier_prices, Decimal("0.75")),
        ),
        listings=inliers,
        rejected_counts=rejected,
    )

"""Source-independent contracts and conservative comparable-listing valuation."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


MAX_LISTING_AGE = timedelta(days=30)
MIN_COMPARABLE_COUNT = 5
YEAR_TOLERANCE = 3
MILEAGE_TOLERANCE_FRACTION = Decimal("0.15")
MILEAGE_TOLERANCE_FLOOR_KM = 5_000
MIN_RAW_LISTINGS_FOR_OUTLIER_FILTERING = 7


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
    title: str | None = None
    match_type: Literal[
        "exact_match", "close_match", "approximate_match", "exact", "approximate"
    ] = "exact_match"
    match_differences: list[str] = Field(default_factory=list)
    fuel_type: str | None = None
    transmission: str | None = None
    mileage_km: int | None = Field(default=None, ge=0, le=2_000_000)
    location: str | None = None
    advertised_price_inr: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    source_url: HttpUrl
    listed_at: datetime | None = None
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
    def require_timezone_aware_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
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
    """Return Tavily discovery only when a backend API key is configured."""
    from app.market.tavily import get_tavily_market_listing_provider

    return get_tavily_market_listing_provider()


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


def _coherent_year_cohort(
    listings: list[MarketListing], target_year: int | None
) -> list[MarketListing]:
    if not listings:
        return []
    windows = [
        [
            item
            for item in listings
            if lower_year <= item.year <= lower_year + YEAR_TOLERANCE
        ]
        for lower_year in sorted({item.year for item in listings})
    ]
    return max(
        windows,
        key=lambda cohort: (
            len(cohort),
            sum(item.match_type == "exact_match" for item in cohort),
            sum(item.match_type == "close_match" for item in cohort),
            -abs(
                (min(item.year for item in cohort) + max(item.year for item in cohort))
                / 2
                - (target_year or cohort[0].year)
            ),
        ),
    )


def _classify(
    listing: MarketListing, criteria: MarketVehicleCriteria
) -> tuple[str, list[str]]:
    differences: list[str] = []
    if _key(listing.make) != _key(criteria.make):
        return "make_mismatch", []
    if _key(listing.model) != _key(criteria.model):
        return "model_mismatch", []
    if criteria.year is not None:
        year_difference = listing.year - criteria.year
        if abs(year_difference) > YEAR_TOLERANCE:
            return "year_mismatch", []
        if year_difference:
            differences.append(f"year: {listing.year} vs {criteria.year}")
    else:
        differences.append("year: target not specified")
    if criteria.variant is not None:
        if _key(listing.variant) != _key(criteria.variant):
            differences.append(
                f"variant: {listing.variant or 'not reported'} vs {criteria.variant}"
            )
    else:
        differences.append("variant: target not specified")
    if listing.fuel_type is None:
        differences.append("fuel_type: not reported")
    elif criteria.fuel_type is None:
        differences.append("fuel_type: target not specified")
    elif _key(listing.fuel_type) != _key(criteria.fuel_type):
        differences.append(
            f"fuel_type: {listing.fuel_type} vs {criteria.fuel_type}"
        )
    if criteria.transmission is not None:
        if _key(listing.transmission) != _key(criteria.transmission):
            differences.append(
                f"transmission: {listing.transmission or 'not reported'} vs {criteria.transmission}"
            )
    if criteria.mileage_km is not None:
        if listing.mileage_km is None:
            differences.append("mileage_km: not reported")
        else:
            allowed_delta = max(
                Decimal(MILEAGE_TOLERANCE_FLOOR_KM),
                Decimal(criteria.mileage_km) * MILEAGE_TOLERANCE_FRACTION,
            )
            if abs(listing.mileage_km - criteria.mileage_km) > allowed_delta:
                differences.append(
                    f"mileage_km: {listing.mileage_km} vs {criteria.mileage_km}"
                )
    if criteria.location is not None:
        if listing.location is None:
            differences.append("location: not reported")
        elif _key(listing.location) != _key(criteria.location):
            differences.append(f"location: {listing.location} vs {criteria.location}")

    if not differences:
        return "exact_match", differences
    close_differences = [
        item
        for item in differences
        if not item.startswith("year:") or item == "year: target not specified"
    ]
    if (
        len(close_differences) == 0
        and (criteria.year is None or abs(listing.year - criteria.year) <= YEAR_TOLERANCE)
        and criteria.fuel_type is not None
        and _key(listing.fuel_type) == _key(criteria.fuel_type)
    ):
        return "close_match", differences
    return "approximate_match", differences


def calculate_market_comparables(
    raw_listings: Iterable[MarketListing | dict[str, Any]],
    criteria: MarketVehicleCriteria,
    *,
    valuation_date: datetime | None = None,
) -> MarketComparableResult:
    """Prefer exact, then close, then approximate same-fuel evidence for the range."""
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
        if listing.listed_at is not None:
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
        match_type, differences = _classify(listing, criteria)
        if match_type in {"make_mismatch", "model_mismatch"}:
            reject(match_type)
            continue
        if match_type == "year_mismatch":
            reject("year_mismatch")
            continue
        if listing.fuel_type is None:
            reject("fuel_not_reported")
            continue
        identity = (listing.source_name.casefold(), listing.listing_id.casefold()) if listing.listing_id else None
        url_key = _listing_url_key(listing.source_url)
        if (identity and identity in seen_ids) or url_key in seen_urls:
            reject("duplicate_listing")
            continue
        if identity:
            seen_ids.add(identity)
        seen_urls.add(url_key)
        candidates.append(
            listing.model_copy(
                update={
                    "match_type": match_type,
                    "match_differences": differences,
                }
            )
        )

    if candidates:
        target_fuel = _key(criteria.fuel_type) or Counter(
            _key(item.fuel_type) for item in candidates if item.fuel_type
        ).most_common(1)[0][0]
        same_fuel = []
        for item in candidates:
            if _key(item.fuel_type) != target_fuel:
                reject("fuel_mismatch")
            else:
                same_fuel.append(item)
        candidates = same_fuel

    exact = _coherent_year_cohort(
        [item for item in candidates if item.match_type == "exact_match"],
        criteria.year,
    )
    exact_and_close = _coherent_year_cohort(
        [
            item
            for item in candidates
            if item.match_type in {"exact_match", "close_match"}
        ],
        criteria.year,
    )
    all_matches = _coherent_year_cohort(candidates, criteria.year)
    cohorts = [exact, exact_and_close, all_matches]
    inliers: list[MarketListing] = []
    outlier_count = 0
    selected_cohort_index = len(cohorts) - 1
    for cohort_index, cohort in enumerate(cohorts):
        prices = sorted(item.advertised_price_inr for item in cohort)
        candidate_inliers = cohort
        if len(prices) >= MIN_RAW_LISTINGS_FOR_OUTLIER_FILTERING:
            q1 = _quantile(prices, Decimal("0.25"))
            q3 = _quantile(prices, Decimal("0.75"))
            spread = q3 - q1
            lower_fence = q1 - Decimal("1.5") * spread
            upper_fence = q3 + Decimal("1.5") * spread
            candidate_inliers = [
                item
                for item in cohort
                if lower_fence <= item.advertised_price_inr <= upper_fence
            ]
        if len(candidate_inliers) >= MIN_COMPARABLE_COUNT:
            inliers = candidate_inliers
            outlier_count = len(cohort) - len(inliers)
            selected_cohort_index = cohort_index
            break
        inliers = candidate_inliers
        outlier_count = len(cohort) - len(inliers)
    if selected_cohort_index == len(cohorts) - 1 and len(all_matches) < len(candidates):
        rejected["year_cohort_mismatch"] = len(candidates) - len(all_matches)
    if outlier_count:
        rejected["price_outlier"] = outlier_count

    if len(inliers) < MIN_COMPARABLE_COUNT:
        data_freshness_days = max(
            ((now - item.retrieved_at).days for item in inliers), default=None
        )
        maximum_listing_age_days = max(
            ((now - item.listed_at).days for item in inliers if item.listed_at is not None), default=None
        )
        return MarketComparableResult(
            status="unavailable" if not inliers else "low_evidence",
            valuation_date=now,
            evidence_count=len(inliers),
            data_freshness_days=data_freshness_days,
            maximum_listing_age_days=maximum_listing_age_days,
            listings=inliers,
            rejected_counts=rejected,
        )

    inlier_prices = sorted(listing.advertised_price_inr for listing in inliers)
    retrieval_ages = [now - listing.retrieved_at for listing in inliers]
    listing_ages = [
        now - listing.listed_at for listing in inliers if listing.listed_at is not None
    ]
    return MarketComparableResult(
        status="available",
        valuation_date=now,
        evidence_count=len(inliers),
        data_freshness_days=max(age.days for age in retrieval_ages),
        maximum_listing_age_days=max((age.days for age in listing_ages), default=None),
        price_range=MarketPriceRange(
            lower_inr=_quantile(inlier_prices, Decimal("0.25")),
            median_inr=_quantile(inlier_prices, Decimal("0.5")),
            upper_inr=_quantile(inlier_prices, Decimal("0.75")),
        ),
        listings=inliers,
        rejected_counts=rejected,
    )

"""Automated market-listing discovery using only permitted sources."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from app.market.ingestion import import_market_listings_from_json
from app.market.sources import (
    MARKET_SOURCE_REGISTRY,
    MarketSourceDefinition,
    get_market_source,
    is_automated_market_source_permitted,
)
from app.pricing.market import (
    MarketListing,
    MarketListingProvider,
    MarketProviderError,
    MarketVehicleCriteria,
)


class MarketSourceError(MarketProviderError):
    """A configured market source failed or is not permitted."""


class MarketSourceRateLimitError(MarketSourceError):
    """The source returned HTTP 429 or an equivalent rate-limit response."""


class MarketSourceTimeoutError(MarketSourceError):
    """The source did not respond before the configured timeout."""


class MarketListingSource(Protocol):
    source_name: str
    automated_access_permitted: bool

    def fetch(
        self,
        criteria: MarketVehicleCriteria,
        *,
        now: datetime | None = None,
        timeout_seconds: float = 10.0,
    ) -> list[MarketListing]:
        ...
def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first(record: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in record and record.get(name) is not None:
            return record.get(name)
    return None


def normalize_source_record(
    source_name: str,
    record: Mapping[str, Any],
    *,
    retrieved_at: datetime | None = None,
) -> dict[str, Any] | None:
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    listed_at = _parse_datetime(_first(record, "listed_at", "listing_date", "posted_at", "date_listed"))
    if listed_at is None:
        return None

    price = _parse_decimal(_first(record, "asking_price_inr", "asking_price", "price_inr", "price"))
    if price is None or price <= 0:
        return None

    make = _clean(_first(record, "make", "manufacturer", "brand"))
    model = _clean(_first(record, "model"))
    year = _parse_int(_first(record, "manufacturing_year", "year", "model_year"))
    source_url = _clean(_first(record, "source_url", "url", "listing_url"))
    if not make or not model or year is None or not source_url:
        return None

    listing_id = _clean(_first(record, "listing_id", "id", "source_listing_id"))
    location = _clean(_first(record, "location", "city"))
    city = _clean(_first(record, "city"))
    state = _clean(_first(record, "state"))
    if location is None and city is not None:
        location = city if state is None else f"{city}, {state}"

    return {
        "source_name": source_name,
        "listing_id": listing_id,
        "make": make,
        "model": model,
        "variant": _clean(_first(record, "variant", "trim")),
        "year": year,
        "fuel_type": _clean(_first(record, "fuel_type", "fuel")),
        "transmission": _clean(_first(record, "transmission", "gearbox")),
        "mileage_km": _parse_int(_first(record, "odometer_km", "mileage_km", "mileage", "km_driven")),
        "location": location,
        "advertised_price_inr": price,
        "source_url": source_url,
        "listed_at": listed_at,
        "retrieved_at": retrieved_at,
    }


def _deduplicate_listings(listings: Iterable[MarketListing]) -> list[MarketListing]:
    seen_ids: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    result: list[MarketListing] = []
    for listing in listings:
        identity = (listing.source_name.casefold(), listing.listing_id.casefold()) if listing.listing_id else None
        url_key = str(listing.source_url).split("?")[0].rstrip("/").casefold()
        if (identity and identity in seen_ids) or url_key in seen_urls:
            continue
        if identity:
            seen_ids.add(identity)
        seen_urls.add(url_key)
        result.append(listing)
    return result

@dataclass
class LocalPermittedDatasetSource:
    source_name: str = "local_permitted_dataset"
    automated_access_permitted: bool = True
    path: Path | str | None = None
    text: str | None = None
    input_format: str | None = None

    def _read_rows(self) -> list[dict[str, Any]]:
        if self.text is not None:
            payload = self.text
        elif self.path is not None:
            payload = Path(self.path).read_text(encoding="utf-8")
        else:
            raise MarketSourceError("A local dataset path or text payload is required.")

        fmt = self.input_format
        if fmt is None:
            fmt = "json" if payload.lstrip().startswith(("[", "{")) else "csv"

        if fmt == "json":
            data = json.loads(payload)
            if isinstance(data, dict):
                for key in ("listings", "results", "data", "items"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                raise MarketSourceError("JSON dataset must contain a list of listings.")
            return [dict(item) for item in data if isinstance(item, Mapping)]
        if fmt == "csv":
            return [dict(row) for row in csv.DictReader(io.StringIO(payload))]
        raise MarketSourceError("input_format must be 'csv' or 'json'.")

    def fetch(
        self,
        criteria: MarketVehicleCriteria,
        *,
        now: datetime | None = None,
        timeout_seconds: float = 10.0,
    ) -> list[MarketListing]:
        retrieved_at = now or datetime.now(timezone.utc)
        rows = self._read_rows()
        listings: list[MarketListing] = []
        for row in rows:
            normalized = normalize_source_record(self.source_name, row, retrieved_at=retrieved_at)
            if normalized is None:
                continue
            try:
                listings.append(MarketListing.model_validate(normalized))
            except (ValidationError, ValueError):
                continue
        return _deduplicate_listings(listings)

class HttpJsonMarketSource:
    def __init__(
        self,
        *,
        source_name: str,
        base_url: str,
        automated_access_permitted: bool = False,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
        params_builder: Callable[[MarketVehicleCriteria], Mapping[str, Any]] | None = None,
        response_parser: Callable[[Any], list[Mapping[str, Any]]] | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        self.source_name = source_name
        self.base_url = base_url
        self.automated_access_permitted = automated_access_permitted
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._params_builder = params_builder
        self._response_parser = response_parser
        self._headers = dict(headers or {})

    def _client_instance(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.timeout_seconds, headers=self._headers)

    def _default_params(self, criteria: MarketVehicleCriteria) -> dict[str, Any]:
        params: dict[str, Any] = {"make": criteria.make, "model": criteria.model}
        if criteria.variant is not None:
            params["variant"] = criteria.variant
        if criteria.year is not None:
            params["year"] = criteria.year
        if criteria.fuel_type is not None:
            params["fuel_type"] = criteria.fuel_type
        if criteria.transmission is not None:
            params["transmission"] = criteria.transmission
        if criteria.location is not None:
            params["location"] = criteria.location
        return params

    def _default_parser(self, payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("results", "listings", "data", "items", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, Mapping)]
        return []

    def fetch(
        self,
        criteria: MarketVehicleCriteria,
        *,
        now: datetime | None = None,
        timeout_seconds: float | None = None,
    ) -> list[MarketListing]:
        if not self.automated_access_permitted:
            raise MarketSourceError(
                f"Automated access is not permitted for source '{self.source_name}'."
            )
        if not is_automated_market_source_permitted(self.source_name):
            raise MarketSourceError(
                f"Source '{self.source_name}' is not registered as permitting automated access."
            )

        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        client = self._client_instance()
        owns_client = self._client is None
        retrieved_at = now or datetime.now(timezone.utc)
        try:
            params = self._params_builder(criteria) if self._params_builder else self._default_params(criteria)
            response = client.get(self.base_url, params=params, timeout=timeout, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise MarketSourceTimeoutError(f"Timed out contacting {self.source_name}.") from exc
        except httpx.HTTPError as exc:
            raise MarketSourceError(f"HTTP error contacting {self.source_name}: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code == 429:
            raise MarketSourceRateLimitError(f"Rate limited by {self.source_name}.")
        if response.status_code >= 400:
            raise MarketSourceError(
                f"{self.source_name} returned HTTP {response.status_code}."
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise MarketSourceError(f"{self.source_name} returned invalid JSON.") from exc

        parser = self._response_parser or self._default_parser
        records = parser(payload)
        listings: list[MarketListing] = []
        for record in records:
            normalized = normalize_source_record(self.source_name, record, retrieved_at=retrieved_at)
            if normalized is None:
                continue
            try:
                listings.append(MarketListing.model_validate(normalized))
            except (ValidationError, ValueError):
                continue
        return _deduplicate_listings(listings)

@dataclass
class AutomatedMarketListingProvider(MarketListingProvider):
    provider_name: str = "automated_market_listings"
    source: MarketListingSource | None = None
    timeout_seconds: float = 10.0

    def fetch_listings(self, criteria: MarketVehicleCriteria) -> Iterable[MarketListing]:
        if self.source is None:
            raise MarketProviderError("No permitted automated market source is configured.")
        return self.source.fetch(criteria, timeout_seconds=self.timeout_seconds)


def discover_market_listings(
    source: MarketListingSource,
    criteria: MarketVehicleCriteria,
    *,
    now: datetime | None = None,
    deduplicate: bool = True,
) -> list[MarketListing]:
    listings = source.fetch(criteria, now=now)
    return _deduplicate_listings(listings) if deduplicate else list(listings)


def market_listing_to_import_row(listing: MarketListing) -> dict[str, Any]:
    return {
        "source_name": listing.source_name,
        "source_listing_id": listing.listing_id,
        "source_url": str(listing.source_url),
        "make": listing.make,
        "model": listing.model,
        "variant": listing.variant,
        "manufacturing_year": listing.year,
        "fuel_type": listing.fuel_type,
        "transmission": listing.transmission,
        "odometer_km": listing.mileage_km,
        "asking_price_inr": str(listing.advertised_price_inr),
        "city": listing.location.split(",")[0].strip() if listing.location else None,
        "state": listing.location.split(",")[1].strip() if listing.location and "," in listing.location else None,
        "listed_at": listing.listed_at.isoformat(),
        "collected_at": listing.retrieved_at.isoformat(),
        "raw_data": listing.model_dump(mode="json"),
    }


def import_discovered_listings(db, listings: Iterable[MarketListing], *, dry_run: bool = False):
    rows = [market_listing_to_import_row(listing) for listing in listings]
    return import_market_listings_from_json(db, rows, dry_run=dry_run)


def get_automated_market_source(name: str | None = None) -> MarketListingSource | None:
    definition = get_market_source(name) if name else None
    if definition is None:
        definition = get_market_source("local_permitted_dataset")
    if definition is None or not definition.automated_access_permitted:
        return None
    if definition.name == "local_permitted_dataset":
        return LocalPermittedDatasetSource(source_name=definition.name)
    return None


def get_market_listing_provider() -> MarketListingProvider | None:
    return None


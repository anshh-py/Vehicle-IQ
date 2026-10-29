"""LangSearch web-search discovery; results are not verified market listings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit
from app.pricing.market import MarketVehicleCriteria


LANGSEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"


class LangSearchError(Exception):
    """Safe, non-secret LangSearch API failure."""


class LangSearchRateLimited(LangSearchError):
    """LangSearch rejected a request due to rate or usage limits."""


class LangSearchInvalidResponse(LangSearchError):
    """LangSearch returned a response outside its documented schema."""


class SearchDiscoveryResult(BaseModel):
    """A search result for follow-up review, never an authorized listing record."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str | None = None
    page_text: str | None = None
    publication_date: str | None = None
    retrieved_at: datetime


class LangSearchDiscoveryClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20.0,
        result_limit: int = 10,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key or not api_key.strip():
            raise LangSearchError("LangSearch API key is not configured.")
        if not 1 <= result_limit <= 50:
            raise ValueError("LangSearch result limit must be between 1 and 50.")
        if not 0 < timeout_seconds <= 60:
            raise ValueError("LangSearch timeout must be greater than 0 and at most 60 seconds.")
        self._api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.result_limit = result_limit
        self.transport = transport

    def search(self, criteria: MarketVehicleCriteria) -> list[SearchDiscoveryResult]:
        payload = {
            "query": self._query(criteria),
            "count": self.result_limit,
            "contents": {"text": True},
            "freshness": "noLimit",
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.timeout_seconds), transport=self.transport
            ) as client:
                response = request_with_response_limit(
                    client,
                    "POST",
                    LANGSEARCH_ENDPOINT,
                    max_bytes=settings.max_external_response_bytes,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
        except ExternalResponseTooLarge:
            raise LangSearchInvalidResponse("LangSearch response exceeded the allowed size.") from None
        except httpx.TimeoutException:
            raise LangSearchError("LangSearch request timed out.") from None
        except httpx.RequestError:
            raise LangSearchError("LangSearch request could not be completed.") from None

        if response.status_code == 429:
            raise LangSearchRateLimited("LangSearch rate or usage limit was reached.")
        if response.is_error:
            raise LangSearchError(f"LangSearch returned HTTP {response.status_code}.")
        try:
            body = response.json()
        except ValueError:
            raise LangSearchInvalidResponse("LangSearch returned invalid JSON.") from None
        if not isinstance(body, dict):
            raise LangSearchInvalidResponse("LangSearch response must be a JSON object.")
        data = body.get("data")
        pages = data.get("webPages") if isinstance(data, dict) else None
        values = pages.get("value") if isinstance(pages, dict) else None
        if not isinstance(values, list):
            raise LangSearchInvalidResponse("LangSearch response is missing data.webPages.value.")

        retrieved_at = datetime.now(timezone.utc)
        results: list[SearchDiscoveryResult] = []
        for item in values:
            parsed = self._parse_result(item, retrieved_at)
            if parsed is not None:
                results.append(parsed)
        return results

    @staticmethod
    def _query(criteria: MarketVehicleCriteria) -> str:
        terms = [criteria.make, criteria.model]
        terms.extend(
            value
            for value in (
                criteria.variant,
                str(criteria.year) if criteria.year is not None else None,
                criteria.fuel_type,
                criteria.transmission,
                f"{criteria.mileage_km} km" if criteria.mileage_km is not None else None,
                criteria.location,
                "used car for sale price",
            )
            if value
        )
        return " ".join(terms)

    @staticmethod
    def _parse_result(item: Any, retrieved_at: datetime) -> SearchDiscoveryResult | None:
        if not isinstance(item, dict):
            return None
        title = item.get("name")
        url = item.get("url")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            return None
        try:
            parts = urlsplit(url)
            if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
                return None
        except ValueError:
            return None
        snippet = item.get("snippet")
        page_text = item.get("text")
        publication_date = item.get("datePublished")
        return SearchDiscoveryResult(
            title=title.strip(),
            url=url,
            snippet=snippet if isinstance(snippet, str) else None,
            page_text=page_text if isinstance(page_text, str) else None,
            publication_date=publication_date if isinstance(publication_date, str) else None,
            retrieved_at=retrieved_at,
        )


def get_langsearch_discovery_client() -> LangSearchDiscoveryClient | None:
    """Return discovery client only when the secret key is configured."""
    secret = settings.langsearch_api_key
    if secret is None or not secret.get_secret_value().strip():
        return None
    return LangSearchDiscoveryClient(
        secret.get_secret_value(),
        timeout_seconds=settings.langsearch_timeout_seconds,
        result_limit=settings.langsearch_result_limit,
    )

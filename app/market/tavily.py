"""Tavily web-search adapter for evidence-backed vehicle listing discovery."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit
from app.pricing.market import (
    MIN_COMPARABLE_COUNT,
    YEAR_TOLERANCE,
    MarketListing,
    MarketVehicleCriteria,
    calculate_market_comparables,
)


TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"
logger = logging.getLogger(__name__)
_PRICE_PATTERN = re.compile(
    r"(?:₹|INR\b|Rs\.?\s*)\s*([\d,]+(?:\.\d+)?)\s*(crores?|cr|lakhs?|lacs?|lac|lakh|l)?(?![a-z])"
    r"|([\d,]+(?:\.\d+)?)\s*(crores?|cr|lakhs?|lacs?|lac|lakh|l)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_MILEAGE_PATTERN = re.compile(
    r"\b([\d,]+(?:\.\d+)?)\s*(?:km|kms|kilomet(?:er|re)s?)\b(?!\s*/)",
    re.IGNORECASE,
)
_MULTIPLIERS = {
    "cr": Decimal("10000000"),
    "crore": Decimal("10000000"),
    "crores": Decimal("10000000"),
    "lac": Decimal("100000"),
    "lacs": Decimal("100000"),
    "lakh": Decimal("100000"),
    "lakhs": Decimal("100000"),
    "l": Decimal("100000"),
}
_FUEL_TERMS = {
    "petrol": ("petrol", "gasoline"),
    "diesel": ("diesel",),
    "cng": ("cng",),
    "lpg": ("lpg",),
    "electric": ("electric", "ev"),
}
_TRANSMISSION_TERMS = {
    "amt": ("amt",),
    "cvt": ("cvt",),
    "dct": ("dct",),
    "automatic": ("automatic", "auto"),
    "manual": ("manual",),
}
_CREDIBLE_LISTING_DOMAINS = {
    "cardekho.com",
    "carwale.com",
    "cars24.com",
    "cartrade.com",
    "droom.in",
    "mahindrafirstchoice.com",
    "olx.in",
    "spinny.com",
    "trovit.co.in",
    "truevalue.marutisuzuki.com",
}
_USED_LISTING_PATH_MARKERS = (
    "used-car",
    "used_car",
    "/used/",
    "cars-for-sale",
    "car-for-sale",
    "car-details",
    "vehicle-listing",
    "/listing/",
    "/item/",
    "cars_c84",
)
_COLLECTION_PATH_MARKERS = {
    "cars_c84",
    "used-cars",
    "used-cars-for-sale",
    "cars-for-sale",
    "car-for-sale",
}
_EDITORIAL_MARKERS = {
    "article",
    "blog",
    "comparison",
    "launched",
    "news",
    "review",
    "video",
    "youtube",
}
_USED_LISTING_MARKERS = (
    "used car",
    "used vehicle",
    "pre-owned",
    "preowned",
    "second hand",
    "second-hand",
)
_NEW_VEHICLE_PRICE_MARKERS = (
    "new car",
    "new vehicle",
    "brand new",
    "ex-showroom",
    "ex showroom",
    "showroom price",
    "on-road price",
)
_FOREIGN_CURRENCY_PRICE_PATTERN = re.compile(
    r"(?:USD|US\$|EUR|GBP|AED|AUD|CAD|SGD|JPY|CNY|[$€£¥])\s*[\d,.]+",
    re.IGNORECASE,
)
_MIN_REALISTIC_PRICE_INR = Decimal("25000")
_MAX_REALISTIC_PRICE_INR = Decimal("50000000")
_NON_ASKING_PRICE_CONTEXT = re.compile(
    r"\b(?:emi|loan|finance|financing|down\s+payment|installments?|"
    r"monthly|per\s+month|kmpl|kms?|kilomet(?:er|re)s?|cc|bhp)\b",
    re.IGNORECASE,
)
_ASKING_PRICE_CONTEXT = re.compile(
    r"(?:asking\s+price|sale\s+price|selling\s+price|price|listed\s+(?:at|for)|"
    r"for\s+sale)\s*[:\-]?\s*[^.!?;\n]{0,25}$",
    re.IGNORECASE,
)
_LOG_REGISTRATION_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9])(?:[A-Z]{2}\s*[-./]?\s*\d{1,2}\s*[-./]?\s*"
    r"[A-Z]{1,4}\s*[-./]?\s*\d{1,4}|\d{2}\s*[-./]?\s*BH\s*[-./]?\s*"
    r"\d{4}\s*[-./]?\s*[A-Z]{1,2})(?![A-Z0-9])"
)
_LOG_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")


def _log_diagnostic(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(
        json.dumps(payload, separators=(",", ":"), default=str),
        extra=payload,
    )


class TavilyError(Exception):
    """Safe Tavily API failure with no upstream content or credentials."""


class TavilyRateLimited(TavilyError):
    """Tavily rejected the request due to a rate or usage limit."""


class TavilyInvalidResponse(TavilyError):
    """Tavily returned data outside the expected response contract."""


class TavilyMarketListingProvider:
    provider_name = "Tavily Web Search"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        result_limit: int = 20,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key or not api_key.strip():
            raise TavilyError("Tavily API key is not configured.")
        if not 0 < timeout_seconds <= 60:
            raise ValueError("Tavily timeout must be greater than 0 and at most 60 seconds.")
        if not 1 <= result_limit <= 50:
            raise ValueError("Tavily result limit must be between 1 and 50.")
        self._api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.result_limit = result_limit
        self.transport = transport
        self.rejected_listing_counts: dict[str, int] = {}
        self.market_research_rejection_counts: dict[str, int] = {}
        self.search_observability: dict[str, int] = {}
        self.market_research_results: list[dict[str, Any]] = []
        self.search_scopes: list[str] = []

    def fetch_listings(self, criteria: MarketVehicleCriteria) -> list[MarketListing]:
        retrieved_at = datetime.now(timezone.utc)
        listings: list[MarketListing] = []
        rejected: dict[str, int] = {}
        research_rejected: dict[str, int] = {}
        seen_urls: set[str] = set()
        research_by_url: dict[str, dict[str, Any]] = {}
        final_result = calculate_market_comparables([], criteria)
        queries = self._queries(criteria)[:2]
        self.market_research_results = []
        self.market_research_rejection_counts = {}
        self.search_observability = {
            "queries_issued": 0,
            "raw_results_returned": 0,
            "listings_extracted": 0,
            "extraction_rejected": 0,
            "relevance_rejected": 0,
            "market_research_rejected": 0,
            "duplicates_removed": 0,
            "outliers_removed": 0,
        }
        self.search_scopes = []
        for tier, query in enumerate(queries):
            self.search_observability["queries_issued"] += 1
            self.search_scopes.append(
                f"{criteria.location} and nearby regions"
                if criteria.location
                else "Location not specified; no geographic fallback"
            )
            results = self._search(query)
            self.search_observability["raw_results_returned"] += len(results)
            query_research_accepted = 0
            query_research_rejected: dict[str, int] = {}
            query_comparable_accepted = 0
            query_comparable_rejected: dict[str, int] = {}
            for result in results:
                research, research_reason = self._market_research_from_result(
                    result, criteria, retrieved_at
                )
                if research is not None:
                    query_research_accepted += 1
                elif research_reason is not None:
                    research_rejected[research_reason] = (
                        research_rejected.get(research_reason, 0) + 1
                    )
                    query_research_rejected[research_reason] = (
                        query_research_rejected.get(research_reason, 0) + 1
                    )
                    self.search_observability["market_research_rejected"] += 1
                listing, reason = self._listing_from_result(result, criteria, retrieved_at)
                if research is not None:
                    url_key = self._url_key(research["source_url"])
                    previous_research = research_by_url.get(url_key)
                    if previous_research is None or self._research_rank(
                        research, criteria
                    ) > self._research_rank(previous_research, criteria):
                        research_by_url[url_key] = research
                comparable_rejection_reason = reason
                comparable_accepted = listing is not None
                if listing is None:
                    if reason is not None:
                        query_comparable_rejected[reason] = (
                            query_comparable_rejected.get(reason, 0) + 1
                        )
                        rejected[reason] = rejected.get(reason, 0) + 1
                        if reason in {
                            "irrelevant_vehicle",
                            "year_mismatch",
                            "non_listing_source",
                            "editorial_or_video_result",
                        }:
                            self.search_observability["relevance_rejected"] += 1
                        else:
                            self.search_observability["extraction_rejected"] += 1
                    self._log_search_result(
                        query,
                        len(results),
                        result,
                        research_reason,
                        research,
                        comparable_rejection_reason,
                        comparable_accepted,
                    )
                    continue
                self.search_observability["listings_extracted"] += 1
                url_key = self._url_key(str(listing.source_url))
                if url_key in seen_urls:
                    rejected["duplicate_listing"] = rejected.get("duplicate_listing", 0) + 1
                    self.search_observability["duplicates_removed"] += 1
                    comparable_rejection_reason = "duplicate_listing"
                    comparable_accepted = False
                    query_comparable_rejected["duplicate_listing"] = (
                        query_comparable_rejected.get("duplicate_listing", 0) + 1
                    )
                    self._log_search_result(
                        query,
                        len(results),
                        result,
                        research_reason,
                        research,
                        comparable_rejection_reason,
                        comparable_accepted,
                    )
                    continue
                seen_urls.add(url_key)
                listings.append(listing)
                query_comparable_accepted += 1
                self._log_search_result(
                    query,
                    len(results),
                    result,
                    research_reason,
                    research,
                    comparable_rejection_reason,
                    comparable_accepted,
                )
            _log_diagnostic(
                "tavily_query_processing_summary",
                query=self._safe_log_text(query, limit=500),
                result_count=len(results),
                market_research_accepted_count=query_research_accepted,
                market_research_rejection_counts=query_research_rejected,
                comparable_accepted_count=query_comparable_accepted,
                comparable_rejection_counts=query_comparable_rejected,
                cumulative_comparable_candidate_count=len(listings),
                cumulative_market_research_count=len(research_by_url),
            )
            final_result = calculate_market_comparables(listings, criteria)
            exact_and_close_count = sum(
                item.match_type in {"exact_match", "close_match"}
                for item in final_result.listings
            )
            if (
                final_result.status == "available"
                and (
                    exact_and_close_count >= MIN_COMPARABLE_COUNT
                    or tier == len(queries) - 1
                )
            ):
                break
        for reason, count in final_result.rejected_counts.items():
            rejected[reason] = rejected.get(reason, 0) + count
            if reason not in {"duplicate_listing", "price_outlier"}:
                self.search_observability["relevance_rejected"] += count
        self.search_observability["outliers_removed"] = final_result.rejected_counts.get(
            "price_outlier", 0
        )
        self.rejected_listing_counts = rejected
        self.market_research_rejection_counts = research_rejected
        self.market_research_results = sorted(
            research_by_url.values(),
            key=lambda item: self._research_rank(item, criteria),
            reverse=True,
        )
        _log_diagnostic(
            "tavily_search_processing_completed",
            final_accepted_listing_count=len(final_result.listings),
            final_market_research_listing_count=len(self.market_research_results),
            final_rejected_listing_counts=dict(rejected),
        )
        return final_result.listings

    @staticmethod
    def _safe_log_text(value: Any, *, limit: int = 500) -> str | None:
        if not isinstance(value, str):
            return None
        sanitized = _LOG_REGISTRATION_PATTERN.sub("[registration redacted]", value)
        sanitized = _LOG_PHONE_PATTERN.sub("[phone redacted]", sanitized)
        return sanitized[:limit]

    @classmethod
    def _safe_log_url(cls, value: Any) -> tuple[str | None, str | None]:
        if not isinstance(value, str):
            return None, None
        try:
            parts = urlsplit(value)
        except ValueError:
            return None, None
        domain = parts.hostname.casefold() if parts.hostname else None
        if domain and domain.startswith("www."):
            domain = domain[4:]
        safe_path = cls._safe_log_text(parts.path, limit=500)
        safe_url = (
            f"{parts.scheme.casefold()}://{domain}{safe_path or ''}"
            if parts.scheme and domain
            else safe_path
        )
        return domain, safe_url

    @classmethod
    def _log_search_result(
        cls,
        query: str,
        result_count: int,
        result: Any,
        research_reason: str | None,
        research: dict[str, Any] | None,
        comparable_reason: str | None,
        comparable_accepted: bool,
    ) -> None:
        item = result if isinstance(result, dict) else {}
        title = item.get("title")
        url = item.get("url")
        domain, safe_url = cls._safe_log_url(url)
        amount, price_status = cls._asking_price_from_fields(
            title if isinstance(title, str) else "",
            item.get("content") if isinstance(item.get("content"), str) else "",
            item.get("raw_content")
            if isinstance(item.get("raw_content"), str)
            else "",
        )
        _log_diagnostic(
            "tavily_search_result_processed",
            query=cls._safe_log_text(query, limit=500),
            result_count=result_count,
            result_domain=domain,
            result_title=cls._safe_log_text(title),
            result_url=safe_url,
            price_extracted=amount is not None,
            extracted_price_inr=format(amount, "f") if amount is not None else None,
            price_parse_status=price_status,
            market_research_accepted=research is not None,
            market_research_rejection_reason=research_reason,
            comparable_accepted=comparable_accepted,
            comparable_rejection_reason=comparable_reason,
        )

    def _search(self, query: str) -> list[Any]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": self.result_limit,
            "include_answer": False,
            "include_raw_content": True,
        }
        try:
            with httpx.Client(
                timeout=httpx.Timeout(self.timeout_seconds),
                transport=self.transport,
            ) as client:
                response = request_with_response_limit(
                    client,
                    "POST",
                    TAVILY_SEARCH_ENDPOINT,
                    max_bytes=settings.max_external_response_bytes,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json=payload,
                )
        except ExternalResponseTooLarge:
            raise TavilyInvalidResponse("Tavily response exceeded the allowed size.") from None
        except httpx.TimeoutException:
            raise TavilyError("Tavily search request timed out.") from None
        except httpx.RequestError:
            raise TavilyError("Tavily search request could not be completed.") from None

        if response.status_code == 429:
            raise TavilyRateLimited("Tavily rate or usage limit was reached.")
        if response.is_error:
            raise TavilyError(f"Tavily returned HTTP {response.status_code}.")
        try:
            body = response.json()
        except ValueError:
            raise TavilyInvalidResponse("Tavily returned invalid JSON.") from None
        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise TavilyInvalidResponse("Tavily response is missing a results list.")
        results = body["results"]
        _log_diagnostic(
            "tavily_search_response",
            query=self._safe_log_text(query, limit=500),
            http_status=response.status_code,
            result_count=len(results),
        )
        return results

    @classmethod
    def _queries(cls, criteria: MarketVehicleCriteria) -> list[str]:
        identity = f'"{criteria.make.strip()} {criteria.model.strip()}"'
        location = f" near {criteria.location}" if criteria.location else ""
        search_suffix = f"used car prices India -Grand -Nios{location}"
        candidates: list[tuple[int, str]] = []
        detailed_terms = [
            str(criteria.year) if criteria.year is not None else None,
            str(criteria.variant).strip() if criteria.variant else None,
            criteria.fuel_type.strip() if criteria.fuel_type else None,
            criteria.transmission.strip() if criteria.transmission else None,
        ]
        detailed = " ".join(term for term in detailed_terms if term)
        candidates.append(
            (
                100
                + (30 if criteria.year is not None else 0)
                + (25 if criteria.variant else 0)
                + (20 if criteria.location else 0)
                + (10 if criteria.fuel_type else 0)
                + (8 if criteria.transmission else 0),
                f"{identity} {detailed} {search_suffix}".replace("  ", " "),
            )
        )
        if criteria.year is not None:
            candidates.append(
                (
                    100 + 30 + (20 if criteria.location else 0),
                    f"{identity} {criteria.year} {search_suffix}",
                )
            )
        if criteria.variant:
            candidates.append(
                (
                    100 + 25 + (20 if criteria.location else 0),
                    f"{identity} {criteria.variant} {search_suffix}",
                )
            )
        if criteria.fuel_type or criteria.transmission:
            mechanical_terms = " ".join(
                term
                for term in (
                    criteria.fuel_type.strip() if criteria.fuel_type else None,
                    criteria.transmission.strip() if criteria.transmission else None,
                )
                if term
            )
            candidates.append(
                (
                    100
                    + (10 if criteria.fuel_type else 0)
                    + (8 if criteria.transmission else 0)
                    + (20 if criteria.location else 0),
                    f"{identity} {mechanical_terms} {search_suffix}",
                )
            )
        ranked: dict[str, int] = {}
        for score, query in candidates:
            ranked[query] = max(score, ranked.get(query, score))
        return [
            query
            for query, _score in sorted(
                ranked.items(), key=lambda item: (-item[1], item[0])
            )[:2]
        ]

    @staticmethod
    def _located_query(query: str, location: str | None) -> str:
        if location:
            return f"{query} near {location}"
        return query

    @classmethod
    def _matches_vehicle_identity(cls, text: str, criteria: MarketVehicleCriteria) -> bool:
        identity = f"{criteria.make} {criteria.model}"
        if not cls._contains_phrase(text, identity):
            return False
        if criteria.model.strip().casefold() != "i10":
            return True
        pattern = re.compile(
            rf"(?<!\w){re.escape(criteria.make.strip())}[\W_]+i10(?!\w)",
            re.IGNORECASE,
        )
        return any(
            not re.search(r"\bgrand\s+$", text[max(0, match.start() - 12):match.start()], re.IGNORECASE)
            and not re.match(r"\s+nios\b", text[match.end():], re.IGNORECASE)
            for match in pattern.finditer(text)
        )

    @classmethod
    def _result_matches_vehicle_identity(
        cls, title: str, content: str, criteria: MarketVehicleCriteria
    ) -> bool:
        if (
            criteria.model.strip().casefold() == "i10"
            and re.search(r"\b(?:grand\s+i10|i10\s+nios)\b", title, re.IGNORECASE)
        ):
            return False
        if cls._matches_vehicle_identity(title, criteria):
            return True
        if cls._contains_phrase(title, criteria.make) and cls._contains_phrase(
            title, criteria.model
        ):
            return False
        return cls._matches_vehicle_identity(content, criteria)

    @staticmethod
    def _url_key(url: str) -> str:
        parts = urlsplit(url)
        host = (parts.hostname or "").casefold()
        if host.startswith("www."):
            host = host[4:]
        return f"{parts.scheme.casefold()}://{host}{parts.path.rstrip('/')}"

    @staticmethod
    def _query(criteria: MarketVehicleCriteria) -> str:
        return TavilyMarketListingProvider._queries(criteria)[0]

    @classmethod
    def _market_research_from_result(
        cls,
        result: Any,
        criteria: MarketVehicleCriteria,
        retrieved_at: datetime,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(result, dict):
            return None, "malformed_search_result"
        title = result.get("title")
        url = result.get("url")
        content = result.get("content")
        raw_content = result.get("raw_content")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            return None, "unusable_market_result"
        content = content if isinstance(content, str) else ""
        raw_content = raw_content if isinstance(raw_content, str) else ""
        try:
            parts = urlsplit(url)
            if (
                parts.scheme.casefold() not in {"http", "https"}
                or not parts.hostname
                or parts.username
                or parts.password
            ):
                return None, "invalid_listing_url"
        except ValueError:
            return None, "invalid_listing_url"

        evidence = f"{title}\n{content}\n{raw_content}"
        if not cls._result_matches_vehicle_identity(
            title, f"{content}\n{raw_content}", criteria
        ):
            return None, "irrelevant_vehicle"
        if not cls._credible_listing_source(parts.hostname):
            return None, "non_listing_source"
        if cls._is_collection_page(parts.path, criteria) or cls._is_collection_title(title):
            return None, "collection_page"
        editorial_text = f"{title} {parts.path}".casefold()
        if any(marker in editorial_text for marker in _EDITORIAL_MARKERS):
            return None, "editorial_or_video_result"
        if not cls._contains_used_listing_marker(evidence):
            path = parts.path.casefold()
            if not any(marker in path for marker in _USED_LISTING_PATH_MARKERS):
                return None, "not_a_used_listing"
        year = next(
            (
                parsed_year
                for text in (title, content, raw_content)
                if (parsed_year := cls._year(text, None)) is not None
            ),
            None,
        )
        variant = next(
            (
                parsed_variant
                for text in (title, content, raw_content)
                if (
                    parsed_variant := cls._variant(text, text, criteria, year)
                )
                is not None
            ),
            None,
        )
        price, price_status = cls._asking_price_from_fields(
            title, content, raw_content
        )
        if price is None:
            if price_status == "new_vehicle":
                return None, "new_or_ex_showroom_price"
            if _FOREIGN_CURRENCY_PRICE_PATTERN.search(evidence):
                return None, "foreign_currency_price"
            return None, f"{price_status}_price"
        if price is not None and not _MIN_REALISTIC_PRICE_INR <= price <= _MAX_REALISTIC_PRICE_INR:
            return None, "implausible_price"
        host = parts.hostname.casefold()
        source_name = host[4:] if host.startswith("www.") else host
        return {
            "source_url": url,
            "source_name": source_name,
            "year": year,
            "variant": variant,
            "asking_price_inr": format(price, "f") if price is not None else None,
            "priced_used_listing": True,
        }, None

    @staticmethod
    def _credible_listing_source(hostname: str) -> bool:
        host = hostname.casefold().removeprefix("www.")
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in _CREDIBLE_LISTING_DOMAINS
        )

    @staticmethod
    def _is_collection_page(path: str, criteria: MarketVehicleCriteria) -> bool:
        segments = [segment.casefold() for segment in path.split("/") if segment]
        if any(
            segment in _COLLECTION_PATH_MARKERS
            or segment.startswith("used-cars-")
            or segment in {"search", "results", "inventory"}
            or re.fullmatch(r"page[-_]\d+", segment)
            for segment in segments
        ):
            return True
        if "used" in segments and any(segment.startswith("under-") for segment in segments):
            return True

        identity_slug = "-".join(
            re.findall(r"[\w]+", f"{criteria.make} {criteria.model}".casefold())
        )
        if "buy-used-car" in segments and segments[-1] in {
            identity_slug,
            criteria.model.casefold().replace(" ", "-"),
        }:
            return True
        if any(
            segments[index:index + 2] == ["used", "cars-for-sale"]
            for index in range(max(0, len(segments) - 1))
        ) and segments[-1] in {
            identity_slug,
            criteria.model.casefold().replace(" ", "-"),
        }:
            return True
        return False

    @staticmethod
    def _is_collection_title(title: str) -> bool:
        normalized = title.strip().casefold()
        return bool(
            re.match(r"^\d{1,3}\s+(?:\d{4}\s+)?used\b", normalized)
            or (
                re.search(r"\bused\b.{0,70}\bcars\b", normalized)
                and re.search(r"\b(?:from|in|listings?|search)\b", normalized)
            )
        )

    @staticmethod
    def _contains_used_listing_marker(text: str) -> bool:
        normalized = text.casefold()
        return any(marker in normalized for marker in _USED_LISTING_MARKERS)

    @staticmethod
    def _contains_new_vehicle_price_marker(text: str) -> bool:
        matches = list(_PRICE_PATTERN.finditer(text))
        return any(
            any(
                marker in TavilyMarketListingProvider._price_context(
                    text, match, index, matches
                ).casefold()
                for marker in _NEW_VEHICLE_PRICE_MARKERS
            )
            for index, match in enumerate(matches)
        )

    @staticmethod
    def _research_rank(
        result: dict[str, Any], criteria: MarketVehicleCriteria
    ) -> tuple[int, int, int, int]:
        year = result.get("year")
        variant = result.get("variant")
        return (
            int(
                bool(criteria.variant)
                and isinstance(variant, str)
                and variant.casefold() == criteria.variant.casefold()
            ),
            int(criteria.year is not None and year == criteria.year),
            -abs(year - criteria.year)
            if criteria.year is not None and isinstance(year, int)
            else -999,
            int(isinstance(variant, str)),
        )

    @classmethod
    def _variant(
        cls,
        evidence: str,
        title: str,
        criteria: MarketVehicleCriteria,
        year: int | None,
    ) -> str | None:
        if criteria.variant and cls._contains_phrase(title, criteria.variant):
            return criteria.variant
        if year is None:
            return None
        tokens = re.findall(r"[\w]+", title)
        model_tokens = re.findall(r"[\w]+", criteria.model)
        normalized_tokens = [token.casefold() for token in tokens]
        normalized_model = [token.casefold() for token in model_tokens]
        start = next(
            (
                index
                for index in range(len(tokens) - len(normalized_model) + 1)
                if normalized_tokens[index:index + len(normalized_model)] == normalized_model
            ),
            None,
        )
        if start is None:
            return None
        stop_words = {
            "used", "car", "cars", "for", "sale", "price", "prices", "petrol",
            "diesel", "cng", "lpg", "electric", "manual", "automatic", "auto",
            "amt", "cvt", "dct", "in", "at", "near", "with", "km", "kms",
        }
        variant_tokens = []
        for token in tokens[start + len(normalized_model):]:
            if token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2100:
                continue
            if token.isdigit() or token.casefold() in stop_words:
                break
            variant_tokens.append(token)
            if len(variant_tokens) == 3:
                break
        return " ".join(variant_tokens) if variant_tokens else None

    @staticmethod
    def _reported_location(evidence: str, target_location: str | None) -> str | None:
        if not target_location:
            return None
        locations = [part.strip() for part in target_location.split(",") if part.strip()]
        return next(
            (location for location in locations if TavilyMarketListingProvider._contains_phrase(evidence, location)),
            None,
        )

    @classmethod
    def _listing_from_result(
        cls,
        result: Any,
        criteria: MarketVehicleCriteria,
        retrieved_at: datetime,
    ) -> tuple[MarketListing | None, str | None]:
        if not isinstance(result, dict):
            return None, "malformed_search_result"
        title = result.get("title")
        url = result.get("url")
        content = result.get("content")
        raw_content = result.get("raw_content")
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            return None, "malformed_search_result"
        content = content if isinstance(content, str) else ""
        raw_content = raw_content if isinstance(raw_content, str) else ""
        try:
            parts = urlsplit(url)
            if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
                return None, "invalid_listing_url"
        except ValueError:
            return None, "invalid_listing_url"

        evidence = f"{title}\n{content}\n{raw_content}"
        host = (parts.hostname or "").casefold().removeprefix("www.")
        if not cls._credible_listing_source(host):
            return None, "non_listing_source"
        if cls._is_collection_page(parts.path, criteria) or cls._is_collection_title(title):
            return None, "collection_page"
        editorial_text = f"{title} {parts.path}".casefold()
        if any(marker in editorial_text for marker in _EDITORIAL_MARKERS):
            return None, "editorial_or_video_result"
        if not cls._result_matches_vehicle_identity(
            title, f"{content}\n{raw_content}", criteria
        ):
            return None, "irrelevant_vehicle"
        if not cls._contains_used_listing_marker(evidence):
            path = parts.path.casefold()
            if not any(marker in path for marker in _USED_LISTING_PATH_MARKERS):
                return None, "not_a_used_listing"
        year = cls._year(evidence, criteria.year)
        if year is None:
            return None, "missing_or_ambiguous_year"
        price, price_status = cls._asking_price_from_fields(
            title, content, raw_content
        )
        if price is None:
            if price_status == "new_vehicle":
                return None, "new_or_ex_showroom_price"
            if _FOREIGN_CURRENCY_PRICE_PATTERN.search(evidence):
                return None, "foreign_currency_price"
            return None, (
                "missing_or_ambiguous_price"
                if price_status in {"missing", "ambiguous"}
                else "implausible_price"
            )
        if not _MIN_REALISTIC_PRICE_INR <= price <= _MAX_REALISTIC_PRICE_INR:
            return None, "implausible_price"

        variant = cls._variant(evidence, title, criteria, year)
        fuel_type = cls._fuel_type(evidence)
        transmission = cls._transmission(evidence)
        mileage = cls._mileage(evidence)
        location = cls._reported_location(evidence, criteria.location)
        host = parts.hostname.casefold()
        source_name = host[4:] if host.startswith("www.") else host
        try:
            listing = MarketListing(
                source_name=source_name,
                make=criteria.make,
                model=criteria.model,
                variant=variant,
                year=year,
                title=title.strip(),
                match_type="approximate_match",
                fuel_type=fuel_type,
                transmission=transmission,
                mileage_km=mileage,
                location=location,
                advertised_price_inr=price,
                source_url=url,
                listed_at=None,
                retrieved_at=retrieved_at,
            )
        except (TypeError, ValueError):
            return None, "malformed_listing"
        return listing, None

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        parts = re.findall(r"[\w]+", phrase.casefold())
        if not parts:
            return False
        pattern = r"[\W_]*".join(re.escape(part) for part in parts)
        return bool(re.search(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE))

    @staticmethod
    def _year(text: str, target_year: int | None) -> int | None:
        years = {int(value) for value in _YEAR_PATTERN.findall(text)}
        if target_year is not None:
            nearby_years = {
                year for year in years if abs(year - target_year) <= YEAR_TOLERANCE
            }
            if len(nearby_years) == 1:
                return next(iter(nearby_years))
            if not nearby_years and len(years) == 1:
                return next(iter(years))
            return None
        return next(iter(years)) if len(years) == 1 else None

    @staticmethod
    def _asking_price(text: str) -> Decimal | None:
        return TavilyMarketListingProvider._asking_price_result(text)[0]

    @staticmethod
    def _asking_price_result(text: str) -> tuple[Decimal | None, str]:
        matches = list(_PRICE_PATTERN.finditer(text))
        prices: set[Decimal] = set()
        new_vehicle_price_found = False
        for index, match in enumerate(matches):
            number = match.group(1) or match.group(3)
            unit = (match.group(2) or match.group(4) or "").casefold()
            try:
                amount = Decimal(number.replace(",", ""))
            except Exception:
                continue
            amount *= _MULTIPLIERS.get(unit, Decimal("1"))
            if amount <= 0 or TavilyMarketListingProvider._is_non_asking_amount(
                text, match
            ):
                continue
            context = TavilyMarketListingProvider._price_context(
                text, match, index, matches
            )
            if any(marker in context.casefold() for marker in _NEW_VEHICLE_PRICE_MARKERS):
                new_vehicle_price_found = True
                continue
            normalized_amount = amount.quantize(Decimal("0.01"))
            prices.add(normalized_amount)
        if not prices:
            return None, "new_vehicle" if new_vehicle_price_found else "missing"
        if len(prices) > 1:
            return None, "ambiguous"
        return next(iter(prices)), "reliable"

    @staticmethod
    def _price_context(
        text: str,
        match: re.Match[str],
        index: int,
        matches: list[re.Match[str]],
    ) -> str:
        start = max(
            text.rfind(separator, 0, match.start()) + 1
            for separator in (".", "!", "?", ";", "\n")
        )
        end_candidates = [
            position
            for separator in (".", "!", "?", ";", "\n")
            if (position := text.find(separator, match.end())) >= 0
        ]
        end = min(end_candidates) if end_candidates else len(text)
        if index:
            start = max(start, matches[index - 1].end())
        if index + 1 < len(matches):
            end = min(end, matches[index + 1].start())
        return text[start:end]

    @staticmethod
    def _is_non_asking_amount(text: str, match: re.Match[str]) -> bool:
        prefix = text[max(0, match.start() - 55):match.start()]
        suffix = text[match.end():match.end() + 24]
        if re.match(
            r"\s*(?:per\s+month|monthly|km|kms|cc|bhp)\b",
            suffix,
            re.IGNORECASE,
        ):
            return True
        non_asking_matches = list(_NON_ASKING_PRICE_CONTEXT.finditer(prefix))
        if not non_asking_matches:
            return False
        asking_matches = list(_ASKING_PRICE_CONTEXT.finditer(prefix))
        return (
            not asking_matches
            or non_asking_matches[-1].start() > asking_matches[-1].start()
        )

    @classmethod
    def _asking_price_from_fields(
        cls, title: str, content: str, raw_content: str
    ) -> tuple[Decimal | None, str]:
        primary_text = f"{title}\n{content}"
        amount, status = cls._asking_price_result(primary_text)
        if status == "reliable":
            return amount, status
        if status == "ambiguous":
            contextual = cls._contextual_asking_prices(primary_text)
            if len(contextual) == 1:
                return next(iter(contextual)), "reliable"
            return None, "ambiguous"

        amount, raw_status = cls._asking_price_result(raw_content)
        if raw_status == "reliable":
            return amount, raw_status
        if raw_status == "ambiguous":
            contextual = cls._contextual_asking_prices(raw_content)
            if len(contextual) == 1:
                return next(iter(contextual)), "reliable"
        if status == "new_vehicle" or raw_status == "new_vehicle":
            return None, "new_vehicle"
        return None, "ambiguous" if raw_status == "ambiguous" else raw_status

    @staticmethod
    def _contextual_asking_prices(text: str) -> set[Decimal]:
        prices: set[Decimal] = set()
        matches = list(_PRICE_PATTERN.finditer(text))
        for index, match in enumerate(matches):
            prefix = text[max(0, match.start() - 55):match.start()]
            if not _ASKING_PRICE_CONTEXT.search(prefix):
                continue
            if TavilyMarketListingProvider._is_non_asking_amount(text, match):
                continue
            context = TavilyMarketListingProvider._price_context(
                text, match, index, matches
            )
            if any(marker in context.casefold() for marker in _NEW_VEHICLE_PRICE_MARKERS):
                continue
            number = match.group(1) or match.group(3)
            unit = (match.group(2) or match.group(4) or "").casefold()
            try:
                amount = Decimal(number.replace(",", ""))
            except Exception:
                continue
            amount *= _MULTIPLIERS.get(unit, Decimal("1"))
            if amount > 0:
                prices.add(amount.quantize(Decimal("0.01")))
        return prices

    @staticmethod
    def _mileage(text: str) -> int | None:
        values: set[int] = set()
        for raw in _MILEAGE_PATTERN.findall(text):
            try:
                values.add(int(Decimal(raw.replace(",", ""))))
            except Exception:
                continue
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _fuel_type(text: str) -> str | None:
        found = {
            fuel
            for fuel, terms in _FUEL_TERMS.items()
            if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) for term in terms)
        }
        return next(iter(found)) if len(found) == 1 else None

    @staticmethod
    def _transmission(text: str) -> str | None:
        found = {
            transmission
            for transmission, terms in _TRANSMISSION_TERMS.items()
            if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) for term in terms)
        }
        return next(iter(found)) if len(found) == 1 else None


def get_tavily_market_listing_provider() -> TavilyMarketListingProvider | None:
    """Construct the market provider only when its backend secret is configured."""
    secret = settings.tavily_api_key
    if secret is None or not secret.get_secret_value().strip():
        return None
    return TavilyMarketListingProvider(
        secret.get_secret_value(),
        timeout_seconds=settings.tavily_timeout_seconds,
        result_limit=settings.tavily_result_limit,
    )

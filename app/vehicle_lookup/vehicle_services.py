"""Isolated Challan and FASTag HTTP adapters for unverified endpoint candidates."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit
from app.schemas.vehicle_services import ChallanDetails, ChallanRecord, FastagDetails
from app.vehicle_lookup.provider import (
    VehicleLookupAuthenticationError,
    VehicleLookupInvalidResponse,
    VehicleLookupRejected,
    VehicleLookupUnavailable,
)


class ChallanProvider(Protocol):
    provider_name: str

    def lookup(self, registration_number: str) -> ChallanDetails: ...


class FastagProvider(Protocol):
    provider_name: str

    def lookup(self, registration_number: str) -> FastagDetails: ...


class _JSONHTTPProvider:
    registration_pattern = re.compile(
        r"^(?:[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{1,4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})$"
    )

    def __init__(
        self,
        base_url: str,
        query_parameter: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Vehicle service URL must be HTTPS and contain no credentials.")
        self.base_url = base_url
        self.query_parameter = query_parameter
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _fetch(self, registration_number: str) -> dict[str, Any] | list[Any]:
        normalized = re.sub(r"[\s-]", "", registration_number).upper()
        if not self.registration_pattern.fullmatch(normalized):
            raise VehicleLookupRejected("Registration number has an unsupported format.")
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds), transport=self.transport) as client:
                response = request_with_response_limit(
                    client,
                    "GET",
                    self.base_url,
                    max_bytes=settings.max_external_response_bytes,
                    params={self.query_parameter: normalized},
                )
        except ExternalResponseTooLarge:
            raise VehicleLookupInvalidResponse("Vehicle service response exceeded the allowed size.") from None
        except httpx.TimeoutException:
            raise VehicleLookupUnavailable("Vehicle service timed out.") from None
        except httpx.RequestError:
            raise VehicleLookupUnavailable("Vehicle service is unavailable.") from None

        if response.status_code in {401, 403}:
            raise VehicleLookupAuthenticationError("Vehicle service authentication failed.")
        if response.status_code == 429 or response.status_code >= 500:
            raise VehicleLookupUnavailable("Vehicle service is unavailable.")
        if response.status_code in {400, 404, 422}:
            raise VehicleLookupRejected("Vehicle service rejected the lookup.")
        if not response.is_success:
            raise VehicleLookupInvalidResponse("Vehicle service returned an HTTP error.")
        try:
            payload = response.json()
        except ValueError:
            raise VehicleLookupInvalidResponse("Vehicle service returned malformed JSON.") from None
        if not isinstance(payload, (dict, list)):
            raise VehicleLookupInvalidResponse("Vehicle service returned an unexpected response.")
        if isinstance(payload, dict):
            if payload.get("success") is False or payload.get("ok") is False or payload.get("error"):
                raise VehicleLookupRejected("Vehicle service did not complete the lookup.")
            status = payload.get("status")
            if isinstance(status, str) and status.strip().casefold() in {
                "error", "failed", "failure", "not_found", "unauthorized"
            }:
                if status.strip().casefold() == "unauthorized":
                    raise VehicleLookupAuthenticationError("Vehicle service authentication failed.")
                raise VehicleLookupRejected("Vehicle service did not complete the lookup.")
            for candidate in (status, payload.get("code")):
                if isinstance(candidate, str) and candidate.isdigit():
                    candidate = int(candidate)
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    if candidate in {401, 403}:
                        raise VehicleLookupAuthenticationError("Vehicle service authentication failed.")
                    if candidate == 429 or candidate >= 500:
                        raise VehicleLookupUnavailable("Vehicle service is unavailable.")
                    if candidate not in {0, 1} and not 200 <= candidate < 300:
                        raise VehicleLookupRejected("Vehicle service did not complete the lookup.")
        return payload


class ChallanHTTPProvider(_JSONHTTPProvider):
    provider_name = "challan_candidate"
    _CONTAINERS = {"data", "result", "challans", "violations", "records", "challandetails"}
    _ALIASES = {
        "status": {"status", "challanstatus"},
        "violation_description": {"violationdescription", "description", "challandescription", "offencename"},
        "amount": {"amount", "challanamount", "fineamount", "penaltyamount"},
        "currency": {"currency", "currencycode"},
        "date": {"date", "challandate", "violationdate", "issuedate"},
        "payment_status": {"paymentstatus", "paidstatus", "challanpaymentstatus"},
    }

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0, transport=None):
        super().__init__(base_url, "vehicle", timeout_seconds=timeout_seconds, transport=transport)

    def lookup(self, registration_number: str) -> ChallanDetails:
        payload = self._fetch(registration_number)
        rows, explicit_empty = self._record_rows(payload, self._CONTAINERS)
        if not rows and not explicit_empty:
            raise VehicleLookupInvalidResponse("Challan provider returned no supported result structure.")
        records: list[ChallanRecord] = []
        for row in rows:
            normalized = self._normalize_record(row, self._ALIASES)
            if any(value is not None for value in normalized.values()):
                try:
                    records.append(ChallanRecord.model_validate(normalized))
                except Exception:
                    raise VehicleLookupInvalidResponse("Challan provider returned invalid supported fields.") from None
        overall_status = self._find_value(payload, {"status", "challanstatus"})
        details = ChallanDetails(status=self._text(overall_status), records=records)
        if rows and not records and not explicit_empty:
            raise VehicleLookupInvalidResponse("Challan provider returned no usable challan records.")
        if not records and not explicit_empty and not details.status:
            raise VehicleLookupInvalidResponse("Challan provider returned no supported challan fields.")
        return details

    @classmethod
    def _record_rows(cls, payload: dict[str, Any] | list[Any], containers: set[str]):
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)], len(payload) == 0
        queue: list[Any] = [payload]
        for _ in range(5):
            next_queue: list[Any] = []
            for item in queue:
                if isinstance(item, dict):
                    for key, value in item.items():
                        normalized_key = cls._key(key)
                        if normalized_key in containers and isinstance(value, list):
                            return [row for row in value if isinstance(row, dict)], len(value) == 0
                        if isinstance(value, (dict, list)):
                            next_queue.append(value)
                elif isinstance(item, list):
                    next_queue.extend(item[:20])
            queue = next_queue
        mapping = cls._find_record_mapping(payload, cls._ALIASES)
        return ([mapping] if mapping else []), False

    @classmethod
    def _find_record_mapping(cls, payload: Any, aliases: dict[str, set[str]]) -> dict[str, Any] | None:
        candidates = cls._walk(payload)
        for mapping in reversed(candidates):
            keys = {cls._key(key) for key in mapping}
            if any(keys & names for names in aliases.values()):
                return mapping
        return None

    @classmethod
    def _normalize_record(cls, row: dict[str, Any], aliases: dict[str, set[str]]) -> dict[str, Any]:
        result = {}
        for target, keys in aliases.items():
            raw = next((value for key, value in row.items() if cls._key(key) in keys and value not in (None, "")), None)
            if target == "amount":
                result[target] = cls._amount(raw)
            elif target == "date":
                result[target] = cls._date(raw)
            else:
                result[target] = cls._text(raw)
        return result

    @classmethod
    def _find_value(cls, payload: Any, aliases: set[str]) -> Any:
        for mapping in cls._walk(payload):
            for key, value in mapping.items():
                if cls._key(key) in aliases and value not in (None, ""):
                    return value
        return None

    @staticmethod
    def _walk(payload: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def visit(value: Any, depth: int = 0) -> None:
            if depth > 5:
                return
            if isinstance(value, dict):
                found.append(value)
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value[:20]:
                    visit(child, depth + 1)

        visit(payload)
        return found

    @staticmethod
    def _key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).casefold())

    @staticmethod
    def _text(value: Any) -> str | None:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text[:500] or None

    @classmethod
    def _amount(cls, value: Any) -> Decimal | None:
        if isinstance(value, bool) or value is None:
            return None
        text = re.sub(r"[,₹\\s]", "", str(value))
        match = re.search(r"\d+(?:\.\d{1,2})?", text)
        if not match:
            return None
        try:
            amount = Decimal(match.group())
            return amount if amount.is_finite() and amount >= 0 else None
        except InvalidOperation:
            return None

    @staticmethod
    def _date(value: Any) -> date | None:
        return _parse_date(value)


class FastagHTTPProvider(_JSONHTTPProvider):
    provider_name = "fastag_candidate"
    _ALIASES = {
        "status": {"status", "tagstatus", "fastagstatus"},
        "issuing_bank": {"issuingbank", "bank", "bankname", "issuerbank"},
        "issue_date": {"issuedate", "tagissuedate", "fastagissuedate"},
        "vehicle_class": {"vehicleclass", "vehclass", "class"},
    }
    _CONTAINERS = {"data", "result", "fastag", "tag", "vehicledetails", "response", "details"}

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0, transport=None):
        super().__init__(base_url, "veh", timeout_seconds=timeout_seconds, transport=transport)

    def lookup(self, registration_number: str) -> FastagDetails:
        payload = self._fetch(registration_number)
        flattened = ChallanHTTPProvider._walk(payload)
        values: dict[str, Any] = {}
        for target, aliases in self._ALIASES.items():
            raw = next(
                (
                    value
                    for mapping in flattened
                    for key, value in mapping.items()
                    if ChallanHTTPProvider._key(key) in aliases and value not in (None, "")
                ),
                None,
            )
            values[target] = _parse_date(raw) if target == "issue_date" else ChallanHTTPProvider._text(raw)
        if not any(value is not None for value in values.values()):
            raise VehicleLookupInvalidResponse("FASTag provider returned no supported fields.")
        try:
            return FastagDetails.model_validate(values)
        except Exception:
            raise VehicleLookupInvalidResponse("FASTag provider returned invalid supported fields.") from None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d/%b/%Y"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def get_challan_provider() -> ChallanProvider | None:
    if not settings.challan_lookup_enabled:
        return None
    return ChallanHTTPProvider(
        settings.challan_lookup_base_url,
        timeout_seconds=settings.challan_lookup_timeout_seconds,
    )


def get_fastag_provider() -> FastagProvider | None:
    if not settings.fastag_lookup_enabled:
        return None
    return FastagHTTPProvider(
        settings.fastag_lookup_base_url,
        timeout_seconds=settings.fastag_lookup_timeout_seconds,
    )

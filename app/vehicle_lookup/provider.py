from datetime import date, datetime
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit
from app.schemas.vehicle_lookup import VehicleRegistrationDetails


class VehicleLookupError(Exception):
    """A registration lookup provider failed to return usable data."""


class VehicleLookupUnavailable(VehicleLookupError):
    """The configured provider could not be reached or is temporarily unavailable."""


class VehicleLookupAuthenticationError(VehicleLookupUnavailable):
    """The configured candidate rejected authentication."""


class VehicleLookupRejected(VehicleLookupError):
    """The provider rejected the registration lookup request."""


class VehicleLookupInvalidResponse(VehicleLookupError):
    """The provider response did not match the supported JSON shape."""


class VehicleRegistrationProvider(Protocol):
    provider_name: str

    def lookup(self, registration_number: str) -> VehicleRegistrationDetails:
        """Return only vehicle facts actually supplied by an authorized provider."""


class VehicleDetailsHTTPProvider:
    """HTTP adapter for the supplied, unaudited vehicle-details endpoint candidate.

    The response parser uses a strict allowlist and never persists the raw payload.
    No authorization header is sent because none is documented in the project.
    """

    provider_name = "vehiclex_candidate"
    _REGISTRATION = re.compile(
        r"^(?:[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{1,4}|[0-9]{2}BH[0-9]{4}[A-Z]{1,2})$"
    )
    _ALIASES = {
        "make": {"make", "manufacturer", "manufacturername", "makerdescription", "brand"},
        "model": {"model", "modeldesc", "modeldescription", "vehiclemodel"},
        "variant": {"variant", "modelvariant", "version"},
        "fuel_type": {"fuel", "fueltype", "fueltypedescription"},
        "registration_date": {"registrationdate", "dateofregistration", "regdate"},
        "manufacturing_year": {"manufacturingyear", "manufacturedyear", "manufactureyear", "mfgyear"},
        "vehicle_class": {"vehicleclass", "vehclass", "class"},
        "registration_valid_until": {"registrationvaliduntil", "registrationvalidity", "regvalidupto"},
        "rto": {"rto", "rtoname", "rtooffice"},
        "fitness_expiry": {"fitupto", "fitnessexpiry", "fitnessvaliduntil"},
        "puc_expiry": {"pucvaliduntil", "puccupto", "pucexpiry"},
        "insurance_expiry": {"insuranceexpiry", "insurancevaliduntil", "insuranceupto", "insurancevalidupto"},
        "blacklisted": {"blacklisted", "blackliststatus", "isblacklisted"},
    }
    _CONTAINERS = {"data", "result", "vehicle", "vehicledetails", "response", "details"}

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Vehicle lookup URL must be an HTTPS URL without credentials.")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def lookup(self, registration_number: str) -> VehicleRegistrationDetails:
        normalized = re.sub(r"[\s-]", "", registration_number).upper()
        if not self._REGISTRATION.fullmatch(normalized):
            raise VehicleLookupRejected("Registration number has an unsupported format.")
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds), transport=self.transport) as client:
                response = request_with_response_limit(
                    client,
                    "GET",
                    self.base_url,
                    max_bytes=settings.max_external_response_bytes,
                    params={"regn": normalized},
                )
        except ExternalResponseTooLarge:
            raise VehicleLookupInvalidResponse("Vehicle registration provider response exceeded the allowed size.") from None
        except httpx.TimeoutException:
            raise VehicleLookupUnavailable("Vehicle registration provider timed out.") from None
        except httpx.RequestError:
            raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.") from None

        if response.status_code in {401, 403}:
            raise VehicleLookupAuthenticationError("Vehicle registration provider authentication failed.")
        if response.status_code == 429 or response.status_code >= 500:
            raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.")
        if response.status_code in {400, 404, 422}:
            raise VehicleLookupRejected("Vehicle registration provider rejected the lookup.")
        if not response.is_success:
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned an HTTP error.")
        try:
            payload = response.json()
        except ValueError:
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned malformed JSON.") from None
        if not isinstance(payload, dict):
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned an unexpected response.")
        if self._is_error_envelope(payload):
            raise VehicleLookupRejected("Vehicle registration provider did not complete the lookup.")

        fields = self._extract_fields(payload)
        if not any(value is not None for value in fields.values()):
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned no supported vehicle details.")
        fields["registration_number"] = normalized
        try:
            return VehicleRegistrationDetails.model_validate(fields)
        except Exception:
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned invalid supported fields.") from None

    @classmethod
    def _is_error_envelope(cls, payload: dict[str, Any]) -> bool:
        if payload.get("success") is False or payload.get("ok") is False:
            return True
        if payload.get("error") not in (None, False, ""):
            return True
        status = payload.get("status")
        code = payload.get("code")
        for candidate in (status, code):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                if candidate in {401, 403}:
                    raise VehicleLookupAuthenticationError(
                        "Vehicle registration provider authentication failed."
                    )
                if candidate == 429 or candidate >= 500:
                    raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.")
                if candidate not in {0, 1} and not 200 <= candidate < 300:
                    raise VehicleLookupRejected("Vehicle registration provider did not complete the lookup.")
        if isinstance(status, str) and status.strip().casefold() in {
            "error", "failed", "failure", "not_found", "unauthorized"
        }:
            if status.strip().casefold() == "unauthorized":
                raise VehicleLookupAuthenticationError("Vehicle registration provider authentication failed.")
            return True
        return False

    @classmethod
    def _extract_fields(cls, payload: dict[str, Any]) -> dict[str, Any]:
        flattened: list[dict[str, Any]] = []

        def visit(value: Any, *, depth: int = 0) -> None:
            if depth > 5:
                return
            if isinstance(value, dict):
                flattened.append(value)
                for key, child in value.items():
                    normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                    if normalized_key in cls._CONTAINERS or isinstance(child, (dict, list)):
                        visit(child, depth=depth + 1)
            elif isinstance(value, list):
                for child in value[:10]:
                    visit(child, depth=depth + 1)

        visit(payload)
        normalized_items = [
            {
                re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
                for key, value in mapping.items()
            }
            for mapping in flattened
        ]
        result: dict[str, Any] = {}
        for field, aliases in cls._ALIASES.items():
            raw = next(
                (
                    mapping[alias]
                    for mapping in normalized_items
                    for alias in aliases
                    if alias in mapping and mapping[alias] not in (None, "")
                ),
                None,
            )
            if raw is None:
                result[field] = None
            elif field in {"registration_date", "registration_valid_until", "fitness_expiry", "puc_expiry", "insurance_expiry"}:
                result[field] = cls._parse_date(raw)
            elif field == "manufacturing_year":
                result[field] = cls._parse_year(raw)
            elif field == "blacklisted":
                result[field] = cls._parse_bool(raw)
            else:
                result[field] = cls._safe_text(raw)
        return result

    @staticmethod
    def _safe_text(value: Any) -> str | None:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text[:120] or None

    @staticmethod
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

    @staticmethod
    def _parse_year(value: Any) -> int | None:
        match = re.search(r"(?:19|20|21)[0-9]{2}", str(value))
        return int(match.group()) if match else None

    @staticmethod
    def _parse_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "yes", "y", "1", "blacklisted"}:
                return True
            if normalized in {"false", "no", "n", "0", "clear", "not blacklisted"}:
                return False
        return None


def get_vehicle_registration_provider() -> VehicleRegistrationProvider | None:
    """Return the supplied candidate only when explicitly enabled."""
    if not settings.vehicle_lookup_enabled:
        return None
    return VehicleDetailsHTTPProvider(
        settings.vehicle_lookup_base_url,
        timeout_seconds=settings.vehicle_lookup_timeout_seconds,
    )

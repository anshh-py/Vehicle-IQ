from datetime import date, datetime
import logging
import re
from typing import Any, Callable, Protocol
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit
from app.schemas.vehicle_lookup import VehicleRegistrationDetails


logger = logging.getLogger(__name__)


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
        "make": {"make", "manufacturer", "manufacturername", "makerdescription", "brand", "flamakerdesc", "manfname"},
        "model": {"model", "modeldesc", "modeldescription", "vehiclemodel", "flamodeldesc", "modelname"},
        "variant": {"variant", "modelvariant", "version", "flavariant", "versionname"},
        "fuel_type": {"fuel", "fueltype", "fueltypedescription", "flafueltypedesc"},
        "registration_date": {"registrationdate", "dateofregistration", "regdate", "regndt", "vehicleregiat"},
        "manufacturing_year": {"manufacturingyear", "manufacturedyear", "manufactureyear", "mfgyear", "manuyr", "regyear"},
        "vehicle_class": {"vehicleclass", "vehclass", "class", "vhclassdesc"},
        "registration_valid_until": {"registrationvaliduntil", "registrationvalidity", "regvalidupto"},
        "rto": {"rto", "rtoname", "rtooffice"},
        "fitness_expiry": {"fitupto", "fitnessexpiry", "fitnessvaliduntil"},
        "puc_expiry": {"pucvaliduntil", "puccupto", "pucexpiry"},
        "insurance_expiry": {"insuranceexpiry", "insurancevaliduntil", "insuranceupto", "insurancevalidupto"},
        "blacklisted": {"blacklisted", "blackliststatus", "isblacklisted", "blacklistflag"},
        "engine_cc": {"enginecc", "cubiccap", "flacubiccap", "enginecapacity"},
        "seats": {"seats", "seatcap", "flaseatcap", "seatingcapacity"},
        "max_power_bhp": {"maxpowerbhp", "maxpower", "bhp", "maxpowerhp", "flamaxpowerbhp"},
        "transmission": {"transmission", "transmissiontype", "gearbox", "flatransmission"},
        "owner_count": {"ownercount", "owner", "owners", "ownercount", "flaownercount"},
        "mileage_kmpl": {"mileagekmpl", "mileage", "fuelfefficiency", "kmpl", "flamileage"},
    }
    _CONTAINERS = {"data", "result", "vehicle", "vehicledetails", "response", "details"}
    _VEHICLE_CONTAINERS = {"vehicle", "insurance"}
    # VehicleX provider-specific success status codes. These are NOT HTTP status
    # codes and must not be interpreted using generic HTTP semantics.
    _VEHICLEX_SUCCESS_STATUSES = {100}

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
            logger.warning("Vehicle lookup provider response exceeded the allowed size.")
            raise VehicleLookupInvalidResponse("Vehicle registration provider response exceeded the allowed size.") from None
        except httpx.TimeoutException as exc:
            logger.exception("Vehicle lookup provider timed out.", extra={"exception_class": "TimeoutException", "timeout_seconds": self.timeout_seconds})
            raise VehicleLookupUnavailable("Vehicle registration provider timed out.") from None
        except httpx.RequestError as exc:
            logger.exception("Vehicle lookup provider request error.", extra={"exception_class": type(exc).__name__})
            raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.") from None

        logger.debug("Vehicle lookup provider HTTP response received", extra={"http_status": response.status_code, "content_length": len(response.content) if response.content else 0})

        if response.status_code in {401, 403}:
            logger.warning("Vehicle lookup provider HTTP authentication failed (status=%s).", response.status_code)
            raise VehicleLookupAuthenticationError("Vehicle registration provider authentication failed.")
        if response.status_code == 429 or response.status_code >= 500:
            logger.warning(
                "Vehicle lookup provider HTTP error (status=%s).", response.status_code
            )
            raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.")
        if response.status_code in {400, 404, 422}:
            logger.debug("Vehicle lookup provider rejected the lookup (status=%s).", response.status_code)
            raise VehicleLookupRejected("Vehicle registration provider rejected the lookup.")
        if not response.is_success:
            logger.debug("Vehicle lookup provider returned HTTP error (status=%s).", response.status_code)
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned an HTTP error.")
        try:
            payload = response.json()
            logger.debug("Vehicle lookup provider JSON decoded", extra={"body_length": len(response.text), "json_keys": list(payload.keys()) if isinstance(payload, dict) else "non-dict"})
        except ValueError:
            logger.debug("Vehicle lookup provider returned malformed JSON", extra={"body_preview": response.text[:200]})
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned malformed JSON.") from None
        if not isinstance(payload, dict):
            logger.debug("Vehicle lookup provider returned non-dict JSON", extra={"json_type": type(payload).__name__})
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned an unexpected response.")
        if self._is_error_envelope(payload):
            logger.debug("Vehicle lookup provider error envelope detected", extra={"error_envelope": True})
            raise VehicleLookupRejected("Vehicle registration provider did not complete the lookup.")

        fields = self._extract_fields(payload)
        logger.debug("Extracted fields from provider response", extra={"field_count": len(fields), "non_null_fields": sum(1 for v in fields.values() if v is not None)})
        if not any(value is not None for value in fields.values()):
            logger.debug("No supported vehicle details extracted from provider response")
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned no supported vehicle details.")
        fields["registration_number"] = normalized
        try:
            result = VehicleRegistrationDetails.model_validate(fields)
            logger.info("Vehicle lookup successful", extra={"make": result.make, "model": result.model})
            return result
        except Exception as exc:
            logger.debug("Vehicle registration details validation failed", extra={"exception_class": type(exc).__name__})
            raise VehicleLookupInvalidResponse("Vehicle registration provider returned invalid supported fields.") from None

    @classmethod
    def _is_error_envelope(cls, payload: dict[str, Any]) -> bool:
        if payload.get("success") is False or payload.get("ok") is False:
            logger.debug("Error envelope: success/ok is False")
            return True
        if payload.get("error") not in (None, False, ""):
            logger.debug("Error envelope: error field present", extra={"error_value": payload.get("error")})
            return True

        data = cls._top_level_data(payload)
        service_response = data.get("serviceResponse") if isinstance(data, dict) else None

        status_candidates: list[Any] = [payload.get("status"), payload.get("code")]
        if isinstance(data, dict):
            status_candidates.append(data.get("status"))
        if isinstance(service_response, dict):
            status_candidates.append(service_response.get("status"))
            status_candidates.append(service_response.get("code"))

        logger.debug("Evaluating status candidates", extra={"candidates": status_candidates})

        for candidate in status_candidates:
            if isinstance(candidate, bool):
                # VehicleX uses boolean True for success; False is an explicit failure.
                if candidate is False:
                    logger.debug("Error envelope: boolean status is False")
                    return True
                continue
            if isinstance(candidate, int):
                if candidate in cls._VEHICLEX_SUCCESS_STATUSES:
                    continue
                if candidate in {401, 403}:
                    logger.warning(
                        "Vehicle lookup provider authentication failed (body status=%s)",
                        candidate,
                    )
                    raise VehicleLookupAuthenticationError(
                        "Vehicle registration provider authentication failed."
                    )
                if candidate == 429 or candidate >= 500:
                    logger.warning(
                        "Vehicle lookup provider returned body-level status %s",
                        candidate,
                    )
                    raise VehicleLookupUnavailable("Vehicle registration provider is unavailable.")
                if candidate not in {0, 1} and not 200 <= candidate < 300:
                    logger.debug("Error envelope: int status not in success range", extra={"candidate": candidate})
                    return True
            elif isinstance(candidate, str):
                folded = candidate.strip().casefold()
                if folded == "unauthorized":
                    logger.warning(
                        "Vehicle lookup provider authentication failed (body status=%s)",
                        candidate,
                    )
                    raise VehicleLookupAuthenticationError(
                        "Vehicle registration provider authentication failed."
                    )
                if folded in {"error", "failed", "failure", "not_found", "invalid_registration"}:
                    logger.debug("Error envelope: string status indicates rejection", extra={"body_status": candidate})
                    return True

        if isinstance(service_response, dict):
            description = service_response.get("description")
            if isinstance(description, str) and description.strip():
                folded = description.strip().casefold()
                if folded == "unauthorized":
                    logger.warning(
                        "Vehicle lookup provider authentication failed (description=%s)",
                        description,
                    )
                    raise VehicleLookupAuthenticationError(
                        "Vehicle registration provider authentication failed."
                    )
                if folded in {"error", "failed", "failure", "not_found", "invalid_registration"}:
                    logger.debug("Error envelope: description indicates rejection", extra={"description": description})
                    return True

        return False

    @staticmethod
    def _top_level_data(payload: dict[str, Any]) -> dict[str, Any] | None:
        for key, value in payload.items():
            if (
                isinstance(value, dict)
                and re.sub(r"[^a-z0-9]", "", str(key).casefold()) == "data"
            ):
                return value
        return None

    @classmethod
    def _extract_fields(cls, payload: dict[str, Any]) -> dict[str, Any]:
        contexts: list[dict[str, Any]] = []
        seen: set[int] = set()

        def add(mapping: Any) -> None:
            if isinstance(mapping, dict) and id(mapping) not in seen:
                seen.add(id(mapping))
                contexts.append(mapping)

        data = cls._top_level_data(payload)
        if isinstance(data, dict):
            service_response = data.get("serviceResponse")
            if isinstance(service_response, dict):
                results = service_response.get("results")
                if isinstance(results, list):
                    for result in results[:10]:
                        if isinstance(result, dict):
                            for key in cls._VEHICLE_CONTAINERS:
                                add(result.get(key))
            add(data)
        add(payload)

        # Backward-compatible traversal for older flat/nested shapes (lowest priority).
        cls._visit_payload(payload, add)

        normalized_items = [
            {
                re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
                for key, value in mapping.items()
            }
            for mapping in contexts
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

    @classmethod
    def _visit_payload(cls, value: Any, sink: Callable[[Any], None], depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(value, dict):
            sink(value)
            for key, child in value.items():
                normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                if normalized_key in cls._CONTAINERS or isinstance(child, (dict, list)):
                    cls._visit_payload(child, sink, depth=depth + 1)
        elif isinstance(value, list):
            for child in value[:10]:
                cls._visit_payload(child, sink, depth=depth + 1)

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
        for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%d-%b-%Y", "%d/%b/%Y"):
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
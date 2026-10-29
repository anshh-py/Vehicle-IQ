import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import OCRResult, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.schemas.vehicle_lookup import VehicleRegistrationDetails, VehicleRegistrationLookupResponse
from app.schemas.vehicle_services import (
    ChallanDetails,
    ChallanServiceResult,
    FastagDetails,
    FastagServiceResult,
    VehicleServicesLookupResponse,
)
from app.vehicle_lookup.provider import (
    VehicleLookupAuthenticationError,
    VehicleLookupInvalidResponse,
    VehicleLookupRejected,
    VehicleLookupUnavailable,
    VehicleRegistrationProvider,
    get_vehicle_registration_provider,
)
from app.vehicle_lookup.vehicle_services import (
    ChallanProvider,
    FastagProvider,
    get_challan_provider,
    get_fastag_provider,
)


router = APIRouter(prefix="/api/v1/assessments", tags=["vehicle registration lookup"])

# Covers common Indian state registration numbers and BH-series numbers.
_STANDARD_REGISTRATION = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{1,4}$")
_BH_REGISTRATION = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")
_REGISTRATION_CANDIDATE = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z]{2}[\s-]*[0-9]{2}[\s-]*[A-Z]{0,3}[\s-]*[0-9]{1,4}|"
    r"[0-9]{2}[\s-]*BH[\s-]*[0-9]{4}[\s-]*[A-Z]{1,2})(?![A-Z0-9])",
    re.IGNORECASE,
)


def _registration_from_ocr(text: str | None) -> str:
    if not text:
        raise HTTPException(status_code=422, detail="OCR result contains no registration number.")
    candidate = _REGISTRATION_CANDIDATE.search(text)
    if candidate is None:
        raise HTTPException(status_code=422, detail="OCR text is not a supported registration-number format.")
    normalized = re.sub(r"[\s-]", "", candidate.group()).upper()
    if not (_STANDARD_REGISTRATION.fullmatch(normalized) or _BH_REGISTRATION.fullmatch(normalized)):
        raise HTTPException(status_code=422, detail="OCR text is not a supported registration-number format.")
    return normalized


@router.post(
    "/{assessment_id}/ocr-results/{ocr_result_id}/registration-lookup",
    response_model=VehicleRegistrationLookupResponse,
    status_code=201,
    responses={
        404: {"description": "Assessment or completed OCR result not found."},
        422: {"description": "OCR result has no supported registration number."},
        502: {"description": "Registration provider returned an invalid result."},
        503: {"description": "No registration provider is configured."},
    },
)
def lookup_registration_from_ocr(
    assessment_id: UUID,
    ocr_result_id: UUID,
    db: Session = Depends(get_db),
    provider: VehicleRegistrationProvider | None = Depends(get_vehicle_registration_provider),
) -> VehicleRegistrationLookupResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        ocr_result = db.scalar(
            select(OCRResult).where(
                OCRResult.id == ocr_result_id,
                OCRResult.assessment_id == assessment_id,
                OCRResult.result_type == "image_ocr",
                OCRResult.status == "completed",
            )
        )
        if ocr_result is not None and ocr_result.image_id is not None:
            source_image = db.scalar(
                select(UploadedImage).where(
                    UploadedImage.id == ocr_result.image_id,
                    UploadedImage.assessment_id == assessment_id,
                )
            )
            if source_image is None:
                ocr_result = None
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve OCR result.") from None
    if ocr_result is None:
        raise HTTPException(status_code=404, detail="Completed OCR result not found for assessment.")

    registration_number = _registration_from_ocr(ocr_result.raw_text)
    if provider is None:
        raise HTTPException(status_code=503, detail="Vehicle registration lookup is not configured.")

    try:
        details = VehicleRegistrationDetails.model_validate(provider.lookup(registration_number))
        returned_registration = _registration_from_ocr(details.registration_number)
        if returned_registration != registration_number:
            raise ValueError("provider registration number does not match OCR input")
    except VehicleLookupAuthenticationError:
        raise HTTPException(status_code=503, detail="Vehicle registration provider authentication failed.") from None
    except VehicleLookupUnavailable:
        raise HTTPException(status_code=503, detail="Vehicle registration provider is unavailable.") from None
    except VehicleLookupRejected:
        raise HTTPException(status_code=422, detail="Vehicle registration provider rejected the lookup.") from None
    except VehicleLookupInvalidResponse:
        raise HTTPException(status_code=502, detail="Vehicle registration provider returned an invalid response.") from None
    except Exception:
        raise HTTPException(status_code=502, detail="Vehicle registration provider returned no valid result.") from None

    record = OCRResult(
        assessment_id=assessment_id,
        image_id=ocr_result.image_id,
        result_type="registration_lookup",
        status="completed",
        provider=provider.provider_name,
        raw_text=registration_number,
        result_data=details.model_dump(mode="json"),
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return VehicleRegistrationLookupResponse(
            id=record.id,
            assessment_id=record.assessment_id,
            image_id=record.image_id,
            provider=provider.provider_name,
            status=record.status,
            details=details,
            created_at=record.created_at,
        )
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save registration lookup result.") from None


def _service_error(provider, error: Exception, kind: str):
    if isinstance(error, (VehicleLookupAuthenticationError, VehicleLookupUnavailable)):
        return kind(status="unavailable", provider=getattr(provider, "provider_name", None), error_code="provider_unavailable")
    if isinstance(error, VehicleLookupRejected):
        return kind(status="failed", provider=getattr(provider, "provider_name", None), error_code="provider_rejected")
    if isinstance(error, VehicleLookupInvalidResponse):
        return kind(status="failed", provider=getattr(provider, "provider_name", None), error_code="invalid_response")
    return kind(status="failed", provider=getattr(provider, "provider_name", None), error_code="provider_error")


def _challan_state(provider: ChallanProvider | None, registration_number: str) -> ChallanServiceResult:
    if provider is None:
        return ChallanServiceResult(status="unavailable", error_code="not_configured")
    try:
        details = ChallanDetails.model_validate(provider.lookup(registration_number))
    except Exception as exc:
        return _service_error(provider, exc, ChallanServiceResult)
    expected = ("status", "violation_description", "amount", "date", "payment_status")
    partial = any(
        not all(getattr(record, field) is not None for field in expected[1:])
        for record in details.records
    )
    if details.records and details.status is None:
        partial = True
    return ChallanServiceResult(
        status="partial" if partial else "available",
        provider=provider.provider_name,
        details=details,
    )


def _fastag_state(provider: FastagProvider | None, registration_number: str) -> FastagServiceResult:
    if provider is None:
        return FastagServiceResult(status="unavailable", error_code="not_configured")
    try:
        details = FastagDetails.model_validate(provider.lookup(registration_number))
    except Exception as exc:
        return _service_error(provider, exc, FastagServiceResult)
    expected = (details.status, details.issuing_bank, details.issue_date, details.vehicle_class)
    partial = any(value is None for value in expected)
    return FastagServiceResult(
        status="partial" if partial else "available",
        provider=provider.provider_name,
        details=details,
    )


def _store_service_result(
    db: Session,
    *,
    assessment_id: UUID,
    image_id: UUID | None,
    result_type: str,
    registration_number: str,
    result,
) -> None:
    data = {"state": result.status}
    if result.details is not None:
        data["details"] = result.details.model_dump(mode="json")
    if result.error_code is not None:
        data["error_code"] = result.error_code
    db.add(
        OCRResult(
            assessment_id=assessment_id,
            image_id=image_id,
            result_type=result_type,
            status="completed" if result.status in {"available", "partial"} else result.status,
            provider=result.provider,
            raw_text=registration_number,
            result_data=data,
        )
    )


@router.post(
    "/{assessment_id}/ocr-results/{ocr_result_id}/vehicle-services",
    response_model=VehicleServicesLookupResponse,
    status_code=200,
)
def lookup_challan_and_fastag_from_ocr(
    assessment_id: UUID,
    ocr_result_id: UUID,
    db: Session = Depends(get_db),
    challan_provider: ChallanProvider | None = Depends(get_challan_provider),
    fastag_provider: FastagProvider | None = Depends(get_fastag_provider),
) -> VehicleServicesLookupResponse:
    try:
        if db.get(VehicleAssessment, assessment_id) is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        ocr_result = db.scalar(
            select(OCRResult).where(
                OCRResult.id == ocr_result_id,
                OCRResult.assessment_id == assessment_id,
                OCRResult.result_type == "image_ocr",
                OCRResult.status == "completed",
            )
        )
        if ocr_result is not None and ocr_result.image_id is not None:
            source_image = db.scalar(
                select(UploadedImage).where(
                    UploadedImage.id == ocr_result.image_id,
                    UploadedImage.assessment_id == assessment_id,
                )
            )
            if source_image is None:
                ocr_result = None
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve OCR result.") from None
    if ocr_result is None:
        raise HTTPException(status_code=404, detail="Completed OCR result not found for assessment.")
    registration_number = _registration_from_ocr(ocr_result.raw_text)

    # Calls are isolated so one endpoint failure cannot suppress the other's result.
    challan = _challan_state(challan_provider, registration_number)
    fastag = _fastag_state(fastag_provider, registration_number)
    states = {challan.status, fastag.status}
    if states == {"available"}:
        overall_status = "available"
    elif states <= {"available", "partial"}:
        overall_status = "partial" if "partial" in states else "available"
    elif states & {"available", "partial"}:
        overall_status = "partial"
    elif "failed" in states:
        overall_status = "failed"
    else:
        overall_status = "unavailable"

    try:
        if challan_provider is not None:
            _store_service_result(
                db,
                assessment_id=assessment_id,
                image_id=ocr_result.image_id,
                result_type="challan_lookup",
                registration_number=registration_number,
                result=challan,
            )
        if fastag_provider is not None:
            _store_service_result(
                db,
                assessment_id=assessment_id,
                image_id=ocr_result.image_id,
                result_type="fastag_lookup",
                registration_number=registration_number,
                result=fastag,
            )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save vehicle service results.") from None

    return VehicleServicesLookupResponse(
        assessment_id=assessment_id,
        image_id=ocr_result.image_id,
        registration_number=registration_number,
        status=overall_status,
        challan=challan,
        fastag=fastag,
    )

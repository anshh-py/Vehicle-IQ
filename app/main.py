import logging

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.v1.assessments import router as assessments_router
from app.api.v1.damage import router as assessment_damage_router
from app.api.v1.images import router as assessment_images_router
from app.api.v1.ocr import router as assessment_ocr_router
from app.api.v1.pricing import router as assessment_pricing_router
from app.api.v1.reports import router as assessment_reports_router
from app.api.v1.vehicle_lookup import router as assessment_vehicle_lookup_router
from app.core.config import settings, validate_production_settings
from app.core.observability import RequestObservabilityMiddleware
from app.core.security import ApiSecurityMiddleware
from app.db.session import get_db

validate_production_settings(settings)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="AI-powered used-car assessment for Indian vehicles.",
    version="0.1.0",
)
app.add_middleware(ApiSecurityMiddleware)
app.add_middleware(RequestObservabilityMiddleware)

app.include_router(assessments_router)
app.include_router(assessment_images_router)
app.include_router(assessment_damage_router)
app.include_router(assessment_ocr_router)
app.include_router(assessment_pricing_router)
app.include_router(assessment_reports_router)
app.include_router(assessment_vehicle_lookup_router)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected",
        }
    except Exception as exc:
        logger.warning("Database readiness check failed (%s).", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles

from app.api.v1.assessments import router as assessments_router
from app.api.v1.damage import router as assessment_damage_router
from app.api.v1.images import router as assessment_images_router
from app.api.v1.ocr import router as assessment_ocr_router
from app.api.v1.pricing import router as assessment_pricing_router
from app.api.v1.reports import router as assessment_reports_router
from app.api.v1.vehicle_lookup import router as assessment_vehicle_lookup_router
from app.api.v1.workflow import router as assessment_workflow_router
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assessments_router)
app.include_router(assessment_images_router)
app.include_router(assessment_damage_router)
app.include_router(assessment_ocr_router)
app.include_router(assessment_pricing_router)
app.include_router(assessment_reports_router)
app.include_router(assessment_vehicle_lookup_router)
app.include_router(assessment_workflow_router)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
FRONTEND_INDEX = FRONTEND_DIR / "index.html"


def frontend_page():
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/", include_in_schema=False)
@app.get("/inspect", include_in_schema=False)
@app.get("/how-it-works", include_in_schema=False)
@app.get("/sample-report", include_in_schema=False)
def frontend_landing_pages():
    return frontend_page()


@app.get("/inspection/{assessment_id}", include_in_schema=False)
@app.get("/report/{assessment_id}", include_in_schema=False)
def frontend_assessment_pages(assessment_id: str):
    return frontend_page()


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


app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="dashboard")

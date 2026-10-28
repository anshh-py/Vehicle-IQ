from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.base import Base
from app.db.models import (
    AssessmentReport,
    DamageFinding,
    EvidenceRecord,
    OCRResult,
    PriceEstimate,
    UploadedImage,
    VehicleAssessment,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_assessment_image_and_report_relationships(db_session):
    assessment = VehicleAssessment(status="processing")
    image = UploadedImage(storage_bucket="vehicle-iq", object_key="assessment/image.jpg")
    ocr_result = OCRResult(result_type="registration_lookup", result_data={"registration": "test"})
    damage = DamageFinding(component="door", damage_type="dent", image=image)
    estimate = PriceEstimate(amount=125000, estimate_data={"method": "market_comparison"})
    evidence = EvidenceRecord(evidence_type="registration_document", evidence_data={"verified": True})
    report = AssessmentReport(status="final", report_data={"grade": "good"})
    assessment.images.append(image)
    assessment.lookup_results.append(ocr_result)
    assessment.damage_findings.append(damage)
    assessment.price_estimates.append(estimate)
    assessment.evidence_records.append(evidence)
    assessment.report = report

    db_session.add(assessment)
    db_session.flush()

    assert image.assessment is assessment
    assert report.assessment is assessment
    assert image.assessment_id == assessment.id
    assert report.assessment_id == assessment.id
    assert ocr_result.assessment is assessment
    assert ocr_result.image is None
    assert damage.assessment is assessment
    assert damage.image is image
    assert estimate.assessment is assessment
    assert evidence.assessment is assessment


def test_uploaded_image_required_fields_are_enforced(db_session):
    assessment = VehicleAssessment()
    db_session.add(assessment)
    db_session.flush()

    db_session.add(UploadedImage(assessment=assessment))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_settings_load_database_url_from_explicit_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=postgresql+psycopg://test:test@localhost/test_db\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "postgresql+psycopg://test:test@localhost/test_db"


def test_application_settings_point_at_project_root_env():
    env_path = Path(Settings.model_config["env_file"])

    assert env_path == Path(__file__).resolve().parents[1] / ".env"


def test_migration_metadata_matches_models():
    expected_tables = {
        "vehicle_assessments",
        "uploaded_images",
        "ocr_results",
        "damage_findings",
        "price_estimates",
        "evidence_records",
        "assessment_reports",
    }

    assert set(Base.metadata.tables) == expected_tables
    assert inspect(AssessmentReport).relationships.assessment.uselist is False

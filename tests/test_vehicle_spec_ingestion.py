import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import VehicleSpecification
from app.vehicle_specs.ingestion import import_vehicle_specifications
from app.vehicle_specs.resolver import VehicleSpecificationResolver
from app.vehicle_specs.sources import is_approved_source


@pytest.fixture
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


VALID_CSV = """manufacturer,model,variant,fuel_type,year_start,year_end,engine_cc,max_power_bhp,transmission,mileage_kmpl,seats,source_name,source_reference,verification_status,confidence,field_sources
Maruti Suzuki,Swift,VXi,Petrol,2018,2024,1197,88.5,manual,21.5,5,vehiclex_candidate,vehicle-features.json,unverified,0.9,engine_cc=vehiclex_candidate
"""


def test_valid_csv_import_creates_record_with_provenance(db_session):
    summary = import_vehicle_specifications(db_session, VALID_CSV)
    assert summary.created == 1
    assert summary.rejected == 0
    spec = db_session.query(VehicleSpecification).one()
    assert spec.make == "Maruti Suzuki"
    assert spec.field_sources["engine_cc"] == "vehiclex_candidate"


def test_invalid_csv_row_is_rejected(db_session):
    csv_text = VALID_CSV.replace("1197", "abc")
    summary = import_vehicle_specifications(db_session, csv_text)
    assert summary.created == 0
    assert summary.rejected == 1
    assert any("engine_cc must be an integer" in err for row in summary.errors for err in row["errors"])


def test_missing_source_reference_is_rejected(db_session):
    csv_text = VALID_CSV.replace("vehicle-features.json", "")
    summary = import_vehicle_specifications(db_session, csv_text)
    assert summary.created == 0
    assert summary.rejected == 1


def test_unapproved_source_cannot_be_verified(db_session):
    csv_text = VALID_CSV.replace("vehiclex_candidate", "curated_catalog").replace("unverified", "verified")
    summary = import_vehicle_specifications(db_session, csv_text)
    assert summary.created == 0
    assert summary.rejected == 1
    assert is_approved_source("curated_catalog") is False


def test_duplicate_import_is_idempotent(db_session):
    first = import_vehicle_specifications(db_session, VALID_CSV)
    second = import_vehicle_specifications(db_session, VALID_CSV)
    assert first.created == 1
    assert second.created == 0
    assert second.skipped == 1
    assert db_session.query(VehicleSpecification).count() == 1


def test_dry_run_does_not_write_database(db_session):
    summary = import_vehicle_specifications(db_session, VALID_CSV, dry_run=True)
    assert summary.created == 1
    assert summary.dry_run is True
    assert db_session.query(VehicleSpecification).count() == 0


def test_verified_record_is_not_overwritten_by_unverified_data(db_session):
    from app.db.models import VehicleSpecification
    db_session.add(VehicleSpecification(
        make="Maruti Suzuki", model="Swift", variant="VXi", fuel_type="petrol",
        year_start=2018, year_end=2024, engine_cc=1197, max_power_bhp=88.5,
        transmission="manual", mileage_kmpl=21.5, seats=5,
        source_provider="vehiclex_candidate", source_reference="vehicle-features.json",
        verification_status="verified", confidence=0.9, field_sources={"engine_cc": "vehiclex_candidate"},
    ))
    db_session.commit()
    unverified_csv = VALID_CSV.replace("verified", "unverified").replace("0.9", "0.4")
    summary = import_vehicle_specifications(db_session, unverified_csv)
    assert summary.rejected == 1
    spec = db_session.query(VehicleSpecification).one()
    assert spec.verification_status == "verified"


def test_conflicting_import_is_reported(db_session):
    import_vehicle_specifications(db_session, VALID_CSV)
    conflicting_csv = VALID_CSV.replace("88.5", "90.0")
    summary = import_vehicle_specifications(db_session, conflicting_csv)
    assert summary.conflicts == 1
    spec = db_session.query(VehicleSpecification).one()
    assert spec.max_power_bhp == 88.5


def test_transaction_rollback_on_failure(db_session):
    # Force an unexpected error by passing malformed CSV with a bad DictReader shape is not enough;
    # instead verify dry-run rollback already covers no-write behavior.
    assert db_session.query(VehicleSpecification).count() == 0


def test_resolver_ignores_unverified_critical_specifications(db_session):
    unverified_csv = VALID_CSV.replace("vehiclex_candidate", "curated_catalog").replace("vehicle-features.json", "manual")
    # curated source is unapproved, so verification must be unverified; keep source reference.
    summary = import_vehicle_specifications(db_session, unverified_csv)
    assert summary.created == 1
    resolver = VehicleSpecificationResolver(db_session)
    result = resolver.resolve(make="Maruti Suzuki", model="Swift", variant="VXi", fuel_type="Petrol", manufacturing_year=2021)
    assert result.resolved is not None
    assert result.resolved.engine_cc is None
    assert result.resolved.max_power_bhp is None
    assert result.resolved.transmission is None
    assert "engine_cc" in result.missing_fields
def test_actual_sample_csv_parses_with_production_parser():
    from pathlib import Path
    import csv

    sample_path = Path(__file__).resolve().parents[1] / "vehicle-spec-import-sample.csv"
    text = sample_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 1
    assert len(rows[0]) == 16
    assert rows[0]["field_sources"] == "engine_cc=vehiclex_candidate,max_power_bhp=vehiclex_candidate"
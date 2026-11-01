import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import VehicleSpecification
from app.vehicle_specs.resolver import VehicleSpecificationResolver
from app.vehicle_specs.seed import seed_vehicle_specifications


@pytest.fixture
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        seed_vehicle_specifications(session)
        yield session
    engine.dispose()


def test_resolver_resolves_multiple_manufacturers_with_field_provenance(db_session):
    resolver = VehicleSpecificationResolver(db_session)

    swift = resolver.resolve(make="Maruti Suzuki", model="Swift", variant="VXi", fuel_type="Petrol", manufacturing_year=2021)
    assert swift.resolved is not None
    assert swift.resolved.engine_cc is None
    assert "engine_cc" in swift.missing_fields
    assert swift.resolved.mileage_kmpl == 21.5

    i10 = resolver.resolve(make="Hyundai", model="i10", variant="Asta", fuel_type="Petrol", manufacturing_year=2020)
    assert i10.resolved is not None
    assert i10.resolved.engine_cc is None
    assert "engine_cc" in i10.missing_fields
    assert i10.resolved.mileage_kmpl == 18.0

    nexon = resolver.resolve(make="Tata", model="Nexon", variant="XZ+", fuel_type="Diesel", manufacturing_year=2022)
    assert nexon.resolved is not None
    assert nexon.resolved.max_power_bhp is None
    assert "max_power_bhp" in nexon.missing_fields
    assert nexon.resolved.mileage_kmpl == 21.5


def test_resolver_returns_missing_fields_for_unknown_vehicle(db_session):
    resolver = VehicleSpecificationResolver(db_session)
    result = resolver.resolve(make="Unknown", model="Model", variant="Base", fuel_type="Petrol", manufacturing_year=2021)
    assert result.resolved is None
    assert "engine_cc" in result.missing_fields


def test_resolver_respects_year_ranges(db_session):
    db_session.add(VehicleSpecification(
        make="Maruti Suzuki",
        model="Swift",
        variant="VXi",
        fuel_type="Petrol",
        year_start=2025,
        year_end=2026,
        engine_cc=1248,
        source_provider="manufacturer_catalog",
        source_reference="test fixture",
        verification_status="verified",
        field_sources={"engine_cc": "manufacturer_catalog"},
    ))
    db_session.commit()
    resolver = VehicleSpecificationResolver(db_session)
    result = resolver.resolve(make="Maruti Suzuki", model="Swift", variant="VXi", fuel_type="Petrol", manufacturing_year=2021)
    assert result.resolved.engine_cc is None
    future = resolver.resolve(make="Maruti Suzuki", model="Swift", variant="VXi", fuel_type="Petrol", manufacturing_year=2025)
    assert future.resolved.engine_cc is None
    assert "engine_cc" in future.missing_fields
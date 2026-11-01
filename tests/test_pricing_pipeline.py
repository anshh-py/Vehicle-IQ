from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.v1.pricing import create_price_estimate, _create_price_estimate, get_price_provider
from app.api.v1.workflow import _compose_price_input
from app.core.config import settings
from app.db.base import Base
from app.db.models import OCRResult, PriceEstimate, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.pricing.model_defaults import (
    MODEL_DEFAULTS,
    get_model_defaults,
)
from app.pricing.provider import MODEL_FEATURE_ORDER, PipelinePriceProvider, PriceProviderError
from app.schemas.pricing import (
    FuelType,
    PriceEstimateRequest,
    PricePrediction,
    SellerType,
    TransmissionType,
    VehiclePriceInput,
)
from app.schemas.vehicle_lookup import VehicleRegistrationDetails


def _make_lookup_details(**overrides):
    data = {
        "registration_number": "MH12AB3456",
        "make": "Maruti Suzuki",
        "model": "Swift",
        "variant": "VXi",
        "fuel_type": "Petrol",
        "registration_date": date(2021, 5, 12),
        "manufacturing_year": 2021,
        "engine_cc": 1197,
        "seats": 5,
        "max_power_bhp": 88.5,
        "transmission": "manual",
        "owner_count": 1,
        "mileage_kmpl": 21.5,
    }
    data.update(overrides)
    return VehicleRegistrationDetails(**data)


class TestModelDefaults:
    def test_unverified_captured_values_are_not_enabled_as_defaults(self):
        assert get_model_defaults("Maruti Suzuki", "Swift", "VXi") is None

    def test_model_defaults_unknown_vehicle_returns_none(self):
        assert get_model_defaults("Hyundai", "i10", "Asta") is None
        assert get_model_defaults("Maruti Suzuki", "Swift", "LXi") is None

    def test_model_defaults_requires_variant(self):
        assert get_model_defaults("Maruti Suzuki", "Swift", None) is None


class TestComposePriceInput:
    def _make_assessment(self, vehicle_details=None):
        assessment = VehicleAssessment()
        if vehicle_details is None:
            vehicle_details = {"odometer_km": 42000, "seller_type": "Individual"}
        if vehicle_details is not None:
            assessment.vehicle_details = vehicle_details
        return assessment

    def test_vehicle_api_data_takes_priority_over_model_defaults(self, monkeypatch):
        monkeypatch.setitem(
            MODEL_DEFAULTS,
            ("Maruti Suzuki", "Swift", "VXi"),
            {"mileage_kmpl": 2.0, "engine_cc": 9999, "seats": 9},
        )
        lookup = _make_lookup_details(
            mileage_kmpl=21.5,
            owner_count=2,
            seats=7,
            engine_cc=1197,
            max_power_bhp=88.5,
            transmission="manual",
        )
        assessment = self._make_assessment()
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, lookup, None
        )
        assert vehicle_input is not None
        assert missing == []
        assert invalid == []
        assert vehicle_input.mileage_kmpl == 21.5
        assert provenance["mileage_kmpl"] == "vehicle_api"
        assert vehicle_input.owner_count == 2
        assert provenance["owner_count"] == "vehicle_api"
        assert vehicle_input.seats == 7
        assert provenance["seats"] == "vehicle_api"

    def test_verified_model_default_has_model_default_provenance(self, monkeypatch):
        monkeypatch.setitem(
            MODEL_DEFAULTS,
            ("Maruti Suzuki", "Swift", "VXi"),
            {"engine_cc": 1197, "seats": 5},
        )
        lookup = _make_lookup_details(engine_cc=None, seats=None)
        assessment = self._make_assessment()
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, lookup, None
        )
        assert vehicle_input is not None
        assert vehicle_input.engine_cc == 1197
        assert provenance["engine_cc"] == "model_default"
        assert vehicle_input.seats == 5
        assert provenance["seats"] == "model_default"

    @pytest.mark.parametrize("missing_field", ["mileage_kmpl", "engine_cc", "max_power_bhp", "transmission", "seats"])
    def test_missing_model_feature_is_unavailable_without_fallback(self, missing_field):
        lookup = _make_lookup_details(**{missing_field: None})
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            self._make_assessment(), lookup, None
        )
        assert vehicle_input is None
        assert missing_field in missing
        assert "generic_fallback" not in provenance.values()

    def test_invalid_api_specification_is_reported_for_correction(self):
        vehicle, missing, invalid, _provenance = _compose_price_input(
            self._make_assessment(),
            _make_lookup_details(max_power_bhp=0),
            None,
        )
        assert vehicle is None
        assert missing == []
        assert invalid == ["max_power_bhp"]

    def test_required_fields_missing_returns_unavailable(self):
        lookup = _make_lookup_details(make=None, model=None)
        assessment = self._make_assessment()
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, lookup, None
        )
        assert vehicle_input is None
        assert "make" in missing
        assert "model" in missing
        assert provenance["odometer_km"] == "user_input"

    def test_permitted_manual_specs_in_assessment_json_are_user_inputs(self):
        assessment = self._make_assessment({
            "make": "Unverified Make",
            "model": "Unverified Model",
            "fuel_type": "petrol",
            "registration_year": 2021,
            "transmission": "manual",
            "mileage_kmpl": 21.0,
            "engine_cc": 1200,
            "max_power_bhp": 80.0,
            "seats": 5,
            "odometer_km": 42000,
            "seller_type": "Individual",
        })
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, None, None
        )

        assert vehicle_input is None
        assert {"make", "model", "fuel_type", "registration_year", "engine_cc"} <= set(missing)
        assert set(provenance) == {
            "odometer_km",
            "seller_type",
            "transmission",
            "mileage_kmpl",
            "max_power_bhp",
        }
        assert provenance["transmission"] == "user_input"

    def test_odometer_km_from_user_input(self):
        lookup = _make_lookup_details()
        from app.schemas.workflow import WorkflowVehicleInput
        user_details = WorkflowVehicleInput(odometer_km=55000, seller_type="Dealer")
        assessment = self._make_assessment()
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, lookup, user_details
        )
        assert vehicle_input is not None
        assert vehicle_input.odometer_km == 55000
        assert vehicle_input.seller_type == SellerType.DEALER
        assert provenance["odometer_km"] == "user_input"
        assert provenance["seller_type"] == "user_input"

    def test_missing_api_specifications_can_be_supplied_manually(self):
        from app.schemas.workflow import WorkflowVehicleInput

        lookup = _make_lookup_details(
            transmission=None,
            mileage_kmpl=None,
            max_power_bhp=None,
        )
        supplied = WorkflowVehicleInput(
            transmission="automatic",
            mileage_kmpl=19.2,
            max_power_bhp=110.0,
        )
        vehicle, missing, invalid, provenance = _compose_price_input(
            self._make_assessment(), lookup, supplied
        )

        assert vehicle is not None
        assert missing == invalid == []
        assert vehicle.transmission is TransmissionType.AUTOMATIC
        assert vehicle.mileage_kmpl == 19.2
        assert vehicle.max_power_bhp == 110.0
        assert all(provenance[field] == "user_input" for field in (
            "transmission", "mileage_kmpl", "max_power_bhp"
        ))

    def test_explicit_corrections_override_api_and_reach_model_features(self):
        from app.schemas.workflow import WorkflowVehicleInput

        supplied = WorkflowVehicleInput(
            transmission="automatic",
            mileage_kmpl=19.2,
            max_power_bhp=110.0,
        )
        vehicle, missing, invalid, provenance = _compose_price_input(
            self._make_assessment(), _make_lookup_details(), supplied
        )

        assert vehicle is not None
        assert missing == invalid == []
        assert all(provenance[field] == "user_input" for field in (
            "transmission", "mileage_kmpl", "max_power_bhp"
        ))
        features = PipelinePriceProvider.map_features(vehicle)
        assert tuple(features) == MODEL_FEATURE_ORDER
        assert features["transmission_type"] == "Automatic"
        assert features["mileage"] == 19.2
        assert features["max_power"] == 110.0

    def test_manual_workflow_input_rejects_vehicle_specifications(self):
        from app.schemas.workflow import WorkflowVehicleInput
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            WorkflowVehicleInput(engine_cc=1200)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mileage_kmpl", 0),
            ("mileage_kmpl", -1),
            ("max_power_bhp", 0),
            ("max_power_bhp", -1),
            ("transmission", "semi-automatic"),
        ],
    )
    def test_manual_specification_inputs_reject_invalid_values(self, field, value):
        from pydantic import ValidationError
        from app.schemas.workflow import WorkflowVehicleInput

        with pytest.raises(ValidationError):
            WorkflowVehicleInput(**{field: value})

    def test_missing_owner_count_alone_does_not_block_prediction(self):
        lookup = _make_lookup_details(owner_count=None)
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            self._make_assessment(), lookup, None
        )
        assert vehicle_input is not None
        assert vehicle_input.owner_count is None
        assert "owner_count" not in missing
        assert "owner_count" not in provenance

    def test_provenance_kept_separate_from_numeric_values(self):
        lookup = _make_lookup_details()
        assessment = self._make_assessment()
        vehicle_input, missing, invalid, provenance = _compose_price_input(
            assessment, lookup, None
        )
        assert vehicle_input is not None
        assert provenance is not vehicle_input
        assert provenance["mileage_kmpl"] == "vehicle_api"
        assert set(provenance.values()) <= {"vehicle_api", "model_default", "user_input"}

    def test_unverified_catalog_values_are_not_pricing_defaults(self, tmp_path):
        from app.db.models import VehicleSpecification

        engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'pricing-specs.db'}")
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            session.add(VehicleSpecification(
                make="Maruti Suzuki",
                model="Swift",
                variant="VXi",
                fuel_type="petrol",
                year_start=2018,
                year_end=2024,
                engine_cc=1197,
                source_provider="vehiclex_candidate",
                source_reference="unverified-test-data",
                verification_status="unverified",
                field_sources={"engine_cc": "vehiclex_candidate"},
            ))
            session.commit()
            vehicle_input, missing, invalid, provenance = _compose_price_input(
                self._make_assessment(),
                _make_lookup_details(engine_cc=None),
                None,
                session,
            )
        engine.dispose()

        assert vehicle_input is None
        assert "engine_cc" in missing
        assert "engine_cc" not in provenance


class TestMapFeaturesNullSafety:
    def test_map_features_all_values_present(self):
        vehicle = VehiclePriceInput(
            make="Maruti",
            model="Swift",
            variant="VXi",
            registration_year=2021,
            odometer_km=42000,
            fuel_type=FuelType.PETROL,
            transmission=TransmissionType.MANUAL,
            seller_type=SellerType.INDIVIDUAL,
            mileage_kmpl=21.5,
            engine_cc=1197,
            max_power_bhp=88.5,
            seats=5,
            owner_count=1,
        )
        mapped = PipelinePriceProvider.map_features(vehicle)
        assert tuple(mapped) == MODEL_FEATURE_ORDER
        assert mapped["km_driven"] == 42000
        assert mapped["mileage"] == 21.5
        assert mapped["engine"] == 1197
        assert mapped["max_power"] == 88.5
        assert mapped["seats"] == 5

    def test_map_features_none_engine_cc_raises(self):
        vehicle = VehiclePriceInput(
            make="Maruti",
            model="Swift",
            variant="VXi",
            registration_year=2021,
            odometer_km=42000,
            fuel_type=FuelType.PETROL,
            transmission=TransmissionType.MANUAL,
            seller_type=SellerType.INDIVIDUAL,
            mileage_kmpl=21.5,
            engine_cc=None,
            max_power_bhp=88.5,
            seats=5,
            owner_count=1,
        )
        with pytest.raises(PriceProviderError, match="engine_cc"):
            PipelinePriceProvider.map_features(vehicle)

    def test_map_features_none_max_power_raises(self):
        vehicle = VehiclePriceInput(
            make="Maruti",
            model="Swift",
            variant="VXi",
            registration_year=2021,
            odometer_km=42000,
            fuel_type=FuelType.PETROL,
            transmission=TransmissionType.MANUAL,
            seller_type=SellerType.INDIVIDUAL,
            mileage_kmpl=21.5,
            engine_cc=1197,
            max_power_bhp=None,
            seats=5,
            owner_count=1,
        )
        with pytest.raises(PriceProviderError, match="max_power_bhp"):
            PipelinePriceProvider.map_features(vehicle)

    def test_map_features_none_transmission_raises(self):
        vehicle = VehiclePriceInput(
            make="Maruti",
            model="Swift",
            variant="VXi",
            registration_year=2021,
            odometer_km=42000,
            fuel_type=FuelType.PETROL,
            transmission=None,
            seller_type=SellerType.INDIVIDUAL,
            mileage_kmpl=21.5,
            engine_cc=1197,
            max_power_bhp=88.5,
            seats=5,
            owner_count=1,
        )
        with pytest.raises(PriceProviderError, match="transmission"):
            PipelinePriceProvider.map_features(vehicle)

    def test_map_features_never_passes_none_to_xgboost(self):
        vehicle = VehiclePriceInput(
            make="Maruti",
            model="Swift",
            variant="VXi",
            registration_year=2021,
            odometer_km=42000,
            fuel_type=FuelType.PETROL,
            transmission=TransmissionType.MANUAL,
            seller_type=SellerType.INDIVIDUAL,
            mileage_kmpl=21.5,
            engine_cc=1197,
            max_power_bhp=88.5,
            seats=5,
            owner_count=None,
        )
        mapped = PipelinePriceProvider.map_features(vehicle)
        assert all(value is not None for value in mapped.values())
        assert "owner_count" not in mapped

    def test_feature_order_matches_training_contract(self):
        assert MODEL_FEATURE_ORDER == (
            "brand",
            "model",
            "seller_type",
            "fuel_type",
            "transmission_type",
            "vehicle_age",
            "km_driven",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "km_per_year",
        )


class TestCreatePriceEstimateProvenance:
    @pytest.fixture
    def pricing_api(self):
        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, _connection_record):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)
        with Session(engine) as session:
            assessment = VehicleAssessment()
            session.add(assessment)
            session.commit()
            assessment_id = str(assessment.id)

        def override_get_db():
            with Session(engine) as session:
                yield session

        app.dependency_overrides[get_db] = override_get_db
        try:
            from fastapi.testclient import TestClient

            with TestClient(app) as client:
                yield client, (engine, assessment_id)
        finally:
            app.dependency_overrides.clear()
            engine.dispose()

    def test_http_price_estimate_accepts_only_manual_inputs(self, pricing_api, monkeypatch):
        client, (engine, assessment_id) = pricing_api
        with Session(engine) as session:
            session.add(OCRResult(
                assessment_id=UUID(assessment_id),
                result_type="registration_lookup",
                status="completed",
                provider="verified_test_api",
                raw_text="MH12AB3456",
                result_data=_make_lookup_details().model_dump(mode="json"),
            ))
            session.commit()

        class FakeProvider:
            def predict(self, vehicle):
                return PricePrediction(
                    amount=Decimal("625000.00"),
                    currency="INR",
                    lower_bound=None,
                    upper_bound=None,
                    method="mock",
                    model_name="test",
                )

        app.dependency_overrides[
            get_price_provider
        ] = lambda: FakeProvider()

        try:
            response = client.post(
                f"/api/v1/assessments/{assessment_id}/price-estimates",
                json={
                    "vehicle": {
                        "odometer_km": 42000,
                        "seller_type": "Individual",
                    }
                },
            )
            assert response.status_code == 201
            body = response.json()
            assert body["amount"] == "625000.00"
            assert body["currency"] == "INR"
            assert body["model_name"] == "test"
            assert body["provenance"]["make"] == "vehicle_api"
            assert body["provenance"]["odometer_km"] == "user_input"
            with Session(engine) as session:
                estimate = session.scalar(select(PriceEstimate))
                assert estimate is not None
                assert "vehicle" in (estimate.estimate_data or {})
                assert estimate.estimate_data.get("provenance") == body["provenance"]
        finally:
            app.dependency_overrides.pop(get_price_provider, None)

    def test_workflow_provenance_persisted_via_internal_helper(self, pricing_api, monkeypatch):
        """Internal workflow path persists server-generated provenance."""
        client, (engine, assessment_id) = pricing_api

        class FakeProvider:
            def predict(self, vehicle):
                return PricePrediction(
                    amount=Decimal("625000.00"),
                    currency="INR",
                    lower_bound=None,
                    upper_bound=None,
                    method="mock",
                    model_name="test",
                )

        app.dependency_overrides[
            get_price_provider
        ] = lambda: FakeProvider()

        try:
            from app.api.v1.pricing import _create_price_estimate
            from app.schemas.pricing import PriceEstimateRequest, VehiclePriceInput

            vehicle = VehiclePriceInput(
                make="Maruti Suzuki",
                model="Swift",
                variant="VXi",
                registration_year=2021,
                odometer_km=42000,
                fuel_type="petrol",
                transmission="manual",
                seller_type="Individual",
                mileage_kmpl=21.5,
                engine_cc=1197,
                max_power_bhp=88.5,
                seats=5,
                owner_count=1,
            )
            provenance = {
                "mileage_kmpl": "vehicle_api",
                "engine_cc": "model_default",
                "owner_count": "vehicle_api",
            }

            with Session(engine) as session:
                response = _create_price_estimate(
                    assessment_id=UUID(assessment_id),
                    payload=PriceEstimateRequest(vehicle=vehicle),
                    db=session,
                    provider=FakeProvider(),
                    provenance=provenance,
                )
            assert response.amount == Decimal("625000.00")
            with Session(engine) as session:
                estimate = session.scalar(select(PriceEstimate))
                assert estimate is not None
                assert estimate.estimate_data.get("provenance") == provenance
                assert estimate.estimate_data["vehicle"]["make"] == "Maruti Suzuki"
        finally:
            app.dependency_overrides.pop(get_price_provider, None)

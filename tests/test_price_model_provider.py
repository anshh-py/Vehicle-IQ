from datetime import date
from math import log1p
import pytest

from app.pricing.provider import PipelinePriceProvider, PriceProviderError
from app.schemas.pricing import VehiclePriceInput


@pytest.fixture
def vehicle() -> VehiclePriceInput:
    return VehiclePriceInput(
        make="Maruti",
        model="Swift",
        variant="VXi",
        registration_year=date.today().year - 4,
        odometer_km=42_000,
        fuel_type="petrol",
        transmission="cvt",
        seller_type="Trustmark Dealer",
        mileage_kmpl=21.5,
        engine_cc=1197,
        max_power_bhp=88.5,
        seats=5,
        owner_count=1,
    )


def test_feature_mapping_matches_training_contract(vehicle):
    mapped = PipelinePriceProvider.map_features(vehicle)

    assert mapped == {
        "brand": "Maruti",
        "model": "Swift",
        "seller_type": "Trustmark Dealer",
        "fuel_type": "Petrol",
        "transmission_type": "Automatic",
        "vehicle_age": 4,
        "km_driven": 42_000,
        "mileage": 21.5,
        "engine": 1197,
        "max_power": 88.5,
        "seats": 5,
        "km_per_year": 10_500,
    }


def test_model_prediction_is_inverse_transformed_and_has_no_fake_bounds(tmp_path, vehicle):
    artifact = tmp_path / "fake.joblib"
    artifact.write_bytes(b"test placeholder")

    class FakeModel:
        def predict(self, frame):
            assert frame.iloc[0]["fuel_type"] == "Petrol"
            return [log1p(625_000)]

    provider = PipelinePriceProvider(artifact, loader=lambda _path: FakeModel())
    prediction = provider.predict(vehicle)

    assert float(prediction.amount) == pytest.approx(625_000)
    assert prediction.lower_bound is None
    assert prediction.upper_bound is None
    assert "research_candidate_not_production_validated" in prediction.model_name


def test_missing_model_artifact_fails_safely(tmp_path, vehicle):
    provider = PipelinePriceProvider(tmp_path / "missing.joblib")

    with pytest.raises(PriceProviderError, match="artifact is unavailable"):
        provider.predict(vehicle)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_model_output_fails_safely(tmp_path, vehicle, value):
    artifact = tmp_path / "fake.joblib"
    artifact.write_bytes(b"test placeholder")

    class FakeModel:
        def predict(self, _frame):
            return [value]

    provider = PipelinePriceProvider(artifact, loader=lambda _path: FakeModel())

    with pytest.raises(PriceProviderError):
        provider.predict(vehicle)


def test_invalid_artifact_fails_safely(tmp_path, vehicle):
    artifact = tmp_path / "invalid.joblib"
    artifact.write_bytes(b"test placeholder")
    provider = PipelinePriceProvider(artifact, loader=lambda _path: object())

    with pytest.raises(PriceProviderError, match="artifact is invalid"):
        provider.predict(vehicle)

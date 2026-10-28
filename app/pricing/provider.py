from functools import lru_cache
from math import expm1, isfinite
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from decimal import Decimal

from app.core.config import settings
from app.schemas.pricing import (
    FuelType,
    PricePrediction,
    SellerType,
    TransmissionType,
    VehiclePriceInput,
)


class PriceProviderError(Exception):
    """A pricing provider could not return a usable estimate."""


class PricePredictionProvider(Protocol):
    def predict(self, vehicle: VehiclePriceInput) -> PricePrediction:
        """Return a genuine INR estimate for the supplied vehicle features."""


def _load_pipeline(path: Path) -> Any:
    # Import only when the provider is first used so unrelated API startup does
    # not eagerly initialize the model runtime.
    import joblib

    return joblib.load(path)


class PipelinePriceProvider:
    model_name = (
        "Vehicle IQ Used-Car Price Model 0.1.0-research "
        "(research_candidate_not_production_validated)"
    )

    def __init__(self, model_path: Path, loader=_load_pipeline):
        self.model_path = model_path
        self._loader = loader
        self._model: Any | None = None
        self._load_lock = Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if not self.model_path.is_file():
                        raise PriceProviderError("Configured price model artifact is unavailable.")
                    try:
                        self._model = self._loader(self.model_path)
                        if not callable(getattr(self._model, "predict", None)):
                            raise ValueError("artifact has no predict method")
                    except Exception as exc:
                        raise PriceProviderError("Configured price model artifact is invalid.") from exc
        return self._model

    @staticmethod
    def map_features(vehicle: VehiclePriceInput) -> dict[str, Any]:
        from datetime import date

        vehicle_age = date.today().year - vehicle.registration_year
        fuel = {
            FuelType.PETROL: "Petrol",
            FuelType.DIESEL: "Diesel",
            FuelType.CNG: "CNG",
            FuelType.LPG: "LPG",
            FuelType.ELECTRIC: "Electric",
        }[vehicle.fuel_type]
        transmission = (
            "Manual" if vehicle.transmission is TransmissionType.MANUAL else "Automatic"
        )
        return {
            "brand": vehicle.make,
            "model": vehicle.model,
            "seller_type": vehicle.seller_type.value,
            "fuel_type": fuel,
            "transmission_type": transmission,
            "vehicle_age": vehicle_age,
            "km_driven": vehicle.odometer_km,
            "mileage": vehicle.mileage_kmpl,
            "engine": vehicle.engine_cc,
            "max_power": vehicle.max_power_bhp,
            "seats": vehicle.seats,
            "km_per_year": vehicle.odometer_km / max(vehicle_age, 1),
        }

    def predict(self, vehicle: VehiclePriceInput) -> PricePrediction:
        features = self.map_features(vehicle)
        try:
            import pandas as pd

            log_price_values = self._get_model().predict(pd.DataFrame([features]))
            if len(log_price_values) != 1:
                raise ValueError("model returned an unexpected number of predictions")
            log_price = float(log_price_values[0])
            if not isfinite(log_price):
                raise ValueError("model returned a non-finite value")
            amount = expm1(log_price)
            if not isfinite(amount) or amount <= 0:
                raise ValueError("model returned an invalid price")
            return PricePrediction(
                amount=Decimal(str(amount)).quantize(Decimal("0.01")),
                currency="INR",
                lower_bound=None,
                upper_bound=None,
                method="xgboost_pipeline_log1p_expm1",
                model_name=self.model_name,
            )
        except PriceProviderError:
            raise
        except Exception as exc:
            raise PriceProviderError("Price model inference did not return a valid estimate.") from exc


@lru_cache(maxsize=1)
def get_price_provider() -> PricePredictionProvider:
    """Return the configured, lazily loaded research pipeline provider."""
    return PipelinePriceProvider(settings.price_model_path)

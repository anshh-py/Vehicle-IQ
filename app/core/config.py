from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Vehicle IQ"
    app_env: str = "development"
    api_access_token: SecretStr | None = None
    api_rate_limit_per_minute: int = Field(default=600, gt=0, le=10000)
    database_url: str
    image_storage_dir: Path = PROJECT_ROOT / ".local-storage" / "vehicle-images"
    max_image_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0, le=100 * 1024 * 1024)
    max_external_response_bytes: int = Field(default=2 * 1024 * 1024, gt=0, le=10 * 1024 * 1024)
    buyer_price_near_threshold_percent: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Maximum absolute deviation from the ML estimate considered near-priced.",
    )
    langsearch_api_key: SecretStr | None = None
    langsearch_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    langsearch_result_limit: int = Field(default=10, ge=1, le=50)
    ocr_service_url: str | None = None
    ocr_service_provider: Literal["baidu_unlimited_ocr", "huggingface_gradio", "paddleocr_local"] = "baidu_unlimited_ocr"
    ocr_gradio_space_url: str = "https://baidu-unlimited-ocr.hf.space"
    ocr_gradio_mode: Literal["gundam", "base"] = "gundam"
    ocr_paddle_detection_model_dir: Path = (
        PROJECT_ROOT
        / ".local-storage"
        / "paddleocr-smoke-bos"
        / "paddlex"
        / "official_models"
        / "PP-OCRv5_mobile_det"
    )
    ocr_paddle_recognition_model_dir: Path = (
        PROJECT_ROOT
        / ".local-storage"
        / "paddleocr-smoke-bos"
        / "paddlex"
        / "official_models"
        / "en_PP-OCRv5_mobile_rec"
    )
    damage_inference_python: Path = (
        PROJECT_ROOT / ".local-storage" / "damage-poc" / ".venv" / "Scripts" / "python.exe"
    )
    damage_checkpoint_path: Path = PROJECT_ROOT / ".local-storage" / "damage-poc" / "trained.pt"
    damage_inference_timeout_seconds: int = Field(default=180, gt=0, le=600)
    vehicle_lookup_enabled: bool = False
    vehicle_lookup_base_url: str = "https://vehiclex.profilework239.workers.dev/"
    vehicle_lookup_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    challan_lookup_enabled: bool = False
    challan_lookup_base_url: str = "https://challan-info-new-abhigyan.vercel.app/challaninfo"
    challan_lookup_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    fastag_lookup_enabled: bool = False
    fastag_lookup_base_url: str = "https://abhigyan-fastag-info.vercel.app/vehtofastaginfo"
    fastag_lookup_timeout_seconds: float = Field(default=10.0, gt=0, le=30)
    price_model_path: Path = Path(
        r"C:\Users\ansha\vehicle-iq-model-research\artifacts\vehicle_iq_price_model.joblib"
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("image_storage_dir")
    @classmethod
    def resolve_image_storage_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        return value.resolve()

    @field_validator("ocr_paddle_detection_model_dir", "ocr_paddle_recognition_model_dir")
    @classmethod
    def resolve_ocr_model_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        return value.resolve()

    @field_validator("damage_inference_python", "damage_checkpoint_path", "price_model_path")
    @classmethod
    def resolve_damage_inference_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        return value.resolve()


settings = Settings()


def validate_production_settings(config: Settings = settings) -> None:
    """Fail startup with setting names only when production prerequisites are absent."""
    if config.app_env.casefold() != "production":
        return

    missing_or_invalid: list[str] = []
    access_token = config.api_access_token.get_secret_value() if config.api_access_token else ""
    if len(access_token) < 32:
        missing_or_invalid.append("API_ACCESS_TOKEN (at least 32 characters)")

    try:
        database_url = make_url(config.database_url)
        if database_url.drivername != "postgresql+psycopg":
            missing_or_invalid.append("DATABASE_URL (must use postgresql+psycopg)")
    except (TypeError, ValueError):
        missing_or_invalid.append("DATABASE_URL (invalid SQLAlchemy URL)")

    if "image_storage_dir" not in config.model_fields_set:
        missing_or_invalid.append("IMAGE_STORAGE_DIR (must point to a persistent mounted volume)")

    if missing_or_invalid:
        required = ", ".join(missing_or_invalid)
        raise RuntimeError(f"Production configuration is incomplete or invalid: {required}.")

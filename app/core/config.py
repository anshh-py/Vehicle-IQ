import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

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
    image_storage_provider: Literal["local", "supabase"] = "local"
    supabase_url: str | None = None
    supabase_storage_bucket: str | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_storage_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    cors_allowed_origins: str = ""
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
    tavily_api_key: SecretStr | None = None
    tavily_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    tavily_result_limit: int = Field(default=20, ge=1, le=50)
    ocr_service_url: str | None = None
    ocr_service_provider: Literal["baidu_unlimited_ocr", "huggingface_gradio", "paddleocr_local"] = "baidu_unlimited_ocr"
    ocr_gradio_space_url: str = "https://baidu-unlimited-ocr.hf.space"
    ocr_timeout_seconds: float = Field(default=180.0, gt=0, le=300)
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
    damage_inference_enabled: bool | None = None
    damage_inference_python: Path = Field(
        default_factory=lambda: PROJECT_ROOT
        / ".local-storage"
        / "damage-poc"
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
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
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    market_listings_source_path: Path | None = None
    market_listings_source_format: str | None = None
    llm_enabled: bool = False
    price_model_path: Path = PROJECT_ROOT / "artifacts" / "vehicle_iq_price_model.joblib"

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
        elif str(database_url.query.get("sslmode", "")).casefold() not in {
            "require", "verify-ca", "verify-full"
        }:
            missing_or_invalid.append("DATABASE_URL (must require TLS with sslmode=require or stronger)")
    except (TypeError, ValueError):
        missing_or_invalid.append("DATABASE_URL (invalid SQLAlchemy URL)")

    if config.image_storage_provider != "supabase":
        missing_or_invalid.append("IMAGE_STORAGE_PROVIDER (must be supabase in production)")
    supabase_url = _valid_https_url(config.supabase_url)
    if supabase_url is None or supabase_url.path not in {"", "/"} or supabase_url.query or supabase_url.fragment:
        missing_or_invalid.append("SUPABASE_URL (must be an HTTPS project URL)")
    bucket = (config.supabase_storage_bucket or "").strip()
    if not bucket or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in bucket):
        missing_or_invalid.append("SUPABASE_STORAGE_BUCKET (required)")
    if config.supabase_service_role_key is None or not config.supabase_service_role_key.get_secret_value().strip():
        missing_or_invalid.append("SUPABASE_SERVICE_ROLE_KEY (required server-side)")

    if config.ocr_service_provider == "paddleocr_local":
        missing_or_invalid.append("OCR_SERVICE_PROVIDER (must select a remote provider in production)")
    elif config.ocr_service_provider == "baidu_unlimited_ocr":
        if _valid_https_url(config.ocr_service_url) is None:
            missing_or_invalid.append("OCR_SERVICE_URL (must be an HTTPS URL for the remote OCR service)")
    elif config.ocr_service_provider == "huggingface_gradio":
        if _valid_https_url(config.ocr_gradio_space_url) is None:
            missing_or_invalid.append("OCR_GRADIO_SPACE_URL (must be an HTTPS URL)")

    origins = [origin.strip() for origin in config.cors_allowed_origins.split(",") if origin.strip()]
    if not origins or any(not _is_exact_https_origin(origin) for origin in origins):
        missing_or_invalid.append("CORS_ALLOWED_ORIGINS (must contain exact HTTPS origins)")

    if missing_or_invalid:
        required = ", ".join(missing_or_invalid)
        raise RuntimeError(f"Production configuration is incomplete or invalid: {required}.")


def _valid_https_url(value: str | None):
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        parsed.port  # Access validates malformed port numbers.
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            return None
        return parsed
    except ValueError:
        return None


def _is_exact_https_origin(value: str) -> bool:
    parsed = _valid_https_url(value)
    return parsed is not None and parsed.path == "" and not parsed.query and not parsed.fragment

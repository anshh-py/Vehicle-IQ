import os
import re
from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings


def test_render_python_version_is_pinned_to_supported_312_patch():
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8") == "3.12.16\n"


def test_production_requirements_exclude_optional_local_ocr_packages():
    production = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").casefold()
    local_ocr = (PROJECT_ROOT / "requirements-local-ocr.txt").read_text(encoding="utf-8").casefold()

    requirement_names = {
        match.group(1).replace("_", "-")
        for match in re.finditer(r"(?m)^\s*([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*[<>=!~]", production)
    }
    assert not {"paddlepaddle", "paddleocr", "gradio-client"} & requirement_names
    assert "paddlepaddle==3.3.1" in local_ocr
    assert "paddleocr==3.3.0" in local_ocr


def test_price_model_path_default_is_repository_relative():
    settings = Settings(_env_file=None, database_url="sqlite+pysqlite:///:memory:")

    assert settings.price_model_path == PROJECT_ROOT / "artifacts" / "vehicle_iq_price_model.joblib"


def test_damage_runtime_default_is_platform_portable_and_opt_in_in_production():
    settings = Settings(
        _env_file=None,
        database_url="sqlite+pysqlite:///:memory:",
        app_env="production",
    )

    expected_runtime_suffix = (
        Path(".local-storage") / "damage-poc" / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else Path(".local-storage") / "damage-poc" / ".venv" / "bin" / "python"
    )
    assert settings.damage_inference_python == PROJECT_ROOT / expected_runtime_suffix
    assert settings.damage_inference_enabled is None


def test_paddle_package_import_is_not_eager_in_ocr_client_source():
    source = (PROJECT_ROOT / "app" / "ocr" / "client.py").read_text(encoding="utf-8")
    import_at = source.index("from paddleocr import PaddleOCR")
    class_at = source.index("class PaddleOCRLocalClient")

    assert import_at > class_at
    assert "from paddleocr import PaddleOCR" not in source[:class_at]


def test_deployment_docs_describe_root_ocr_isolation_and_python_pin():
    documentation = (PROJECT_ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")

    assert "3.12.16" in documentation
    assert "requirements-local-ocr.txt" in documentation
    assert "OCR_SERVICE_URL" in documentation
    assert "DAMAGE_INFERENCE_ENABLED=false" in documentation

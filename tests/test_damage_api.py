from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import MODEL_NAME, DamageWorkerUnavailable
from app.db.base import Base
from app.db.models import DamageFinding, EvidenceRecord, OCRResult, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.api.v1.damage import get_damage_provider


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 20), color=(200, 200, 200)).save(output, format="PNG")
    return output.getvalue()


def vehicle_png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (415, 378), color=(200, 200, 200)).save(output, format="PNG")
    return output.getvalue()


class FakeDamageProvider:
    provider_name = "yolo11m_local"
    def __init__(self, result=None, error=None):
        self.result = result or DamageInferenceResult(
            detections=(
                DamageDetection("dent", 0.91, (1.0, 2.0, 10.0, 11.0)),
                DamageDetection("broken_lamp", 0.84, (11.0, 3.0, 20.0, 15.0)),
            ),
            inference_seconds=1.25,
        )
        self.error = error
        self.calls = []

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        self.calls.append((content, extension))
        if self.error:
            raise self.error
        return self.result


class SequenceDamageProvider(FakeDamageProvider):
    def __init__(self, results):
        super().__init__()
        self.results = iter(results)

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        self.calls.append((content, extension))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class ExternalGradioDamageProvider(FakeDamageProvider):
    provider_name = "huggingface_gradio_damage"
    model_name = "huggingface_gradio_damage"


class DamageAPIFixture:
    def __init__(self, client, engine, assessment_id, image_id, provider, content):
        self.client = client
        self.engine = engine
        self.assessment_id = assessment_id
        self.image_id = image_id
        self.provider = provider
        self.content = content


@contextmanager
def damage_api(tmp_path, *, content=None, provider=None) -> Generator[DamageAPIFixture, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    content = png_bytes() if content is None else content
    with Session(engine) as session:
        assessment = VehicleAssessment()
        session.add(assessment)
        session.flush()
        image = UploadedImage(
            assessment_id=assessment.id,
            storage_provider="postgresql",
            storage_bucket="vehicle-images",
            object_key=uuid4().hex,
            content=content,
            original_filename="vehicle.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="test-sha256",
        )
        session.add(image)
        session.commit()
        assessment_id, image_id = str(assessment.id), str(image.id)

    fake = provider or FakeDamageProvider()

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_damage_provider] = lambda: fake
    try:
        with TestClient(app) as client:
            yield DamageAPIFixture(client, engine, assessment_id, image_id, fake, content)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.mark.parametrize("category", ["vehicle_exterior", "damage_closeup"])
def test_explicit_eligible_categories_invoke_provider_even_when_ocr_failed(damage_api_fixture, category):
    api = damage_api_fixture
    content = vehicle_png_bytes()
    uploaded = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images",
        files={"file": ("unnamed.png", content, "image/png")},
        data={"image_type": category},
    )
    assert uploaded.status_code == 201
    image_id = uploaded.json()["id"]
    with Session(api.engine) as session:
        session.add(OCRResult(
            assessment_id=UUID(api.assessment_id), image_id=UUID(image_id),
            result_type="image_ocr", status="failed", provider="test-ocr",
            result_data={"error_code": "inference_failed"},
        ))
        session.commit()

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{image_id}/damage"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed_with_findings"
    assert payload["image_type"] == category
    assert payload["inspection_outcome"] == "completed_with_findings"
    assert len(api.provider.calls) == 1
    assert api.provider.calls[0][0] == content

    report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
    assert report.status_code == 200
    image_report = next(item for item in report.json()["report_data"]["images"] if item["id"] == image_id)
    assert image_report["image_type"] == category
    assert image_report["damage_eligible"] is True
    assert image_report["classification_source"] == "user_input"


@pytest.mark.parametrize(
    ("category", "outcome"),
    [("registration_plate", "ineligible"), ("document", "ineligible"), ("vehicle_interior", "ineligible"), ("unknown", "classification_uncertain")],
)
def test_explicit_ineligible_or_unknown_category_never_invokes_provider(damage_api_fixture, category, outcome):
    api = damage_api_fixture
    uploaded = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images",
        files={"file": ("unnamed.png", vehicle_png_bytes(), "image/png")},
        data={"image_type": category},
    )
    assert uploaded.status_code == 201
    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{uploaded.json()['id']}/damage"
    )
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["inspection_outcome"] == outcome
    assert api.provider.calls == []


@pytest.fixture
def damage_api_fixture(tmp_path):
    with damage_api(tmp_path) as api:
        yield api


def count_findings(engine) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(DamageFinding)) or 0


def add_stored_image(api, content: bytes) -> str:
    with Session(api.engine) as session:
        image = UploadedImage(
            assessment_id=UUID(api.assessment_id),
            storage_provider="postgresql",
            storage_bucket="vehicle-images",
            object_key=uuid4().hex,
            content=content,
            original_filename="additional-vehicle.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="additional-test-sha256",
        )
        session.add(image)
        session.commit()
        return str(image.id)


def test_damage_endpoint_persists_only_model_results(damage_api_fixture):
    api = damage_api_fixture

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "yolo11m_local"
    assert payload["status"] == "completed_with_findings"
    assert payload["reused"] is False
    assert [finding["damage_type"] for finding in payload["findings"]] == ["dent", "broken_lamp"]
    assert [finding["confidence"] for finding in payload["findings"]] == [0.91, 0.84]
    assert payload["findings"][0]["component"] is None
    assert payload["findings"][0]["bounding_box"] == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 11.0,
        "coordinate_space": "original_image_pixels",
    }
    assert api.provider.calls == [(api.content, "png")]
    assert count_findings(api.engine) == 2


def test_repeated_damage_request_reuses_existing_findings(damage_api_fixture):
    api = damage_api_fixture
    path = f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"

    first = api.client.post(path)
    second = api.client.post(path)

    assert first.status_code == second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["findings"] == first.json()["findings"]
    assert len(api.provider.calls) == 1
    assert count_findings(api.engine) == 2


def test_legacy_completed_zero_detection_record_maps_to_no_findings(damage_api_fixture):
    api = damage_api_fixture
    with Session(api.engine) as session:
        session.add(
            EvidenceRecord(
                assessment_id=UUID(api.assessment_id),
                image_id=UUID(api.image_id),
                evidence_type="damage_detection",
                source_reference=MODEL_NAME,
                evidence_data={"status": "completed", "detection_count": 0},
            )
        )
        session.commit()

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed_no_findings"
    assert response.json()["reused"] is True
    assert api.provider.calls == []


def test_damage_endpoint_returns_404_for_image_outside_assessment(damage_api_fixture):
    api = damage_api_fixture
    with Session(api.engine) as session:
        other_assessment = VehicleAssessment()
        session.add(other_assessment)
        session.commit()
        other_assessment_id = str(other_assessment.id)

    response = api.client.post(
        f"/api/v1/assessments/{other_assessment_id}/images/{api.image_id}/damage"
    )

    assert response.status_code == 404
    assert api.provider.calls == []
    assert count_findings(api.engine) == 0


def test_damage_endpoint_returns_404_for_missing_image(damage_api_fixture):
    api = damage_api_fixture

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{uuid4()}/damage"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found for assessment."
    assert api.provider.calls == []


def test_damage_endpoint_rejects_unsupported_stored_image_format(tmp_path):
    output = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(output, format="BMP")
    with damage_api(tmp_path, content=output.getvalue()) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "unavailable"
        assert response.json()["reason"] == "Stored image is corrupt or unsupported."
        assert api.provider.calls == []
        assert count_findings(api.engine) == 0
        with Session(api.engine) as session:
            marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
            assert marker.evidence_data["status"] == "unavailable"


def test_damage_endpoint_rejects_corrupt_stored_image(tmp_path):
    with damage_api(tmp_path, content=b"not an image") as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "unavailable"
        assert api.provider.calls == []
        assert count_findings(api.engine) == 0
        with Session(api.engine) as session:
            marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
            assert marker.evidence_data["status"] == "unavailable"


def test_damage_endpoint_persists_unavailable_model_outcome(tmp_path):
    with damage_api(tmp_path, provider=FakeDamageProvider(error=DamageWorkerUnavailable())) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )
        assert response.status_code == 200
        assert response.json()["status"] == "unavailable"
        assert response.json()["reason"] == "Damage inference is unavailable."
        assert count_findings(api.engine) == 0
        with Session(api.engine) as session:
            marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
            assert marker.evidence_data["status"] == "unavailable"


def test_damage_endpoint_persists_unexpected_inference_exception_as_failed(tmp_path):
    with damage_api(
        tmp_path,
        provider=FakeDamageProvider(error=RuntimeError("private inference detail")),
    ) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert response.json()["reason"] == "Damage inference failed."
        assert "private inference detail" not in response.text
        with Session(api.engine) as session:
            marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
            assert marker.evidence_data["status"] == "failed"


def test_batch_damage_processes_ten_owned_images_and_report_aggregates_findings(damage_api_fixture):
    api = damage_api_fixture
    image_ids = [api.image_id] + [add_stored_image(api, api.content) for _ in range(9)]

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed_with_findings"
    assert payload["processed_count"] == 10
    assert payload["failed_count"] == 0
    assert len(payload["results"]) == 10
    assert all(item["status"] == "completed_with_findings" and len(item["findings"]) == 2 for item in payload["results"])
    assert {finding["image_id"] for item in payload["results"] for finding in item["findings"]} == set(image_ids)
    assert count_findings(api.engine) == 20
    assert len(api.provider.calls) == 10

    repeated = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": image_ids},
    )
    assert repeated.status_code == 200
    assert all(item["reused"] for item in repeated.json()["results"])
    assert count_findings(api.engine) == 20
    assert len(api.provider.calls) == 10

    report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
    assert report.status_code == 200
    report_data = report.json()["report_data"]
    assert report_data["stages"]["damage_detection"]["finding_count"] == 20
    assert report_data["stages"]["damage_detection"]["images_with_findings"] == 10


def test_batch_damage_rejects_more_than_ten_and_duplicate_image_ids(damage_api_fixture):
    api = damage_api_fixture
    too_many = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [str(uuid4()) for _ in range(11)]},
    )
    duplicate = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id, api.image_id]},
    )

    assert too_many.status_code == duplicate.status_code == 422
    assert api.provider.calls == []


def test_batch_damage_reports_cross_assessment_image_failure(damage_api_fixture):
    api = damage_api_fixture
    with Session(api.engine) as session:
        other_assessment = VehicleAssessment()
        session.add(other_assessment)
        session.commit()
        other_assessment_id = str(other_assessment.id)

    response = api.client.post(
        f"/api/v1/assessments/{other_assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["results"][0]["error"] == "Image not found for assessment."
    assert api.provider.calls == []
    assert count_findings(api.engine) == 0


def test_batch_damage_empty_detection_is_success_without_fabricated_findings(
    damage_api_fixture, monkeypatch
):
    api = damage_api_fixture
    api.provider.result = DamageInferenceResult(detections=(), inference_seconds=0.2)

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed_no_findings"
    assert response.json()["results"][0]["status"] == "completed_no_findings"
    assert response.json()["results"][0]["findings"] == []
    assert count_findings(api.engine) == 0
    assert response.json()["results"][0]["reused"] is False

    repeated = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )
    assert repeated.status_code == 200
    assert repeated.json()["results"][0]["reused"] is True
    assert len(api.provider.calls) == 1
    with Session(api.engine) as session:
        marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
        assert marker.evidence_data["status"] == "completed_no_findings"
        assert marker.evidence_data["detection_count"] == 0
        assert marker.evidence_data["inspection_outcome"] == "completed_no_findings"
        assert marker.evidence_data["damage_eligible"] is True
        assert marker.evidence_data["condition_category"] == "no_visible_damage"
        assert marker.evidence_data["inference_seconds"] == pytest.approx(0.2)

    from app.llm.schemas import LLMReportNarrative

    narrative = LLMReportNarrative(
        vehicle_overview="Vehicle.",
        overall_condition_summary="This vehicle is damage-free.",
        detected_damage_explanation="No issues were found.",
        potential_inspection_risks="None.",
        buyer_recommendations="Proceed.",
        price_estimate_explanation="Model estimate.",
        missing_information_and_limitations="None.",
    )
    monkeypatch.setattr(
        "app.api.v1.reports.generate_llm_narrative",
        lambda _report_data: narrative,
    )
    report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
    assert report.status_code == 200
    assert report.json()["report_data"]["stages"]["damage_detection"]["status"] == "completed_no_findings"
    assert report.json()["report_data"]["stages"]["damage_detection"]["image_results"][0]["status"] == "completed_no_findings"
    assert report.json()["report_data"]["llm_narrative"]["detected_damage_explanation"] == (
        "No damage was detected in the supplied images. This does not establish "
        "that the vehicle is damage-free."
    )
    assert report.json()["report_data"]["llm_narrative"]["overall_condition_summary"] == (
        "No damage was detected in the supplied images. This does not establish "
        "that the vehicle is damage-free."
    )


def test_batch_damage_continues_after_corrupt_image_and_inference_failure(tmp_path):
    with damage_api(tmp_path, provider=FakeDamageProvider()) as api:
        corrupt_id = add_stored_image(api, b"not an image")
        second_id = add_stored_image(api, api.content)
        third_id = add_stored_image(api, api.content)
        api.provider = SequenceDamageProvider(
            [DamageWorkerUnavailable("worker unavailable"), api.provider.result]
        )
        app.dependency_overrides[get_damage_provider] = lambda: api.provider

        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/damage",
            json={"image_ids": [corrupt_id, second_id, third_id]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed_with_findings"
        assert body["processed_count"] == 1
        assert body["failed_count"] == 0
        assert body["unavailable_count"] == 2
        assert body["results"][0]["status"] == "unavailable"
        assert body["results"][1]["status"] == "unavailable"
        assert body["results"][2]["status"] == "completed_with_findings"
        assert body["results"][0]["error"] == "Stored image is corrupt or unsupported."
        assert body["results"][1]["error"] == "Damage inference is unavailable."
        assert len(body["results"][2]["findings"]) == 2
        assert len(api.provider.calls) == 2
        assert count_findings(api.engine) == 2

        report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
        assert report.status_code == 200
        assert report.json()["report_data"]["stages"]["damage_detection"]["status"] == "completed_with_findings"
        assert report.json()["report_data"]["stages"]["damage_detection"]["unavailable_count"] == 3
def png_plate_bytes():
    output = BytesIO()
    Image.new("RGB", (141, 59), color=(200, 200, 200)).save(output, format="PNG")
    return output.getvalue()


def test_ineligible_plate_image_is_not_evaluated_and_provider_not_called():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    content = png_plate_bytes()
    with Session(engine) as session:
        assessment = VehicleAssessment()
        session.add(assessment)
        session.flush()
        image = UploadedImage(
            assessment_id=assessment.id,
            storage_provider="postgresql",
            storage_bucket="vehicle-images",
            object_key=uuid4().hex,
            content=content,
            original_filename="plate.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="test-sha256",
        )
        session.add(image)
        session.commit()
        assessment_id, image_id = str(assessment.id), str(image.id)

    fake = FakeDamageProvider()

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_damage_provider] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/damage")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "unavailable"
            assert payload["findings"] == []
            assert payload["image_type"] == "registration_plate"
            assert len(fake.calls) == 0
            second = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/damage")
            assert second.status_code == 200
            assert second.json()["reused"] is True
            with Session(engine) as session:
                markers = session.scalars(
                    select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_eligibility")
                ).all()
                assert len(markers) == 1
                scan = session.scalar(
                    select(EvidenceRecord).where(
                        EvidenceRecord.evidence_type == "damage_detection"
                    )
                )
                assert scan.evidence_data["status"] == "unavailable"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_external_gradio_provider_persists_external_model_provenance(tmp_path):
    with damage_api(tmp_path, provider=ExternalGradioDamageProvider()) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "huggingface_gradio_damage"
        assert payload["findings"]
        assert all(f["model_name"] == "huggingface_gradio_damage" for f in payload["findings"])

        with Session(api.engine) as session:
            findings = session.scalars(
                select(DamageFinding).where(DamageFinding.assessment_id == UUID(api.assessment_id))
            ).all()
            assert findings
            assert all(f.model_name == "huggingface_gradio_damage" for f in findings)
            scan = session.scalar(
                select(EvidenceRecord).where(
                    EvidenceRecord.evidence_type == "damage_detection",
                    EvidenceRecord.assessment_id == UUID(api.assessment_id),
                )
            )
            assert scan is not None
            assert scan.source_reference == "huggingface_gradio_damage"


def test_local_provider_keeps_local_model_provenance(damage_api_fixture):
    api = damage_api_fixture
    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "yolo11m_local"
    assert all(f["model_name"] == MODEL_NAME for f in payload["findings"])

    with Session(api.engine) as session:
        findings = session.scalars(
            select(DamageFinding).where(DamageFinding.assessment_id == UUID(api.assessment_id))
        ).all()
        assert findings
        assert all(f.model_name == MODEL_NAME for f in findings)
        scan = session.scalar(
            select(EvidenceRecord).where(
                EvidenceRecord.evidence_type == "damage_detection",
                EvidenceRecord.assessment_id == UUID(api.assessment_id),
            )
        )
        assert scan is not None
        assert scan.source_reference == MODEL_NAME


def test_external_provider_findings_are_reused_without_duplication(tmp_path):
    with damage_api(tmp_path, provider=ExternalGradioDamageProvider()) as api:
        path = f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        first = api.client.post(path)
        second = api.client.post(path)

        assert first.status_code == second.status_code == 200
        assert second.json()["reused"] is True
        assert second.json()["findings"] == first.json()["findings"]
        assert len(api.provider.calls) == 1

        with Session(api.engine) as session:
            count = session.scalar(select(func.count()).select_from(DamageFinding)) or 0
            assert count == len(first.json()["findings"])


def test_gradio_no_damage_response_is_not_a_finding_and_condition_clean(tmp_path):
    from app.damage.gradio_provider import GradioDamageProvider

    class LocalFakeClient:
        def __init__(self):
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            headers = ["Class", "Confidence", "YOLO [x_center, y_center, w, h]"]
            return (
                "/tmp/a.webp",
                {"headers": headers, "data": [["no damage", 0.989, "[0.5, 0.5, 1.0, 1.0]"]]},
                "/tmp/a.csv",
                "/tmp/a.txt",
            )

    fake = LocalFakeClient()
    provider = GradioDamageProvider(client_factory=lambda: fake)
    with damage_api(tmp_path, provider=provider) as api:
        path = f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        response = api.client.post(path)
        assert response.status_code == 200
        payload = response.json()
        assert payload["provider"] == "huggingface_gradio_damage"
        assert payload["status"] == "completed_no_findings"
        assert payload["findings"] == []
        assert payload["condition_category"] == "no_visible_damage"
        assert payload["requires_human_review"] is False
        assert count_findings(api.engine) == 0

        with Session(api.engine) as session:
            scan = session.scalar(
                select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection")
            )
            assert scan is not None
            assert scan.evidence_data["status"] == "completed_no_findings"
            assert scan.evidence_data["detection_count"] == 0
            assert scan.evidence_data["condition_category"] == "no_visible_damage"
            assert scan.source_reference == "huggingface_gradio_damage"


def test_batch_clean_and_damaged_images_aggregate_distinctly(tmp_path):
    # Two images: the first yields an empty result (clean vehicle after the
    # provider filters the "no damage" sentinel); the second yields real damage.
    results = [
        DamageInferenceResult(detections=(), inference_seconds=0.3),
        DamageInferenceResult(
            detections=(DamageDetection("dent", 0.91, (1.0, 2.0, 10.0, 11.0)),),
            inference_seconds=0.5,
        ),
    ]
    with damage_api(tmp_path, provider=SequenceDamageProvider(results)) as api:
        clean_id = api.image_id
        damaged_id = add_stored_image(api, api.content)
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/damage",
            json={"image_ids": [clean_id, damaged_id]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed_with_findings"
        assert body["processed_count"] == 2
        assert body["failed_count"] == 0
        assert body["unavailable_count"] == 0
        by_image = {r["image_id"]: r for r in body["results"]}
        assert by_image[clean_id]["status"] == "completed_no_findings"
        assert by_image[clean_id]["findings"] == []
        assert by_image[clean_id]["condition_category"] == "no_visible_damage"
        assert by_image[damaged_id]["status"] == "completed_with_findings"
        assert len(by_image[damaged_id]["findings"]) == 1
        assert by_image[damaged_id]["findings"][0]["damage_type"] == "dent"
        assert count_findings(api.engine) == 1


def test_valuation_not_withheld_for_no_visible_damage_condition():
    from app.pricing.condition import assess_condition, pricing_decision

    condition = assess_condition(
        [{"damage_eligible": True, "status": "completed_no_findings"}], []
    )
    assert condition["category"] == "no_visible_damage"
    assert condition["requires_human_review"] is False

    decision = pricing_decision(2000000, "test-model", condition)
    assert decision["pricing_eligible"] is True
    assert decision["status"] == "available"
    assert decision["customer_facing_estimate_inr"] == "2000000"


def test_valuation_withheld_for_minor_cosmetic_damage_condition():
    from app.pricing.condition import assess_condition, pricing_decision

    # "dent" is in the supported minor set -> minor_cosmetic (still withheld).
    condition = assess_condition(
        [{"damage_eligible": True, "status": "completed_with_findings"}],
        [{"damage_type": "dent"}],
    )
    assert condition["category"] == "minor_cosmetic"
    assert condition["requires_human_review"] is True

    decision = pricing_decision(2000000, "test-model", condition)
    assert decision["pricing_eligible"] is False
    assert decision["status"] == "withheld_insufficient_adjustment_evidence"


def test_valuation_withheld_for_moderate_damage_condition():
    from app.pricing.condition import assess_condition, pricing_decision

    # A non-minor class (e.g. "broken lamp") -> moderate_damage (withheld).
    condition = assess_condition(
        [{"damage_eligible": True, "status": "completed_with_findings"}],
        [{"damage_type": "broken lamp"}],
    )
    assert condition["category"] == "moderate_damage"
    assert condition["requires_human_review"] is True

    decision = pricing_decision(2000000, "test-model", condition)
    assert decision["pricing_eligible"] is False
    assert decision["status"] == "withheld_insufficient_adjustment_evidence"


def test_generic_filename_large_vehicle_photo_invokes_provider(damage_api_fixture):
    api = damage_api_fixture
    content = vehicle_png_bytes()
    uploaded = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images",
        files={"file": ("IMG_0001.png", content, "image/png")},
    )
    assert uploaded.status_code == 201
    image_id = uploaded.json()["id"]
    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{image_id}/damage"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed_with_findings"
    assert payload["image_type"] == "vehicle_exterior"
    assert payload["inspection_outcome"] == "completed_with_findings"
    assert len(api.provider.calls) == 1
    assert api.provider.calls[0][0] == content


def test_rejected_image_logs_structured_non_sensitive_record(damage_api_fixture, caplog):
    import logging

    api = damage_api_fixture
    uploaded = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images",
        files={"file": ("IMG_0001.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 201
    image_id = uploaded.json()["id"]
    with caplog.at_level(logging.INFO, logger="app.api.v1.damage"):
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{image_id}/damage"
        )
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["inspection_outcome"] == "classification_uncertain"
    assert response.json()["damage_eligible"] is False
    assert api.provider.calls == []

    records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "damage_image_classification_rejected"
    ]
    assert records
    record = records[0]
    assert record.image_type == "unknown"
    assert record.rejection_reason == "insufficient_classification_evidence"
    assert record.classification_method == "insufficient_evidence"
    assert record.classification_source == "automatic"
    assert record.message == "Damage image classification rejected for damage inference"

import sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.api.v1.ocr import get_ocr_client
from app.api.v1.damage import get_damage_provider
from app.api.v1.vehicle_lookup import get_vehicle_registration_provider, get_challan_provider, get_fastag_provider
from app.pricing.provider import get_price_provider
from datetime import date
from decimal import Decimal

engine = create_engine('sqlite+pysqlite:///:memory:', connect_args={'check_same_thread': False}, poolclass=StaticPool)
@event.listens_for(engine, 'connect')
def enable_foreign_keys(dbapi_connection, _connection_record):
    dbapi_connection.execute('PRAGMA foreign_keys=ON')
Base.metadata.create_all(engine)
def override_get_db():
    with Session(engine) as session:
        yield session
app.dependency_overrides[get_db] = override_get_db

class FakeOCR:
    provider_name = 'mock_ocr'
    def recognize(self, _content, _filename, _content_type):
        from app.ocr.client import OCRRecognition
        return OCRRecognition('RJ45CA1879', 0.98)

class FakeVehicleDetails:
    provider_name = 'mock_vehicle_details'
    def lookup(self, registration_number):
        from app.schemas.vehicle_lookup import VehicleRegistrationDetails
        return VehicleRegistrationDetails(
            registration_number=registration_number,
            make='Test Make',
            model='Test Model',
            variant='X',
            fuel_type='Petrol',
            registration_date=date(2021, 5, 12),
            manufacturing_year=2021,
            engine_cc=1500,
            seats=5,
            max_power_bhp=100.0,
            transmission='Manual',
            mileage_kmpl=15.0,
            owner_count=1,
        )

class FakeChallan:
    provider_name = 'mock_challan'
    def lookup(self, _registration_number):
        return None

class FailedFastag:
    provider_name = 'mock_fastag'
    def lookup(self, _registration_number):
        return None

class FakeDamage:
    provider_name = 'mock_damage'
    def detect(self, _content, _extension):
        from app.damage.inference import DamageDetection, DamageInferenceResult
        return DamageInferenceResult(
            detections=(DamageDetection('dent', 0.9, (1.0, 2.0, 12.0, 14.0)),),
            inference_seconds=0.01,
        )

class FakePrice:
    def predict(self, _vehicle):
        from app.schemas.pricing import PricePrediction
        from decimal import Decimal
        return PricePrediction(
            amount=Decimal('625000.00'),
            currency='INR',
            lower_bound=None,
            upper_bound=None,
            method='mock',
            model_name='test_price_model',
        )

app.dependency_overrides[get_ocr_client] = lambda: FakeOCR()
app.dependency_overrides[get_vehicle_registration_provider] = lambda: FakeVehicleDetails()
app.dependency_overrides[get_challan_provider] = lambda: FakeChallan()
app.dependency_overrides[get_fastag_provider] = lambda: FailedFastag()
app.dependency_overrides[get_damage_provider] = lambda: FakeDamage()
app.dependency_overrides[get_price_provider] = lambda: FakePrice()

with TestClient(app) as client:
    # Create assessment
    payload = {'asking_price_inr': '700000'}
    created = client.post('/api/v1/assessments', json=payload)
    print('Created:', created.status_code)
    assessment_id = created.json()['id']
    
    # Upload image
    from io import BytesIO
    from PIL import Image
    output = BytesIO()
    Image.new('RGB', (32, 24), color=(180, 180, 180)).save(output, format='PNG')
    png_data = output.getvalue()
    
    uploaded = client.post(f'/api/v1/assessments/{assessment_id}/images', files={'file': ('vehicle.png', png_data, 'image/png')})
    print('Uploaded:', uploaded.status_code)
    image_id = uploaded.json()['id']
    
    # Run workflow first time
    vehicle_payload = {"odometer_km": 50000, "seller_type": "Individual"}
    response1 = client.post(f'/api/v1/assessments/{assessment_id}/workflow', json={'image_ids': [image_id], 'vehicle': vehicle_payload})
    print('Workflow 1:', response1.status_code)
    body1 = response1.json()
    print('Status 1:', body1['status'])
    for stage_name, stage in body1['stages'].items():
        print('  ' + stage_name + ': ' + stage['status'])
    
    # Check stored report
    from app.db.models import AssessmentReport
    from sqlalchemy import select
    from uuid import UUID
    with Session(engine) as session:
        report = session.scalar(select(AssessmentReport).where(AssessmentReport.assessment_id == UUID(assessment_id)))
        if report and report.report_data:
            print('\nStored report stages:')
            for k, v in report.report_data.get('stages', {}).items():
                print(f'  {k}: {v.get("status") if isinstance(v, dict) else v}')
    
    # Run workflow second time
    response2 = client.post(f'/api/v1/assessments/{assessment_id}/workflow', json={'image_ids': [image_id], 'vehicle': vehicle_payload})
    print('\nWorkflow 2:', response2.status_code)
    body2 = response2.json()
    print('Status 2:', body2['status'])
    for stage_name, stage in body2['stages'].items():
        print('  ' + stage_name + ': ' + stage['status'])
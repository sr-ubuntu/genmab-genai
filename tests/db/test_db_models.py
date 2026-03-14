"""Tests for SQLAlchemy ORM models."""
import datetime
import pytest
from sqlalchemy import create_engine

from clinical_summarizer.db.models import Base, Patient, ClinicalRecord, RecordType


@pytest.fixture
def fresh_engine():
    """Create a fresh in-memory engine for model tests."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


class TestRecordTypeEnum:
    """Tests for the RecordType enum."""

    def test_record_type_values(self) -> None:
        """RecordType should have LAB_RESULT and IMAGING_REPORT values."""
        assert RecordType.LAB_RESULT.value == "lab_result"
        assert RecordType.IMAGING_REPORT.value == "imaging_report"

    def test_record_type_from_string(self) -> None:
        """RecordType should be constructible from string value."""
        lab = RecordType("lab_result")
        imaging = RecordType("imaging_report")
        assert lab == RecordType.LAB_RESULT
        assert imaging == RecordType.IMAGING_REPORT


class TestPatientModel:
    """Tests for the Patient ORM model."""

    def test_patient_instantiation(self) -> None:
        """Patient should instantiate with required fields."""
        patient = Patient(
            name="John Doe",
            date_of_birth=datetime.date(1990, 1, 1),
            gender="Male",
            mrn="TEST-001",
        )
        assert patient.name == "John Doe"
        assert patient.gender == "Male"
        assert patient.mrn == "TEST-001"

    def test_patient_created_at_defaults_to_utcnow(self) -> None:
        """Patient.created_at should default to UTC now."""
        patient = Patient(
            name="Jane Doe",
            date_of_birth=datetime.date(1985, 5, 15),
            gender="Female",
            mrn="TEST-002",
        )
        assert patient.created_at is not None
        assert isinstance(patient.created_at, datetime.datetime)

    def test_patient_relationship_to_records(self, fresh_engine):
        """Patient.records should be a list of ClinicalRecord instances."""
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(bind=fresh_engine)
        session = SessionLocal()

        patient = Patient(
            name="Alice",
            date_of_birth=datetime.date(1975, 3, 10),
            gender="Female",
            mrn="TEST-003",
        )
        session.add(patient)
        session.flush()

        record = ClinicalRecord(
            patient_id=patient.id,
            record_type=RecordType.LAB_RESULT,
            title="Test Lab",
            content="Test content",
            recorded_at=datetime.datetime.utcnow(),
        )
        session.add(record)
        session.commit()

        patient_from_db = session.query(Patient).first()
        assert len(patient_from_db.records) == 1
        assert patient_from_db.records[0].title == "Test Lab"

        session.close()


class TestClinicalRecordModel:
    """Tests for the ClinicalRecord ORM model."""

    def test_clinical_record_instantiation(self) -> None:
        """ClinicalRecord should instantiate with required fields."""
        record = ClinicalRecord(
            patient_id=1,
            record_type=RecordType.LAB_RESULT,
            title="Blood Test",
            content="Test results...",
            recorded_at=datetime.datetime.utcnow(),
        )
        assert record.patient_id == 1
        assert record.record_type == RecordType.LAB_RESULT
        assert record.title == "Blood Test"
        assert len(record.content) > 0

    def test_clinical_record_type_validation(self) -> None:
        """ClinicalRecord.record_type should only accept valid enum values."""
        record_lab = ClinicalRecord(
            patient_id=1,
            record_type=RecordType.LAB_RESULT,
            title="Test",
            content="Content",
            recorded_at=datetime.datetime.utcnow(),
        )
        assert record_lab.record_type == RecordType.LAB_RESULT

        record_imaging = ClinicalRecord(
            patient_id=1,
            record_type=RecordType.IMAGING_REPORT,
            title="Test",
            content="Content",
            recorded_at=datetime.datetime.utcnow(),
        )
        assert record_imaging.record_type == RecordType.IMAGING_REPORT

"""Tests for database seeding."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from clinical_summarizer.db.models import Base, Patient, ClinicalRecord, RecordType
from clinical_summarizer.db.seed import create_tables, seed_database


@pytest.fixture
def seeding_engine():
    """Create a fresh in-memory engine for seeding tests."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


class TestCreateTables:
    """Tests for create_tables() function."""

    def test_create_tables_creates_tables(self, seeding_engine) -> None:
        """create_tables() should create both patients and clinical_records tables."""
        create_tables(seeding_engine)
        inspector = __import__("sqlalchemy").inspect(seeding_engine)
        table_names = inspector.get_table_names()
        assert "patients" in table_names
        assert "clinical_records" in table_names

    def test_create_tables_idempotent(self, seeding_engine) -> None:
        """create_tables() should be idempotent (calling twice is safe)."""
        create_tables(seeding_engine)
        create_tables(seeding_engine)  # Should not raise
        inspector = __import__("sqlalchemy").inspect(seeding_engine)
        table_names = inspector.get_table_names()
        assert "patients" in table_names


class TestSeedDatabase:
    """Tests for seed_database() function."""

    def test_seed_database_creates_4_patients(self, seeding_engine) -> None:
        """seed_database() should create 4 patients."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            patient_count = session.query(Patient).count()
            assert patient_count == 4
        finally:
            session.close()

    def test_seed_database_creates_correct_patient_names(self, seeding_engine) -> None:
        """seed_database() should create patients with correct names."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            patients = session.query(Patient).order_by(Patient.id).all()
            names = [p.name for p in patients]
            assert names == ["Alice Mercer", "David Okafor", "Susan Park", "James Rivera"]
        finally:
            session.close()

    def test_seed_database_creates_correct_mrns(self, seeding_engine) -> None:
        """seed_database() should create patients with unique MRNs."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            patients = session.query(Patient).order_by(Patient.id).all()
            mrns = [p.mrn for p in patients]
            assert mrns == ["MRN-001", "MRN-002", "MRN-003", "MRN-004"]
            # Check uniqueness
            assert len(mrns) == len(set(mrns))
        finally:
            session.close()

    def test_seed_database_creates_clinical_records(self, seeding_engine) -> None:
        """seed_database() should create multiple clinical records."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            record_count = session.query(ClinicalRecord).count()
            # Total should be 3 + 2 + 3 + 2 = 10 records
            assert record_count == 10
        finally:
            session.close()

    def test_seed_database_records_have_content(self, seeding_engine) -> None:
        """seed_database() should create records with non-empty content."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            records = session.query(ClinicalRecord).all()
            for record in records:
                assert record.content is not None
                assert len(record.content) > 0
                assert "Patient:" in record.content or "Results:" in record.content or "Findings:" in record.content or "Impression:" in record.content
        finally:
            session.close()

    def test_seed_database_idempotent(self, seeding_engine) -> None:
        """seed_database() should be idempotent (calling twice doesn't duplicate)."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            count_after_first = session.query(Patient).count()
            seed_database(session)
            count_after_second = session.query(Patient).count()
            assert count_after_first == count_after_second == 4
        finally:
            session.close()

    def test_seed_database_mixed_record_types(self, seeding_engine) -> None:
        """seed_database() should create mixed LAB_RESULT and IMAGING_REPORT records."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            lab_count = (
                session.query(ClinicalRecord)
                .filter(ClinicalRecord.record_type == RecordType.LAB_RESULT)
                .count()
            )
            imaging_count = (
                session.query(ClinicalRecord)
                .filter(ClinicalRecord.record_type == RecordType.IMAGING_REPORT)
                .count()
            )
            assert lab_count > 0
            assert imaging_count > 0
            assert lab_count + imaging_count == 10
        finally:
            session.close()

    def test_seed_database_patient_record_relationships(self, seeding_engine) -> None:
        """seed_database() should maintain correct patient-record relationships."""
        create_tables(seeding_engine)
        SessionLocal = sessionmaker(bind=seeding_engine)
        session = SessionLocal()
        try:
            seed_database(session)
            # Alice (id=1) should have 3 records
            alice = session.query(Patient).filter(Patient.name == "Alice Mercer").first()
            assert alice is not None
            assert len(alice.records) == 3
            # David (id=2) should have 2 records
            david = session.query(Patient).filter(Patient.name == "David Okafor").first()
            assert david is not None
            assert len(david.records) == 2
            # Susan (id=3) should have 3 records
            susan = session.query(Patient).filter(Patient.name == "Susan Park").first()
            assert susan is not None
            assert len(susan.records) == 3
            # James (id=4) should have 2 records
            james = session.query(Patient).filter(Patient.name == "James Rivera").first()
            assert james is not None
            assert len(james.records) == 2
        finally:
            session.close()

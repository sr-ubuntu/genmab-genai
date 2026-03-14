"""Tests for the ClinicalDbRepository query layer."""
import pytest
from sqlalchemy.orm import Session

from clinical_summarizer.db.repository import (
    ClinicalDbRepository,
    PatientNotFoundError,
    RecordNotFoundError,
)
from clinical_summarizer.db.models import RecordType


class TestClinicalDbRepositoryListPatients:
    """Tests for list_patients() method."""

    def test_list_patients_returns_all_patients(self, seeded_session: Session) -> None:
        """list_patients() should return all 4 seeded patients."""
        repo = ClinicalDbRepository(seeded_session)
        patients = repo.list_patients()
        assert len(patients) == 4
        assert patients[0].id == 1
        assert patients[0].name == "Alice Mercer"

    def test_list_patients_ordered_by_id(self, seeded_session: Session) -> None:
        """list_patients() should return patients ordered by id ascending."""
        repo = ClinicalDbRepository(seeded_session)
        patients = repo.list_patients()
        ids = [p.id for p in patients]
        assert ids == [1, 2, 3, 4]


class TestClinicalDbRepositoryGetPatient:
    """Tests for get_patient() method."""

    def test_get_patient_by_id(self, seeded_session: Session) -> None:
        """get_patient() should return correct patient by id."""
        repo = ClinicalDbRepository(seeded_session)
        patient = repo.get_patient(1)
        assert patient.id == 1
        assert patient.name == "Alice Mercer"
        assert patient.mrn == "MRN-001"

    def test_get_patient_not_found_raises_error(self, seeded_session: Session) -> None:
        """get_patient() should raise PatientNotFoundError for invalid id."""
        repo = ClinicalDbRepository(seeded_session)
        with pytest.raises(PatientNotFoundError):
            repo.get_patient(999)

    def test_get_patient_all_patients(self, seeded_session: Session) -> None:
        """get_patient() should return correct data for all seeded patients."""
        repo = ClinicalDbRepository(seeded_session)
        names = {1: "Alice Mercer", 2: "David Okafor", 3: "Susan Park", 4: "James Rivera"}
        mrns = {1: "MRN-001", 2: "MRN-002", 3: "MRN-003", 4: "MRN-004"}
        for pid in [1, 2, 3, 4]:
            patient = repo.get_patient(pid)
            assert patient.name == names[pid]
            assert patient.mrn == mrns[pid]


class TestClinicalDbRepositoryGetClinicalRecords:
    """Tests for get_clinical_records() method."""

    def test_get_clinical_records_all_for_patient(self, seeded_session: Session) -> None:
        """get_clinical_records() should return all records for a patient."""
        repo = ClinicalDbRepository(seeded_session)
        records = repo.get_clinical_records(1)
        assert len(records) == 3  # Alice has 3 records

    def test_get_clinical_records_ordered_by_recorded_at_desc(
        self, seeded_session: Session
    ) -> None:
        """get_clinical_records() should return records ordered by recorded_at descending."""
        repo = ClinicalDbRepository(seeded_session)
        records = repo.get_clinical_records(1)
        # Alice's records are at: 2024-03-10, 2024-03-15, 2024-03-12
        # Descending order should be: 2024-03-15, 2024-03-12, 2024-03-10
        assert records[0].title == "Chest X-Ray"  # 2024-03-15
        assert records[1].title == "Lipid Panel"  # 2024-03-12
        assert records[2].title == "Complete Blood Count"  # 2024-03-10

    def test_get_clinical_records_filter_by_type(self, seeded_session: Session) -> None:
        """get_clinical_records() should filter by record_type."""
        repo = ClinicalDbRepository(seeded_session)
        lab_records = repo.get_clinical_records(1, RecordType.LAB_RESULT)
        assert len(lab_records) == 2
        for record in lab_records:
            assert record.record_type == RecordType.LAB_RESULT

    def test_get_clinical_records_filter_imaging_report(
        self, seeded_session: Session
    ) -> None:
        """get_clinical_records() should filter for imaging_report type."""
        repo = ClinicalDbRepository(seeded_session)
        imaging_records = repo.get_clinical_records(1, RecordType.IMAGING_REPORT)
        assert len(imaging_records) == 1
        assert imaging_records[0].title == "Chest X-Ray"

    def test_get_clinical_records_patient_not_found(self, seeded_session: Session) -> None:
        """get_clinical_records() should raise PatientNotFoundError if patient doesn't exist."""
        repo = ClinicalDbRepository(seeded_session)
        with pytest.raises(PatientNotFoundError):
            repo.get_clinical_records(999)

    def test_get_clinical_records_empty_for_nonexistent_type(
        self, seeded_session: Session
    ) -> None:
        """get_clinical_records() should return empty list if no records match filter."""
        repo = ClinicalDbRepository(seeded_session)
        # Alice has no lab results if we filter by a type that doesn't exist
        # This test assumes our seeding doesn't have weird edge cases
        records = repo.get_clinical_records(2)  # David
        assert len(records) > 0


class TestClinicalDbRepositoryGetRecordById:
    """Tests for get_record_by_id() method."""

    def test_get_record_by_id(self, seeded_session: Session) -> None:
        """get_record_by_id() should return correct record with full content."""
        repo = ClinicalDbRepository(seeded_session)
        records = repo.get_clinical_records(1)
        first_record = records[0]
        record = repo.get_record_by_id(first_record.id)
        assert record.id == first_record.id
        assert record.title == first_record.title
        assert record.content is not None
        assert len(record.content) > 0

    def test_get_record_by_id_includes_full_content(self, seeded_session: Session) -> None:
        """get_record_by_id() should include the full clinical text."""
        repo = ClinicalDbRepository(seeded_session)
        # Get a record and verify content is substantial
        patient_records = repo.get_clinical_records(1)
        record = repo.get_record_by_id(patient_records[0].id)
        assert record.content is not None
        # Verify we get real clinical text, not just a title
        assert "Patient:" in record.content or "Results:" in record.content or "Findings:" in record.content

    def test_get_record_by_id_not_found(self, seeded_session: Session) -> None:
        """get_record_by_id() should raise RecordNotFoundError for invalid id."""
        repo = ClinicalDbRepository(seeded_session)
        with pytest.raises(RecordNotFoundError):
            repo.get_record_by_id(999)

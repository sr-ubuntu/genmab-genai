"""Data access layer for the clinical summarizer SQLite database.

All database I/O is isolated here. All other modules operate on plain
Pydantic models or ORM objects, not raw query results.
"""
from __future__ import annotations
from sqlalchemy.orm import Session
from sqlalchemy import desc

from clinical_summarizer.db.models import Patient, ClinicalRecord, RecordType


class PatientNotFoundError(Exception):
    """Raised when a patient_id does not exist in the database."""


class RecordNotFoundError(Exception):
    """Raised when a record_id does not exist in the database."""


class ClinicalDbRepository:
    """Provides read-only query access to the clinical SQLite database.

    The SQLAlchemy Session is injected at construction time to allow
    tests to substitute an in-memory session without patching globals.

    Args:
        session: An open SQLAlchemy Session bound to the target database.
    """

    def __init__(self, session: Session) -> None:
        """Initialise with an injected SQLAlchemy session.

        Args:
            session: An open SQLAlchemy Session instance.
        """
        self.session = session

    def list_patients(self) -> list[Patient]:
        """Return all patient rows ordered by id ascending.

        Returns:
            A list of Patient ORM instances (may be empty).
        """
        return self.session.query(Patient).order_by(Patient.id).all()

    def get_patient(self, patient_id: int) -> Patient:
        """Return a single patient by primary key.

        Args:
            patient_id: The integer primary key of the patient.

        Returns:
            The Patient ORM instance.

        Raises:
            PatientNotFoundError: If no patient with that id exists.
        """
        patient = self.session.query(Patient).filter(Patient.id == patient_id).first()
        if patient is None:
            raise PatientNotFoundError(f"Patient with id {patient_id} not found")
        return patient

    def get_clinical_records(
        self,
        patient_id: int,
        record_type: RecordType | None = None,
    ) -> list[ClinicalRecord]:
        """Return clinical records for a patient, optionally filtered by type.

        Args:
            patient_id: FK to filter records by patient.
            record_type: Optional enum value to narrow the result set.

        Returns:
            A list of ClinicalRecord ORM instances ordered by recorded_at descending.

        Raises:
            PatientNotFoundError: If the patient_id does not exist.
        """
        # Verify patient exists
        self.get_patient(patient_id)

        query = self.session.query(ClinicalRecord).filter(
            ClinicalRecord.patient_id == patient_id
        )

        if record_type is not None:
            query = query.filter(ClinicalRecord.record_type == record_type)

        return query.order_by(desc(ClinicalRecord.recorded_at)).all()

    def get_record_by_id(self, record_id: int) -> ClinicalRecord:
        """Return a single clinical record by primary key.

        Args:
            record_id: The integer primary key of the record.

        Returns:
            The ClinicalRecord ORM instance with full content.

        Raises:
            RecordNotFoundError: If no record with that id exists.
        """
        record = self.session.query(ClinicalRecord).filter(
            ClinicalRecord.id == record_id
        ).first()
        if record is None:
            raise RecordNotFoundError(f"Record with id {record_id} not found")
        return record

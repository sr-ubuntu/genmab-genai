"""SQLAlchemy ORM models for the clinical summarizer database.

Defines the `patients` and `clinical_records` tables with relationships.
"""
from __future__ import annotations
import enum
import datetime
from sqlalchemy import String, Text, ForeignKey, Enum as SAEnum, DateTime, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class RecordType(str, enum.Enum):
    """Allowed values for ClinicalRecord.record_type."""

    LAB_RESULT = "lab_result"
    IMAGING_REPORT = "imaging_report"


class Patient(Base):
    """ORM model representing a patient row.

    Attributes:
        id: Auto-incrementing primary key.
        name: Patient full name.
        date_of_birth: ISO date of birth.
        gender: Biological sex string (e.g. "Male", "Female").
        mrn: Medical record number — unique per patient.
        created_at: UTC timestamp of row insertion.
        records: Back-reference to associated ClinicalRecord rows.
    """

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    date_of_birth: Mapped[datetime.date]
    gender: Mapped[str] = mapped_column(String(50))
    mrn: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    records: Mapped[list[ClinicalRecord]] = relationship(
        "ClinicalRecord", back_populates="patient", cascade="all, delete-orphan"
    )


class ClinicalRecord(Base):
    """ORM model representing a clinical record row.

    Attributes:
        id: Auto-incrementing primary key.
        patient_id: FK to patients.id.
        record_type: Enum — "lab_result" or "imaging_report".
        title: Short descriptive title of the record.
        content: Full raw clinical text.
        recorded_at: UTC timestamp of the clinical event.
        patient: Back-reference to the owning Patient.
    """

    __tablename__ = "clinical_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    record_type: Mapped[RecordType] = mapped_column(SAEnum(RecordType))
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    recorded_at: Mapped[datetime.datetime]
    patient: Mapped[Patient] = relationship("Patient", back_populates="records")

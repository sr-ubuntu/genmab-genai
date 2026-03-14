"""MCP server exposing clinical database tools via fastmcp.

This server provides MCP tools to query patient and clinical record data
from the SQLite database. It's designed to be run as a subprocess with
stdio transport by the agent/API server.

Run with: python mcp_db_server.py
Transport: stdio (consumed as a subprocess by the agent)
"""
from __future__ import annotations
import logging
from contextlib import contextmanager

import fastmcp

from clinical_summarizer.db import get_engine, get_session_factory
from clinical_summarizer.db.repository import (
    ClinicalDbRepository,
    PatientNotFoundError,
    RecordNotFoundError,
)
from clinical_summarizer.db.models import RecordType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-db-server")

mcp = fastmcp.FastMCP("clinical-db")

# Module-level engine and session factory for reuse across tool calls
_engine = None
_session_factory = None


def _init_db():
    """Initialize the database engine and session factory."""
    global _engine, _session_factory
    if _engine is None:
        _engine = get_engine("sqlite:///clinical.db")
        _session_factory = get_session_factory(_engine)


@contextmanager
def _get_session_context():
    """Context manager for database sessions.

    Yields:
        An open SQLAlchemy Session.
    """
    _init_db()
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


@mcp.tool()
def list_patients() -> list[dict]:
    """Return all patients in the database.

    Returns a list of dicts, each with patient identifiers and metadata.

    Returns:
        A list of dicts with keys: id, name, date_of_birth, gender, mrn.
    """
    with _get_session_context() as session:
        repo = ClinicalDbRepository(session)
        patients = repo.list_patients()
        return [
            {
                "id": p.id,
                "name": p.name,
                "date_of_birth": str(p.date_of_birth),
                "gender": p.gender,
                "mrn": p.mrn,
            }
            for p in patients
        ]


@mcp.tool()
def get_patient(patient_id: int) -> dict:
    """Return details for a single patient.

    Args:
        patient_id: The integer primary key of the patient.

    Returns:
        A dict with keys: id, name, date_of_birth, gender, mrn, created_at.

    Raises:
        ValueError: If the patient does not exist.
    """
    with _get_session_context() as session:
        repo = ClinicalDbRepository(session)
        try:
            patient = repo.get_patient(patient_id)
            return {
                "id": patient.id,
                "name": patient.name,
                "date_of_birth": str(patient.date_of_birth),
                "gender": patient.gender,
                "mrn": patient.mrn,
                "created_at": patient.created_at.isoformat(),
            }
        except PatientNotFoundError as e:
            raise ValueError(str(e)) from e


@mcp.tool()
def get_clinical_records(
    patient_id: int, record_type: str | None = None
) -> list[dict]:
    """Return clinical records for a patient, optionally filtered by type.

    Args:
        patient_id: The patient whose records to retrieve.
        record_type: Optional filter — "lab_result" or "imaging_report".

    Returns:
        A list of dicts with keys: id, patient_id, record_type, title, recorded_at.
        Note: content is excluded for brevity; use get_record_by_id for full text.

    Raises:
        ValueError: If patient_id does not exist or record_type is invalid.
    """
    with _get_session_context() as session:
        repo = ClinicalDbRepository(session)
        try:
            # Convert record_type string to enum if provided
            enum_type = None
            if record_type is not None:
                try:
                    enum_type = RecordType(record_type)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid record_type '{record_type}'. Must be 'lab_result' or 'imaging_report'."
                    ) from e

            records = repo.get_clinical_records(patient_id, enum_type)
            return [
                {
                    "id": r.id,
                    "patient_id": r.patient_id,
                    "record_type": r.record_type.value,
                    "title": r.title,
                    "recorded_at": r.recorded_at.isoformat(),
                }
                for r in records
            ]
        except PatientNotFoundError as e:
            raise ValueError(str(e)) from e


@mcp.tool()
def get_record_by_id(record_id: int) -> dict:
    """Return a single clinical record including its full content.

    Args:
        record_id: The integer primary key of the record.

    Returns:
        A dict with keys: id, patient_id, record_type, title, content, recorded_at.

    Raises:
        ValueError: If the record does not exist.
    """
    with _get_session_context() as session:
        repo = ClinicalDbRepository(session)
        try:
            record = repo.get_record_by_id(record_id)
            return {
                "id": record.id,
                "patient_id": record.patient_id,
                "record_type": record.record_type.value,
                "title": record.title,
                "content": record.content,
                "recorded_at": record.recorded_at.isoformat(),
            }
        except RecordNotFoundError as e:
            raise ValueError(str(e)) from e


if __name__ == "__main__":
    logger.info("Starting clinical database MCP server")
    mcp.run()

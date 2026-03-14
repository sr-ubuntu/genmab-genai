"""Tests for the MCP database server."""
import datetime
from unittest.mock import MagicMock, patch
import pytest

from clinical_summarizer.db.models import Patient, ClinicalRecord, RecordType
from clinical_summarizer.db.repository import PatientNotFoundError, RecordNotFoundError


@pytest.fixture
def mock_patient():
    """Create a mock Patient ORM instance."""
    patient = MagicMock(spec=Patient)
    patient.id = 1
    patient.name = "Alice Mercer"
    patient.date_of_birth = datetime.date(1968, 4, 12)
    patient.gender = "Female"
    patient.mrn = "MRN-001"
    patient.created_at = datetime.datetime(2024, 3, 1, 10, 0, 0)
    return patient


@pytest.fixture
def mock_record():
    """Create a mock ClinicalRecord ORM instance."""
    record = MagicMock(spec=ClinicalRecord)
    record.id = 1
    record.patient_id = 1
    record.record_type = RecordType.LAB_RESULT
    record.title = "Blood Test"
    record.content = "Patient ABC, Results: WBC 7.2, RBC 4.8..."
    record.recorded_at = datetime.datetime(2024, 3, 10, 9, 30, 0)
    return record


class TestMcpDbServerListPatients:
    """Tests for the list_patients MCP tool."""

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_list_patients_returns_list_of_dicts(
        self, mock_session_context, mock_repo_class, mock_patient
    ) -> None:
        """list_patients() should return a list of patient dicts."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.list_patients.return_value = [mock_patient]
        mock_repo_class.return_value = mock_repo

        # Import and call the tool
        import mcp_db_server

        result = mcp_db_server.list_patients()

        # Verify result shape
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Alice Mercer"
        assert result[0]["mrn"] == "MRN-001"


class TestMcpDbServerGetPatient:
    """Tests for the get_patient MCP tool."""

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_patient_returns_patient_dict(
        self, mock_session_context, mock_repo_class, mock_patient
    ) -> None:
        """get_patient() should return a dict with patient details."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_patient.return_value = mock_patient
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        result = mcp_db_server.get_patient(1)

        assert result["id"] == 1
        assert result["name"] == "Alice Mercer"
        assert result["mrn"] == "MRN-001"
        assert "created_at" in result

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_patient_not_found_raises_value_error(
        self, mock_session_context, mock_repo_class
    ) -> None:
        """get_patient() should raise ValueError if patient doesn't exist."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_patient.side_effect = PatientNotFoundError("Patient not found")
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        with pytest.raises(ValueError):
            mcp_db_server.get_patient(999)


class TestMcpDbServerGetClinicalRecords:
    """Tests for the get_clinical_records MCP tool."""

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_clinical_records_returns_list(
        self, mock_session_context, mock_repo_class, mock_record
    ) -> None:
        """get_clinical_records() should return a list of record dicts."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_clinical_records.return_value = [mock_record]
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        result = mcp_db_server.get_clinical_records(1)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["record_type"] == "lab_result"
        assert result[0]["title"] == "Blood Test"
        # Content should be excluded from list view
        assert "content" not in result[0]

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_clinical_records_with_filter(
        self, mock_session_context, mock_repo_class, mock_record
    ) -> None:
        """get_clinical_records() should support record_type filtering."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_clinical_records.return_value = [mock_record]
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        result = mcp_db_server.get_clinical_records(1, "lab_result")

        # Verify the repo was called with the correct enum
        call_args = mock_repo.get_clinical_records.call_args
        assert call_args[0][0] == 1  # patient_id
        assert call_args[0][1] == RecordType.LAB_RESULT  # converted to enum

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_clinical_records_invalid_type_raises_error(
        self, mock_session_context, mock_repo_class
    ) -> None:
        """get_clinical_records() should raise ValueError for invalid record_type."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        with pytest.raises(ValueError, match="Invalid record_type"):
            mcp_db_server.get_clinical_records(1, "invalid_type")


class TestMcpDbServerGetRecordById:
    """Tests for the get_record_by_id MCP tool."""

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_record_by_id_returns_full_record(
        self, mock_session_context, mock_repo_class, mock_record
    ) -> None:
        """get_record_by_id() should return a record dict with full content."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_record_by_id.return_value = mock_record
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        result = mcp_db_server.get_record_by_id(1)

        assert result["id"] == 1
        assert result["title"] == "Blood Test"
        # Content SHOULD be included in full record view
        assert result["content"] == "Patient ABC, Results: WBC 7.2, RBC 4.8..."
        assert result["record_type"] == "lab_result"

    @patch("mcp_db_server.ClinicalDbRepository")
    @patch("mcp_db_server._get_session_context")
    def test_get_record_by_id_not_found_raises_value_error(
        self, mock_session_context, mock_repo_class
    ) -> None:
        """get_record_by_id() should raise ValueError if record doesn't exist."""
        mock_session = MagicMock()
        mock_session_context.return_value.__enter__.return_value = mock_session
        mock_session_context.return_value.__exit__.return_value = None

        mock_repo = MagicMock()
        mock_repo.get_record_by_id.side_effect = RecordNotFoundError("Record not found")
        mock_repo_class.return_value = mock_repo

        import mcp_db_server

        with pytest.raises(ValueError):
            mcp_db_server.get_record_by_id(999)

"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest, ErrorResponse


class TestClinicalSummaryRequest:
    """Tests for the ClinicalSummaryRequest model."""

    def test_valid_request(self) -> None:
        """Happy path: all valid fields are accepted."""
        req = ClinicalSummaryRequest(text="Patient has elevated BP.", request_id="req-1")
        assert req.text == "Patient has elevated BP."
        assert req.request_id == "req-1"

    def test_text_is_stripped(self) -> None:
        """Leading and trailing whitespace is removed from text."""
        req = ClinicalSummaryRequest(text="  Some note.  ", request_id="r1")
        assert req.text == "Some note."

    def test_request_id_is_stripped(self) -> None:
        """Leading and trailing whitespace is removed from request_id."""
        req = ClinicalSummaryRequest(text="note", request_id="  id-1  ")
        assert req.request_id == "id-1"

    def test_empty_text_raises(self) -> None:
        """Empty text after stripping raises a ValidationError."""
        with pytest.raises(ValidationError, match="blank"):
            ClinicalSummaryRequest(text="   ", request_id="r1")

    def test_text_too_long_raises(self) -> None:
        """Text exceeding 50,000 characters raises a ValidationError."""
        with pytest.raises(ValidationError, match="maximum of 50000"):
            ClinicalSummaryRequest(text="x" * 50_001, request_id="r1")

    def test_text_at_max_length_is_accepted(self) -> None:
        """Text of exactly 50,000 characters is valid."""
        req = ClinicalSummaryRequest(text="x" * 50_000, request_id="r1")
        assert len(req.text) == 50_000

    def test_request_id_required(self) -> None:
        """Omitting request_id raises a ValidationError."""
        with pytest.raises(ValidationError):
            ClinicalSummaryRequest(text="note")  # type: ignore[call-arg]

    def test_text_required(self) -> None:
        """Omitting text raises a ValidationError."""
        with pytest.raises(ValidationError):
            ClinicalSummaryRequest(request_id="r1")  # type: ignore[call-arg]


class TestClinicalSummary:
    """Tests for the ClinicalSummary model."""

    def test_valid_summary(self) -> None:
        """Happy path: all fields populated correctly."""
        summary = ClinicalSummary(
            document_type="blood_test",
            summary="Short overview.",
            key_findings=["Finding A", "Finding B"],
            abnormal_results=["Abnormal A"],
            recommended_follow_up=["Follow up A"],
        )
        assert summary.document_type == "blood_test"
        assert len(summary.key_findings) == 2

    def test_empty_lists_accepted(self) -> None:
        """All list fields may be empty."""
        summary = ClinicalSummary(
            document_type="pcp_note",
            summary="Normal visit.",
            key_findings=[],
            abnormal_results=[],
            recommended_follow_up=[],
        )
        assert summary.key_findings == []
        assert summary.abnormal_results == []
        assert summary.recommended_follow_up == []

    def test_round_trip_serialization(self) -> None:
        """model_dump_json / model_validate round-trip preserves all data."""
        original = ClinicalSummary(
            document_type="colonoscopy_report",
            summary="Two polyps found and removed.",
            key_findings=["Polyp in sigmoid colon"],
            abnormal_results=["Tubular adenoma"],
            recommended_follow_up=["Repeat colonoscopy in 3 years"],
        )
        restored = ClinicalSummary.model_validate_json(original.model_dump_json())
        assert restored == original

    def test_missing_field_raises(self) -> None:
        """Omitting a required field raises a ValidationError."""
        with pytest.raises(ValidationError):
            ClinicalSummary(  # type: ignore[call-arg]
                summary="Missing document_type.",
                key_findings=[],
                abnormal_results=[],
                recommended_follow_up=[],
            )


class TestErrorResponse:
    """Tests for the ErrorResponse model."""

    def test_valid_with_request_id(self) -> None:
        """ErrorResponse with all fields serialises correctly."""
        resp = ErrorResponse(error="ValidationError", message="Bad input", request_id="r1")
        assert resp.request_id == "r1"

    def test_request_id_optional(self) -> None:
        """request_id defaults to None when omitted."""
        resp = ErrorResponse(error="BedrockServiceError", message="Upstream failure")
        assert resp.request_id is None

    def test_serialization(self) -> None:
        """model_dump_json produces valid JSON with expected keys."""
        import json

        resp = ErrorResponse(error="RateLimitError", message="Throttled", request_id="r2")
        data = json.loads(resp.model_dump_json())
        assert data["error"] == "RateLimitError"
        assert data["request_id"] == "r2"

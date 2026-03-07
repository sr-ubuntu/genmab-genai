"""Integration tests for the Lambda handler in main.py."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import main as lambda_module
from clinical_summarizer.exceptions import (
    BedrockServiceError,
    RateLimitError,
    SummaryParsingError,
)
from clinical_summarizer.models import ClinicalSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(body: dict | str | None = None) -> dict:
    """Build a minimal API Gateway proxy event."""
    if isinstance(body, dict):
        body = json.dumps(body)
    return {"body": body, "httpMethod": "POST"}


def _make_context(request_id: str = "lambda-req-123") -> SimpleNamespace:
    """Build a minimal Lambda context object."""
    return SimpleNamespace(aws_request_id=request_id)


def _valid_summary() -> ClinicalSummary:
    return ClinicalSummary(
        document_type="pcp_note",
        summary="Patient seen for fatigue and elevated blood pressure.",
        key_findings=["Fatigue", "BP 152/94"],
        abnormal_results=["Elevated BP"],
        recommended_follow_up=["Start lisinopril"],
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHandlerHappyPath:
    """Tests for successful 200 responses."""

    def test_returns_200_on_valid_request(self) -> None:
        """Valid request returns HTTP 200."""
        with patch.object(lambda_module._service, "summarize", return_value=_valid_summary()):
            event = _make_event({"text": "Patient note here.", "request_id": "req-1"})
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 200

    def test_response_body_is_valid_json(self) -> None:
        """Response body is valid JSON."""
        with patch.object(lambda_module._service, "summarize", return_value=_valid_summary()):
            event = _make_event({"text": "Patient note.", "request_id": "req-1"})
            response = lambda_module.handler(event, _make_context())
        data = json.loads(response["body"])
        assert "document_type" in data

    def test_response_contains_all_summary_fields(self) -> None:
        """All ClinicalSummary fields are present in the response body."""
        with patch.object(lambda_module._service, "summarize", return_value=_valid_summary()):
            event = _make_event({"text": "Patient note.", "request_id": "req-1"})
            response = lambda_module.handler(event, _make_context())
        data = json.loads(response["body"])
        expected = ("document_type", "summary", "key_findings", "abnormal_results", "recommended_follow_up")  # noqa: E501
        for field in expected:
            assert field in data, f"Field '{field}' missing from response"

    def test_cors_headers_present(self) -> None:
        """CORS headers are included in a successful response."""
        with patch.object(lambda_module._service, "summarize", return_value=_valid_summary()):
            event = _make_event({"text": "Patient note.", "request_id": "req-1"})
            response = lambda_module.handler(event, _make_context())
        assert response["headers"]["Access-Control-Allow-Origin"] == "*"


# ---------------------------------------------------------------------------
# 400 / 422 validation errors
# ---------------------------------------------------------------------------


class TestHandlerValidationErrors:
    """Tests for 400 and 422 error responses."""

    def test_missing_body_returns_400(self) -> None:
        """No body in the event returns HTTP 400."""
        response = lambda_module.handler({"httpMethod": "POST"}, _make_context())
        assert response["statusCode"] == 400

    def test_invalid_json_body_returns_400(self) -> None:
        """Malformed JSON body returns HTTP 400."""
        event = {"body": "{not valid json}", "httpMethod": "POST"}
        response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 400

    def test_empty_text_returns_422(self) -> None:
        """Empty clinical text after stripping returns HTTP 422."""
        event = _make_event({"text": "   ", "request_id": "r1"})
        response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 422

    def test_missing_text_field_returns_422(self) -> None:
        """Request body missing the text field returns HTTP 422."""
        event = _make_event({"request_id": "r1"})
        response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 422

    def test_missing_request_id_returns_422(self) -> None:
        """Request body missing request_id returns HTTP 422."""
        event = _make_event({"text": "Patient note."})
        response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 422

    def test_error_body_contains_error_field(self) -> None:
        """Error response body includes an 'error' field."""
        event = _make_event({"request_id": "r1"})
        response = lambda_module.handler(event, _make_context())
        data = json.loads(response["body"])
        assert "error" in data

    def test_cors_headers_on_error(self) -> None:
        """CORS headers are present on error responses."""
        event = _make_event({"request_id": "r1"})
        response = lambda_module.handler(event, _make_context())
        assert response["headers"]["Access-Control-Allow-Origin"] == "*"


# ---------------------------------------------------------------------------
# Upstream / service errors
# ---------------------------------------------------------------------------


class TestHandlerUpstreamErrors:
    """Tests for 429, 500, and 502 error responses."""

    def test_rate_limit_returns_429(self) -> None:
        """RateLimitError from the service returns HTTP 429."""
        err = RateLimitError("Throttled")
        with patch.object(lambda_module._service, "summarize", side_effect=err):
            event = _make_event({"text": "note", "request_id": "r1"})
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 429

    def test_bedrock_service_error_returns_502(self) -> None:
        """BedrockServiceError from the service returns HTTP 502."""
        with patch.object(
            lambda_module._service, "summarize", side_effect=BedrockServiceError("Bedrock down")
        ):
            event = _make_event({"text": "note", "request_id": "r1"})
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 502

    def test_parsing_error_returns_500(self) -> None:
        """SummaryParsingError from the service returns HTTP 500."""
        with patch.object(
            lambda_module._service, "summarize", side_effect=SummaryParsingError("Bad JSON")
        ):
            event = _make_event({"text": "note", "request_id": "r1"})
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 500

    def test_unexpected_exception_returns_500(self) -> None:
        """Unexpected exceptions return HTTP 500."""
        with patch.object(
            lambda_module._service, "summarize", side_effect=RuntimeError("boom")
        ):
            event = _make_event({"text": "note", "request_id": "r1"})
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 500

    def test_dict_body_is_accepted(self) -> None:
        """Pre-parsed dict body (local test invocation) is handled correctly."""
        with patch.object(lambda_module._service, "summarize", return_value=_valid_summary()):
            event = {"body": {"text": "Patient note.", "request_id": "r1"}, "httpMethod": "POST"}
            response = lambda_module.handler(event, _make_context())
        assert response["statusCode"] == 200

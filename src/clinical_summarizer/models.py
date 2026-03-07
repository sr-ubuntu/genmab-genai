"""Pydantic v2 data models for the clinical summarization API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

_MAX_TEXT_LENGTH = 50_000


class ClinicalSummaryRequest(BaseModel):
    """Inbound payload for the summarization endpoint.

    Attributes:
        text: The raw clinical document to summarise (1–50,000 characters).
        request_id: Caller-supplied identifier used for log correlation.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    request_id: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only clinical text."""
        if not value:
            raise ValueError("Clinical text must not be blank.")
        return value

    @field_validator("text")
    @classmethod
    def text_length_within_bounds(cls, value: str) -> str:
        """Enforce the maximum character limit to control Bedrock token usage."""
        if len(value) > _MAX_TEXT_LENGTH:
            raise ValueError(
                f"Clinical text exceeds the maximum of {_MAX_TEXT_LENGTH} characters "
                f"(received {len(value)})."
            )
        return value


class ClinicalSummary(BaseModel):
    """Structured summary produced from a clinical document.

    All fields are populated exclusively from the source text by the AI model;
    no values are fabricated or inferred from external knowledge.

    Attributes:
        document_type: Auto-detected category of the clinical document
            (e.g. ``"pcp_note"``, ``"blood_test"``, ``"colonoscopy_report"``).
        summary: Concise 2-3 sentence plain-English overview of the document.
        key_findings: List of notable clinical observations or measurements.
        abnormal_results: Subset of findings that are outside normal ranges
            or clinically significant.
        recommended_follow_up: Actionable next steps explicitly stated in the document.
    """

    document_type: str
    summary: str
    key_findings: list[str]
    abnormal_results: list[str]
    recommended_follow_up: list[str]


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all non-2xx responses.

    Attributes:
        error: Exception class name (e.g. ``"ValidationError"``).
        message: Human-readable description of the failure.
        request_id: Caller-supplied request identifier, when available.
    """

    error: str
    message: str
    request_id: str | None = None

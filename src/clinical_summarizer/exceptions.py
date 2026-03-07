"""Custom exception hierarchy for the clinical summarization service."""

from __future__ import annotations


class ClinicalSummarizerError(Exception):
    """Base exception for all clinical summarizer errors.

    Attributes:
        message: Human-readable description of the error.
        context: Optional structured metadata for logging and debugging.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        """Initialise with a message and optional context dict."""
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted string including context when present."""
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


class ValidationError(ClinicalSummarizerError):
    """Raised when the incoming request fails validation.

    Examples: empty clinical text, missing required fields, malformed JSON body.
    """


class BedrockServiceError(ClinicalSummarizerError):
    """Raised when an Amazon Bedrock API call fails.

    Wraps the underlying boto3 ClientError and preserves the HTTP status code
    in the context dict under the key ``status_code``.
    """


class RateLimitError(BedrockServiceError):
    """Raised when Bedrock returns a ThrottlingException.

    Signals the caller to back off and retry after a delay.
    """


class SummaryParsingError(ClinicalSummarizerError):
    """Raised when the Bedrock response cannot be parsed into a ClinicalSummary.

    Occurs when the model returns malformed JSON or a JSON object that does not
    conform to the expected schema.
    """

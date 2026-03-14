"""HTTP client for the AWS API Gateway /summarize endpoint.

This module provides an integration point with the deployed AWS Lambda
clinical summarizer. Instead of calling the LLM directly, it makes an
HTTP POST request to the API Gateway endpoint, which internally handles
prompt building, Bedrock invocation, and response parsing.
"""
from __future__ import annotations
import httpx

from clinical_summarizer.models import ClinicalSummary
from clinical_summarizer.exceptions import (
    BedrockServiceError,
    RateLimitError,
    ValidationError,
    SummaryParsingError,
)


class ApiGatewayClient:
    """HTTP client for the AWS API Gateway /summarize endpoint.

    This client calls the full API Gateway → Lambda → Bedrock pipeline,
    bypassing local prompt building and parsing (handled server-side).
    Reuses the existing exception hierarchy to map HTTP errors to business
    exceptions, so callers can handle API Gateway failures the same way as
    direct Bedrock errors.

    Args:
        base_url: Full URL of the POST /summarize endpoint.
                  e.g. "https://abc123.execute-api.us-east-1.amazonaws.com/prod/summarize"
        timeout: HTTP request timeout in seconds (default 30.0).
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """Initialise with the API Gateway endpoint URL.

        Args:
            base_url: The full URL of the /summarize endpoint.
            timeout: HTTP request timeout (default 30.0 seconds).
        """
        self._base_url = base_url.rstrip("/")  # Remove trailing slash if present
        self._timeout = timeout

    def summarize(self, text: str, request_id: str) -> ClinicalSummary:
        """POST text to the API Gateway and return the parsed ClinicalSummary.

        Args:
            text: Raw clinical document text.
            request_id: Correlation ID for logging.

        Returns:
            Parsed ClinicalSummary from the API response.

        Raises:
            ValidationError: If the request is invalid (text blank, exceeds length, etc.).
            RateLimitError: If the API Gateway returns HTTP 429 (rate limited).
            BedrockServiceError: On HTTP 5xx, network errors, or other API failures.
            SummaryParsingError: If the response JSON can't be parsed as ClinicalSummary.
        """
        payload = {
            "text": text,
            "request_id": request_id,
        }

        try:
            # Make the HTTP POST request to the API Gateway
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(self._base_url, json=payload)

            # Handle different HTTP status codes
            if response.status_code == 429:
                # Rate limited by API Gateway or Lambda
                raise RateLimitError(
                    message="API Gateway rate limited (HTTP 429)",
                    context={"request_id": request_id, "status_code": 429},
                )

            if response.status_code in (400, 422):
                # Validation error (bad input, text too long, etc.)
                try:
                    error_json = response.json()
                    raise ValidationError(
                        message=error_json.get("message", f"HTTP {response.status_code}"),
                        context={"request_id": request_id, "status_code": response.status_code},
                    )
                except Exception:
                    raise ValidationError(
                        message=f"Invalid request (HTTP {response.status_code})",
                        context={"request_id": request_id, "status_code": response.status_code},
                    ) from None

            if response.status_code >= 500:
                # Server error (Lambda failure, Bedrock error, etc.)
                raise BedrockServiceError(
                    message="API Gateway returned server error",
                    context={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "url": self._base_url,
                    },
                )

            if response.status_code != 200:
                # Unexpected status code
                raise BedrockServiceError(
                    message=f"Unexpected HTTP status {response.status_code}",
                    context={
                        "request_id": request_id,
                        "status_code": response.status_code,
                    },
                )

            # Parse the successful response (200)
            try:
                response_json = response.json()
                summary = ClinicalSummary.model_validate(response_json)
                return summary
            except ValueError as e:
                # Pydantic validation error — response shape doesn't match ClinicalSummary
                raise SummaryParsingError(
                    message=f"Response JSON does not match ClinicalSummary schema: {str(e)}",
                    context={
                        "request_id": request_id,
                        "response_body": response.text[:500],  # Include truncated response
                    },
                ) from e

        except httpx.TimeoutException as e:
            # Network timeout
            raise BedrockServiceError(
                message="API Gateway request timed out",
                context={"request_id": request_id, "timeout": self._timeout},
            ) from e

        except httpx.NetworkError as e:
            # Network error (connection refused, DNS failure, etc.)
            raise BedrockServiceError(
                message="Network error calling API Gateway",
                context={"request_id": request_id, "url": self._base_url},
            ) from e

        except (ValidationError, RateLimitError, BedrockServiceError, SummaryParsingError):
            # Re-raise our custom exceptions as-is
            raise

        except Exception as e:
            # Unexpected error (e.g., invalid JSON response)
            raise BedrockServiceError(
                message="Unexpected error calling API Gateway",
                context={"request_id": request_id, "error": str(e)},
            ) from e

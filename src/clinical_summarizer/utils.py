"""Shared helpers for Lambda event parsing and HTTP response construction."""

from __future__ import annotations

import json
import os

from clinical_summarizer.exceptions import ClinicalSummarizerError, ValidationError
from clinical_summarizer.models import ClinicalSummary, ErrorResponse

_DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

_CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def parse_api_gateway_body(event: dict) -> dict:
    """Extract and JSON-parse the body from an API Gateway proxy event.

    Handles both string bodies (the default from API Gateway) and pre-parsed
    dict bodies (common in local test invocations).

    Args:
        event: The raw Lambda event dict from API Gateway.

    Returns:
        The parsed request body as a plain dict.

    Raises:
        ValidationError: If the body is absent or contains malformed JSON.
    """
    body = event.get("body")
    if body is None:
        raise ValidationError(
            "Request body is missing.",
            context={"event_keys": list(event.keys())},
        )
    if isinstance(body, dict):
        return body
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "Request body is not valid JSON.",
            context={"json_error": str(exc)},
        ) from exc


def build_success_response(summary: ClinicalSummary) -> dict:
    """Build an API Gateway proxy response for a successful summarization.

    Args:
        summary: The structured clinical summary to return.

    Returns:
        A dict conforming to the API Gateway Lambda proxy integration response
        format, with CORS headers and the summary serialised as JSON.
    """
    return {
        "statusCode": 200,
        "headers": _CORS_HEADERS,
        "body": summary.model_dump_json(),
    }


def build_error_response(
    error: ClinicalSummarizerError,
    status_code: int,
    request_id: str | None,
) -> dict:
    """Build an API Gateway proxy response for an error condition.

    Args:
        error: The exception that caused the failure.
        status_code: The HTTP status code to return.
        request_id: The caller-supplied or Lambda-assigned request identifier.

    Returns:
        A dict conforming to the API Gateway Lambda proxy integration response
        format, with CORS headers and an :class:`ErrorResponse` body.
    """
    envelope = ErrorResponse(
        error=type(error).__name__,
        message=error.message,
        request_id=request_id,
    )
    return {
        "statusCode": status_code,
        "headers": _CORS_HEADERS,
        "body": envelope.model_dump_json(),
    }


def get_model_id_from_env() -> str:
    """Read the Bedrock model ID from the environment.

    Returns:
        The value of ``BEDROCK_MODEL_ID`` if set, otherwise the default
        Claude 3 Haiku model identifier.
    """
    return os.environ.get("BEDROCK_MODEL_ID", _DEFAULT_MODEL_ID)

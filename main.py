"""AWS Lambda entry point for the clinical text summarization API.

Exposes a single ``handler`` function that is invoked by API Gateway.
All heavy dependencies (boto3 client, service graph) are constructed once at
module load time and reused across warm Lambda invocations to minimise latency.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from pydantic import ValidationError as PydanticValidationError

from clinical_summarizer.core import PromptBuilder, SummaryParser
from clinical_summarizer.exceptions import (
    BedrockServiceError,
    ClinicalSummarizerError,
    RateLimitError,
    SummaryParsingError,
    ValidationError,
)
from clinical_summarizer.models import ClinicalSummaryRequest
from clinical_summarizer.bedrock_client import BedrockRepository
from clinical_summarizer.services import SummarizationService
from clinical_summarizer.utils import (
    build_error_response,
    build_success_response,
    get_model_id_from_env,
    parse_api_gateway_body,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("clinical-summarizer")

# ---------------------------------------------------------------------------
# Service singleton — built once on cold start, reused on warm invocations
# ---------------------------------------------------------------------------


def _build_service() -> SummarizationService:
    """Wire all collaborators and return the configured service.

    Reads configuration from environment variables so the same deployment
    package can target different models or regions without code changes.
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = get_model_id_from_env()

    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    repository = BedrockRepository(bedrock_client=bedrock_client, model_id=model_id)
    prompt_builder = PromptBuilder()
    summary_parser = SummaryParser()

    return SummarizationService(
        repository=repository,
        prompt_builder=prompt_builder,
        summary_parser=summary_parser,
        logger=_logger,
    )


_service = _build_service()

# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda handler for the POST /summarize endpoint.

    Parses the API Gateway proxy event, validates the request payload,
    runs the summarization pipeline, and returns a correctly formatted
    API Gateway proxy response.

    HTTP status code mapping:
    - 200: Successful summarization.
    - 400: Missing or malformed request body.
    - 422: Request payload fails field-level validation.
    - 429: Bedrock rate limit exceeded.
    - 500: Unexpected error or model response parsing failure.
    - 502: Bedrock service error (upstream failure).

    Args:
        event: API Gateway Lambda proxy integration event.
        context: Lambda runtime context (provides ``aws_request_id``).

    Returns:
        API Gateway proxy integration response dict.
    """
    lambda_request_id: str | None = getattr(context, "aws_request_id", None)
    caller_request_id: str | None = None

    try:
        body = parse_api_gateway_body(event)
        request = ClinicalSummaryRequest.model_validate(body)
        caller_request_id = request.request_id

        summary = _service.summarize(request)
        return build_success_response(summary)

    except PydanticValidationError as exc:
        _logger.warning(
            "Request validation failed",
            extra={"errors": exc.errors(), "request_id": caller_request_id},
        )
        wrapped = ValidationError(str(exc))
        return build_error_response(wrapped, 422, caller_request_id or lambda_request_id)

    except ValidationError as exc:
        _logger.warning("Validation error", extra={"error": str(exc)})
        return build_error_response(exc, 400, lambda_request_id)

    except RateLimitError as exc:
        _logger.warning("Bedrock rate limit hit", extra={"error": str(exc)})
        return build_error_response(exc, 429, caller_request_id or lambda_request_id)

    except BedrockServiceError as exc:
        _logger.error("Bedrock service error", extra={"error": str(exc)})
        return build_error_response(exc, 502, caller_request_id or lambda_request_id)

    except SummaryParsingError as exc:
        _logger.error("Failed to parse Bedrock response", extra={"error": str(exc)})
        return build_error_response(exc, 500, caller_request_id or lambda_request_id)

    except Exception as exc:
        _logger.exception("Unexpected error in Lambda handler")
        fallback = ClinicalSummarizerError(str(exc))
        return build_error_response(fallback, 500, lambda_request_id)

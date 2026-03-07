"""Amazon Bedrock integration layer.

All AWS I/O is isolated here. Every other module in the package operates on
plain Python objects and has no knowledge of boto3 or AWS APIs.
"""

from __future__ import annotations

import json
from typing import Any

import botocore.exceptions

from clinical_summarizer.exceptions import BedrockServiceError, RateLimitError

_ANTHROPIC_VERSION = "bedrock-2023-05-31"
_THROTTLING_ERROR_CODE = "ThrottlingException"


class BedrockRepository:
    """Invokes Amazon Bedrock foundation models via the boto3 runtime client.

    The boto3 client is injected at construction time so that tests can
    substitute a mock without patching module-level globals.

    Args:
        bedrock_client: A boto3 ``bedrock-runtime`` client instance.
        model_id: The Bedrock model identifier to invoke.
        max_tokens: Maximum number of tokens in the model response.
        temperature: Sampling temperature; 0.0 produces deterministic output.
    """

    def __init__(
        self,
        bedrock_client: Any,
        model_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        """Initialise the repository with a boto3 client and model configuration."""
        self._client = bedrock_client
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    def invoke_model(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """Invoke the configured Bedrock model and return its text response.

        Constructs the Anthropic Messages API payload, calls Bedrock, and
        extracts the first content block's text.

        Args:
            messages: A list of conversation turns in Messages API format.
            system_prompt: The system-role instruction string.

        Returns:
            The raw text content of the first response content block.

        Raises:
            RateLimitError: When Bedrock returns a ThrottlingException.
            BedrockServiceError: For any other Bedrock or boto3 failure.
        """
        payload = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": system_prompt,
            "messages": messages,
        }
        try:
            response = self._client.invoke_model(
                body=json.dumps(payload),
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
            return str(response_body["content"][0]["text"])
        except botocore.exceptions.ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            status_code = exc.response["ResponseMetadata"]["HTTPStatusCode"]
            if error_code == _THROTTLING_ERROR_CODE:
                raise RateLimitError(
                    "Bedrock request was throttled. Retry after a short delay.",
                    context={"model_id": self._model_id, "error_code": error_code},
                ) from exc
            raise BedrockServiceError(
                f"Bedrock returned an error: {error_code}",
                context={
                    "model_id": self._model_id,
                    "error_code": error_code,
                    "status_code": status_code,
                },
            ) from exc
        except Exception as exc:
            raise BedrockServiceError(
                f"Unexpected error while invoking Bedrock: {exc}",
                context={"model_id": self._model_id},
            ) from exc

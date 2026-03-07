"""Tests for the BedrockRepository integration layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from clinical_summarizer.exceptions import BedrockServiceError, RateLimitError
from clinical_summarizer.bedrock_client import BedrockRepository

_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
_SYSTEM = "You are a clinical assistant."
_MESSAGES = [{"role": "user", "content": "Summarise this."}]


def _make_client_error(code: str, status: int = 400) -> botocore.exceptions.ClientError:
    """Helper to construct a botocore ClientError for a given error code."""
    return botocore.exceptions.ClientError(
        error_response={
            "Error": {"Code": code, "Message": "error"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation_name="InvokeModel",
    )


def _make_body_mock(text: str) -> MagicMock:
    """Return a mock streaming body that yields the given text as a Bedrock response."""
    body = MagicMock()
    payload = json.dumps({"content": [{"text": text}]}).encode()
    body.read.return_value = payload
    return body


@pytest.fixture()
def client() -> MagicMock:
    """A fresh boto3 client mock."""
    return MagicMock()


@pytest.fixture()
def repo(client: MagicMock) -> BedrockRepository:
    """A BedrockRepository with the mock client injected."""
    return BedrockRepository(bedrock_client=client, model_id=_MODEL_ID)


class TestBedrockRepositoryHappyPath:
    """Tests for successful Bedrock invocations."""

    def test_returns_text_from_response(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """invoke_model returns the text from the first content block."""
        client.invoke_model.return_value = {"body": _make_body_mock("Hello JSON")}
        result = repo.invoke_model(_MESSAGES, _SYSTEM)
        assert result == "Hello JSON"

    def test_calls_invoke_model_with_correct_model_id(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """invoke_model passes the configured modelId to boto3."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        call_kwargs = client.invoke_model.call_args.kwargs
        assert call_kwargs["modelId"] == _MODEL_ID

    def test_calls_invoke_model_with_correct_content_type(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """invoke_model sets contentType and accept to application/json."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        call_kwargs = client.invoke_model.call_args.kwargs
        assert call_kwargs["contentType"] == "application/json"
        assert call_kwargs["accept"] == "application/json"

    def test_payload_includes_anthropic_version(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """The Bedrock payload includes the required anthropic_version field."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        body_str = client.invoke_model.call_args.kwargs["body"]
        payload = json.loads(body_str)
        assert payload["anthropic_version"] == "bedrock-2023-05-31"

    def test_payload_uses_zero_temperature(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """Default temperature is 0.0 for deterministic clinical output."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        body_str = client.invoke_model.call_args.kwargs["body"]
        payload = json.loads(body_str)
        assert payload["temperature"] == 0.0

    def test_payload_includes_system_prompt(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """The system prompt is included in the Bedrock payload."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        body_str = client.invoke_model.call_args.kwargs["body"]
        payload = json.loads(body_str)
        assert payload["system"] == _SYSTEM

    def test_payload_includes_messages(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """The messages list is passed through to the Bedrock payload."""
        client.invoke_model.return_value = {"body": _make_body_mock("ok")}
        repo.invoke_model(_MESSAGES, _SYSTEM)
        body_str = client.invoke_model.call_args.kwargs["body"]
        payload = json.loads(body_str)
        assert payload["messages"] == _MESSAGES


class TestBedrockRepositoryErrorHandling:
    """Tests for error mapping in BedrockRepository."""

    def test_throttling_raises_rate_limit_error(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """ThrottlingException is mapped to RateLimitError."""
        client.invoke_model.side_effect = _make_client_error("ThrottlingException", 429)
        with pytest.raises(RateLimitError):
            repo.invoke_model(_MESSAGES, _SYSTEM)

    def test_other_client_error_raises_bedrock_service_error(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """Non-throttling ClientErrors are mapped to BedrockServiceError."""
        client.invoke_model.side_effect = _make_client_error("ValidationException", 400)
        with pytest.raises(BedrockServiceError):
            repo.invoke_model(_MESSAGES, _SYSTEM)

    def test_bedrock_service_error_not_rate_limit(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """Non-throttling errors do NOT raise RateLimitError."""
        client.invoke_model.side_effect = _make_client_error("AccessDeniedException", 403)
        with pytest.raises(BedrockServiceError) as exc_info:
            repo.invoke_model(_MESSAGES, _SYSTEM)
        assert not isinstance(exc_info.value, RateLimitError)

    def test_unexpected_exception_raises_bedrock_service_error(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """Any unexpected exception is wrapped in BedrockServiceError."""
        client.invoke_model.side_effect = RuntimeError("network failure")
        with pytest.raises(BedrockServiceError, match="Unexpected error"):
            repo.invoke_model(_MESSAGES, _SYSTEM)

    def test_rate_limit_error_context_contains_model_id(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """RateLimitError context dict includes the model_id."""
        client.invoke_model.side_effect = _make_client_error("ThrottlingException", 429)
        with pytest.raises(RateLimitError) as exc_info:
            repo.invoke_model(_MESSAGES, _SYSTEM)
        assert exc_info.value.context["model_id"] == _MODEL_ID

    def test_bedrock_service_error_context_contains_status_code(
        self, repo: BedrockRepository, client: MagicMock
    ) -> None:
        """BedrockServiceError context dict includes the HTTP status code."""
        client.invoke_model.side_effect = _make_client_error("ServiceUnavailableException", 503)
        with pytest.raises(BedrockServiceError) as exc_info:
            repo.invoke_model(_MESSAGES, _SYSTEM)
        assert exc_info.value.context["status_code"] == 503

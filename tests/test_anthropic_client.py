"""Tests for AnthropicRepository."""
import pytest
from unittest.mock import MagicMock, patch
import anthropic

from clinical_summarizer.anthropic_client import AnthropicRepository
from clinical_summarizer.exceptions import BedrockServiceError, RateLimitError


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    return MagicMock(spec=anthropic.Anthropic)


@pytest.fixture
def repository(mock_anthropic_client):
    """Create an AnthropicRepository with a mock client."""
    return AnthropicRepository(
        client=mock_anthropic_client,
        model="claude-3-5-haiku-20241022",
        max_tokens=2048,
        temperature=0.0,
    )


class TestAnthropicRepositoryHappyPath:
    """Tests for successful AnthropicRepository.invoke_model() calls."""

    def test_invoke_model_returns_text(self, repository, mock_anthropic_client) -> None:
        """invoke_model() should return the text from the response."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test summary output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test prompt"}]
        system_prompt = "You are a clinical summarizer."

        result = repository.invoke_model(messages, system_prompt)

        assert result == "Test summary output"

    def test_invoke_model_calls_create_with_correct_params(
        self, repository, mock_anthropic_client
    ) -> None:
        """invoke_model() should call messages.create with correct parameters."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        messages = [{"role": "user", "content": "Test"}]
        system_prompt = "System instruction"

        repository.invoke_model(messages, system_prompt)

        mock_anthropic_client.messages.create.assert_called_once_with(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            temperature=0.0,
            system="System instruction",
            messages=messages,
        )

    def test_invoke_model_uses_configured_model(
        self, mock_anthropic_client
    ) -> None:
        """invoke_model() should use the model specified at construction."""
        repo = AnthropicRepository(
            client=mock_anthropic_client,
            model="claude-3-5-sonnet-20241022",
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        # Verify the sonnet model was used
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-5-sonnet-20241022"

    def test_invoke_model_uses_configured_max_tokens(
        self, mock_anthropic_client
    ) -> None:
        """invoke_model() should use max_tokens specified at construction."""
        repo = AnthropicRepository(
            client=mock_anthropic_client,
            model="claude-3-5-haiku-20241022",
            max_tokens=1000,
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 1000

    def test_invoke_model_uses_configured_temperature(
        self, mock_anthropic_client
    ) -> None:
        """invoke_model() should use temperature specified at construction."""
        repo = AnthropicRepository(
            client=mock_anthropic_client,
            model="claude-3-5-haiku-20241022",
            temperature=0.7,
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0.7


class TestAnthropicRepositoryErrorHandling:
    """Tests for error handling in AnthropicRepository."""

    def test_invoke_model_rate_limit_raises_rate_limit_error(
        self, repository, mock_anthropic_client
    ) -> None:
        """invoke_model() should raise RateLimitError on Anthropic 429."""
        mock_anthropic_client.messages.create.side_effect = (
            anthropic.RateLimitError("Rate limited")
        )

        with pytest.raises(RateLimitError):
            repository.invoke_model([{"role": "user", "content": "Test"}], "System")

    def test_invoke_model_rate_limit_error_has_context(
        self, repository, mock_anthropic_client
    ) -> None:
        """RateLimitError should include model_id in context."""
        mock_anthropic_client.messages.create.side_effect = (
            anthropic.RateLimitError("Rate limited")
        )

        with pytest.raises(RateLimitError) as exc_info:
            repository.invoke_model([{"role": "user", "content": "Test"}], "System")

        assert exc_info.value.context["model_id"] == "claude-3-5-haiku-20241022"

    def test_invoke_model_api_error_raises_bedrock_service_error(
        self, repository, mock_anthropic_client
    ) -> None:
        """invoke_model() should raise BedrockServiceError on other API errors."""
        # Create a mock APIStatusError
        error = anthropic.APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500),
            body={},
        )
        mock_anthropic_client.messages.create.side_effect = error

        with pytest.raises(BedrockServiceError):
            repository.invoke_model([{"role": "user", "content": "Test"}], "System")

    def test_invoke_model_generic_exception_raises_bedrock_service_error(
        self, repository, mock_anthropic_client
    ) -> None:
        """invoke_model() should raise BedrockServiceError for generic exceptions."""
        mock_anthropic_client.messages.create.side_effect = RuntimeError(
            "Unexpected error"
        )

        with pytest.raises(BedrockServiceError):
            repository.invoke_model([{"role": "user", "content": "Test"}], "System")

    def test_invoke_model_bedrock_error_has_context(
        self, repository, mock_anthropic_client
    ) -> None:
        """BedrockServiceError should include context information."""
        error = anthropic.APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500),
            body={},
        )
        mock_anthropic_client.messages.create.side_effect = error

        with pytest.raises(BedrockServiceError) as exc_info:
            repository.invoke_model([{"role": "user", "content": "Test"}], "System")

        assert exc_info.value.context["model_id"] == "claude-3-5-haiku-20241022"
        assert "status_code" in exc_info.value.context


class TestAnthropicRepositoryDefaults:
    """Tests for default configuration values."""

    def test_default_model_is_haiku(self, mock_anthropic_client) -> None:
        """Default model should be claude-3-5-haiku."""
        repo = AnthropicRepository(client=mock_anthropic_client)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-3-5-haiku-20241022"

    def test_default_max_tokens_is_2048(self, mock_anthropic_client) -> None:
        """Default max_tokens should be 2048."""
        repo = AnthropicRepository(client=mock_anthropic_client)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 2048

    def test_default_temperature_is_zero(self, mock_anthropic_client) -> None:
        """Default temperature should be 0.0 for deterministic output."""
        repo = AnthropicRepository(client=mock_anthropic_client)
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Output")]
        mock_anthropic_client.messages.create.return_value = mock_response

        repo.invoke_model([{"role": "user", "content": "Test"}], "System")

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0.0

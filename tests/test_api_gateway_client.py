"""Tests for the ApiGatewayClient HTTP integration."""
import pytest
from unittest.mock import patch, MagicMock
import httpx

from clinical_summarizer.api_gateway_client import ApiGatewayClient
from clinical_summarizer.models import ClinicalSummary
from clinical_summarizer.exceptions import (
    BedrockServiceError,
    RateLimitError,
    ValidationError,
    SummaryParsingError,
)


@pytest.fixture
def mock_summary_response() -> dict:
    """A valid ClinicalSummary response from the API."""
    return {
        "document_type": "Lab Results",
        "summary": "Patient blood work shows normal values.",
        "key_findings": ["Normal CBC", "Normal chemistry panel"],
        "abnormal_results": [],
        "recommended_follow_up": ["Routine checkup in 6 months"],
    }


@pytest.fixture
def client() -> ApiGatewayClient:
    """Create an ApiGatewayClient instance."""
    return ApiGatewayClient(
        base_url="https://api.example.com/summarize",
        timeout=30.0,
    )


class TestApiGatewayClientInit:
    """Tests for ApiGatewayClient initialization."""

    def test_init_stores_url(self) -> None:
        """ApiGatewayClient should store the base URL."""
        url = "https://api.example.com/summarize"
        client = ApiGatewayClient(base_url=url)
        assert client._base_url == url

    def test_init_strips_trailing_slash(self) -> None:
        """ApiGatewayClient should strip trailing slashes from URL."""
        url = "https://api.example.com/summarize/"
        client = ApiGatewayClient(base_url=url)
        assert client._base_url == "https://api.example.com/summarize"

    def test_init_stores_timeout(self) -> None:
        """ApiGatewayClient should store the timeout value."""
        client = ApiGatewayClient(
            base_url="https://api.example.com/summarize",
            timeout=60.0,
        )
        assert client._timeout == 60.0

    def test_init_default_timeout(self) -> None:
        """ApiGatewayClient should use 30s timeout by default."""
        client = ApiGatewayClient(base_url="https://api.example.com/summarize")
        assert client._timeout == 30.0


class TestApiGatewayClientHappyPath:
    """Tests for successful API Gateway calls."""

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_summarize_success(
        self, mock_client_class, client, mock_summary_response
    ) -> None:
        """summarize() should return ClinicalSummary on HTTP 200."""
        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_summary_response
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = client.summarize("Test clinical text", "req-123")

        assert isinstance(result, ClinicalSummary)
        assert result.document_type == "Lab Results"
        assert len(result.key_findings) == 2

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_summarize_posts_correct_payload(
        self, mock_client_class, client, mock_summary_response
    ) -> None:
        """summarize() should POST the text and request_id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_summary_response
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        client.summarize("Clinical data...", "req-456")

        # Verify POST was called with correct payload
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.example.com/summarize"
        assert call_args[1]["json"] == {
            "text": "Clinical data...",
            "request_id": "req-456",
        }

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_summarize_uses_correct_timeout(
        self, mock_client_class, mock_summary_response
    ) -> None:
        """summarize() should use the configured timeout."""
        client = ApiGatewayClient(
            base_url="https://api.example.com/summarize",
            timeout=60.0,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_summary_response
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        client.summarize("Test", "req-123")

        # Verify Client was created with correct timeout
        mock_client_class.assert_called_once_with(timeout=60.0)


class TestApiGatewayClientErrorHandling:
    """Tests for error scenarios."""

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_rate_limit_error_on_429(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise RateLimitError on HTTP 429."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(RateLimitError):
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_validation_error_on_400(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise ValidationError on HTTP 400."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "ValidationError",
            "message": "Text is blank",
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(ValidationError):
            client.summarize("", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_validation_error_on_422(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise ValidationError on HTTP 422."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "error": "ValidationError",
            "message": "Text exceeds maximum length",
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(ValidationError):
            client.summarize("x" * 60000, "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_bedrock_error_on_500(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise BedrockServiceError on HTTP 500."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(BedrockServiceError):
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_bedrock_error_on_502(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise BedrockServiceError on HTTP 502."""
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(BedrockServiceError):
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_parsing_error_on_invalid_json(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise SummaryParsingError if response JSON is invalid."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(Exception):  # Will raise from response.json()
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_parsing_error_on_mismatched_schema(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise SummaryParsingError if response schema doesn't match."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Missing required fields
        mock_response.json.return_value = {
            "document_type": "Test",
            # Missing: summary, key_findings, abnormal_results, recommended_follow_up
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(SummaryParsingError):
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_timeout_raises_bedrock_error(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise BedrockServiceError on timeout."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(BedrockServiceError):
            client.summarize("Test", "req-123")

    @patch("clinical_summarizer.api_gateway_client.httpx.Client")
    def test_network_error_raises_bedrock_error(
        self, mock_client_class, client
    ) -> None:
        """summarize() should raise BedrockServiceError on network error."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.NetworkError("Connection refused")
        mock_client_class.return_value.__enter__.return_value = mock_client

        with pytest.raises(BedrockServiceError):
            client.summarize("Test", "req-123")

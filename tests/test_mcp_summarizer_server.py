"""Tests for the MCP summarizer server."""
from unittest.mock import MagicMock, patch
import pytest

from clinical_summarizer.exceptions import ValidationError, SummaryParsingError
from clinical_summarizer.models import ClinicalSummary


@pytest.fixture
def mock_clinical_summary():
    """Create a mock ClinicalSummary instance."""
    return ClinicalSummary(
        document_type="Lab Results",
        summary="Patient blood work shows normal CBC.",
        key_findings=["All values normal", "No abnormalities"],
        abnormal_results=[],
        recommended_follow_up=["Routine checkup in 6 months"],
    )


class TestMcpSummarizerServerSummarizeDocumentWithApiGateway:
    """Tests for summarize_document with AWS API Gateway backend."""

    @patch.dict("os.environ", {"API_GATEWAY_URL": "https://api.example.com/summarize"})
    @patch("mcp_summarizer_server._get_client")
    def test_summarize_with_api_gateway(
        self, mock_get_client, mock_clinical_summary
    ) -> None:
        """summarize_document() should work with API Gateway backend."""
        mock_client = MagicMock()
        mock_client.summarize.return_value = mock_clinical_summary
        mock_get_client.return_value = mock_client

        import mcp_summarizer_server

        result = mcp_summarizer_server.summarize_document("Test text", "req-123")

        assert isinstance(result, dict)
        assert result["document_type"] == "Lab Results"
        mock_client.summarize.assert_called_once_with("Test text", "req-123")


class TestMcpSummarizerServerSummarizeDocumentWithAnthropicLocal:
    """Tests for summarize_document with local Anthropic backend."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server._get_client")
    def test_summarize_with_local_service(
        self, mock_get_client, mock_clinical_summary
    ) -> None:
        """summarize_document() should work with local SummarizationService backend."""
        mock_service = MagicMock()
        mock_service.summarize.return_value = mock_clinical_summary
        mock_get_client.return_value = mock_service

        import mcp_summarizer_server

        result = mcp_summarizer_server.summarize_document(
            "Blood test results...", "req-123"
        )

        assert isinstance(result, dict)
        assert result["document_type"] == "Lab Results"
        assert "All values normal" in result["key_findings"]


class TestMcpSummarizerServerErrorHandling:
    """Tests for error scenarios."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server._get_client")
    def test_validation_error_raises_value_error(
        self, mock_get_client
    ) -> None:
        """summarize_document() should raise ValueError on ValidationError."""
        mock_service = MagicMock()
        mock_service.summarize.side_effect = ValidationError(
            message="text is empty or blank",
            context={},
        )
        mock_get_client.return_value = mock_service

        import mcp_summarizer_server

        with pytest.raises(ValueError):
            mcp_summarizer_server.summarize_document("   ", "req-789")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server._get_client")
    def test_parsing_error_raises_runtime_error(
        self, mock_get_client
    ) -> None:
        """summarize_document() should raise RuntimeError on SummaryParsingError."""
        mock_service = MagicMock()
        mock_service.summarize.side_effect = SummaryParsingError(
            message="Invalid JSON in response",
            context={},
        )
        mock_get_client.return_value = mock_service

        import mcp_summarizer_server

        with pytest.raises(RuntimeError):
            mcp_summarizer_server.summarize_document("Valid text", "req-999")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server._get_client")
    def test_unexpected_exception_raises_runtime_error(
        self, mock_get_client
    ) -> None:
        """summarize_document() should raise RuntimeError for unexpected errors."""
        mock_service = MagicMock()
        mock_service.summarize.side_effect = RuntimeError("Network error")
        mock_get_client.return_value = mock_service

        import mcp_summarizer_server

        with pytest.raises(RuntimeError):
            mcp_summarizer_server.summarize_document("Test", "req-111")


class TestMcpSummarizerServerGetClient:
    """Tests for the _get_client() function."""

    @patch.dict("os.environ", {"API_GATEWAY_URL": "https://api.example.com/summarize"})
    def test_get_client_api_gateway_priority(self) -> None:
        """_get_client() should prefer API_GATEWAY_URL over ANTHROPIC_API_KEY."""
        import mcp_summarizer_server

        mcp_summarizer_server._client = None

        client = mcp_summarizer_server._get_client()

        # Should return ApiGatewayClient
        from clinical_summarizer.api_gateway_client import ApiGatewayClient

        assert isinstance(client, ApiGatewayClient)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server.anthropic.Anthropic")
    @patch("mcp_summarizer_server.AnthropicRepository")
    def test_get_client_anthropic_fallback(
        self, mock_repo_class, mock_anthropic_class
    ) -> None:
        """_get_client() should use Anthropic if API_GATEWAY_URL not set."""
        import mcp_summarizer_server

        mcp_summarizer_server._client = None

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client = mcp_summarizer_server._get_client()

        # Should return SummarizationService
        from clinical_summarizer.services import SummarizationService

        assert isinstance(client, SummarizationService)

    @patch.dict("os.environ", {}, clear=True)
    def test_get_client_requires_env_var(self) -> None:
        """_get_client() should raise ValueError if neither env var is set."""
        import mcp_summarizer_server

        mcp_summarizer_server._client = None

        with pytest.raises(ValueError, match="API_GATEWAY_URL.*ANTHROPIC_API_KEY"):
            mcp_summarizer_server._get_client()

    @patch.dict("os.environ", {"API_GATEWAY_URL": "https://api.example.com/summarize"})
    def test_get_client_caches_api_gateway(self) -> None:
        """_get_client() should cache the API Gateway client."""
        import mcp_summarizer_server

        mcp_summarizer_server._client = None

        client1 = mcp_summarizer_server._get_client()
        client2 = mcp_summarizer_server._get_client()

        # Should return same instance
        assert client1 is client2

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("mcp_summarizer_server.anthropic.Anthropic")
    @patch("mcp_summarizer_server.AnthropicRepository")
    def test_get_client_caches_service(
        self, mock_repo_class, mock_anthropic_class
    ) -> None:
        """_get_client() should cache the SummarizationService."""
        import mcp_summarizer_server

        mcp_summarizer_server._client = None

        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        client1 = mcp_summarizer_server._get_client()
        client2 = mcp_summarizer_server._get_client()

        # Anthropic client should be created only once
        mock_anthropic_class.assert_called_once_with(api_key="test-key")
        # Same service instance returned
        assert client1 is client2

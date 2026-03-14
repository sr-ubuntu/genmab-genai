"""Tests for the FastAPI API server."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from api_server import app, QueryRequest


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


class TestApiServerHealth:
    """Tests for the /health endpoint."""

    def test_health_endpoint(self, client) -> None:
        """GET /health should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestApiServerQuery:
    """Tests for the /query endpoint."""

    @patch("api_server._run_agent")
    def test_query_endpoint_returns_response(self, mock_run_agent, client) -> None:
        """POST /query should return a QueryResponse with the agent's answer."""
        mock_run_agent.return_value = "Test response from agent"

        # Make async mock work with TestClient (which is synchronous)
        async_mock = AsyncMock(return_value="Test response from agent")

        with patch("api_server._run_agent", async_mock):
            response = client.post(
                "/query",
                json={"query": "Test query"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response from agent"
        assert "session_id" in data

    def test_query_endpoint_generates_session_id(self, client) -> None:
        """POST /query should generate a session_id if not provided."""
        with patch("api_server._run_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = "Response"

            response = client.post(
                "/query",
                json={"query": "Test query"},
            )

        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0

    def test_query_endpoint_uses_provided_session_id(self, client) -> None:
        """POST /query should use the provided session_id."""
        provided_session_id = "test-session-123"

        with patch("api_server._run_agent", new_callable=AsyncMock) as mock_agent:
            mock_agent.return_value = "Response"

            response = client.post(
                "/query",
                json={"query": "Test query", "session_id": provided_session_id},
            )

        data = response.json()
        assert data["session_id"] == provided_session_id

    def test_query_endpoint_missing_query_returns_422(self, client) -> None:
        """POST /query without 'query' field should return 422."""
        response = client.post(
            "/query",
            json={},  # Missing required 'query' field
        )

        assert response.status_code == 422

    def test_query_endpoint_invalid_json_returns_422(self, client) -> None:
        """POST /query with invalid JSON should return 422."""
        response = client.post(
            "/query",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @patch("api_server._run_agent")
    def test_query_endpoint_agent_error_returns_502(
        self, mock_run_agent, client
    ) -> None:
        """POST /query should return 502 if agent fails."""
        async_mock = AsyncMock(side_effect=RuntimeError("Agent failed"))

        with patch("api_server._run_agent", async_mock):
            response = client.post(
                "/query",
                json={"query": "Test query"},
            )

        assert response.status_code == 502

    @patch("api_server._run_agent")
    def test_query_endpoint_config_error_returns_500(
        self, mock_run_agent, client
    ) -> None:
        """POST /query should return 500 on config error."""
        async_mock = AsyncMock(
            side_effect=ValueError("ANTHROPIC_API_KEY not set")
        )

        with patch("api_server._run_agent", async_mock):
            response = client.post(
                "/query",
                json={"query": "Test query"},
            )

        assert response.status_code == 500


class TestQueryRequest:
    """Tests for the QueryRequest model."""

    def test_query_request_required_fields(self) -> None:
        """QueryRequest should require 'query' field."""
        with pytest.raises(ValueError):
            QueryRequest()  # type: ignore

    def test_query_request_query_only(self) -> None:
        """QueryRequest should accept just 'query' field."""
        req = QueryRequest(query="Test query")
        assert req.query == "Test query"
        assert req.session_id is None

    def test_query_request_with_session_id(self) -> None:
        """QueryRequest should accept optional 'session_id' field."""
        req = QueryRequest(query="Test", session_id="sess-123")
        assert req.query == "Test"
        assert req.session_id == "sess-123"

"""MCP server exposing the clinical summarization tool via fastmcp.

This server provides MCP tools to summarize clinical documents. It supports
two backends via environment variables:
1. API_GATEWAY_URL — calls the AWS API Gateway endpoint (→ Lambda → Bedrock)
2. ANTHROPIC_API_KEY — calls the Anthropic API directly (for local dev/test)

If API_GATEWAY_URL is set, it takes priority. If neither is set, the server
will raise an error at startup.

Run with: python mcp_summarizer_server.py
Transport: stdio (consumed as a subprocess by the agent)
"""
from __future__ import annotations
import logging
import os
from typing import Union

import fastmcp
import anthropic

from clinical_summarizer.anthropic_client import AnthropicRepository
from clinical_summarizer.api_gateway_client import ApiGatewayClient
from clinical_summarizer.core import PromptBuilder, SummaryParser
from clinical_summarizer.services import SummarizationService
from clinical_summarizer.models import ClinicalSummaryRequest
from clinical_summarizer.exceptions import ClinicalSummarizerError, ValidationError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-summarizer-server")

mcp = fastmcp.FastMCP("clinical-summarizer")

# Module-level client cache (lazy singleton)
# Can be either a SummarizationService (local) or ApiGatewayClient (remote)
_client: Union[SummarizationService, ApiGatewayClient, None] = None


def _get_client() -> Union[SummarizationService, ApiGatewayClient]:
    """Get or initialize the summarization backend.

    Prefers AWS API Gateway over local Anthropic API based on environment:
    1. If API_GATEWAY_URL is set → use ApiGatewayClient (calls Lambda → Bedrock)
    2. Else if ANTHROPIC_API_KEY is set → use SummarizationService + AnthropicRepository
    3. Else raise ValueError

    Returns:
        Either a SummarizationService or ApiGatewayClient instance.

    Raises:
        ValueError: If neither API_GATEWAY_URL nor ANTHROPIC_API_KEY is set.
    """
    global _client
    if _client is not None:
        return _client

    # Check for API Gateway endpoint (takes priority)
    api_gateway_url = os.environ.get("API_GATEWAY_URL")
    if api_gateway_url:
        _client = ApiGatewayClient(base_url=api_gateway_url)
        logger.info(f"Using AWS API Gateway backend: {api_gateway_url}")
        return _client

    # Fall back to local Anthropic API
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Neither API_GATEWAY_URL nor ANTHROPIC_API_KEY environment variables are set. "
            "Please set one before using the summarizer service. "
            "API_GATEWAY_URL takes priority if both are set."
        )

    logger.info("Using Anthropic API backend (local)")

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Wire up the DI graph for local summarization
    repository = AnthropicRepository(
        client=client,
        model="claude-3-5-haiku-20241022",
        max_tokens=2048,
        temperature=0.0,
    )
    prompt_builder = PromptBuilder()
    summary_parser = SummaryParser()

    _client = SummarizationService(
        repository=repository,
        prompt_builder=prompt_builder,
        summary_parser=summary_parser,
        logger=logger,
    )

    return _client


@mcp.tool()
def summarize_document(text: str, request_id: str) -> dict:
    """Summarize a clinical document and return a structured result.

    Routes to either the AWS API Gateway endpoint (→ Lambda → Bedrock) or
    the local Anthropic API depending on environment configuration.

    Args:
        text: The raw clinical document text (1-50,000 characters).
        request_id: Caller-supplied identifier used for log correlation.

    Returns:
        A dict with keys: document_type, summary, key_findings,
        abnormal_results, recommended_follow_up.

    Raises:
        ValueError: If text is blank or exceeds length limits.
        RuntimeError: If the API call fails or response parsing fails.
    """
    try:
        # Get the client (either API Gateway or Anthropic)
        client = _get_client()

        # Route to the appropriate backend
        if isinstance(client, ApiGatewayClient):
            # AWS API Gateway backend (text validation happens server-side)
            summary = client.summarize(text, request_id)
        else:
            # Local Anthropic backend (validate here)
            request = ClinicalSummaryRequest(text=text, request_id=request_id)
            summary = client.summarize(request)

        # Return as a plain dict for MCP JSON serialization
        return summary.model_dump()

    except ValidationError as e:
        # Input validation error (blank text, exceeds max length, etc.)
        logger.error(f"Validation error for request {request_id}: {e.message}")
        raise ValueError(e.message) from e

    except ClinicalSummarizerError as e:
        # Any clinical summarizer error (API, parsing, etc.)
        logger.error(f"Summarizer error for request {request_id}: {e.message}")
        raise RuntimeError(e.message) from e

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error for request {request_id}: {str(e)}")
        raise RuntimeError(f"Summarization failed: {str(e)}") from e


if __name__ == "__main__":
    logger.info("Starting clinical summarizer MCP server")
    mcp.run()

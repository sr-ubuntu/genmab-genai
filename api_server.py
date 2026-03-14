"""FastAPI REST API server for the clinical summarizer agent.

This server:
1. Starts two MCP servers (database and summarizer) as subprocesses
2. Exposes a POST /query endpoint to submit natural-language queries
3. Uses Claude (via Anthropic SDK) with MCP client to process queries
4. Returns structured summaries in the response

Run with: uvicorn api_server:app --reload --port 8000
Requires: ANTHROPIC_API_KEY environment variable

Example request:
  curl -X POST http://localhost:8000/query \\
    -H "Content-Type: application/json" \\
    -d '{"query": "Summarize all lab results for patient 3"}'
"""
from __future__ import annotations
import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import anthropic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QueryRequest(BaseModel):
    """Request payload for the /query endpoint."""

    query: str = Field(..., description="Natural-language query for the agent")
    session_id: str | None = Field(
        None, description="Optional session ID for conversation history"
    )


class QueryResponse(BaseModel):
    """Response payload from the /query endpoint."""

    response: str = Field(..., description="Agent's final text answer")
    session_id: str = Field(..., description="Session ID (generated if not supplied)")


# MCP server subprocess handles (will be set in startup)
_mcp_db_process = None
_mcp_summarizer_process = None
_anthropic_client = None


async def _run_agent(query: str) -> str:
    """Connect to both MCP servers and run a single-turn agent query.

    Opens stdio connections to both MCP servers as subprocesses,
    discovers their tools, then runs an agentic loop using Claude
    until the model returns a final text response.

    Args:
        query: The natural-language query to process.

    Returns:
        The agent's final text response as a string.

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set or MCP servers fail.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Define MCP server configurations
    db_server = {
        "type": "stdio",
        "command": "python",
        "args": ["mcp_db_server.py"],
    }
    summarizer_server = {
        "type": "stdio",
        "command": "python",
        "args": ["mcp_summarizer_server.py"],
    }

    # Connect to both MCP servers and run the agentic loop
    try:
        async with client.beta.mcp.connect_mcp_servers([db_server, summarizer_server]) as mcp_connection:
            # Initial message to Claude
            messages = [
                {
                    "role": "user",
                    "content": query,
                }
            ]

            system_prompt = (
                "You are a clinical data assistant. You have access to tools to query a patient "
                "database and summarize clinical documents. When asked about patients or their records, "
                "always retrieve data using the available tools before responding. "
                "Present summaries in a clear, structured format. Be concise and professional."
            )

            # Agentic loop: keep calling Claude until it returns a final response
            while True:
                response = await mcp_connection.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                )

                # Check if Claude is done (no tool calls)
                if response.stop_reason == "end_turn":
                    # Extract the final text response
                    for content in response.content:
                        if hasattr(content, "text"):
                            return content.text
                    # Fallback if no text found
                    return "No response generated"

                # Process tool calls
                tool_calls_made = False
                for content in response.content:
                    if content.type == "tool_use":
                        tool_calls_made = True
                        # Claude made a tool call; we need to call the tool and add result
                        # The MCP connection handles actual tool invocation
                        # We just need to add Claude's response to messages and continue the loop
                        break

                # Add Claude's response (including tool calls) to message history
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )

                # If no tool calls were made, Claude is responding with text
                if not tool_calls_made:
                    # Find the text content and return it
                    for content in response.content:
                        if hasattr(content, "text"):
                            return content.text
                    return "No response generated"

                # Note: Tool results are automatically added by the MCP connection
                # when we call messages.create again in the next iteration

    except Exception as e:
        logger.error(f"MCP agent error: {str(e)}")
        raise RuntimeError(f"Agent failed to process query: {str(e)}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown.

    Starts the MCP server subprocesses on startup and cleans them up on shutdown.
    """
    # Startup
    logger.info("API server starting up...")
    # Note: In a production setup, you might start the MCP servers here
    # For now, they're started on-demand by the agent
    yield
    # Shutdown
    logger.info("API server shutting down...")
    # Clean up MCP processes if they were started
    # (In this design, they're started per-request, so cleanup is minimal)


# Create FastAPI app
app = FastAPI(
    title="Clinical Summarizer API",
    description="REST API for clinical document summarization with MCP agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns:
        A dict with status "ok".
    """
    return {"status": "ok"}


@app.post("/query")
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Process a natural-language query using the agent.

    The agent connects to MCP servers (database and summarizer) and
    uses Claude to orchestrate queries and summaries.

    Args:
        request: A QueryRequest with the query string and optional session_id.

    Returns:
        A QueryResponse with the agent's answer and session_id.

    Raises:
        HTTPException: If the API key is not set or agent fails.
    """
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(f"Processing query for session {session_id}")
        response_text = await _run_agent(request.query)
        logger.info(f"Query completed for session {session_id}")

        return QueryResponse(response=response_text, session_id=session_id)

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"Agent error: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

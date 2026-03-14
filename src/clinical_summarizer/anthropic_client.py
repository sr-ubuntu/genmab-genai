"""Anthropic SDK integration layer for local / MCP use.

Implements the same LLMRepository protocol as BedrockRepository so that
SummarizationService can be wired with this repository without any changes
to the service layer. All Anthropic API I/O is isolated here.
"""
from __future__ import annotations
import anthropic

from clinical_summarizer.exceptions import BedrockServiceError, RateLimitError


_DEFAULT_MODEL = "claude-3-5-haiku-20241022"


class AnthropicRepository:
    """Invokes Claude via the Anthropic Python SDK.

    Drop-in replacement for BedrockRepository in non-Lambda environments
    (e.g., local dev, MCP servers). Accepts an injected anthropic.Anthropic
    client so tests can substitute a mock without patching module-level globals.

    Args:
        client: An anthropic.Anthropic client instance.
        model: The Claude model string to invoke.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature; 0.0 for deterministic output.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        """Initialise with an injected Anthropic client and model config.

        Args:
            client: Initialised anthropic.Anthropic instance.
            model: Model identifier string (e.g. "claude-3-5-sonnet-20241022").
            max_tokens: Maximum output tokens (default 2048).
            temperature: Sampling temperature (default 0.0 for deterministic).
        """
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def invoke_model(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """Call the Anthropic Messages API and return the text response.

        Implements the LLMRepository protocol — matches the exact signature
        of BedrockRepository.invoke_model for seamless substitution.

        Args:
            messages: Conversation turns in Messages API format.
                Each dict has keys "role" ("user" or "assistant") and "content".
            system_prompt: The system-role instruction string.

        Returns:
            The raw text content of the first response content block.

        Raises:
            RateLimitError: When Anthropic returns a 429 rate limit response.
            BedrockServiceError: For any other Anthropic API failure.
                Reuses BedrockServiceError to maintain interface compatibility.
        """
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=messages,
            )
            # Extract text from the first content block
            return response.content[0].text
        except anthropic.RateLimitError as e:
            raise RateLimitError(
                message="Anthropic API rate limited",
                context={"model_id": self._model, "error": str(e)},
            ) from e
        except anthropic.APIStatusError as e:
            # Map other API errors to BedrockServiceError
            error_context = {
                "model_id": self._model,
                "status_code": e.status_code if hasattr(e, "status_code") else "unknown",
                "error": str(e),
            }
            raise BedrockServiceError(
                message="Anthropic API error",
                context=error_context,
            ) from e
        except Exception as e:
            # Catch any other exceptions (network, JSON parsing, etc.)
            raise BedrockServiceError(
                message="Anthropic API call failed",
                context={"model_id": self._model, "error": str(e)},
            ) from e

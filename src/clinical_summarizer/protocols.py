"""Protocols (structural interfaces) for the clinical summarizer.

Using Protocol instead of ABC allows both BedrockRepository and
AnthropicRepository to satisfy the interface without inheritance.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMRepository(Protocol):
    """Structural interface for LLM backends used by SummarizationService.

    Any class implementing invoke_model with this exact signature
    satisfies this protocol without subclassing.

    This allows both BedrockRepository (AWS Bedrock backend) and
    AnthropicRepository (Anthropic SDK backend) to be used interchangeably
    with SummarizationService.
    """

    def invoke_model(self, messages: list[dict[str, str]], system_prompt: str) -> str:
        """Invoke the LLM and return its raw text response.

        Args:
            messages: Conversation turns in Messages API format.
                Each dict has keys "role" ("user" or "assistant") and "content".
            system_prompt: The system-role instruction string.

        Returns:
            The raw text content of the first response content block.

        Raises:
            RateLimitError: When the LLM API returns a rate limit / throttling error.
            BedrockServiceError: For any other LLM API failure.
        """
        ...

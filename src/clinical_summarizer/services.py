"""Orchestration layer: wires the prompt, Bedrock call, and parsing together."""

from __future__ import annotations

import logging

from clinical_summarizer.core import PromptBuilder, SummaryParser
from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest
from clinical_summarizer.protocols import LLMRepository


class SummarizationService:
    """Orchestrates the end-to-end clinical text summarization pipeline.

    Accepts a validated request, delegates prompt construction to
    :class:`PromptBuilder`, the LLM call to any :class:`LLMRepository` backend,
    and response parsing to :class:`SummaryParser`. All collaborators are
    injected so the service is fully testable without AWS credentials.

    Args:
        repository: LLM integration (Bedrock or Anthropic) responsible for all API I/O.
        prompt_builder: Builds the messages payload for the LLM.
        summary_parser: Parses the raw LLM response into a structured summary.
        logger: Standard Python logger used for structured log output.
    """

    def __init__(
        self,
        repository: LLMRepository,
        prompt_builder: PromptBuilder,
        summary_parser: SummaryParser,
        logger: logging.Logger,
    ) -> None:
        """Initialise the service with all required collaborators."""
        self._repository = repository
        self._prompt_builder = prompt_builder
        self._summary_parser = summary_parser
        self._logger = logger

    def summarize(self, request: ClinicalSummaryRequest) -> ClinicalSummary:
        """Run the full summarization pipeline for a single request.

        Args:
            request: A validated inbound summarization request.

        Returns:
            A structured :class:`ClinicalSummary` derived from the document.

        Raises:
            BedrockServiceError: If the Bedrock API call fails.
            RateLimitError: If Bedrock throttles the request.
            SummaryParsingError: If the model response cannot be parsed.
        """
        self._logger.info(
            "Summarization requested",
            extra={
                "request_id": request.request_id,
                "text_length": len(request.text),
            },
        )

        messages, system_prompt = self._prompt_builder.build_prompt(request)
        raw_response = self._repository.invoke_model(messages, system_prompt)
        summary = self._summary_parser.parse(raw_response)

        self._logger.info(
            "Summarization complete",
            extra={
                "request_id": request.request_id,
                "document_type": summary.document_type,
                "key_findings_count": len(summary.key_findings),
                "abnormal_results_count": len(summary.abnormal_results),
            },
        )
        return summary

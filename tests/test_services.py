"""Tests for the SummarizationService orchestration layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clinical_summarizer.core import PromptBuilder, SummaryParser
from clinical_summarizer.exceptions import BedrockServiceError, RateLimitError, SummaryParsingError
from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest
from clinical_summarizer.services import SummarizationService


@pytest.fixture()
def service(
    mock_repository: MagicMock,
    mock_logger: MagicMock,
) -> SummarizationService:
    """A SummarizationService with real PromptBuilder/SummaryParser and mock I/O."""
    return SummarizationService(
        repository=mock_repository,
        prompt_builder=PromptBuilder(),
        summary_parser=SummaryParser(),
        logger=mock_logger,
    )


class TestSummarizationServiceHappyPath:
    """Tests for successful summarization flows."""

    def test_returns_clinical_summary(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
    ) -> None:
        """summarize returns a ClinicalSummary on success."""
        result = service.summarize(valid_request)
        assert isinstance(result, ClinicalSummary)

    def test_summary_has_document_type(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
    ) -> None:
        """The returned summary has a non-empty document_type."""
        result = service.summarize(valid_request)
        assert result.document_type

    def test_repository_invoked_once(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
        mock_repository: MagicMock,
    ) -> None:
        """BedrockRepository.invoke_model is called exactly once per request."""
        service.summarize(valid_request)
        mock_repository.invoke_model.assert_called_once()

    def test_repository_receives_messages_list(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
        mock_repository: MagicMock,
    ) -> None:
        """The first argument to invoke_model is a non-empty messages list."""
        service.summarize(valid_request)
        call_args = mock_repository.invoke_model.call_args
        messages = call_args.args[0]
        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_logger_info_called_on_start(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
        mock_logger: MagicMock,
    ) -> None:
        """Logger.info is called at the start of summarization."""
        service.summarize(valid_request)
        assert mock_logger.info.call_count >= 1

    def test_logger_info_called_on_completion(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
        mock_logger: MagicMock,
    ) -> None:
        """Logger.info is called twice: once at start, once on completion."""
        service.summarize(valid_request)
        assert mock_logger.info.call_count == 2

    def test_request_id_in_log_extra(
        self,
        service: SummarizationService,
        valid_request: ClinicalSummaryRequest,
        mock_logger: MagicMock,
    ) -> None:
        """The request_id appears in the logger extra dict."""
        service.summarize(valid_request)
        first_call_extra = mock_logger.info.call_args_list[0].kwargs.get("extra", {})
        assert first_call_extra.get("request_id") == valid_request.request_id


class TestSummarizationServiceErrorPropagation:
    """Tests that exceptions from collaborators are propagated correctly."""

    def test_propagates_bedrock_service_error(
        self,
        mock_repository: MagicMock,
        mock_logger: MagicMock,
        valid_request: ClinicalSummaryRequest,
    ) -> None:
        """BedrockServiceError from the repository is not swallowed."""
        mock_repository.invoke_model.side_effect = BedrockServiceError("Bedrock down")
        svc = SummarizationService(
            repository=mock_repository,
            prompt_builder=PromptBuilder(),
            summary_parser=SummaryParser(),
            logger=mock_logger,
        )
        with pytest.raises(BedrockServiceError):
            svc.summarize(valid_request)

    def test_propagates_rate_limit_error(
        self,
        mock_repository: MagicMock,
        mock_logger: MagicMock,
        valid_request: ClinicalSummaryRequest,
    ) -> None:
        """RateLimitError from the repository is not swallowed."""
        mock_repository.invoke_model.side_effect = RateLimitError("Throttled")
        svc = SummarizationService(
            repository=mock_repository,
            prompt_builder=PromptBuilder(),
            summary_parser=SummaryParser(),
            logger=mock_logger,
        )
        with pytest.raises(RateLimitError):
            svc.summarize(valid_request)

    def test_propagates_summary_parsing_error(
        self,
        mock_repository: MagicMock,
        mock_logger: MagicMock,
        valid_request: ClinicalSummaryRequest,
    ) -> None:
        """SummaryParsingError from the parser is not swallowed."""
        mock_repository.invoke_model.return_value = "not valid json"
        svc = SummarizationService(
            repository=mock_repository,
            prompt_builder=PromptBuilder(),
            summary_parser=SummaryParser(),
            logger=mock_logger,
        )
        with pytest.raises(SummaryParsingError):
            svc.summarize(valid_request)

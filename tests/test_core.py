"""Tests for the pure business logic layer (PromptBuilder and SummaryParser)."""

from __future__ import annotations

import json

import pytest

from clinical_summarizer.core import SYSTEM_PROMPT, PromptBuilder, SummaryParser
from clinical_summarizer.exceptions import SummaryParsingError
from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest

# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    """Tests for PromptBuilder.build_prompt."""

    @pytest.fixture()
    def builder(self) -> PromptBuilder:
        return PromptBuilder()

    @pytest.fixture()
    def req(self) -> ClinicalSummaryRequest:
        return ClinicalSummaryRequest(text="Patient has elevated BP.", request_id="r1")

    def test_returns_tuple_of_messages_and_system(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """build_prompt returns a (list, str) tuple."""
        messages, system = builder.build_prompt(req)
        assert isinstance(messages, list)
        assert isinstance(system, str)

    def test_single_user_message(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """The messages list contains exactly one user-role dict."""
        messages, _ = builder.build_prompt(req)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_clinical_text_in_prompt(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """The clinical text appears verbatim inside the user message."""
        messages, _ = builder.build_prompt(req)
        assert "Patient has elevated BP." in messages[0]["content"]

    def test_json_schema_fields_in_prompt(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """All expected output field names appear in the prompt schema."""
        messages, _ = builder.build_prompt(req)
        content = messages[0]["content"]
        expected_fields = ("document_type", "summary", "key_findings", "abnormal_results", "recommended_follow_up")  # noqa: E501
        for field in expected_fields:
            assert field in content, f"Field '{field}' missing from prompt"

    def test_system_prompt_constant_returned(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """The system prompt is the module-level SYSTEM_PROMPT constant."""
        _, system = builder.build_prompt(req)
        assert system == SYSTEM_PROMPT

    def test_clinical_document_xml_tag(
        self, builder: PromptBuilder, req: ClinicalSummaryRequest
    ) -> None:
        """The prompt wraps clinical text in <clinical_document> XML tags."""
        messages, _ = builder.build_prompt(req)
        content = messages[0]["content"]
        assert "<clinical_document>" in content
        assert "</clinical_document>" in content

    def test_different_texts_produce_different_prompts(
        self, builder: PromptBuilder
    ) -> None:
        """Two requests with different texts produce different prompts."""
        req_a = ClinicalSummaryRequest(text="Text A", request_id="r1")
        req_b = ClinicalSummaryRequest(text="Text B", request_id="r2")
        msgs_a, _ = builder.build_prompt(req_a)
        msgs_b, _ = builder.build_prompt(req_b)
        assert msgs_a[0]["content"] != msgs_b[0]["content"]


# ---------------------------------------------------------------------------
# SummaryParser tests
# ---------------------------------------------------------------------------


class TestSummaryParser:
    """Tests for SummaryParser.parse."""

    @pytest.fixture()
    def parser(self) -> SummaryParser:
        return SummaryParser()

    @pytest.fixture()
    def valid_json(self) -> str:
        return json.dumps(
            {
                "document_type": "pcp_note",
                "summary": "Patient seen for fatigue and elevated BP.",
                "key_findings": ["Fatigue reported", "BP 152/94 mmHg"],
                "abnormal_results": ["Elevated blood pressure"],
                "recommended_follow_up": ["Start lisinopril", "Repeat BP check in 4 weeks"],
            }
        )

    def test_happy_path(self, parser: SummaryParser, valid_json: str) -> None:
        """Valid JSON string is parsed into a ClinicalSummary."""
        summary = parser.parse(valid_json)
        assert isinstance(summary, ClinicalSummary)
        assert summary.document_type == "pcp_note"

    def test_all_fields_populated(self, parser: SummaryParser, valid_json: str) -> None:
        """All ClinicalSummary fields are correctly populated from JSON."""
        summary = parser.parse(valid_json)
        assert summary.summary == "Patient seen for fatigue and elevated BP."
        assert "Fatigue reported" in summary.key_findings
        assert "Elevated blood pressure" in summary.abnormal_results
        assert "Start lisinopril" in summary.recommended_follow_up

    def test_strips_markdown_fences(self, parser: SummaryParser, valid_json: str) -> None:
        """Markdown code fences wrapping the JSON are stripped before parsing."""
        fenced = f"```json\n{valid_json}\n```"
        summary = parser.parse(fenced)
        assert summary.document_type == "pcp_note"

    def test_strips_plain_fences(self, parser: SummaryParser, valid_json: str) -> None:
        """Plain triple-backtick fences (no language tag) are also stripped."""
        fenced = f"```\n{valid_json}\n```"
        summary = parser.parse(fenced)
        assert summary.document_type == "pcp_note"

    def test_empty_lists_accepted(self, parser: SummaryParser) -> None:
        """All list fields may be empty arrays in the JSON."""
        raw = json.dumps(
            {
                "document_type": "blood_test",
                "summary": "All results normal.",
                "key_findings": [],
                "abnormal_results": [],
                "recommended_follow_up": [],
            }
        )
        summary = parser.parse(raw)
        assert summary.key_findings == []

    def test_invalid_json_raises_summary_parsing_error(
        self, parser: SummaryParser
    ) -> None:
        """Non-JSON text raises SummaryParsingError."""
        with pytest.raises(SummaryParsingError, match="not valid JSON"):
            parser.parse("This is not JSON at all.")

    def test_json_wrong_schema_raises_summary_parsing_error(
        self, parser: SummaryParser
    ) -> None:
        """Valid JSON that does not match the ClinicalSummary schema raises SummaryParsingError."""
        wrong_shape = json.dumps({"unexpected_field": "value"})
        with pytest.raises(SummaryParsingError, match="schema"):
            parser.parse(wrong_shape)

    def test_document_type_present_in_output(
        self, parser: SummaryParser, valid_json: str
    ) -> None:
        """The document_type field is always present in the parsed output."""
        summary = parser.parse(valid_json)
        assert summary.document_type

    def test_partial_json_raises_summary_parsing_error(
        self, parser: SummaryParser
    ) -> None:
        """Truncated JSON raises SummaryParsingError."""
        with pytest.raises(SummaryParsingError):
            parser.parse('{"document_type": "blood_test"')

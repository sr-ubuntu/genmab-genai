"""Pure business logic: prompt construction and response parsing.

This module contains zero I/O. All classes accept plain data and return plain
data, making them trivially testable without mocking any AWS services.
"""

from __future__ import annotations

import json
import re

from clinical_summarizer.exceptions import SummaryParsingError
from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical document summarization assistant. Your task is to analyze "
    "clinical documents — including blood test reports, colonoscopy reports, and "
    "physician notes — and produce structured summaries. "
    "You MUST respond with valid JSON only. Do NOT include any prose, explanation, "
    "or markdown code fences outside the JSON object."
)

_USER_PROMPT_TEMPLATE = """\
Analyze the following clinical document and return a JSON object matching this exact schema:

{{
  "document_type": "<string: detected category, e.g. pcp_note, blood_test, colonoscopy_report>",
  "summary": "<string: 2-3 sentence plain-English overview of the document>",
  "key_findings": ["<string: notable clinical observation or measurement>"],
  "abnormal_results": ["<string: finding outside normal range or clinically significant>"],
  "recommended_follow_up": ["<string: actionable next step explicitly stated in the document>"]
}}

Rules:
- All list fields must be JSON arrays (use [] if there are no items for that field).
- Never fabricate values not present in the source document.
- Use clinical terminology appropriate for a medical professional audience.
- Respond with the JSON object only — no surrounding text.

<clinical_document>
{clinical_text}
</clinical_document>"""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Constructs the Bedrock messages payload from a ClinicalSummaryRequest.

    This class is stateless; a single instance may be shared safely across
    concurrent invocations.
    """

    def build_prompt(
        self, request: ClinicalSummaryRequest
    ) -> tuple[list[dict[str, str]], str]:
        """Build the messages list and system prompt for the Bedrock Messages API.

        Args:
            request: A validated inbound summarization request.

        Returns:
            A tuple of ``(messages, system_prompt)`` where ``messages`` is a
            list containing a single user turn and ``system_prompt`` is the
            role-definition string to pass as the ``system`` field.
        """
        user_content = _USER_PROMPT_TEMPLATE.format(clinical_text=request.text)
        messages = [{"role": "user", "content": user_content}]
        return messages, SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SummaryParser
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class SummaryParser:
    """Parses the raw string returned by Bedrock into a ClinicalSummary.

    This class is stateless; a single instance may be shared safely across
    concurrent invocations.
    """

    def parse(self, raw: str) -> ClinicalSummary:
        """Parse the model's raw text output into a validated ClinicalSummary.

        Args:
            raw: The raw text response from the Bedrock model.

        Returns:
            A fully validated :class:`ClinicalSummary` instance.

        Raises:
            SummaryParsingError: If the response is not valid JSON or does not
                conform to the expected schema.
        """
        json_str = self._extract_json(raw)
        parsed = self._load_json(json_str)
        return self._validate_schema(parsed)

    def _extract_json(self, raw: str) -> str:
        """Strip accidental markdown code fences and return the JSON substring."""
        match = _FENCE_PATTERN.search(raw)
        if match:
            return match.group(1).strip()
        return raw.strip()

    def _load_json(self, json_str: str) -> dict:
        """Deserialise the JSON string, raising SummaryParsingError on failure."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise SummaryParsingError(
                "Bedrock response is not valid JSON.",
                context={"raw_response": json_str[:500], "json_error": str(exc)},
            ) from exc

    def _validate_schema(self, parsed: dict) -> ClinicalSummary:
        """Validate the parsed dict against the ClinicalSummary schema."""
        try:
            return ClinicalSummary.model_validate(parsed)
        except Exception as exc:
            raise SummaryParsingError(
                "Bedrock response does not match the expected summary schema.",
                context={"validation_error": str(exc)},
            ) from exc

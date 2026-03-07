"""Shared pytest fixtures for the clinical summarizer test suite."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from clinical_summarizer.models import ClinicalSummary, ClinicalSummaryRequest
from clinical_summarizer.bedrock_client import BedrockRepository

# ---------------------------------------------------------------------------
# Sample clinical text fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blood_test_text() -> str:
    """Realistic synthetic blood test report."""
    return (
        "Patient: Jane Smith, DOB: 1972-08-20. CBC and metabolic panel results dated 2026-02-28. "
        "WBC: 11.2 K/uL (ref 4.5-11.0) — mildly elevated. RBC: 4.1 M/uL (ref 4.2-5.4) — low. "
        "Hemoglobin: 10.8 g/dL (ref 12.0-16.0) — below normal, consistent with mild anemia. "
        "Platelets: 210 K/uL (ref 150-400) — normal. "
        "Sodium: 138 mEq/L (ref 136-145). Potassium: 4.1 mEq/L (ref 3.5-5.0). "
        "Creatinine: 1.1 mg/dL (ref 0.6-1.1). eGFR: 68 mL/min — borderline. "
        "Total Cholesterol: 245 mg/dL (ref <200) — elevated. LDL: 162 mg/dL (ref <130) — elevated. "
        "Physician recommends repeat CBC in 6 weeks, iron studies, and lipid management consult."
    )


@pytest.fixture()
def colonoscopy_text() -> str:
    """Realistic synthetic colonoscopy report."""
    return (
        "Procedure: Colonoscopy. Patient: Robert Nguyen, 58 y/o male. Date: 2026-03-01. "
        "Indication: Screening. Bowel prep: Adequate. Cecum reached: Yes. "
        "Findings: Two 4mm hyperplastic polyps in the sigmoid colon — removed by cold snare. "
        "One 8mm tubular adenoma in the ascending colon — removed by hot snare polypectomy. "
        "No bleeding, perforation, or other complications. "
        "Impression: Low-risk adenoma. Recommend repeat colonoscopy in 3 years."
    )


@pytest.fixture()
def pcp_note_text() -> str:
    """Realistic synthetic PCP visit note."""
    return (
        "S: Patient presents with 2-week history of fatigue and intermittent headaches. "
        "BP elevated at last two visits. No chest pain, no dyspnea. "
        "O: BP 152/94 mmHg (elevated), HR 78 bpm, weight 91 kg (BMI 29.4). "
        "A: Stage 1 hypertension, fatigue — etiology unclear. "
        "P: Start lisinopril 10 mg daily. Order CBC, BMP, TSH. "
        "Home BP monitoring twice daily. Follow up in 4 weeks."
    )


# ---------------------------------------------------------------------------
# Request fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_request(blood_test_text: str) -> ClinicalSummaryRequest:
    """A fully valid ClinicalSummaryRequest."""
    return ClinicalSummaryRequest(text=blood_test_text, request_id="req-abc-123")


# ---------------------------------------------------------------------------
# Bedrock response fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_summary_dict() -> dict:
    """A valid ClinicalSummary as a plain dict (mirrors expected Bedrock JSON output)."""
    return {
        "document_type": "blood_test",
        "summary": (
            "Blood panel for a 53-year-old female showing mild anemia and elevated cholesterol. "
            "Borderline eGFR and mildly elevated WBC also noted. Follow-up recommended."
        ),
        "key_findings": [
            "Hemoglobin 10.8 g/dL (below normal 12.0-16.0)",
            "Total Cholesterol 245 mg/dL (elevated, ref <200)",
            "LDL 162 mg/dL (elevated, ref <130)",
            "WBC 11.2 K/uL (mildly elevated)",
            "eGFR 68 mL/min (borderline)",
        ],
        "abnormal_results": [
            "Hemoglobin below normal range — mild anemia",
            "Total Cholesterol elevated above recommended threshold",
            "LDL elevated above recommended threshold",
        ],
        "recommended_follow_up": [
            "Repeat CBC in 6 weeks",
            "Iron studies to evaluate anemia etiology",
            "Lipid management consultation",
        ],
    }


@pytest.fixture()
def valid_bedrock_response(valid_summary_dict: dict) -> str:
    """JSON string as would be returned by Bedrock."""
    return json.dumps(valid_summary_dict)


@pytest.fixture()
def valid_summary(valid_summary_dict: dict) -> ClinicalSummary:
    """A fully validated ClinicalSummary instance."""
    return ClinicalSummary.model_validate(valid_summary_dict)


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_bedrock_client(valid_bedrock_response: str) -> MagicMock:
    """A MagicMock boto3 bedrock-runtime client pre-configured to return a valid response."""
    client = MagicMock()
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps(
        {"content": [{"text": valid_bedrock_response}]}
    ).encode()
    client.invoke_model.return_value = {"body": body_mock}
    return client


@pytest.fixture()
def mock_repository(valid_bedrock_response: str) -> MagicMock:
    """A MagicMock BedrockRepository that returns a valid Bedrock JSON string."""
    repo = MagicMock(spec=BedrockRepository)
    repo.invoke_model.return_value = valid_bedrock_response
    return repo


@pytest.fixture()
def mock_logger() -> MagicMock:
    """A MagicMock logger compatible with logging.Logger."""
    return MagicMock()

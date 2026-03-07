# Clinical Text Summarization API

A serverless Generative AI API that transforms unstructured clinical text into
structured JSON summaries using AWS Lambda and Amazon Bedrock (Claude 3 Haiku).

---

## Problem Statement

Healthcare systems generate large volumes of unstructured clinical text —
physician notes, lab interpretations, colonoscopy reports, patient summaries —
that contain critical information buried in narrative form. This API extracts
that information automatically and returns it as machine-readable JSON.

---

## Architecture

```
Client
  │  POST /summarize  { "text": "...", "request_id": "..." }
  ▼
API Gateway (REST)
  │
  ▼
AWS Lambda  (main.handler)
  │
  ├── PromptBuilder   builds Bedrock messages payload       [core.py]
  ├── BedrockRepository  invokes Claude 3 Haiku             [bedrock_client.py]
  ├── SummaryParser   parses raw JSON response              [core.py]
  └── SummarizationService  orchestrates the pipeline       [services.py]
  │
  ▼
Amazon Bedrock  (anthropic.claude-3-haiku-20240307-v1:0)
```

---

## API Contract

### Request

```http
POST /summarize
Content-Type: application/json

{
  "text": "Patient seen for fatigue and elevated blood pressure ...",
  "request_id": "abbaaa"
}
```

| Field        | Type   | Required | Description                              |
|--------------|--------|----------|------------------------------------------|
| `text`       | string | Yes      | Raw clinical document (1–50,000 chars)   |
| `request_id` | string | Yes      | Caller-supplied ID for log correlation   |

### Success Response — 200 OK

```json
{
  "document_type": "pcp_note",
  "summary": "Patient seen for fatigue and elevated blood pressure. PCP advised repeat labs and home BP monitoring.",
  "key_findings": [
    "Fatigue reported",
    "Elevated blood pressure noted"
  ],
  "abnormal_results": [
    "Elevated blood pressure"
  ],
  "recommended_follow_up": [
    "Repeat labs",
    "Home blood pressure monitoring"
  ]
}
```

| Field                   | Type          | Description                                   |
|-------------------------|---------------|-----------------------------------------------|
| `document_type`         | string        | Auto-detected category (e.g. `pcp_note`)      |
| `summary`               | string        | 2-3 sentence plain-English overview           |
| `key_findings`          | list[string]  | Notable observations from the document        |
| `abnormal_results`      | list[string]  | Findings outside normal range                 |
| `recommended_follow_up` | list[string]  | Actionable next steps from the document       |

### Error Responses

| Status | Condition                                  |
|--------|--------------------------------------------|
| 400    | Missing or malformed request body          |
| 422    | Field validation failure (blank text, etc.)|
| 429    | Bedrock rate limit exceeded                |
| 500    | Model response parsing failure             |
| 502    | Bedrock service error                      |

Error body:
```json
{
  "error": "ValidationError",
  "message": "Clinical text must not be blank.",
  "request_id": "abbaaa"
}
```

---

## Project Structure

```
├── src/clinical_summarizer/
│   ├── exceptions.py      Custom exception hierarchy
│   ├── models.py          Pydantic v2 data models
│   ├── core.py            PromptBuilder + SummaryParser (pure, no I/O)
│   ├── bedrock_client.py  Amazon Bedrock integration
│   ├── services.py        Orchestration layer
│   └── utils.py           Lambda event parsing + response helpers
├── terraform/
│   ├── main.tf            AWS infrastructure (Lambda, API Gateway, IAM, CloudWatch)
│   ├── variables.tf       Input variables with defaults
│   └── outputs.tf         Outputs: API URL, Lambda ARN, log group
├── scripts/
│   ├── build.sh           Build lambda.zip on Mac/Linux
│   └── build.ps1          Build lambda.zip on Windows (PowerShell)
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_core.py
│   ├── test_bedrock_client.py
│   ├── test_services.py
│   └── test_main.py
├── docs/
│   └── assumptions.md  Design decisions and trade-offs
├── main.py             Lambda handler entry point
├── pyproject.toml
└── requirements.txt
```

---

## Local Setup

### Prerequisites

- Python 3.12
- AWS credentials with `bedrock:InvokeModel` permission (for live testing only)

### Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

Expected output: 70 tests pass, 100% coverage.

### Lint

```bash
ruff check src/ tests/ main.py
```

### Local Lambda Simulation (requires AWS credentials)

```bash
python -c "
from main import handler
event = {
    'body': '{\"text\": \"Patient presents with elevated WBC...\", \"request_id\": \"test-123\"}',
    'httpMethod': 'POST'
}
import types
ctx = types.SimpleNamespace(aws_request_id='local-001')
import json; print(json.dumps(json.loads(handler(event, ctx)['body']), indent=2))
"
```

---

## Deployment

### Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.6
- AWS credentials configured (`aws configure` or environment variables)
- Claude 3 Haiku activated in your account — open the [Bedrock Chat Playground](https://console.aws.amazon.com/bedrock/home#/chat-playground), select **Claude 3 Haiku**, and send a message. First-time Anthropic users will be prompted to submit use case details before the model becomes available.

### Step 1 — Build the Lambda package

```bash
# Mac/Linux
bash scripts/build.sh

# Windows (PowerShell)
.\scripts\build.ps1
```

This creates `lambda.zip` in the project root containing the application code
and all runtime dependencies compiled for Linux x86_64.

### Step 2 — Deploy with Terraform

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

Terraform will output the API URL when complete:

```
Outputs:
  api_url = "https://abc123.execute-api.us-east-1.amazonaws.com/prod/summarize"
```

### Step 3 — Test the live endpoint

```bash
curl -X POST <api_url> \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient presents with fatigue and elevated BP 152/94. Start lisinopril 10mg.",
    "request_id": "test-001"
  }'
```

### Tear down

```bash
cd terraform
terraform destroy
```

### Terraform variables

| Variable              | Default                                  | Description                    |
|-----------------------|------------------------------------------|--------------------------------|
| `aws_region`          | `us-east-1`                              | Deployment region              |
| `environment`         | `prod`                                   | Environment tag                |
| `bedrock_model_id`    | `anthropic.claude-3-haiku-20240307-v1:0` | Bedrock model to use           |
| `lambda_memory_mb`    | `512`                                    | Lambda memory in MB            |
| `lambda_timeout_sec`  | `30`                                     | Lambda timeout in seconds      |
| `log_retention_days`  | `14`                                     | CloudWatch log retention       |

Override any variable:
```bash
terraform apply -var="environment=staging" -var="lambda_memory_mb=1024"
```

### What Terraform creates

| Resource | Name |
|----------|------|
| IAM Role + Policy (Lambda) | `clinical-summarizer-prod-role` |
| IAM Role (API Gateway → CloudWatch) | `clinical-summarizer-prod-apigw-cw-role` |
| API Gateway Account settings | CloudWatch role ARN configured account-wide |
| Lambda Function | `clinical-summarizer-prod` |
| CloudWatch Log Group | `/aws/lambda/clinical-summarizer-prod` |
| API Gateway REST API | `clinical-summarizer-prod` |
| API Gateway Stage | `prod` |



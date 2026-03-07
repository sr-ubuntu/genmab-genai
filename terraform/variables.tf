variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment tag (e.g. prod, staging, dev)."
  type        = string
  default     = "prod"
}

variable "bedrock_model_id" {
  description = "Amazon Bedrock model identifier used for summarization."
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}

variable "lambda_memory_mb" {
  description = "Lambda function memory allocation in MB."
  type        = number
  default     = 512
}

variable "lambda_timeout_sec" {
  description = "Lambda function timeout in seconds."
  type        = number
  default     = 30
}

variable "log_retention_days" {
  description = "Number of days to retain Lambda logs in CloudWatch."
  type        = number
  default     = 14
}

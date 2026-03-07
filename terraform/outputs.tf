output "api_url" {
  description = "Full URL of the POST /summarize endpoint."
  value       = "${aws_api_gateway_stage.api.invoke_url}/summarize"
}

output "lambda_function_name" {
  description = "Name of the deployed Lambda function."
  value       = aws_lambda_function.summarizer.function_name
}

output "lambda_function_arn" {
  description = "ARN of the deployed Lambda function."
  value       = aws_lambda_function.summarizer.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for Lambda logs."
  value       = aws_cloudwatch_log_group.lambda.name
}

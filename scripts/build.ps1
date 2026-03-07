# Build the Lambda deployment package (lambda.zip) on Windows.
# Run from the project root:
#   .\scripts\build.ps1
#
# Requires: Python/pip on PATH, zip support via Compress-Archive (built into PowerShell 5+).
# Uses --platform manylinux2014_x86_64 so the package runs on AWS Lambda (Linux x86_64)
# even when built on Windows.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PackageDir = "package"
$ZipFile    = "lambda.zip"

Write-Host "==> Cleaning previous build..."
if (Test-Path $PackageDir) { Remove-Item -Recurse -Force $PackageDir }
if (Test-Path $ZipFile)    { Remove-Item -Force $ZipFile }
New-Item -ItemType Directory -Path $PackageDir | Out-Null

Write-Host "==> Installing runtime dependencies for Linux x86_64..."
pip install `
    boto3 `
    pydantic `
    "aws-lambda-powertools" `
    --target ".\$PackageDir" `
    --platform manylinux2014_x86_64 `
    --python-version 3.12 `
    --only-binary=:all: `
    --upgrade `
    --quiet

Write-Host "==> Copying application source..."
Copy-Item -Recurse "src\clinical_summarizer" "$PackageDir\"
Copy-Item "main.py" "$PackageDir\"

Write-Host "==> Creating $ZipFile..."
Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipFile

$Size = (Get-Item $ZipFile).Length / 1MB
Write-Host ("==> Done. Package size: {0:N1} MB" -f $Size)
Write-Host "    Path: $(Resolve-Path $ZipFile)"

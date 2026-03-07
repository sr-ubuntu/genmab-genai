#!/usr/bin/env bash
# Build the Lambda deployment package (lambda.zip).
# Must be run from the project root.
# Requires: pip, zip
#
# NOTE: Uses --platform manylinux2014_x86_64 so the package runs on
# AWS Lambda (Linux x86_64) even when built on macOS or Windows.

set -euo pipefail

PACKAGE_DIR="package"
ZIP_FILE="lambda.zip"

echo "==> Cleaning previous build..."
rm -rf "$PACKAGE_DIR" "$ZIP_FILE"
mkdir -p "$PACKAGE_DIR"

echo "==> Installing runtime dependencies for Linux x86_64..."
pip install \
  boto3 \
  pydantic \
  "aws-lambda-powertools" \
  --target "./$PACKAGE_DIR" \
  --platform manylinux2014_x86_64 \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  --quiet

echo "==> Copying application source..."
cp -r src/clinical_summarizer "$PACKAGE_DIR/"
cp main.py "$PACKAGE_DIR/"

echo "==> Creating $ZIP_FILE..."
cd "$PACKAGE_DIR"
zip -r "../$ZIP_FILE" . --quiet
cd ..

echo "==> Done. Package size: $(du -sh $ZIP_FILE | cut -f1)"
echo "    Path: $(pwd)/$ZIP_FILE"

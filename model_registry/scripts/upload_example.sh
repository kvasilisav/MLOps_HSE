#!/bin/bash
FILE="${1:-/tmp/dummy_model.bin}"
echo "test" > "$FILE" 2>/dev/null || true

curl -X POST "http://localhost:8000/models" \
  -F "name=example_model" \
  -F "team=mlds_1" \
  -F "metadata={\"dataset\": \"train_v1\", \"metric\": 0.95}" \
  -F "tags={\"task\": \"classification\", \"env\": \"staging\"}" \
  -F "status=staging" \
  -F "file=@$FILE"

echo ""

#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
if [ ! -d onnx_model ] || [ -z "$(ls -A onnx_model 2>/dev/null)" ]; then
  echo "Exporting ONNX..."
  python export_onnx.py
fi
echo "Part 3 — Dynamic batching. Starting server on :8002..."
uvicorn part3_batching.app:app --host 0.0.0.0 --port 8002

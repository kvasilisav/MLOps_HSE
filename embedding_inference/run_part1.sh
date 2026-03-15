#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "Part 1 — Basic (HF). Starting server on :8000..."
uvicorn part1_basic.app:app --host 0.0.0.0 --port 8000

#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Run Part 1 server in another terminal: ./run_part1.sh"
echo "Then: python part1_basic/benchmark.py"
echo ""
echo "Run Part 2 server: ./run_part2.sh"
echo "Then: python part2_onnx/benchmark.py"
echo ""
echo "Run Part 3 server: ./run_part3.sh"
echo "Then: python part3_batching/benchmark.py"

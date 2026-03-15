#!/bin/bash
set -e
cd "$(dirname "$0")"

DOCKER_COMPOSE="docker compose"
command -v docker-compose &>/dev/null && DOCKER_COMPOSE="docker-compose"

IMAGE_NAME=embedding_inference

docker build -t "$IMAGE_NAME" .
echo "Контейнер запущен. Внутри:"
echo "  1) Экспорт ONNX (один раз): python export_onnx.py"
echo "  2) Часть 1: ./run_part1.sh &   затем  python part1_basic/benchmark.py"
echo "  3) Часть 2: ./run_part2.sh &   затем  python part2_onnx/benchmark.py"
echo "  4) Часть 3: ./run_part3.sh &   затем  python part3_batching/benchmark.py"
echo ""
docker run -it --rm -v "$(pwd):/app" -w /app -p 8000:8000 -p 8001:8001 -p 8002:8002 "$IMAGE_NAME" /bin/bash

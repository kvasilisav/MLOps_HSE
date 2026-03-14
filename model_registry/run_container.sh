#!/bin/bash
set -e
cd "$(dirname "$0")"

DOCKER_COMPOSE="docker compose"
command -v docker-compose &>/dev/null && DOCKER_COMPOSE="docker-compose"

mkdir -p data
docker build -t model_registry .
docker run -it --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -e REGISTRY_DB=/app/data/registry.db \
  -e REGISTRY_STORAGE=/app/data/models_storage \
  --name "${USER}_model_registry" \
  model_registry

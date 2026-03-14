#!/bin/bash
set -e
cd "$(dirname "$0")"

DOCKER_COMPOSE="docker compose"
command -v docker-compose &>/dev/null && DOCKER_COMPOSE="docker-compose"

RUN="-v $(pwd):/app -w /app"
if [ "${1:-}" = "shell" ]; then
  $DOCKER_COMPOSE run --rm $RUN registry /bin/bash
else
  $DOCKER_COMPOSE build
  $DOCKER_COMPOSE run --rm $RUN registry python -m pytest tests/ -v
fi

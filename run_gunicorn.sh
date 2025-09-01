#!/usr/bin/env bash
set -euo pipefail

: "${UVICORN_WORKERS:=4}"
: "${PORT:=8080}"

exec gunicorn backend.api.main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers "${UVICORN_WORKERS}" \
  --bind 0.0.0.0:"${PORT}" \
  --timeout 60 \
  --graceful-timeout 30 \
  --log-level info



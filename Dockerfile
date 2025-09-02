# Use stable Python 3.12 (same as training environment)
FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    UVICORN_WORKERS=2

# system deps (Biopython/scipy sometimes need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential procps curl && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
COPY requirements-backend.txt .
RUN pip install -r requirements.txt

# copy app
COPY . .

# Copy models into image (including ClinVar models)
COPY models /app/models

# Set default env vars with UNDERSCORES (no hyphens) - can be overridden by platform secrets
ENV EWCLV1_MODEL_PATH=/app/models/disorder/ewclv1.pkl \
    EWCLV1_M_MODEL_PATH=/app/models/disorder/ewclv1-M.pkl \
    EWCLV1_P3_MODEL_PATH=/app/models/pdb/ewclv1p3.pkl \
    EWCLV1_C_MODEL_PATH=/app/models/clinvar/ewclv1-C.pkl \
    EWCLV1_C_FEATURES_PATH=/app/models/clinvar/EWCLv1-C_features.json \
    MAX_BODY_BYTES=100000000 \
    PORT=8080

EXPOSE $PORT

# Health check (works with both Fly.io and Railway)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/healthz || exit 1

# Use PORT env var for both platforms
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

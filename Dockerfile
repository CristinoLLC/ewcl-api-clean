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

# Create model directory and copy entire tree for robustness (avoid per-file COPY fragility)
RUN mkdir -p /app/models
COPY models/ /app/models/
# List copied model artifacts for debugging
RUN find /app/models -maxdepth 3 -type f -printf "%P\n" | sort

# Set default env vars (can be overridden). Use consistent lowercase filenames.
ENV EWCLV1_MODEL_PATH=/app/models/disorder/ewclv1.pkl \
    EWCLV1_M_MODEL_PATH=/app/models/disorder/ewclv1-M.pkl \
    EWCLV1_P3_MODEL_PATH=/app/models/pdb/ewclv1p3.pkl \
    EWCLV1_C_MODEL_PATH=/app/models/clinvar/ewclv1-c.pkl \
    EWCLV1_C_FEATURES_PATH=/app/models/clinvar/ewclv1-c_features.json \
    MAX_BODY_BYTES=100000000 \
    PORT=8080

EXPOSE 8080

# Health check (works with both Fly.io and Railway)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Use a shell so ${PORT} expands (Railway sets PORT). Fallback to 8080 locally.
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

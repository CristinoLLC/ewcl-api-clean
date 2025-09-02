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

# Create target directories explicitly
RUN mkdir -p /app/models/disorder /app/models/clinvar /app/models/pdb

# Copy models from actual repo structure - disorder models
COPY models/disorder/ewclv1.pkl /app/models/disorder/ewclv1.pkl
COPY models/disorder/ewclv1-M.pkl /app/models/disorder/ewclv1-M.pkl

# Copy PDB model
COPY models/pdb/ewclv1p3.pkl /app/models/pdb/ewclv1p3.pkl

# Copy ClinVar models and features
COPY models/clinvar/ewclv1c.pkl /app/models/clinvar/ewclv1c.pkl
COPY models/clinvar/EWCLv1-C_features.json /app/models/clinvar/EWCLv1-C_features.json

# Copy additional backend_bundle models if they exist
COPY backend_bundle/models/EWCLv1C_Gate.pkl /app/models/clinvar/EWCLv1C_Gate.pkl

# Set default env vars with UNDERSCORES (no hyphens) - can be overridden by platform secrets
ENV EWCLV1_MODEL_PATH=/app/models/disorder/ewclv1.pkl \
    EWCLV1_M_MODEL_PATH=/app/models/disorder/ewclv1-M.pkl \
    EWCLV1_P3_MODEL_PATH=/app/models/pdb/ewclv1p3.pkl \
    EWCLV1_C_MODEL_PATH=/app/models/clinvar/ewclv1c.pkl \
    EWCLV1_C_FEATURES_PATH=/app/models/clinvar/EWCLv1-C_features.json \
    MAX_BODY_BYTES=100000000 \
    PORT=8080

EXPOSE 8080

# Health check (works with both Fly.io and Railway)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Use Python to start the app - this way main.py handles PORT env var properly
CMD ["python", "main.py"]

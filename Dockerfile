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

# Download model files from GitHub (raw content) to avoid LFS issues
ARG EWCLV1_MODEL_URL=https://github.com/CristinoLLC/ewcl-api-clean/raw/main/models/disorder/ewclv1.pkl
ARG EWCLV1M_MODEL_URL=https://github.com/CristinoLLC/ewcl-api-clean/raw/main/models/disorder/ewclv1-M.pkl
ARG EWCLV1P3_MODEL_URL=https://github.com/CristinoLLC/ewcl-api-clean/raw/main/models/pdb/ewclv1p3.pkl
ARG EWCLV1C_FULL_MODEL_URL=https://github.com/CristinoLLC/ewcl-api-clean/raw/main/models/clinvar/C_Full_model.pkl
ARG EWCLV1C_43_MODEL_URL=https://github.com/CristinoLLC/ewcl-api-clean/raw/main/models/clinvar/C_43_model.pkl

# Create model directories and download real model files
RUN mkdir -p /app/models/disorder /app/models/pdb /app/models/clinvar

# Download model files (these are now regular Git objects, not LFS pointers)
RUN echo "Downloading model files..." && \
    curl -fL "$EWCLV1_MODEL_URL" -o /app/models/disorder/ewclv1.pkl && \
    echo "Downloaded ewclv1.pkl: $(ls -lh /app/models/disorder/ewclv1.pkl)" && \
    curl -fL "$EWCLV1M_MODEL_URL" -o /app/models/disorder/ewclv1-M.pkl && \
    echo "Downloaded ewclv1-M.pkl: $(ls -lh /app/models/disorder/ewclv1-M.pkl)" && \
    curl -fL "$EWCLV1P3_MODEL_URL" -o /app/models/pdb/ewclv1p3.pkl && \
    echo "Downloaded ewclv1p3.pkl: $(ls -lh /app/models/pdb/ewclv1p3.pkl)" && \
    curl -fL "$EWCLV1C_FULL_MODEL_URL" -o /app/models/clinvar/C_Full_model.pkl && \
    echo "Downloaded C_Full_model.pkl: $(ls -lh /app/models/clinvar/C_Full_model.pkl)" && \
    curl -fL "$EWCLV1C_43_MODEL_URL" -o /app/models/clinvar/C_43_model.pkl && \
    echo "Downloaded C_43_model.pkl: $(ls -lh /app/models/clinvar/C_43_model.pkl)" && \
    echo "All model downloads completed successfully"

# List downloaded model artifacts for debugging
RUN find /app/models -maxdepth 3 -type f -printf "%P %s\n" | sort

# Set default env vars (can be overridden). Correct names for loader.
ENV EWCLV1_MODEL_PATH=/app/models/disorder/ewclv1.pkl \
    EWCLV1M_MODEL_PATH=/app/models/disorder/ewclv1-M.pkl \
    EWCLV1P3_MODEL_PATH=/app/models/pdb/ewclv1p3.pkl \
    EWCLV1C_MODEL_PATH=/app/models/clinvar/C_Full_model.pkl \
    EWCLV1_C_43_MODEL_PATH=/app/models/clinvar/C_43_model.pkl \
    MAX_BODY_BYTES=100000000 \
    PORT=8080

EXPOSE 8080

# Health check (works with both Fly.io and Railway)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Simple CMD - Python handles PORT reading
CMD ["python", "main.py"]

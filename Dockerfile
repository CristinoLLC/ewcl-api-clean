# Use stable Python 3.12 (same as training environment)
FROM python:3.12.10-slim

# Prevents Python from writing .pyc files / buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better Docker layer caching)
COPY requirements.txt .
COPY requirements-backend.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend
COPY . .

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Start the app with Gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--timeout", "60", "--graceful-timeout", "30", "--log-level", "info"]

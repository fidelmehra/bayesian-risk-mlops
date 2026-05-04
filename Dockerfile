# ============================================================
# Bayesian Risk-MLOps  —  Multi-stage Docker build
# Author: Fidel Mehra
# ============================================================
# Stage 1: builder — install all dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed for scipy / statsmodels compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: runtime image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Fidel Mehra" \
      description="Bayesian Risk-MLOps prediction API" \
      version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/    ./app/
COPY src/    ./src/
COPY config/ ./config/

# Non-root user for security
RUN addgroup --system mlops && adduser --system --ingroup mlops mlops
RUN chown -R mlops:mlops /app
USER mlops

# Create data directories (model artefacts mounted at runtime)
RUN mkdir -p data/models data/processed data/raw reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command: launch the FastAPI service with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

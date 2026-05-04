# ============================================================
# Bayesian Risk-MLOps  —  Makefile
# Author: Fidel Mehra
#
# Usage
# -----
#   make install        Install Python dependencies
#   make ingest         Download & cache raw OHLCV data
#   make features       Build feature matrix
#   make train          Train Bayesian Ridge model
#   make evaluate       Run evaluation & generate report
#   make predict        Run inference on latest observation
#   make serve          Start FastAPI server (dev mode)
#   make docker-build   Build Docker image
#   make docker-up      Start all services via docker-compose
#   make docker-down    Stop docker-compose services
#   make test           Run pytest suite
#   make lint           Run ruff + black + isort checks
#   make format         Auto-format code with black + isort
#   make clean          Remove generated artefacts
#   make pipeline       End-to-end: ingest → features → train → evaluate
# ============================================================

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
UVICORN     ?= uvicorn
CFG         ?= config/config.yaml
IMAGE       ?= bayesian-risk-mlops:latest

.PHONY: install ingest features train evaluate predict serve \
        docker-build docker-up docker-down \
        test lint format clean pipeline help

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Done."

# ------------------------------------------------------------
# Data pipeline steps
# ------------------------------------------------------------
ingest:
	@echo "==> Ingesting raw OHLCV data..."
	$(PYTHON) src/ingestion/ingest_data.py

features: ingest
	@echo "==> Building feature matrix..."
	$(PYTHON) src/features/build_features.py

train: features
	@echo "==> Training Bayesian Ridge model..."
	$(PYTHON) src/train_model.py

evaluate: train
	@echo "==> Evaluating model and generating risk report..."
	$(PYTHON) src/evaluate_model.py

predict:
	@echo "==> Running inference on latest observation..."
	$(PYTHON) src/predict.py

pipeline: ingest features train evaluate
	@echo "==> Full pipeline complete."

# ------------------------------------------------------------
# API server
# ------------------------------------------------------------
serve:
	@echo "==> Starting FastAPI dev server on http://localhost:8000 ..."
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

# ------------------------------------------------------------
# Docker
# ------------------------------------------------------------
docker-build:
	@echo "==> Building Docker image $(IMAGE)..."
	docker build -t $(IMAGE) --target runtime .

docker-up:
	@echo "==> Starting services with docker-compose..."
	docker compose up -d

docker-down:
	@echo "==> Stopping docker-compose services..."
	docker compose down

# ------------------------------------------------------------
# Testing & linting
# ------------------------------------------------------------
test:
	@echo "==> Running test suite..."
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=app --cov-report=term-missing

lint:
	@echo "==> Running linters..."
	$(PYTHON) -m ruff check src/ app/ tests/ --output-format concise
	$(PYTHON) -m black --check src/ app/ tests/
	$(PYTHON) -m isort --check-only src/ app/ tests/

format:
	@echo "==> Auto-formatting code..."
	$(PYTHON) -m black src/ app/ tests/
	$(PYTHON) -m isort src/ app/ tests/

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	@echo "==> Cleaning generated artefacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov coverage.xml .coverage
	@echo "Artefacts cleaned."

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Run 'make pipeline' for a full end-to-end execution."

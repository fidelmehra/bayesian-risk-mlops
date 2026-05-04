"""tests/test_api.py
Bayesian Risk-MLOps — FastAPI Endpoint Tests
Author: Fidel Mehra

Uses httpx + pytest-asyncio to exercise all API routes
with a mocked predict_latest / predict_from_features.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Patch config loading before importing app
MOCK_CFG = {
    "data": {
        "primary_ticker": "SPY",
        "start_date": "2018-01-01",
        "end_date": "2023-12-31",
        "raw_dir": "data/raw",
        "cache_dir": "data/raw",
        "processed_dir": "data/processed",
        "models_dir": "data/models",
        "reports_dir": "reports",
    },
    "model": {
        "target_horizon_days": 5,
    },
    "mlflow": {
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment_name": "test_exp",
        "registered_model_name": "test_model",
    },
}

MOCK_RESULT = MagicMock()
MOCK_RESULT.to_dict.return_value = {
    "ticker": "SPY",
    "horizon_days": 5,
    "predicted_log_return": 0.004,
    "predicted_return_pct": 0.4,
    "uncertainty_std": 0.008,
    "lower_95": -0.012,
    "upper_95": 0.020,
    "var_95": -0.009,
    "signal": "LONG",
    "confidence_score": 0.69,
}


@pytest.fixture(scope="module")
def client():
    with (
        patch("yaml.safe_load", return_value=MOCK_CFG),
        patch("builtins.open", MagicMock()),
        patch("app.main.predict_latest", return_value=MOCK_RESULT),
        patch("app.main.predict_from_features", return_value=[MOCK_RESULT]),
    ):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert "ticker" in data
        assert "model" in data


# ---------------------------------------------------------------------------
# /predict/latest
# ---------------------------------------------------------------------------

class TestPredictLatestEndpoint:
    def test_status_200(self, client):
        resp = client.get("/predict/latest")
        assert resp.status_code == 200

    def test_required_fields_present(self, client):
        resp = client.get("/predict/latest")
        data = resp.json()
        required = [
            "ticker", "horizon_days", "predicted_log_return",
            "predicted_return_pct", "uncertainty_std",
            "lower_95", "upper_95", "var_95", "signal", "confidence_score",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_signal_valid(self, client):
        resp = client.get("/predict/latest")
        assert resp.json()["signal"] in ("LONG", "FLAT")

    def test_confidence_score_range(self, client):
        resp = client.get("/predict/latest")
        score = resp.json()["confidence_score"]
        assert 0.0 <= score <= 1.0

    def test_pi_ordering(self, client):
        resp = client.get("/predict/latest")
        data = resp.json()
        assert data["lower_95"] <= data["upper_95"]


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredictPostEndpoint:
    PAYLOAD = {
        "features": {
            "returns_1d": 0.010,
            "returns_5d": 0.025,
            "rsi_14": 58.3,
            "gk_vol_20": 0.015,
            "bb_pband": 0.72,
        }
    }

    def test_status_200(self, client):
        resp = client.post("/predict", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post("/predict", json=self.PAYLOAD)
        data = resp.json()
        assert isinstance(data["predicted_log_return"], float)
        assert isinstance(data["horizon_days"], int)

    def test_missing_features_key(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_status_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert "bayesian_risk" in resp.text


# ---------------------------------------------------------------------------
# /docs  (Swagger UI)
# ---------------------------------------------------------------------------

class TestDocsEndpoint:
    def test_swagger_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200

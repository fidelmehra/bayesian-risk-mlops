"""app/main.py
Bayesian Risk-MLOps — FastAPI Prediction Service
Author: Fidel Mehra

Endpoints
---------
GET  /health          – liveness probe
GET  /predict/latest  – inference on most recent feature row
POST /predict         – inference on caller-supplied feature dict
GET  /metrics         – Prometheus-compatible plaintext metrics
"""

import time
import logging
import yaml
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predict import predict_latest, predict_from_features, PredictionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
cfg: dict = yaml.safe_load(_CFG_PATH.open())

app = FastAPI(
    title="Bayesian Risk-MLOps API",
    description=(
        "Probabilistic return forecasting service for equity time-series. "
        "Exposes Bayesian point estimates, uncertainty bands, VaR, and trading signals."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory counters for /metrics
_request_count: dict[str, int] = {}
_latency_sum: dict[str, float] = {}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    path = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    _request_count[path] = _request_count.get(path, 0) + 1
    _latency_sum[path] = _latency_sum.get(path, 0.0) + elapsed
    return response


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    ticker: str
    model: str


class PredictResponse(BaseModel):
    ticker: str
    horizon_days: int
    predicted_log_return: float = Field(..., description="E[log(P_{t+h}/P_t)]")
    predicted_return_pct: float = Field(..., description="Point estimate as percentage return")
    uncertainty_std: float = Field(..., description="Bayesian posterior predictive std")
    lower_95: float = Field(..., description="95 %% prediction interval lower bound")
    upper_95: float = Field(..., description="95 %% prediction interval upper bound")
    var_95: float = Field(..., description="Parametric VaR at 95 %% confidence")
    signal: str = Field(..., description="LONG or FLAT")
    confidence_score: float = Field(..., description="P(return > 0)")


class FeaturePayload(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Key-value map of feature_name -> value (same schema as training)",
        example={"returns_1d": 0.012, "rsi_14": 54.3, "bb_pband": 0.6},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    """Liveness probe. Returns 200 if the service is up."""
    return HealthResponse(
        ticker=cfg["data"]["primary_ticker"],
        model="BayesianRidge",
    )


@app.get("/predict/latest", response_model=PredictResponse, tags=["Inference"])
def predict_latest_route():
    """
    Run inference on the **most recent** row of the pre-computed feature matrix.
    Returns a full probabilistic forecast including VaR and signal.
    """
    try:
        result: PredictionResult = predict_latest(cfg)
        return PredictResponse(**result.to_dict())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model or feature file not found: {exc}")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_from_payload(payload: FeaturePayload):
    """
    Run inference on a **caller-supplied** feature dictionary.
    Keys must match the training schema exactly.
    """
    try:
        df = pd.DataFrame([payload.features])
        results = predict_from_features(df, cfg)
        return PredictResponse(**results[0].to_dict())
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing feature: {exc}")
    except Exception as exc:
        logger.exception("Prediction from payload failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", tags=["Ops"])
def prometheus_metrics():
    """
    Minimal Prometheus-compatible plaintext metrics.
    For production use the `prometheus-fastapi-instrumentator` package.
    """
    lines = ["# HELP bayesian_risk_requests_total Total HTTP requests per path"]
    lines.append("# TYPE bayesian_risk_requests_total counter")
    for path, count in _request_count.items():
        lines.append(f'bayesian_risk_requests_total{{path="{path}"}} {count}')
    lines.append("# HELP bayesian_risk_latency_seconds_total Cumulative latency per path")
    lines.append("# TYPE bayesian_risk_latency_seconds_total counter")
    for path, lat in _latency_sum.items():
        lines.append(f'bayesian_risk_latency_seconds_total{{path="{path}"}} {lat:.6f}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for local dev:  uvicorn app.main:app --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

"""predict.py
Bayesian Risk-MLOps — Inference Utilities
Author: Fidel Mehra

Exposes a clean predict() function used by both the FastAPI app
and the CLI.  Returns point estimate + uncertainty band.
"""

import logging
import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PIPELINE_CACHE: dict = {}


@dataclass
class PredictionResult:
    ticker: str
    horizon_days: int
    predicted_log_return: float
    predicted_return_pct: float
    uncertainty_std: float
    lower_95: float
    upper_95: float
    var_95: float
    signal: str  # "LONG" | "FLAT"
    confidence_score: float  # P(return > 0)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_pipeline(model_path: Path):
    """Cached model loader."""
    key = str(model_path)
    if key not in _PIPELINE_CACHE:
        logger.info("Loading pipeline from %s", model_path)
        _PIPELINE_CACHE[key] = joblib.load(model_path)
    return _PIPELINE_CACHE[key]


def predict_from_features(
    features: pd.DataFrame,
    cfg: dict,
    model_path: Optional[Path] = None,
) -> list[PredictionResult]:
    """
    Run inference on a pre-built feature DataFrame.

    Parameters
    ----------
    features : pd.DataFrame
        One or more rows of the feature matrix (same schema as training).
    cfg : dict
        Project configuration.
    model_path : Path, optional
        Override the default model artefact path.

    Returns
    -------
    list[PredictionResult]
    """
    if model_path is None:
        model_path = Path(cfg["data"]["models_dir"]) / "bayesian_ridge_pipeline.pkl"

    pipe = _load_pipeline(model_path)
    primary = cfg["data"]["primary_ticker"]
    horizon = cfg["model"]["target_horizon_days"]

    br = pipe.named_steps["bayes_ridge"]
    scaler = pipe.named_steps["scaler"]

    X_scaled = scaler.transform(features)
    y_mean, y_std = br.predict(X_scaled, return_std=True)

    z = stats.norm.ppf(0.975)  # 95 % two-tailed
    results = []
    for i in range(len(features)):
        mu = float(y_mean[i])
        sigma = float(y_std[i])
        lower = mu - z * sigma
        upper = mu + z * sigma

        # VaR: worst expected loss at 95 % confidence
        var_95 = float(stats.norm.ppf(0.05, loc=mu, scale=sigma))

        # P(positive return)
        p_positive = float(1 - stats.norm.cdf(0, loc=mu, scale=sigma))
        signal = "LONG" if mu > 0 else "FLAT"

        results.append(PredictionResult(
            ticker=primary,
            horizon_days=horizon,
            predicted_log_return=mu,
            predicted_return_pct=float((np.exp(mu) - 1) * 100),
            uncertainty_std=sigma,
            lower_95=lower,
            upper_95=upper,
            var_95=var_95,
            signal=signal,
            confidence_score=p_positive,
        ))

    return results


def predict_latest(cfg: dict, model_path: Optional[Path] = None) -> PredictionResult:
    """
    Convenience wrapper: loads the most recent row of the feature matrix
    and returns a single PredictionResult for the latest observation.
    """
    primary = cfg["data"]["primary_ticker"]
    feat_path = Path(cfg["data"]["processed_dir"]) / f"{primary}_features.parquet"
    df = pd.read_parquet(feat_path)

    # Drop non-feature columns if present
    feature_cols = [c for c in df.columns if c != "close"]
    latest = df[feature_cols].iloc[[-1]]
    results = predict_from_features(latest, cfg, model_path)
    result = results[0]
    logger.info(
        "Latest prediction | ticker=%s horizon=%dd "
        "log_ret=%.5f pct=%.3f%% signal=%s p_pos=%.3f",
        result.ticker, result.horizon_days,
        result.predicted_log_return, result.predicted_return_pct,
        result.signal, result.confidence_score,
    )
    return result


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/config.yaml"))
    r = predict_latest(cfg)
    print(r.to_dict())

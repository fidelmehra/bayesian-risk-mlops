"""train_model.py
Bayesian Risk-MLOps — Model Training Pipeline
Author: Fidel Mehra

Trains a Bayesian Ridge Regression model on the engineered feature matrix
and persists the artefact via MLflow.
"""

import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and construct forward-return target."""
    primary = cfg["data"]["primary_ticker"]
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    feat_path = Path(cfg["data"]["processed_dir"]) / f"{primary}_features.parquet"

    df = pd.read_parquet(feat_path)
    horizon = cfg["model"]["target_horizon_days"]

    # Target: forward log-return over *horizon* trading days
    df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c not in ["close", "target"]]
    X = df[feature_cols]
    y = df["target"]
    logger.info("Loaded %d samples, %d features", len(X), X.shape[1])
    return X, y


def build_pipeline(cfg: dict) -> Pipeline:
    """Construct sklearn pipeline with scaling + Bayesian Ridge."""
    mc = cfg["model"]
    model = BayesianRidge(
        max_iter=mc["max_iter"],
        tol=mc["tol"],
        alpha_1=mc["alpha_1"],
        alpha_2=mc["alpha_2"],
        lambda_1=mc["lambda_1"],
        lambda_2=mc["lambda_2"],
        compute_score=True,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("bayes_ridge", model),
    ])
    return pipe


def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """Time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    neg_mse = cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_squared_error")
    r2_scores = cross_val_score(pipe, X, y, cv=tscv, scoring="r2")
    metrics = {
        "cv_rmse_mean": float(np.sqrt(-neg_mse.mean())),
        "cv_rmse_std": float(np.sqrt(-neg_mse).std()),
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
    }
    logger.info("CV RMSE: %.6f ± %.6f", metrics["cv_rmse_mean"], metrics["cv_rmse_std"])
    logger.info("CV R2  : %.4f ± %.4f", metrics["cv_r2_mean"], metrics["cv_r2_std"])
    return metrics


def train_and_log(cfg: dict) -> None:
    """Full training run with MLflow tracking."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_data(cfg)

    # Train/test split — last 20 % is held out
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipe = build_pipeline(cfg)
    cv_metrics = evaluate_cv(pipe, X_train, y_train, n_splits=cfg["model"]["cv_splits"])

    with mlflow.start_run(run_name="bayesian_ridge_training") as run:
        # Log hyperparams
        mlflow.log_params(cfg["model"])

        # Fit on full training set
        pipe.fit(X_train, y_train)

        # Hold-out evaluation
        y_pred = pipe.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_r2 = float(r2_score(y_test, y_pred))

        # Log metrics
        metrics = {**cv_metrics, "test_rmse": test_rmse, "test_r2": test_r2}
        mlflow.log_metrics(metrics)

        # Directional accuracy (sign of return)
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test.values)))
        mlflow.log_metric("directional_accuracy", dir_acc)
        logger.info("Directional accuracy: %.4f", dir_acc)

        # Log model
        signature = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )

        # Save model locally as well
        out_dir = Path(cfg["data"]["models_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(pipe, out_dir / "bayesian_ridge_pipeline.pkl")
        logger.info("Model saved to %s", out_dir / "bayesian_ridge_pipeline.pkl")
        logger.info("MLflow run ID: %s", run.info.run_id)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/config.yaml"))
    train_and_log(cfg)

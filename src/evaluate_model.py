"""evaluate_model.py
Bayesian Risk-MLOps — Model Evaluation & Risk Reporting
Author: Fidel Mehra

Loads the trained pipeline and produces a comprehensive risk report:
  - Prediction interval coverage (Bayesian uncertainty)
  - VaR / CVaR of predicted return distribution
  - Brier-style calibration check
  - Residual autocorrelation (Ljung-Box)
  - Walk-forward simulation P&L curve
"""

import logging
import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_artefacts(cfg: dict):
    """Load pipeline and feature matrix."""
    primary = cfg["data"]["primary_ticker"]
    feat_path = Path(cfg["data"]["processed_dir"]) / f"{primary}_features.parquet"
    model_path = Path(cfg["data"]["models_dir"]) / "bayesian_ridge_pipeline.pkl"

    df = pd.read_parquet(feat_path)
    horizon = cfg["model"]["target_horizon_days"]
    df["target"] = np.log(df["close"].shift(-horizon) / df["close"])
    df.dropna(inplace=True)

    feature_cols = [c for c in df.columns if c not in ["close", "target"]]
    X = df[feature_cols]
    y = df["target"]

    pipe = joblib.load(model_path)
    return pipe, X, y, df


def prediction_intervals(pipe, X: pd.DataFrame, alpha: float = 0.05):
    """Return point predictions + Bayesian predictive std."""
    br = pipe.named_steps["bayes_ridge"]
    scaler = pipe.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    y_mean, y_std = br.predict(X_scaled, return_std=True)
    z = stats.norm.ppf(1 - alpha / 2)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    return y_mean, y_std, lower, upper


def var_cvar(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Parametric VaR and CVaR assuming normality."""
    mu, sigma = returns.mean(), returns.std()
    var = stats.norm.ppf(1 - confidence, loc=mu, scale=sigma)
    cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(confidence)) / (1 - confidence)
    return float(var), float(cvar)


def walk_forward_pnl(y_pred: np.ndarray, y_true: np.ndarray) -> pd.Series:
    """Long/flat signal: go long when predicted return > 0."""
    signal = (y_pred > 0).astype(float)
    strategy_returns = signal * y_true
    cum_pnl = pd.Series(np.cumsum(strategy_returns), name="cumulative_log_return")
    return cum_pnl


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> pd.DataFrame:
    """Test residuals for serial autocorrelation."""
    lb = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return lb


def run_evaluation(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    pipe, X, y, df = load_artefacts(cfg)

    # Hold-out (last 20 %)
    split_idx = int(len(X) * 0.80)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    y_mean, y_std, lower, upper = prediction_intervals(pipe, X_test)

    coverage = float(np.mean((y_test.values >= lower) & (y_test.values <= upper)))
    logger.info("95%% PI coverage: %.4f", coverage)

    residuals = y_test.values - y_mean
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    var_95, cvar_95 = var_cvar(y_mean)
    logger.info("Predicted VaR(95%%): %.6f  CVaR(95%%): %.6f", var_95, cvar_95)

    cum_pnl = walk_forward_pnl(y_mean, y_test.values)
    total_return = float(cum_pnl.iloc[-1])
    sharpe = float(cum_pnl.diff().mean() / (cum_pnl.diff().std() + 1e-9) * np.sqrt(252))
    logger.info("Strategy total log-return: %.4f  Sharpe: %.4f", total_return, sharpe)

    lb = ljung_box_test(residuals)
    lb_pvalue = float(lb["lb_pvalue"].values[0])
    logger.info("Ljung-Box p-value (lag 10): %.4f", lb_pvalue)

    # Plots
    report_dir = Path(cfg["data"].get("reports_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bayesian Risk-MLOps — Evaluation Report", fontsize=14)

    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_test.values, y_mean, alpha=0.4, s=10, color="steelblue")
    lims = [min(y_test.min(), y_mean.min()), max(y_test.max(), y_mean.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual log-return")
    ax.set_ylabel("Predicted log-return")
    ax.set_title(f"Actual vs Predicted  (RMSE={rmse:.5f})")

    # 2. Prediction interval plot (first 100 points)
    ax = axes[0, 1]
    idx = np.arange(min(100, len(y_mean)))
    ax.fill_between(idx, lower[:len(idx)], upper[:len(idx)], alpha=0.3, label="95% PI")
    ax.plot(idx, y_mean[:len(idx)], label="Mean pred", linewidth=1)
    ax.plot(idx, y_test.values[:len(idx)], "k.", markersize=3, label="Actual")
    ax.set_title(f"Prediction Intervals (coverage={coverage:.3f})")
    ax.legend(fontsize=8)

    # 3. Walk-forward P&L
    ax = axes[1, 0]
    cum_pnl.plot(ax=ax, color="green")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"Walk-Forward P&L (Sharpe={sharpe:.2f})")
    ax.set_ylabel("Cumulative log-return")

    # 4. Residual histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, color="salmon", edgecolor="white", alpha=0.8)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")

    plt.tight_layout()
    fig_path = report_dir / "evaluation_report.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Report saved to %s", fig_path)

    # Log to MLflow
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_mae": mae,
            "pi_coverage_95": coverage,
            "predicted_var_95": var_95,
            "predicted_cvar_95": cvar_95,
            "strategy_total_log_return": total_return,
            "strategy_sharpe": sharpe,
            "ljung_box_pvalue_lag10": lb_pvalue,
        })
        mlflow.log_artifact(str(fig_path))


if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/config.yaml"))
    run_evaluation(cfg)

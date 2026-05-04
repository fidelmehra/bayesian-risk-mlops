# Bayesian Risk MLOps

> **End-to-end Bayesian deep learning MLOps pipeline for financial risk forecasting.**
> Author: **Fidel Mehra** | MSc Advanced Data Science, Newcastle University

[![CI](https://github.com/fidelmehra/bayesian-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/fidelmehra/bayesian-risk-mlops/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](Dockerfile)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)

---

## Overview

This project implements a production-ready, uncertainty-aware financial risk forecasting system. Classical risk models such as Historical Simulation or parametric VaR assume i.i.d. Gaussian returns and provide only point estimates, ignoring **epistemic uncertainty** (model uncertainty) and **aleatoric uncertainty** (inherent data noise). This pipeline addresses both:

| Component | Method | Purpose |
|---|---|---|
| **Regime Detection** | Hidden Markov Model (2–4 states) | Identify bull/bear/crisis regimes |
| **Volatility Forecasting** | MC Dropout LSTM | Predictive intervals for realised vol |
| **Density Estimation** | Gaussian Process Regression | Smooth posterior over return distribution |
| **Risk Metrics** | Monte Carlo Simulation | VaR (95%, 99%), CVaR, Expected Shortfall |
| **Uncertainty Quant.** | Conformal Prediction + Bayesian CI | Calibrated prediction intervals |
| **Serving** | FastAPI + MLflow Model Registry | Versioned, containerised inference |

---

## Architecture

```
bayesian-risk-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint → Test → Docker Build → Publish
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI lifespan & router registration
│   ├── routers/
│   │   ├── risk.py             # /risk/var, /risk/cvar, /risk/forecast
│   │   └── health.py           # /health, /ready
│   ├── schemas.py              # Pydantic I/O models
│   ├── service.py              # Model loading, inference, uncertainty
│   └── middleware.py           # Logging, timing, CORS
├── config/
│   └── config.yaml             # All hyper-parameters and paths
├── notebooks/
│   ├── 01_eda_returns.ipynb    # Return distribution & tail analysis
│   ├── 02_bayesian_lstm.ipynb  # MC Dropout training & uncertainty viz
│   └── 03_risk_metrics.ipynb   # VaR/CVaR backtesting
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── fetch_data.py       # yfinance multi-ticker downloader
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Returns, vol, skew, tail features
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── hmm_regimes.py      # HMM regime labelling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mc_dropout_lstm.py  # Bayesian LSTM with MC Dropout
│   │   ├── gaussian_process.py # GPR for smooth density estimation
│   │   └── ensemble.py         # Model ensemble + uncertainty aggregation
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── var_cvar.py         # Historical, parametric, MC VaR/CVaR
│   │   └── backtest.py         # Kupiec, Christoffersen backtests
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py            # Full training pipeline + MLflow logging
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # RMSE, Winkler score, coverage, DM test
│   └── utils/
│       ├── __init__.py
│       └── io.py               # Config, parquet, artefact helpers
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_risk.py
│   ├── test_models.py
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Theoretical Background

### 1. Bayesian Uncertainty Decomposition

For a prediction $\hat{y}$, total uncertainty decomposes as:

$$\text{Total Uncertainty} = \underbrace{\mathbb{E}[\sigma^2(x)]}_\text{Aleatoric} + \underbrace{\text{Var}[\mu(x)]}_\text{Epistemic}$$

- **Aleatoric** (irreducible): noise inherent to financial time series — fat tails, micro-structure noise
- **Epistemic** (reducible): model uncertainty captured via MC Dropout approximation to Bayesian inference

### 2. Monte Carlo Dropout as Bayesian Approximation

Dropout at inference time (T forward passes) approximates the posterior predictive distribution:

$$p(y^* | x^*, \mathbf{X}, \mathbf{Y}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y^* | x^*, \hat{\omega}_t)$$

Predictive mean and variance:
$$\mu^* = \frac{1}{T}\sum_t f^{\hat{\omega}_t}(x^*), \quad
\sigma^{*2} = \tau^{-1}\mathbf{I} + \frac{1}{T}\sum_t f^{\hat{\omega}_t}(x^*)^\top f^{\hat{\omega}_t}(x^*) - \mu^{*\top}\mu^*$$

### 3. Value at Risk & Conditional Value at Risk

$$\text{VaR}_{\alpha}(X) = -\inf\{x \in \mathbb{R} : F_X(x) > \alpha\}$$

$$\text{CVaR}_{\alpha}(X) = -\frac{1}{1-\alpha}\int_{\alpha}^{1} \text{VaR}_u(X)\, du = \mathbb{E}[-X \mid -X \geq \text{VaR}_\alpha]$$

CVaR (Expected Shortfall) is coherent (sub-additive) unlike VaR, making it preferred under Basel III/IV.

### 4. Gaussian Process Regression

GPR places a prior over functions: $f(x) \sim \mathcal{GP}(m(x), k(x,x'))$

Posterior predictive:
$$f_* | X, y, X_* \sim \mathcal{N}(\bar{f}_*, \text{cov}(f_*))$$
$$\bar{f}_* = K_{*f}[K_{ff} + \sigma_n^2 I]^{-1}y$$

Kernel: Matérn-5/2 + RBF composite for capturing both smooth and locally-varying volatility dynamics.

### 5. Kupiec Proportion of Failures Test

Tests whether VaR violations match the expected frequency:
$$LR_{POF} = -2\ln\left[\frac{(1-p)^{T-N}p^N}{(1-N/T)^{T-N}(N/T)^N}\right] \sim \chi^2_1$$

where $N$ = number of violations, $T$ = number of observations, $p$ = confidence level.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 4 GB RAM minimum

### Installation

```bash
git clone https://github.com/fidelmehra/bayesian-risk-mlops.git
cd bayesian-risk-mlops
make install
```

### Run Full Pipeline

```bash
# 1. Ingest market data (default: SPY, QQQ, GLD, BTC-USD, VIX)
make ingest

# 2. Build feature matrix
make features

# 3. Fit HMM regime labels
make regimes

# 4. Train MC Dropout LSTM + GP ensemble
make train

# 5. Compute VaR/CVaR risk metrics
make risk

# 6. Serve API locally
make serve
```

### Docker Deployment

```bash
make up         # docker compose up -d (API + MLflow + Redis)
make logs       # tail logs
make down       # tear down
```

API available at `http://localhost:8080/docs`
MLflow UI at `http://localhost:5000`

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/risk/var` | Compute VaR at given confidence level |
| `POST` | `/risk/cvar` | Compute CVaR / Expected Shortfall |
| `POST` | `/risk/forecast` | Predict next-N-day vol with uncertainty bounds |
| `POST` | `/risk/regime` | Detect current market regime |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (model loaded) |
| `GET` | `/metrics` | Prometheus-compatible metrics |

#### Example: VaR Request

```bash
curl -X POST http://localhost:8080/risk/var \
  -H 'Content-Type: application/json' \
  -d '{
    "returns": [-0.012, 0.008, -0.021, 0.015, -0.005],
    "confidence": 0.99,
    "method": "monte_carlo",
    "n_simulations": 10000,
    "horizon_days": 1
  }'
```

#### Example: Forecast Response

```json
{
  "ticker": "SPY",
  "horizon_days": 5,
  "forecast": {
    "mean": [0.0124, 0.0131, 0.0128, 0.0135, 0.0142],
    "lower_95": [0.0089, 0.0091, 0.0087, 0.0092, 0.0098],
    "upper_95": [0.0159, 0.0171, 0.0169, 0.0178, 0.0186],
    "epistemic_std": [0.0018, 0.0021, 0.0023, 0.0024, 0.0026],
    "aleatoric_std": [0.0019, 0.0020, 0.0019, 0.0021, 0.0022]
  },
  "regime": "low_volatility",
  "regime_probability": 0.87,
  "var_99": -0.0234,
  "cvar_99": -0.0312
}
```

---

## Pipeline Stages

### Stage 1: Data Ingestion
- Multi-ticker OHLCV download via `yfinance`
- Handles corporate actions (splits, dividends)
- Stores as compressed Parquet with metadata
- Configurable tickers, date ranges, and intervals

### Stage 2: Feature Engineering
- **Return features**: log returns, rolling mean/std (5/10/20/60 day)
- **Tail features**: rolling skewness, excess kurtosis
- **Volatility estimators**: Close-to-close, Parkinson, Garman-Klass, Rogers-Satchell
- **Momentum**: RSI, ROC, Z-score of returns
- **Market micro-structure**: bid-ask proxy, Amihud illiquidity
- **Cross-asset**: correlation with VIX, gold, bonds

### Stage 3: Regime Labelling (HMM)
- Gaussian HMM with 2–4 hidden states
- States represent: bull low-vol / bear low-vol / high-vol / crisis
- Viterbi decoding for hard labels
- Forward–backward for soft (probabilistic) labels

### Stage 4: Model Training

#### MC Dropout LSTM
```
Input (seq_len, n_features)
    → LSTM(128, dropout=0.3, return_sequences=True)
    → LSTM(64, dropout=0.3)
    → Dense(32, activation='relu')
    → MC Dropout(0.3) [kept active at inference]
    → Dense(1)
```
T=100 stochastic forward passes generate the predictive distribution.

#### Gaussian Process Regression
- Kernel: `Matern52 + RBF + WhiteKernel`
- Sparse GP with inducing points for scalability
- Calibrated credible intervals

#### Ensemble
- Weighted average of LSTM MC samples and GP posterior
- Weights learned via validation NLL minimisation
- Final uncertainty: closed-form mixture-of-Gaussians

### Stage 5: Risk Computation
- **Historical VaR**: empirical quantile of return history
- **Parametric VaR**: assumes Normal/Student-t distribution
- **Monte Carlo VaR**: simulate paths from fitted model, extract quantile
- **CVaR/ES**: tail conditional expectation beyond VaR threshold
- **Stressed VaR**: replay 2008, 2020 scenarios

### Stage 6: Backtesting
- Kupiec POF test for VaR violations
- Christoffersen independence test
- Dynamic Quantile (DQ) test
- Winkler score for interval sharpness
- Coverage probability check

---

## Configuration

All parameters live in `config/config.yaml`. Key sections:

```yaml
data:
  tickers: [SPY, QQQ, GLD, BTC-USD, ^VIX]
  start_date: "2015-01-01"
  end_date: "2024-12-31"

features:
  rolling_windows: [5, 10, 20, 60]
  vol_estimators: [close_to_close, parkinson, garman_klass]

model:
  lstm:
    seq_len: 60
    hidden_sizes: [128, 64]
    dropout: 0.3
    mc_samples: 100
    epochs: 100
    batch_size: 64
    learning_rate: 0.001
  gp:
    kernel: matern52
    n_inducing: 200

risk:
  confidence_levels: [0.95, 0.99]
  horizon_days: [1, 5, 10]
  n_mc_simulations: 50000
```

---

## MLflow Tracking

All experiments are tracked with MLflow:

| Logged Item | Details |
|---|---|
| Hyperparameters | All model/training config |
| Loss curves | Train/val loss per epoch |
| Metrics | RMSE, MAE, Winkler score, coverage |
| Artefacts | Trained weights, scalers, GP kernel params |
| Risk metrics | VaR, CVaR per ticker and horizon |
| Backtests | POF test statistic, p-value |

Access MLflow UI: `http://localhost:5000`

---

## Makefile Targets

```
install          Install all Python dependencies
lint             flake8 + mypy type checking
format           black + isort auto-format
test             pytest with coverage
test-cov         pytest with HTML coverage report
ingest           Fetch and cache market data
features         Build feature matrix
regimes          Fit HMM and compute regime labels
train            Train LSTM + GP ensemble with MLflow logging
risk             Compute VaR/CVaR risk metrics
backtest         Run Kupiec/Christoffersen backtests
serve            Start FastAPI server locally
build            Build Docker image
up               docker compose up -d
down             docker compose down
logs             Tail docker compose logs
clean            Remove artefacts and cache
```

---

## Dependencies

| Category | Libraries |
|---|---|
| Data | `yfinance`, `pandas`, `numpy` |
| Deep Learning | `torch`, `torchvision` |
| Bayesian / GP | `gpytorch`, `botorch` |
| ML | `scikit-learn`, `hmmlearn`, `xgboost` |
| Risk | `scipy`, `statsmodels` |
| Serving | `fastapi`, `uvicorn`, `pydantic` |
| Tracking | `mlflow` |
| Viz | `matplotlib`, `seaborn`, `plotly` |
| Testing | `pytest`, `pytest-cov`, `httpx` |
| Code Quality | `black`, `flake8`, `isort`, `mypy` |

---

## Citation

If you use this work, please cite:

```bibtex
@software{mehra2025bayesian,
  author    = {Fidel Mehra},
  title     = {Bayesian Risk MLOps: Uncertainty-Aware Financial Risk Forecasting},
  year      = {2025},
  url       = {https://github.com/fidelmehra/bayesian-risk-mlops},
  note      = {MSc Advanced Data Science, Newcastle University}
}
```

---

## References

1. Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
2. Artzner, P. et al. (1999). Coherent measures of risk. *Mathematical Finance*.
3. Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *FEDS*.
4. Williams, C. & Rasmussen, C. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
5. Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*.
6. Lakshminarayanan, B. et al. (2017). Simple and scalable predictive uncertainty estimation. *NeurIPS*.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Author: Fidel Mehra — Data Science & Blockchain Expert | MSc Advanced Data Science, Newcastle University*

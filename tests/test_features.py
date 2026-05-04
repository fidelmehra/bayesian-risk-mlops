"""tests/test_features.py
Bayesian Risk-MLOps — Unit Tests: Feature Engineering
Author: Fidel Mehra

Tests build_feature_matrix() and its sub-components
using deterministic synthetic OHLCV data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.build_features import (
    build_feature_matrix,
    compute_rolling_returns,
    compute_garman_klass_vol,
    compute_parkinson_vol,
    compute_rsi,
    compute_bollinger_bands,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300  # rows of synthetic OHLCV


@pytest.fixture(scope="module")
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate a reproducible synthetic OHLCV dataframe."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=N, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=N))
    high = close * (1 + rng.uniform(0.001, 0.02, size=N))
    low = close * (1 - rng.uniform(0.001, 0.02, size=N))
    open_ = low + rng.uniform(0, 1, size=N) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=N).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Tests: individual feature functions
# ---------------------------------------------------------------------------


class TestRollingReturns:
    def test_shape_preserved(self, synthetic_ohlcv):
        result = compute_rolling_returns(synthetic_ohlcv["close"], windows=[1, 5, 20])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == N

    def test_column_names(self, synthetic_ohlcv):
        result = compute_rolling_returns(synthetic_ohlcv["close"], windows=[1, 5])
        assert "returns_1d" in result.columns
        assert "returns_5d" in result.columns

    def test_log_returns_finite(self, synthetic_ohlcv):
        result = compute_rolling_returns(synthetic_ohlcv["close"], windows=[1])
        assert np.isfinite(result["returns_1d"].dropna()).all()


class TestGarmanKlassVol:
    def test_non_negative(self, synthetic_ohlcv):
        result = compute_garman_klass_vol(synthetic_ohlcv, window=20)
        assert (result.dropna() >= 0).all(), "GK vol should be non-negative"

    def test_output_name(self, synthetic_ohlcv):
        result = compute_garman_klass_vol(synthetic_ohlcv, window=20)
        assert result.name == "gk_vol_20"


class TestParkinsonVol:
    def test_non_negative(self, synthetic_ohlcv):
        result = compute_parkinson_vol(synthetic_ohlcv, window=20)
        assert (result.dropna() >= 0).all()

    def test_output_name(self, synthetic_ohlcv):
        result = compute_parkinson_vol(synthetic_ohlcv, window=20)
        assert result.name == "parkinson_vol_20"


class TestRSI:
    def test_range_0_100(self, synthetic_ohlcv):
        result = compute_rsi(synthetic_ohlcv["close"], period=14)
        vals = result.dropna()
        assert (vals >= 0).all() and (vals <= 100).all()

    def test_output_name(self, synthetic_ohlcv):
        result = compute_rsi(synthetic_ohlcv["close"], period=14)
        assert result.name == "rsi_14"


class TestBollingerBands:
    def test_columns_present(self, synthetic_ohlcv):
        result = compute_bollinger_bands(
            synthetic_ohlcv["close"], period=20, std=2.0
        )
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_pband" in result.columns
        assert "bb_bandwidth" in result.columns

    def test_upper_gte_lower(self, synthetic_ohlcv):
        result = compute_bollinger_bands(
            synthetic_ohlcv["close"], period=20, std=2.0
        )
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


# ---------------------------------------------------------------------------
# Tests: full feature matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    def test_returns_dataframe(self, synthetic_ohlcv):
        result = build_feature_matrix(synthetic_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_no_inf(self, synthetic_ohlcv):
        result = build_feature_matrix(synthetic_ohlcv)
        assert not np.isinf(result.values).any(), "Feature matrix contains inf"

    def test_close_column_present(self, synthetic_ohlcv):
        result = build_feature_matrix(synthetic_ohlcv)
        assert "close" in result.columns

    def test_column_count_reasonable(self, synthetic_ohlcv):
        result = build_feature_matrix(synthetic_ohlcv)
        # Expect at least 20 feature columns
        assert result.shape[1] >= 20, f"Only {result.shape[1]} columns found"

    def test_index_preserved(self, synthetic_ohlcv):
        result = build_feature_matrix(synthetic_ohlcv)
        # Index should be a subset of input (NaN rows dropped)
        assert result.index.isin(synthetic_ohlcv.index).all()

    def test_custom_rolling_windows(self, synthetic_ohlcv):
        result = build_feature_matrix(
            synthetic_ohlcv, rolling_windows=[5, 10]
        )
        assert "returns_5d" in result.columns
        assert "returns_10d" in result.columns

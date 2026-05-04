"""Market data ingestion module.

Downloads OHLCV data for multiple tickers using yfinance,
handles splits/dividends, and caches as Parquet.

Author: Fidel Mehra
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _safe_download(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Download a single ticker with retry logic."""
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                raise ValueError(f"Empty data returned for {ticker}")
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            return df
        except Exception as exc:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, ticker, exc)
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"Failed to download {ticker} after {retries} attempts")


def fetch_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for a list of tickers.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    start:
        Start date string in YYYY-MM-DD format.
    end:
        End date string in YYYY-MM-DD format.
    interval:
        Data granularity. Default '1d'.
    cache_dir:
        If provided, cache raw data as Parquet files here.

    Returns
    -------
    dict mapping ticker symbol -> OHLCV DataFrame
    """
    results: Dict[str, pd.DataFrame] = {}

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_path = (
            Path(cache_dir) / f"{ticker.replace('^', '_')}_{start}_{end}.parquet"
            if cache_dir
            else None
        )

        if cache_path and cache_path.exists():
            log.info("Loading %s from cache: %s", ticker, cache_path)
            df = pd.read_parquet(cache_path)
        else:
            log.info("Downloading %s [%s -> %s]", ticker, start, end)
            df = _safe_download(ticker, start=start, end=end, interval=interval)
            if cache_path:
                df.to_parquet(cache_path, index=True)
                log.debug("Cached %s -> %s", ticker, cache_path)

        results[ticker] = df
        log.info("Loaded %s: %d rows, %d cols", ticker, len(df), df.shape[1])

    return results


def compute_log_returns(prices: pd.Series, dropna: bool = True) -> pd.Series:
    """Compute log returns from a price series."""
    lr = np.log(prices).diff()
    return lr.dropna() if dropna else lr


def align_tickers(data: Dict[str, pd.DataFrame], column: str = "close") -> pd.DataFrame:
    """Align multiple tickers on a common date index.

    Parameters
    ----------
    data:
        Dict of ticker -> DataFrame (output of fetch_ohlcv).
    column:
        Column to extract per ticker (default: 'close').

    Returns
    -------
    DataFrame with one column per ticker, aligned on dates.
    """
    series = {}
    for ticker, df in data.items():
        if column in df.columns:
            series[ticker] = df[column]
        else:
            log.warning("Column '%s' not found in %s; skipping.", column, ticker)

    aligned = pd.DataFrame(series).dropna(how="all")
    aligned.index = pd.to_datetime(aligned.index)
    aligned = aligned.sort_index()
    log.info("Aligned price matrix: %d rows x %d tickers", *aligned.shape)
    return aligned


def compute_return_matrix(aligned_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for all tickers simultaneously."""
    return np.log(aligned_prices).diff().dropna()


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load(open("config/config.yaml"))
    dc = cfg["data"]

    all_tickers = (
        dc["tickers"]["equities"]
        + dc["tickers"]["volatility"]
        + dc["tickers"]["crypto"]
        + dc["tickers"]["bonds"]
    )

    raw = fetch_ohlcv(
        tickers=all_tickers,
        start=dc["start_date"],
        end=dc["end_date"],
        interval=dc["interval"],
        cache_dir=dc["cache_dir"],
    )

    prices = align_tickers(raw)
    returns = compute_return_matrix(prices)

    Path(dc["raw_dir"]).mkdir(parents=True, exist_ok=True)
    prices.to_parquet(f"{dc['raw_dir']}/prices.parquet")
    returns.to_parquet(f"{dc['raw_dir']}/returns.parquet")
    log.info("Saved prices and returns to %s", dc["raw_dir"])

"""Relative strength pure functions for M4.

Computes stock-vs-benchmark relative performance metrics using daily OHLCV
DataFrames. All calculations use float internally; pure indicators return float
or pd.Series; Decimal is used only at the service / model boundary.

No network calls, no database access, no side effects.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from mcp_finance.indicators._date_utils import extract_norm_dates


def relative_return(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    period: int,
    as_of_date: datetime.date | str | None = None,
) -> float | None:
    """Compute the N-bar return difference between a stock and its benchmark.

    Formula::

        stock_return    = (stock_close[as_of] - stock_close[as_of - period]) /
                          stock_close[as_of - period]
        bench_return    = (bench_close[as_of] - bench_close[as_of - period]) /
                          bench_close[as_of - period]
        relative_return = stock_return - bench_return

    Contiguity contract (stated in full):
        The window is defined as the trailing ``period + 1`` stock bars ending
        at or before ``as_of_date``. Across this window, *both* frames must
        have **identical dates**. Any mismatch — a gap in either series, a
        stale benchmark missing the stock's last bar, or frames of different
        lengths — causes the function to return ``None``.

    as_of_date handling:
        The snap is defined from the **stock series**:
        - If ``as_of_date`` is ``None``, the last available date in ``stock_df``
          is used.
        - Otherwise the function snaps *backward* to the nearest trading bar
          with date ≤ ``as_of_date`` in ``stock_df``.
        - If ``as_of_date`` is strictly before the earliest stock bar,
          returns ``None``.
        The benchmark must then contain the exact same window. If the benchmark
        is missing the stock's last bar (e.g. benchmark cache is stale), returns
        ``None``.

    Args:
        stock_df:     DataFrame with at least a ``close`` column and either a
                      ``'date'`` column or a ``DatetimeIndex``.
        benchmark_df: Same shape as ``stock_df`` but for the benchmark.
        period:       Number of bars to look back (e.g. 63 for ~3 months).
                      Must be ≥ 1.
        as_of_date:   Reference date for the most-recent bar. Defaults to the
                      last bar in ``stock_df``.

    Returns:
        Signed fractional return difference as a ``float``, or ``None``
        when insufficient data, a date gap, a stale benchmark, or a
        non-positive close is found.

    Raises:
        ValueError: If ``period < 1``.
    """
    if period < 1:
        raise ValueError("period must be at least 1.")

    if stock_df.empty or benchmark_df.empty:
        return None
    if "close" not in stock_df.columns or "close" not in benchmark_df.columns:
        return None

    try:
        stock_dates = extract_norm_dates(stock_df)
        bench_dates = extract_norm_dates(benchmark_df)
    except ValueError:
        return None

    # Uniqueness and monotonicity validation
    if stock_dates.duplicated().any() or bench_dates.duplicated().any():
        return None
    if (
        not stock_dates.is_monotonic_increasing
        or not bench_dates.is_monotonic_increasing
    ):
        return None

    # Snap as_of_date using the stock series
    if as_of_date is None:
        ref_ts = stock_dates.max()
    else:
        try:
            target_ts = pd.Timestamp(as_of_date).normalize()
        except Exception:
            return None
        candidates = stock_dates[stock_dates <= target_ts]
        if candidates.empty:
            return None
        ref_ts = candidates.max()

    # Slice stock to (period + 1) bars ending at ref_ts
    stock_mask = (stock_dates <= ref_ts).to_numpy(dtype=bool)
    stock_slice = stock_df[stock_mask]
    stock_slice_dates = stock_dates[stock_mask]
    if len(stock_slice) < period + 1:
        return None

    stock_slice = stock_slice.iloc[-(period + 1) :]
    stock_slice_dates = stock_slice_dates.iloc[-(period + 1) :]

    # Slice benchmark to bars <= ref_ts, take trailing (period + 1) bars
    bench_mask = (bench_dates <= ref_ts).to_numpy(dtype=bool)
    bench_slice = benchmark_df[bench_mask]
    bench_slice_dates = bench_dates[bench_mask]
    if len(bench_slice) < period + 1:
        return None

    bench_slice = bench_slice.iloc[-(period + 1) :]
    bench_slice_dates = bench_slice_dates.iloc[-(period + 1) :]

    # Contiguity check: both windows must have identical date sequences
    bench_date_vals: np.ndarray = bench_slice_dates.to_numpy()
    stock_date_vals: np.ndarray = stock_slice_dates.to_numpy()
    if not bool(np.array_equal(bench_date_vals, stock_date_vals)):
        return None

    # Extract close arrays (float)
    s_close: np.ndarray = stock_slice["close"].to_numpy(dtype=float)
    b_close: np.ndarray = bench_slice["close"].to_numpy(dtype=float)

    # Check for NaN / non-positive prices
    if np.any(np.isnan(s_close)) or np.any(np.isnan(b_close)):
        return None
    if (
        s_close[0] <= 0.0
        or b_close[0] <= 0.0
        or s_close[-1] <= 0.0
        or b_close[-1] <= 0.0
    ):
        return None

    s_ret = (s_close[-1] - s_close[0]) / s_close[0]
    b_ret = (b_close[-1] - b_close[0]) / b_close[0]

    return float(s_ret - b_ret)


def relative_price_ratio(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_bars: int | None = None,
    as_of_date: datetime.date | str | None = None,
) -> pd.Series | None:
    """Compute a daily price-ratio series: stock_close / benchmark_close.

    Formula::

        ratio[t] = stock_close[t] / benchmark_close[t]

    Note on scale:
        The absolute scale of the raw ratio is arbitrary (determined by nominal
        share prices). Only relative measures derived from the ratio series —
        such as EMA alignment, percent slope, and breakout levels — are
        meaningful.

    Strict window & contiguity contract:
        - The window is defined as the trailing ``lookback_bars`` stock bars
          ending at or before ``as_of_date`` (or all stock bars on or before
          ``as_of_date`` if ``lookback_bars`` is ``None``).
        - Across this window, the benchmark DataFrame must contain **exactly the
          same date sequence**.
        - Any date gap, missing bar, stale benchmark, NaN close, or non-positive
          close in either series causes the function to return ``None`` (no silent
          truncation or partial overlap allowed).

    as_of_date handling:
        The snap is defined from the **stock series**:
        - If ``as_of_date`` is ``None``, the last available date in ``stock_df``
          is used.
        - Otherwise the function snaps *backward* to the nearest trading bar
          with date ≤ ``as_of_date`` in ``stock_df``.
        - If ``as_of_date`` is strictly before the earliest stock bar, returns
          ``None``.

    Args:
        stock_df:      DataFrame with at least a ``close`` column and either a
                       ``'date'`` column or a ``DatetimeIndex``.
        benchmark_df:  Same shape as ``stock_df`` but for the benchmark.
        lookback_bars: Optional number of trailing bars to include (must be ≥ 1).
                       If None, uses all bars on or before as_of_date.
        as_of_date:    Reference date for the most-recent bar (defaults to last
                       bar in ``stock_df``).

    Returns:
        pd.Series indexed by pd.DatetimeIndex of normalized dates, or ``None``
        on any validation failure, gap, NaN, or non-positive price.

    Raises:
        ValueError: If ``lookback_bars is not None and lookback_bars < 1``, or
            if either DataFrame lacks a ``'close'`` column.
    """
    if lookback_bars is not None and lookback_bars < 1:
        raise ValueError("lookback_bars must be at least 1.")

    if "close" not in stock_df.columns:
        raise ValueError("stock_df must contain a 'close' column")
    if "close" not in benchmark_df.columns:
        raise ValueError("benchmark_df must contain a 'close' column")

    if stock_df.empty or benchmark_df.empty:
        return None

    try:
        stock_dates = extract_norm_dates(stock_df)
        bench_dates = extract_norm_dates(benchmark_df)
    except ValueError:
        return None

    # Uniqueness and monotonicity validation
    if stock_dates.duplicated().any() or bench_dates.duplicated().any():
        return None
    if (
        not stock_dates.is_monotonic_increasing
        or not bench_dates.is_monotonic_increasing
    ):
        return None

    # Snap as_of_date using the stock series
    if as_of_date is None:
        ref_ts = stock_dates.max()
    else:
        try:
            target_ts = pd.Timestamp(as_of_date).normalize()
        except Exception:
            return None
        candidates = stock_dates[stock_dates <= target_ts]
        if candidates.empty:
            return None
        ref_ts = candidates.max()

    # Define stock window
    stock_mask = (stock_dates <= ref_ts).to_numpy(dtype=bool)
    stock_slice = stock_df[stock_mask]
    stock_slice_dates = stock_dates[stock_mask]

    if lookback_bars is not None:
        if len(stock_slice) < lookback_bars:
            return None
        stock_slice = stock_slice.iloc[-lookback_bars:]
        stock_slice_dates = stock_slice_dates.iloc[-lookback_bars:]

    window_len = len(stock_slice)
    if window_len == 0:
        return None

    # Define benchmark window across the same span
    bench_mask = (bench_dates <= ref_ts).to_numpy(dtype=bool)
    bench_slice = benchmark_df[bench_mask]
    bench_slice_dates = bench_dates[bench_mask]

    if len(bench_slice) < window_len:
        return None
    bench_slice = bench_slice.iloc[-window_len:]
    bench_slice_dates = bench_slice_dates.iloc[-window_len:]

    # Contiguity check: both windows must have identical date sequences
    bench_date_vals: np.ndarray = bench_slice_dates.to_numpy()
    stock_date_vals: np.ndarray = stock_slice_dates.to_numpy()
    if not bool(np.array_equal(bench_date_vals, stock_date_vals)):
        return None

    # Extract closes as float
    s_close: np.ndarray = stock_slice["close"].to_numpy(dtype=float)
    b_close: np.ndarray = bench_slice["close"].to_numpy(dtype=float)

    # Check for NaN / non-positive prices (None on invalid prices, no NaNs in series)
    if np.any(np.isnan(s_close)) or np.any(np.isnan(b_close)):
        return None
    if np.any(s_close <= 0.0) or np.any(b_close <= 0.0):
        return None

    ratios = s_close / b_close
    return pd.Series(
        ratios, index=pd.DatetimeIndex(stock_slice_dates.to_numpy()), dtype=float
    )

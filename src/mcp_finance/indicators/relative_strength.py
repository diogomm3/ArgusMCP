"""Relative strength pure functions for M4.

Computes stock-vs-benchmark relative performance metrics using daily OHLCV
DataFrames. All calculations use float internally; Decimal is used only at
the model boundary (return types marked ``Decimal``).

No network calls, no database access, no side effects.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from mcp_finance.indicators._date_utils import extract_norm_dates


def relative_return(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    period: int,
    as_of_date: datetime.date | str | None = None,
) -> Decimal | None:
    """Compute the N-bar return difference between a stock and its benchmark.

    Formula::

        stock_return    = (stock_close[as_of] - stock_close[as_of - period]) /
                          stock_close[as_of - period]
        bench_return    = (bench_close[as_of] - bench_close[as_of - period]) /
                          bench_close[as_of - period]
        relative_return = stock_return - bench_return

    Contiguity contract (stated in full):
        After slicing to the ``period + 1`` trailing bars ending at ``as_of_date``,
        *both* frames must have **identical dates** across the whole window.
        Any mismatch — a gap in either series, a missing bar on either side,
        or frames of different lengths — causes the function to return ``None``.

    as_of_date handling:
        If ``as_of_date`` is ``None``, the last available date in
        ``benchmark_df`` is used. Otherwise the function snaps *backward* to
        the nearest trading bar with date ≤ ``as_of_date`` in ``benchmark_df``.
        If no such bar exists, returns ``None``.

    Args:
        stock_df:     DataFrame with at least a ``close`` column and either a
                      ``'date'`` column or a ``DatetimeIndex``.
        benchmark_df: Same shape as ``stock_df`` but for the benchmark.
        period:       Number of bars to look back (e.g. 63 for ~3 months).
                      Must be ≥ 1.
        as_of_date:   Reference date for the most-recent bar. Defaults to the
                      last bar in ``benchmark_df``.

    Returns:
        Signed fractional return difference as a ``Decimal``, or ``None``
        when insufficient data, a date gap, or a non-positive close is found.

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

    # Determine reference date (backward-snap)
    if as_of_date is None:
        ref_ts = bench_dates.max()
    else:
        try:
            target_ts = pd.Timestamp(as_of_date).normalize()
        except Exception:
            return None
        candidates = bench_dates[bench_dates <= target_ts]
        if candidates.empty:
            return None
        ref_ts = candidates.max()

    # Slice to bars on or before ref_ts, then take the last (period + 1) bars
    bench_mask = (bench_dates <= ref_ts).to_numpy(dtype=bool)
    bench_slice = benchmark_df[bench_mask].copy()
    bench_slice_dates = bench_dates[bench_mask]
    if len(bench_slice) < period + 1:
        return None

    bench_slice = bench_slice.iloc[-(period + 1) :]
    bench_slice_dates = bench_slice_dates.iloc[-(period + 1) :]

    # Match stock bars to benchmark dates
    bench_date_set = set(bench_slice_dates.values)
    stock_mask = stock_dates.isin(bench_date_set).to_numpy(dtype=bool)
    stock_slice = stock_df[stock_mask].copy()
    stock_slice_dates = stock_dates[stock_mask]

    if len(stock_slice) < period + 1:
        return None

    # Take the most-recent (period + 1) bars from stock
    stock_slice = stock_slice.iloc[-(period + 1) :]
    stock_slice_dates = stock_slice_dates.iloc[-(period + 1) :]

    # Contiguity check: both windows must have identical date sequences
    bench_date_vals: np.ndarray = bench_slice_dates.to_numpy()
    stock_date_vals: np.ndarray = stock_slice_dates.to_numpy()
    if len(bench_date_vals) != len(stock_date_vals):
        return None
    if not bool(np.array_equal(bench_date_vals, stock_date_vals)):
        return None

    # Extract close arrays (float internally)
    s_close = stock_slice["close"].astype(float).values
    b_close = bench_slice["close"].astype(float).values

    s_start, s_end = s_close[0], s_close[-1]
    b_start, b_end = b_close[0], b_close[-1]

    # Guard against non-positive closes
    if s_start <= 0.0 or b_start <= 0.0:
        return None

    s_ret = (s_end - s_start) / s_start
    b_ret = (b_end - b_start) / b_start

    return Decimal(str(round(s_ret - b_ret, 10)))


def relative_price_ratio(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> "pd.Series[float]":
    """Compute a daily price-ratio series: stock_close / benchmark_close.

    Only dates present in **both** DataFrames are included. The result is
    indexed by the shared normalized dates.

    Formula::

        ratio[t] = stock_close[t] / benchmark_close[t]

    ``NaN`` is returned for any date where either close is zero, NaN, or
    non-positive.

    Args:
        stock_df:     DataFrame with at least a ``close`` column and either a
                      ``'date'`` column or a ``DatetimeIndex``.
        benchmark_df: Same shape as ``stock_df`` but for the benchmark.

    Returns:
        pd.Series indexed by normalized pd.Timestamp. Empty if no dates
        overlap between the two frames.

    Raises:
        ValueError: If either DataFrame is missing a ``close`` column.
    """
    if "close" not in stock_df.columns:
        raise ValueError("stock_df must contain a 'close' column")
    if "close" not in benchmark_df.columns:
        raise ValueError("benchmark_df must contain a 'close' column")

    if stock_df.empty or benchmark_df.empty:
        return pd.Series(dtype=float)

    stock_dates = extract_norm_dates(stock_df)
    bench_dates = extract_norm_dates(benchmark_df)

    # Build date → close float lookups
    stock_map: dict[pd.Timestamp, float] = {
        d: float(c) for d, c in zip(stock_dates.values, stock_df["close"].values)
    }
    bench_map: dict[pd.Timestamp, float] = {
        d: float(c) for d, c in zip(bench_dates.values, benchmark_df["close"].values)
    }

    # Intersect on dates; produce a ratio for each shared date
    shared_dates = sorted(set(stock_map.keys()) & set(bench_map.keys()))
    if not shared_dates:
        return pd.Series(dtype=float)

    ratios: list[float] = []
    for d in shared_dates:
        s_c = stock_map[d]
        b_c = bench_map[d]
        if b_c > 0.0 and s_c >= 0.0 and not (s_c != s_c) and not (b_c != b_c):
            ratios.append(s_c / b_c)
        else:
            ratios.append(float("nan"))

    return pd.Series(ratios, index=pd.DatetimeIndex(shared_dates), dtype=float)

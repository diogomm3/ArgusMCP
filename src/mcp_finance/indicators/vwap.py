"""Anchored Volume-Weighted Average Price (AVWAP) module.

Provides anchored VWAP over daily bars and distance-to-VWAP calculation helpers.

Design Boundaries & Scope Notes:
  - Session VWAP: True session VWAP requires intraday data (tick or minute bars),
    which is outside ArgusMCP's daily-only architecture. This module computes
    anchored VWAP over daily bars using typical price TP = (H + L + C) / 3.
    Session VWAP is explicitly deferred until/if intraday data ingestion is introduced.
  - Candidate Model: Candidate is NOT extended in M3. AVWAP requires a caller-chosen
    anchor date, so there is no single canonical value to store without deciding
    default anchor heuristics.
  - MCP Server Tools: No MCP tool exposes AVWAP directly in M3. Like M2 Volume Profile,
    this is currently a library-only analytical module that will be integrated into
    chart overlays in Phase M6 and/or screening tools.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from mcp_finance.indicators.structure import distance_to_level


def anchored_vwap(
    df: pd.DataFrame,
    anchor_date: datetime.date | str,
) -> pd.Series:
    """Compute Anchored Volume-Weighted Average Price (AVWAP) from a given anchor date.

    Formula:
      TP_t = (High_t + Low_t + Close_t) / 3.0
      AVWAP_t = sum_{i=anchor}^t (TP_i * Vol_i) / sum_{i=anchor}^t Vol_i

    Rules & Edge Cases:
      - Normalization: Dates are converted to normalized pd.Timestamp for consistent
        comparison across datetime.date, datetime.datetime, or string representations.
      - Sorting & Uniqueness: The DataFrame is sorted by date ascending. If duplicate
        dates are found, ValueError is raised.
      - Forward-Snap: If anchor_date falls on a non-trading day (weekend/holiday),
        it snaps forward to the first available trading bar on or after anchor_date
        (>= anchor_date).
      - Zero Volume: While cumulative volume from the anchor bar is 0, AVWAP is
        NaN (never inf). If total volume from the anchor date onward is <= 0,
        ValueError is raised.
      - Pre-anchor bars: All bars strictly before the anchor trading bar are set to NaN.

    Args:
        df: DataFrame containing at least 'high', 'low', 'close', 'volume' columns,
            and either a 'date' column or a DatetimeIndex.
        anchor_date: Date to begin volume weighting
            (datetime.date or 'YYYY-MM-DD' string).

    Returns:
        pd.Series of AVWAP floats aligned to the input DataFrame index (pre-anchor
        values are NaN).

    Raises:
        ValueError: On empty df, missing required columns, duplicate dates, invalid
            date format, anchor_date strictly before the earliest bar, anchor_date
            after the latest bar, or zero total volume from anchor onward.
            Forward-snap applies only when anchor_date is within the series range
            (i.e. >= first bar) but falls on a non-trading day.
    """
    if df.empty:
        raise ValueError("df cannot be empty")

    req_cols = {"high", "low", "close", "volume"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse and normalize anchor_date
    try:
        target_anchor = pd.Timestamp(anchor_date).normalize()
    except Exception as e:
        raise ValueError(f"Invalid anchor_date format: {anchor_date}") from e

    # Extract dates from column or DatetimeIndex
    if "date" in df.columns:
        try:
            norm_dates = pd.to_datetime(df["date"]).dt.normalize()
        except Exception as e:
            raise ValueError("Failed to parse date column") from e
    elif isinstance(df.index, pd.DatetimeIndex):
        norm_dates = pd.Series(df.index.normalize(), index=df.index)
    else:
        raise ValueError(
            "DataFrame must contain a 'date' column or have a DatetimeIndex"
        )

    # Check for duplicate dates
    if norm_dates.duplicated().any():
        raise ValueError("Duplicate dates detected in DataFrame")

    # Defensive sort by date ascending
    orig_index = df.index
    is_sorted = norm_dates.is_monotonic_increasing
    if not is_sorted:
        sort_order = norm_dates.argsort()
        work_df = df.iloc[sort_order].copy()
        norm_dates = norm_dates.iloc[sort_order]
    else:
        work_df = df

    earliest_date = norm_dates.iloc[0]
    latest_date = norm_dates.iloc[-1]

    if target_anchor > latest_date:
        raise ValueError(
            f"anchor_date {anchor_date} is after latest bar ({latest_date.date()})"
        )
    if target_anchor < earliest_date:
        raise ValueError(
            f"anchor_date {anchor_date} is before earliest bar "
            f"({earliest_date.date()}). Extend cache with batch ingestion: "
            f"python -m mcp_finance.market_data.batch --days <N>."
        )

    # Forward-snap: first trading bar on or after target_anchor (within-range gaps only)
    mask_anchor = norm_dates >= target_anchor
    anchor_idx = int(mask_anchor.to_numpy().argmax())

    post_anchor_vols = work_df["volume"].iloc[anchor_idx:].to_numpy(dtype=float)
    if np.sum(post_anchor_vols) <= 0.0:
        raise ValueError(
            "Total volume from anchor date onward must be strictly positive"
        )

    # Compute Typical Price and Typical Price * Volume
    high = work_df["high"].to_numpy(dtype=float)
    low = work_df["low"].to_numpy(dtype=float)
    close = work_df["close"].to_numpy(dtype=float)
    volume = work_df["volume"].to_numpy(dtype=float)

    tp = (high + low + close) / 3.0
    tpv = tp * volume

    n_bars = len(work_df)
    avwap = np.full(n_bars, np.nan, dtype=float)

    # Cumulative accumulation from anchor_idx onward
    cum_tpv = np.cumsum(tpv[anchor_idx:])
    cum_vol = np.cumsum(volume[anchor_idx:])

    # Safe division: where cum_vol > 0, compute AVWAP; otherwise keep NaN
    valid_vol_mask = cum_vol > 0.0
    post_avwap = np.full_like(cum_vol, np.nan)
    post_avwap[valid_vol_mask] = cum_tpv[valid_vol_mask] / cum_vol[valid_vol_mask]

    avwap[anchor_idx:] = post_avwap

    result_series = pd.Series(avwap, index=work_df.index, name="anchored_vwap")
    if not is_sorted:
        result_series = result_series.loc[orig_index]

    return result_series


def price_vs_vwap_distance(
    price: Decimal,
    vwap_val: Decimal,
    atr_val: Decimal | None = None,
) -> tuple[Decimal, Decimal | None]:
    """Calculate signed distance from current price to an Anchored VWAP level.

    Formula:
        pct_distance = (price - vwap_val) / vwap_val
        atr_distance = (price - vwap_val) / atr_val (if atr_val > 0, else None)

    Delegates directly to price_vs_ema_distance for guaranteed contract consistency.

    Args:
        price: Current price.
        vwap_val: VWAP price level.
        atr_val: Optional ATR value for volatility-normalized distance.

    Returns:
        tuple (pct_distance, atr_distance).

    Raises:
        ValueError: If vwap_val is <= 0.
    """
    return distance_to_level(price, vwap_val, atr=atr_val)

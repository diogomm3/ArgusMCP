"""Price and market-structure detection functions.

Pure functions for swing detection, market structure classification,
support/resistance clustering, session gap analysis, and 52-week ranges.

All functions accept normalized pandas DataFrames with lowercase columns:
  open, high, low, close, volume

No network calls, no database access, no side effects.
"""

from decimal import Decimal
from typing import Literal

import numpy as np
import pandas as pd

from mcp_finance.indicators.models import Level, LevelType

MarketStructure = Literal[
    "higher_highs_higher_lows",
    "lower_highs_lower_lows",
    "ranging",
    "insufficient_data",
]


def swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Detect local swing highs and swing lows using symmetrical rolling windows.

    A bar is a swing high if its `high` is strictly greater than all other highs
    in the window [t - lookback, t + lookback].
    A bar is a swing low if its `low` is strictly less than all other lows
    in the window [t - lookback, t + lookback].

    Note on boundary bars:
      The most recent `lookback` bars cannot be confirmed swing points because
      future bars are not yet known. They are set to False/NaN.

    Returns:
        DataFrame with columns:
          - is_swing_high: bool
          - swing_high_price: float (NaN if not swing high)
          - is_swing_low: bool
          - swing_low_price: float (NaN if not swing low)
    """
    n = len(df)
    is_high = pd.Series(False, index=df.index)
    high_price = pd.Series(np.nan, index=df.index, dtype=float)
    is_low = pd.Series(False, index=df.index)
    low_price = pd.Series(np.nan, index=df.index, dtype=float)

    if n < 2 * lookback + 1:
        return pd.DataFrame(
            {
                "is_swing_high": is_high,
                "swing_high_price": high_price,
                "is_swing_low": is_low,
                "swing_low_price": low_price,
            },
            index=df.index,
        )

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    # Symmetrical window check: t must exceed
    # [t - lookback, t - 1] and [t + 1, t + lookback]
    for i in range(lookback, n - lookback):
        curr_high = highs[i]
        curr_low = lows[i]

        # Check swing high (strictly greater than neighbors)
        left_highs = highs[i - lookback : i]
        right_highs = highs[i + 1 : i + lookback + 1]
        if (curr_high > left_highs).all() and (curr_high > right_highs).all():
            is_high.iloc[i] = True
            high_price.iloc[i] = float(curr_high)

        # Check swing low (strictly lower than neighbors)
        left_lows = lows[i - lookback : i]
        right_lows = lows[i + 1 : i + lookback + 1]
        if (curr_low < left_lows).all() and (curr_low < right_lows).all():
            is_low.iloc[i] = True
            low_price.iloc[i] = float(curr_low)

    return pd.DataFrame(
        {
            "is_swing_high": is_high,
            "swing_high_price": high_price,
            "is_swing_low": is_low,
            "swing_low_price": low_price,
        },
        index=df.index,
    )


def market_structure(df: pd.DataFrame, lookback: int = 5) -> MarketStructure:
    """Classify trend structure based on the sequence of confirmed swing points.

    Evaluates the last 2 confirmed swing highs and the last 2 confirmed swing lows:
      - "higher_highs_higher_lows": high[-1] > high[-2] and low[-1] > low[-2]
      - "lower_highs_lower_lows":   high[-1] < high[-2] and low[-1] < low[-2]
      - "ranging":                  mixed structure (e.g. higher high with lower low)
      - "insufficient_data":        fewer than 2 swing highs or 2 swing lows found

    Note:
      `build_candidate_snapshot()` maps "insufficient_data" to None when populating
      Candidate models so downstream callers can follow the standard `is None` pattern.
    """
    swings = swing_highs_lows(df, lookback=lookback)
    recent_highs = swings.loc[swings["is_swing_high"], "swing_high_price"].tolist()
    recent_lows = swings.loc[swings["is_swing_low"], "swing_low_price"].tolist()

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return "insufficient_data"

    h1, h2 = recent_highs[-2], recent_highs[-1]
    l1, l2 = recent_lows[-2], recent_lows[-1]

    if h2 > h1 and l2 > l1:
        return "higher_highs_higher_lows"
    if h2 < h1 and l2 < l1:
        return "lower_highs_lower_lows"
    return "ranging"


def support_resistance_levels(
    df: pd.DataFrame,
    lookback: int = 5,
    tolerance_pct: float = 0.015,
    min_touches: int = 2,
) -> list[Level]:
    """Cluster swing highs and swing lows into support and resistance levels.

    Heuristic clustering algorithm:
      1. Detect confirmed swing highs and swing lows.
      2. Collect each pivot as (price, is_high: bool).
      3. Sort points ascending by price.
      4. Cluster adjacent points where price <= cluster_min * (1 + tolerance_pct).
      5. Filter clusters by touches >= min_touches.
      6. Classify level_type:
         - "SUPPORT": All points in cluster are swing lows.
         - "RESISTANCE": All points in cluster are swing highs.
         - "PIVOT": Cluster contains both swing highs and swing lows (polarity flip).

    Args:
        df: DataFrame with high, low, close.
        lookback: Symmetrical swing lookback window.
        tolerance_pct: Maximum relative price gap (e.g. 0.015 for 1.5%) to
            merge into one level.
        min_touches: Minimum touches to qualify as a valid level (default: 2).

    Returns:
        List of Level objects sorted ascending by representative price.
    """
    swings = swing_highs_lows(df, lookback=lookback)

    high_mask = swings["is_swing_high"].to_numpy()
    high_prices = swings["swing_high_price"].to_numpy()
    low_mask = swings["is_swing_low"].to_numpy()
    low_prices = swings["swing_low_price"].to_numpy()

    points: list[tuple[float, bool]] = []
    for p in high_prices[high_mask]:
        points.append((float(p), True))
    for p in low_prices[low_mask]:
        points.append((float(p), False))

    if not points:
        return []

    # Sort by price ascending
    points.sort(key=lambda p: p[0])

    # Cluster into tolerance buckets
    raw_clusters: list[list[tuple[float, bool]]] = []
    current_cluster: list[tuple[float, bool]] = [points[0]]

    for p in points[1:]:
        cluster_base = current_cluster[0][0]
        if p[0] <= cluster_base * (1.0 + tolerance_pct):
            current_cluster.append(p)
        else:
            raw_clusters.append(current_cluster)
            current_cluster = [p]
    raw_clusters.append(current_cluster)

    levels: list[Level] = []
    for cluster in raw_clusters:
        if len(cluster) < min_touches:
            continue

        prices = [pt[0] for pt in cluster]
        has_highs = any(pt[1] for pt in cluster)
        has_lows = any(not pt[1] for pt in cluster)

        if has_highs and has_lows:
            lvl_type: LevelType = "PIVOT"
        elif has_highs:
            lvl_type = "RESISTANCE"
        else:
            lvl_type = "SUPPORT"

        avg_price = sum(prices) / len(prices)
        levels.append(
            Level(
                price=Decimal(f"{avg_price:.6f}"),
                level_type=lvl_type,
                touches=len(cluster),
                min_price=Decimal(f"{min(prices):.6f}"),
                max_price=Decimal(f"{max(prices):.6f}"),
            )
        )

    return levels


def gap_pct(df: pd.DataFrame) -> "pd.Series[float]":
    """Calculate overnight/session price gap percentage.

    Formula:
        (open[t] - close[t-1]) / close[t-1]

    Returns:
        Series of gap percentages (e.g. 0.02 for +2.0% gap). The first entry is NaN.
    """
    prev_close = df["close"].shift(1)
    gap = (df["open"] - prev_close) / prev_close
    return gap


def gap_analysis(
    df: pd.DataFrame,
    atr_series: "pd.Series[float] | None" = None,
) -> pd.DataFrame:
    """Analyze gap metrics: ATR multiple, days since last gap, and fill status.

    A gap is defined as any session where |gap_pct| >= 0.005 (0.5%).
    A gap-up is filled if low[t] <= close[t-1].
    A gap-down is filled if high[t] >= close[t-1].

    Returns:
        DataFrame with columns:
          - gap_pct: float
          - gap_vs_atr: float | NaN (gap size in dollars divided by ATR)
          - is_gap: bool (|gap_pct| >= 0.5%)
          - is_filled: bool (whether gap was filled during the session)
    """
    prev_close = df["close"].shift(1)
    gap_dollar = df["open"] - prev_close
    gap_percent = gap_dollar / prev_close
    is_gap = gap_percent.abs() >= 0.005

    # Gap fill conditions
    # Gap up: open > prev_close; filled if low <= prev_close
    # Gap down: open < prev_close; filled if high >= prev_close
    gap_up = gap_dollar > 0
    gap_down = gap_dollar < 0
    filled = pd.Series(False, index=df.index)
    filled[gap_up] = df.loc[gap_up, "low"] <= prev_close[gap_up]
    filled[gap_down] = df.loc[gap_down, "high"] >= prev_close[gap_down]

    if atr_series is not None and not atr_series.empty:
        gap_vs_atr = gap_dollar.abs() / atr_series
    else:
        gap_vs_atr = pd.Series(np.nan, index=df.index, dtype=float)

    return pd.DataFrame(
        {
            "gap_pct": gap_percent,
            "gap_vs_atr": gap_vs_atr,
            "is_gap": is_gap,
            "is_filled": filled,
        },
        index=df.index,
    )


def distance_to_level(
    price: Decimal,
    level: Decimal,
    atr: Decimal | None = None,
) -> tuple[Decimal, Decimal | None]:
    """Calculate signed distance from current price to a target level.

    Returns:
        tuple (pct_distance, atr_distance):
          - pct_distance: (price - level) / level
          - atr_distance: (price - level) / atr if atr is provided and > 0, else None
    """
    if level == Decimal("0"):
        raise ValueError("Level price cannot be zero.")

    pct_dist = (price - level) / level
    atr_dist = None
    if atr is not None and atr > Decimal("0"):
        atr_dist = (price - level) / atr

    return pct_dist, atr_dist


def week_52_high_low(
    df: pd.DataFrame, window_bars: int = 252
) -> tuple[Decimal, Decimal]:
    """Compute 52-week (rolling window) high and low prices.

    Uses the trailing `window_bars` (default 252 trading days). If fewer bars
    exist, computes over all available bars.

    Returns:
        tuple (week_52_high, week_52_low) as Decimals.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    tail_df = df.tail(window_bars)
    high_val = float(tail_df["high"].max())
    low_val = float(tail_df["low"].min())

    return Decimal(f"{high_val:.6f}"), Decimal(f"{low_val:.6f}")

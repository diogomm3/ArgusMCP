"""Market regime indicator pure function (Phase M4).

Provides `market_regime(df, as_of_date)` implementing Variant (b) on a fixed
500-bar trailing window.

Analytical Scope & Intent:
    This is a **lagging trend-state label**, not an intraday or entry/exit timing
    signal. Empirical drawdown calibration shows:
    - Leaves BULLISH after a 4–9% market pullback (e.g. 2022 drawdown leaves
      BULLISH on 2022-01-25 at -9.05% from peak; 2025 pullback leaves BULLISH
      on 2025-01-14 at -4.22% from peak).
    - Turns BEARISH near ~13% drawdown (e.g. 2022 turns BEARISH on 2022-05-02
      at -13.24% from peak; 2025 turns BEARISH on 2025-04-11 at -12.89% from peak,
      three days after the April 2025 low).

    Breadth Approximation Note:
    Explicitly **not** true market breadth (e.g. advance-decline lines, NYSE % above
    50-day MA). It is an approximation derived from major benchmark / sector ETF
    price series (such as SPY or QQQ), adhering to the build-vs-buy architectural
    scope of ArgusMCP.

Fixed Window Policy:
    Requires exactly a 500-bar trailing window ending at or before `as_of_date`.
    If fewer than 500 trading bars are available up to `as_of_date`, returns `None`.
    This guarantees 100% deterministic window equivalence across environments
    and eliminates EMA-200 seed-bias drift (<0.01% weight from the initial seed
    at bar 500).

Classification Rule — Variant (b):
    - BULLISH: EMA50 > EMA200 AND slope20(EMA50) > 0.0025
    - BEARISH: EMA50 < EMA200 AND slope20(EMA50) < -0.0025
    - NEUTRAL: All other configurations (including deadband slope, exact equality,
      or opposing alignment/slope).

Sessions in Regime:
    Counts the number of consecutive trading sessions the current label has held
    within the evaluated 500-bar window, capped at 60. This provides downstream
    consumers a stateless heuristic to discount fresh flips without maintaining
    cross-invocation state.

No network calls, no database access, no side effects.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from mcp_finance.indicators._date_utils import extract_norm_dates
from mcp_finance.indicators.functions import ema, ema_slope
from mcp_finance.indicators.models import MarketRegime, RegimeLabel

DEADBAND: float = 0.0025
REQUIRED_WINDOW_BARS: int = 500
MAX_SESSIONS_CAP: int = 60


def market_regime(
    df: pd.DataFrame,
    as_of_date: datetime.date | str | None = None,
) -> MarketRegime | None:
    """Classify market trend regime using EMA-50/200 alignment and EMA-50 slope.

    Args:
        df: Daily OHLCV DataFrame containing at least a ``close`` column
            and either a ``date`` column or a ``pd.DatetimeIndex``.
        as_of_date: Evaluation cutoff date. Snaps backward to the nearest
            trading session with date <= as_of_date. If None, defaults to the
            latest session in ``df``.

    Returns:
        `MarketRegime` instance containing regime label, sessions count,
        and underlying trend/distance metrics, or `None` if fewer than
        500 bars are available, data is invalid, or dates are non-monotonic.
    """
    if df.empty or "close" not in df.columns:
        return None

    try:
        norm_dates = extract_norm_dates(df)
    except ValueError:
        return None

    # Monotonicity and uniqueness checks
    if norm_dates.duplicated().any() or not norm_dates.is_monotonic_increasing:
        return None

    # Snap as_of_date
    if as_of_date is None:
        ref_ts = norm_dates.max()
    else:
        try:
            target_ts = pd.Timestamp(as_of_date).normalize()
        except Exception:
            return None
        candidates = norm_dates[norm_dates <= target_ts]
        if candidates.empty:
            return None
        ref_ts = candidates.max()

    # Slice to dates <= ref_ts
    mask = (norm_dates <= ref_ts).to_numpy(dtype=bool)
    sliced_df = df[mask]
    sliced_dates = norm_dates[mask]

    # Fixed 500-bar window policy: strictly require at least 500 bars
    if len(sliced_df) < REQUIRED_WINDOW_BARS:
        return None

    window_df = sliced_df.iloc[-REQUIRED_WINDOW_BARS:].copy()
    window_dates = sliced_dates.iloc[-REQUIRED_WINDOW_BARS:]

    # Ensure valid positive prices
    closes = window_df["close"].astype(float)
    if (closes <= 0.0).any() or closes.isna().any():
        return None

    # Calculate EMAs and slope
    # Use standard indicator functions: ema(df, period) and ema_slope(series, lookback)
    e50_series = ema(window_df, 50, column="close")
    e200_series = ema(window_df, 200, column="close")
    slope20_series = ema_slope(e50_series, lookback_bars=20)

    cur_close = float(closes.iloc[-1])
    cur_e50 = float(e50_series.iloc[-1])
    cur_e200 = float(e200_series.iloc[-1])
    cur_slope = float(slope20_series.iloc[-1])
    eval_date = window_dates.iloc[-1].date()

    if np.isnan(cur_e50) or np.isnan(cur_e200) or np.isnan(cur_slope):
        return None

    # Variant (b) classification with strict inequalities
    regime: RegimeLabel
    if cur_e50 > cur_e200 and cur_slope > DEADBAND:
        regime = "BULLISH"
    elif cur_e50 < cur_e200 and cur_slope < -DEADBAND:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    # Compute sessions_in_regime (stateless, backwards scan across window, capped at 60)
    is_bull = (e50_series > e200_series) & (slope20_series > DEADBAND)
    is_bear = (e50_series < e200_series) & (slope20_series < -DEADBAND)

    labels = pd.Series("NEUTRAL", index=window_df.index)
    labels[is_bull] = "BULLISH"
    labels[is_bear] = "BEARISH"

    consecutive = 0
    # Walk backwards from the latest session
    for lab in reversed(labels.tolist()):
        if lab == regime:
            consecutive += 1
        else:
            break

    sessions_in_regime = min(max(consecutive, 1), MAX_SESSIONS_CAP)

    # Relative percentage distances
    close_vs_ema50_pct = (cur_close - cur_e50) / cur_e50 * 100.0
    close_vs_ema200_pct = (cur_close - cur_e200) / cur_e200 * 100.0

    return MarketRegime(
        regime=regime,
        sessions_in_regime=sessions_in_regime,
        close_vs_ema50_pct=Decimal(str(round(close_vs_ema50_pct, 4))),
        close_vs_ema200_pct=Decimal(str(round(close_vs_ema200_pct, 4))),
        ema50_slope20=Decimal(str(round(cur_slope, 6))),
        ema50=Decimal(str(round(cur_e50, 4))),
        ema200=Decimal(str(round(cur_e200, 4))),
        close=Decimal(str(round(cur_close, 4))),
        as_of_date=eval_date,
    )

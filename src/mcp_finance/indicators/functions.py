"""Pure technical indicator functions.

All functions accept a DataFrame with lowercase columns:
  open, high, low, close, volume

This is the normalized shape produced by _clean_ohlcv_dataframe() in
market_data/utils.py and by the OhlcvDaily → DataFrame conversion in snapshot.py.

No network calls, no database access, no side effects.
"""

from decimal import Decimal
from typing import Literal

import pandas as pd


def ema(df: pd.DataFrame, period: int, column: str = "close") -> "pd.Series[float]":
    """Exponential Moving Average.

    Uses pandas ewm with adjust=False (recursive form), which matches the
    conventional EMA formula used in charting platforms.

    Returns:
        Series of the same length as df. The first (period - 1) values are NaN
        because there is insufficient history; from row `period` onwards values
        are finite.

    Args:
        df:     DataFrame with at least the `column` column.
        period: EMA span (e.g. 20 for EMA-20).
        column: Price column to smooth (default: "close").
    """
    return df[column].ewm(span=period, min_periods=period, adjust=False).mean()


def rsi(
    df: pd.DataFrame, period: int = 14, column: str = "close"
) -> "pd.Series[float]":
    """Relative Strength Index using Wilder smoothing.

    Wilder's original RSI uses an exponential smoothing equivalent to
    ewm(alpha=1/period, adjust=False) on the up-moves and down-moves.

    Returns:
        Series in [0, 100]. The first `period` values are NaN.

    Args:
        df:     DataFrame with at least the `column` column.
        period: RSI period (default: 14).
        column: Price column (default: "close").
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result: pd.Series[float] = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss == 0 and avg_gain > 0, RSI should be 100; handle NaN/inf.
    result = result.where(avg_loss != 0.0, other=100.0)
    # First row of delta is NaN, so first RSI value is also NaN.
    result.iloc[0] = float("nan")
    return result


def atr(df: pd.DataFrame, period: int = 14) -> "pd.Series[float]":
    """Average True Range using Wilder smoothing.

    True Range = max(high − low, |high − prev_close|, |low − prev_close|)
    ATR = Wilder smoothing of TR over `period`.

    Returns:
        Series in price units. The first value is NaN (no previous close).

    Args:
        df:     DataFrame with columns high, low, close.
        period: ATR period (default: 14).
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """MACD — Moving Average Convergence/Divergence.

    Returns:
        DataFrame with three columns:
          - macd:      EMA(fast) − EMA(slow)
          - signal:    EMA(macd, signal)
          - histogram: macd − signal

    Args:
        df:     DataFrame with at least the `column` column.
        fast:   Fast EMA period (default: 12).
        slow:   Slow EMA period (default: 26).
        signal: Signal EMA period (default: 9).
        column: Price column (default: "close").
    """
    ema_fast = ema(df, fast, column)
    ema_slow = ema(df, slow, column)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        },
        index=df.index,
    )


EmaAlignment = Literal["bullish", "bearish", "mixed"]


def ema_slope(series: "pd.Series[float]", lookback_bars: int = 5) -> "pd.Series[float]":
    """Calculate the total percentage change of a moving average over a window.

    Formula:
        (series[t] - series[t - lookback_bars]) / series[t - lookback_bars]

    Returns:
        Series of total percentage change as a fractional float
        (e.g. 0.02 for +2.0% change over the window). The first `lookback_bars`
        entries are NaN. If the reference value is zero or negative, returns NaN.

    Args:
        series: Pandas series of moving average or price values.
        lookback_bars: Trailing window size (default: 5 bars).
            To obtain the average per-bar rate of change, divide by `lookback_bars`.
    """
    if lookback_bars < 1:
        raise ValueError("lookback_bars must be at least 1.")

    prev = series.shift(lookback_bars)
    slope = (series - prev) / prev
    return slope.where(prev > 0.0, other=float("nan"))


def ema_alignment(
    ema9: float | Decimal | None,
    ema20: float | Decimal | None,
    ema50: float | Decimal | None,
    ema200: float | Decimal | None,
) -> EmaAlignment:
    """Classify the trend alignment across four key EMAs (9, 20, 50, 200).

    Classification rules:
      - "bullish": Strict descending cascade (ema9 > ema20 > ema50 > ema200)
      - "bearish": Strict ascending cascade (ema9 < ema20 < ema50 < ema200)
      - "mixed": Any other relationship, equality, or if ANY value is None.

    Args:
        ema9: Fast 9-period EMA.
        ema20: Swing 20-period EMA.
        ema50: Medium 50-period EMA.
        ema200: Long-term 200-period baseline EMA.

    Returns:
        "bullish", "bearish", or "mixed".
    """
    if ema9 is None or ema20 is None or ema50 is None or ema200 is None:
        return "mixed"

    if ema9 > ema20 > ema50 > ema200:
        return "bullish"
    if ema9 < ema20 < ema50 < ema200:
        return "bearish"
    return "mixed"


def price_vs_ema_distance(
    price: Decimal,
    ema_val: Decimal,
    atr_val: Decimal | None = None,
) -> tuple[Decimal, Decimal | None]:
    """Calculate signed distance from current price to an EMA.

    Formula:
        pct_distance = (price - ema_val) / ema_val
        atr_distance = (price - ema_val) / atr_val (if atr_val > 0, else None)

    Args:
        price: Current price.
        ema_val: Moving average price level.
        atr_val: Optional ATR value for volatility-normalized distance.

    Returns:
        tuple (pct_distance, atr_distance).

    Raises:
        ValueError: If ema_val is <= 0.
    """
    if ema_val <= Decimal("0"):
        raise ValueError("ema_val must be strictly positive.")

    pct_dist = (price - ema_val) / ema_val
    atr_dist = None
    if atr_val is not None and atr_val > Decimal("0"):
        atr_dist = (price - ema_val) / atr_val

    return pct_dist, atr_dist


# ---------------------------------------------------------------------------
# M1.3 — Momentum extensions
# ---------------------------------------------------------------------------


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI.

    Uses Wilder smoothing (alpha = 1/period) on True Range, +DM, and -DM,
    exactly matching the treatment of ATR and RSI in this codebase.

    Warmup note: ADX requires ``2 * period - 1`` bars before the first valid
    value — the DI series itself must warm up (``period`` bars) before DX can
    be smoothed into ADX (another ``period`` bars). The first
    ``2 * period - 2`` rows are therefore NaN, not zero.

    Returns:
        DataFrame with three columns:
          - adx:      Smoothed directional index in [0, 100]. NaN until
                      ``2 * period - 1`` bars are available.
          - plus_di:  +DI directional indicator in [0, 100].
          - minus_di: -DI directional indicator in [0, 100].

    Args:
        df:     DataFrame with columns high, low, close.
        period: ADX smoothing period (default: 14).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    # True Range (same formula as atr())
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional moves
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), other=0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), other=0.0)

    # Wilder smoothing
    smoothed_tr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_plus = plus_dm.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    smoothed_minus = minus_dm.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di_series = 100.0 * smoothed_plus / smoothed_tr
    minus_di_series = 100.0 * smoothed_minus / smoothed_tr

    dx = (
        100.0
        * (plus_di_series - minus_di_series).abs()
        / (plus_di_series + minus_di_series)
    )

    adx_series = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {
            "adx": adx_series,
            "plus_di": plus_di_series,
            "minus_di": minus_di_series,
        },
        index=df.index,
    )


def roc(
    df: pd.DataFrame, period: int = 10, column: str = "close"
) -> "pd.Series[float]":
    """Rate of Change — percentage price change over ``period`` bars.

    Formula:
        ROC[t] = (close[t] - close[t - period]) / close[t - period]

    Returns:
        Series of fractional change (e.g. 0.05 for +5%). The first ``period``
        values are NaN. If the reference price is zero or negative, returns NaN.

    Args:
        df:     DataFrame with at least the ``column`` column.
        period: Look-back window (default: 10).
        column: Price column (default: ``"close"``).
    """
    if period < 1:
        raise ValueError("period must be at least 1.")
    prev = df[column].shift(period)
    result = (df[column] - prev) / prev
    return result.where(prev > 0.0, other=float("nan"))


# ---------------------------------------------------------------------------
# M1.4 — Volatility extensions
# ---------------------------------------------------------------------------


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0, column: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands — rolling mean ± ``num_std`` standard deviations.

    Uses an unbiased (ddof=1) rolling standard deviation, which is standard
    for Bollinger Bands as defined by John Bollinger.

    Returns:
        DataFrame with three columns:
          - upper:  middle + num_std * rolling_std
          - middle: rolling simple moving average (SMA)
          - lower:  middle - num_std * rolling_std

        The first ``period - 1`` rows are NaN.

    Args:
        df:      DataFrame with at least the ``column`` column.
        period:  SMA and standard-deviation window (default: 20).
        num_std: Number of standard deviations for the band width (default: 2.0).
        column:  Price column (default: ``"close"``).
    """
    price = df[column]
    middle = price.rolling(window=period, min_periods=period).mean()
    std = price.rolling(window=period, min_periods=period).std(ddof=1)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower}, index=df.index
    )


def bollinger_bandwidth(bands_df: pd.DataFrame) -> "pd.Series[float]":
    """Bollinger Bandwidth — (upper − lower) / middle.

    Measures band width as a fraction of the middle band. A larger value
    indicates higher volatility; contracting bandwidth often precedes breakouts.

    Args:
        bands_df: DataFrame produced by ``bollinger_bands()`` with columns
                  ``upper``, ``middle``, ``lower``.

    Returns:
        Series of fractional bandwidth. NaN where middle is zero or NaN.
    """
    bw: pd.Series[float] = (bands_df["upper"] - bands_df["lower"]) / bands_df["middle"]
    return bw.where(bands_df["middle"] > 0.0, other=float("nan"))


def atr_expansion_ratio(
    atr_series: "pd.Series[float]", avg_period: int = 50
) -> "pd.Series[float]":
    """ATR Expansion Ratio — current ATR divided by its rolling average.

    Identifies whether recent volatility is elevated (ratio > 1) or compressed
    (ratio < 1) relative to the recent baseline.

    Formula:
        ratio[t] = ATR[t] / SMA(ATR, avg_period)[t]

    Returns:
        Series of ratios. The first ``avg_period - 1`` values are NaN.
        If the rolling average is zero, returns NaN (no division by zero).

    Args:
        atr_series: ATR series (e.g. from ``atr()``).
        avg_period: Window for the rolling ATR mean baseline (default: 50).
    """
    rolling_avg = atr_series.rolling(window=avg_period, min_periods=avg_period).mean()
    ratio: pd.Series[float] = atr_series / rolling_avg
    return ratio.where(rolling_avg > 0.0, other=float("nan"))


def rolling_percentile(series: "pd.Series[float]", window: int) -> "pd.Series[float]":
    """Rolling percentile rank — where does the current value rank in the window?

    For each bar, computes the fraction of values in the trailing ``window``
    that are strictly less than the current value (standard "less-than" rank).

    Formula:
        percentile[t] = count(series[t-window+1..t-1] < series[t]) / (window - 1)

    Returns:
        Series in [0, 1]. The first ``window - 1`` values are NaN.

    Args:
        series: Pandas Series of values to rank.
        window: Rolling window size.
    """
    if window < 2:
        raise ValueError("window must be at least 2 for a meaningful percentile.")

    def _pct(arr: "pd.Series[float]") -> float:
        if len(arr) < window:
            return float("nan")
        current = arr.iloc[-1]
        past = arr.iloc[:-1]
        return float((past < current).sum()) / len(past)

    return series.rolling(window=window, min_periods=window).apply(_pct, raw=False)


# ---------------------------------------------------------------------------
# M1.5 — Volume extensions
# ---------------------------------------------------------------------------


def relative_volume(df: pd.DataFrame, period: int = 20) -> "pd.Series[float]":
    """Relative Volume (RVOL) — today's volume vs. its ``period``-day average.

    Formula:
        RVOL[t] = volume[t] / SMA(volume, period)[t]

    Returns:
        Series of ratios. Values > 1 indicate above-average activity.
        The first ``period - 1`` values are NaN. Returns NaN where the rolling
        average is zero (degenerate case: all-zero volume window).

    Args:
        df:     DataFrame with a ``volume`` column.
        period: Rolling average window (default: 20).
    """
    avg = df["volume"].rolling(window=period, min_periods=period).mean()
    rvol: pd.Series[float] = df["volume"] / avg
    return rvol.where(avg > 0.0, other=float("nan"))


def dollar_volume(df: pd.DataFrame) -> "pd.Series[float]":
    """Dollar Volume — close price multiplied by volume.

    A proxy for liquidity and institutional participation. High dollar volume
    relative to peers indicates a more liquid, tradeable name.

    Formula:
        dollar_volume[t] = close[t] * volume[t]

    Returns:
        Series of dollar-volume values (same length as ``df``). No NaNs
        unless ``close`` or ``volume`` themselves contain NaNs.

    Args:
        df: DataFrame with columns ``close`` and ``volume``.
    """
    return df["close"] * df["volume"]


def obv(df: pd.DataFrame) -> "pd.Series[float]":
    """On-Balance Volume (OBV).

    Accumulates signed volume based on price direction:
      - Close UP   → add volume
      - Close DOWN → subtract volume
      - No change  → no change

    The first bar's OBV equals its own volume (no prior close to compare).

    Returns:
        Cumulative OBV series (same length as ``df``). No NaN values.

    Args:
        df: DataFrame with columns ``close`` and ``volume``.
    """
    direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # First bar has no diff — treat as flat (no contribution).
    direction.iloc[0] = 0
    signed_vol: pd.Series[float] = direction * df["volume"]
    return signed_vol.cumsum()


def mfi(df: pd.DataFrame, period: int = 14) -> "pd.Series[float]":
    """Money Flow Index (MFI) — volume-weighted RSI analogue.

    Algorithm:
        1. Typical Price (TP) = (high + low + close) / 3
        2. Raw Money Flow   = TP * volume
        3. Positive MF      = RMF where TP > prev_TP, else 0
        4. Negative MF      = RMF where TP < prev_TP, else 0
        5. Money Ratio      = sum(pos_MF, period) / sum(neg_MF, period)
        6. MFI              = 100 − 100 / (1 + money_ratio)

    Returns:
        Series in [0, 100]. The first ``period`` values are NaN (insufficient
        rolling history). Where negative money flow is zero, MFI = 100.

    Args:
        df:     DataFrame with columns high, low, close, volume.
        period: Rolling window for money flow sums (default: 14).
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_mf = tp * df["volume"]

    prev_tp = tp.shift(1)
    pos_mf = raw_mf.where(tp > prev_tp, other=0.0)
    neg_mf = raw_mf.where(tp < prev_tp, other=0.0)

    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()

    money_ratio = pos_sum / neg_sum
    result: pd.Series[float] = 100.0 - (100.0 / (1.0 + money_ratio))

    # Where neg_sum == 0 and pos_sum > 0, MFI should be 100.
    result = result.where(neg_sum != 0.0, other=100.0)
    return result

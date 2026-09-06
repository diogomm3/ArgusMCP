"""Pure technical indicator functions.

All functions accept a DataFrame with lowercase columns:
  open, high, low, close, volume

This is the normalized shape produced by _clean_ohlcv_dataframe() in
market_data/utils.py and by the OhlcvDaily → DataFrame conversion in snapshot.py.

No network calls, no database access, no side effects.
"""

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

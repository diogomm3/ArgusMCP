"""Unit tests for the technical indicators module.

Fixture design:
  - FLAT fixture (60 rows, close=50.0): tests that EMA on a constant series
    converges to the constant. ATR = 0 on a perfectly flat series (degenerate),
    so ATR tests use a VOLATILE fixture.
  - LINEAR fixture (60 rows, close = 100 + i*0.5): tests EMA-20 convergence on
    a linear trend. For a perfectly linear series, the recursive EMA eventually
    tracks the series with a lag. The reference value is computed here via the
    same formula and checked for self-consistency (histogram = macd - signal).
  - STEP fixture: 30 bars at 50.0, then 30 bars at 51.0. After the step,
    all gains, no losses → RSI → 100.
  - VOLATILE fixture: alternating highs and lows to test ATR > 0.

Reference values for tight-tolerance tests (EMA, RSI) are computed here via
numpy/pandas directly, then compared back to the functions under test. This
tests that the implementation matches the *formula* rather than just itself —
because the formula is well-specified (ewm adjust=False) and independently
verifiable against charting platforms.
"""

import datetime
import math
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from mcp_finance.indicators.functions import (
    adx,
    atr,
    atr_expansion_ratio,
    bollinger_bands,
    bollinger_bandwidth,
    dollar_volume,
    ema,
    ema_alignment,
    ema_slope,
    macd,
    mfi,
    obv,
    price_vs_ema_distance,
    relative_volume,
    roc,
    rolling_percentile,
    rsi,
)
from mcp_finance.indicators.snapshot import (
    SymbolNotCachedError,
    build_candidate_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def flat_df() -> pd.DataFrame:
    """60 rows, all prices = 50.0. Spread = 0.5 each side."""
    n = 60
    close = np.full(n, 50.0)
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": np.full(n, 1_000_000, dtype=int),
        }
    )


@pytest.fixture()
def linear_df() -> pd.DataFrame:
    """60 rows, close = 100 + i * 0.5 (strictly increasing linear trend)."""
    n = 60
    close = np.array([100.0 + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "open": close - 0.10,
            "high": close + 0.30,
            "low": close - 0.30,
            "close": close,
            "volume": np.full(n, 500_000, dtype=int),
        }
    )


@pytest.fixture()
def step_df() -> pd.DataFrame:
    """60 rows: first 30 at close=50.0, then 30 at close=51.0 (pure gains)."""
    n = 60
    close = np.concatenate([np.full(30, 50.0), np.full(30, 51.0)])
    return pd.DataFrame(
        {
            "open": close - 0.10,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": np.full(n, 800_000, dtype=int),
        }
    )


@pytest.fixture()
def volatile_df() -> pd.DataFrame:
    """60 rows with alternating high-low spreads to guarantee ATR > 0."""
    n = 60
    close = np.array([50.0 + (i % 5) * 2.0 for i in range(n)])
    high = close + 1.5
    low = close - 1.5
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 750_000, dtype=int),
        }
    )


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ema_flat_series_converges(flat_df: pd.DataFrame) -> None:
    """EMA on a constant series should equal the constant (after warmup)."""
    result = ema(flat_df, 20)
    # First (period - 1) values are NaN due to min_periods=20.
    assert result.iloc[:19].isna().all()
    valid = result.dropna()
    assert len(valid) == 41
    np.testing.assert_allclose(valid.values, 50.0, rtol=1e-6)


@pytest.mark.unit
def test_ema_linear_known_value(linear_df: pd.DataFrame) -> None:
    """EMA-20 last value on linear_df must match a reference computed independently.

    Reference: pd.Series.ewm(
        span=20, min_periods=20, adjust=False
    ).mean() on the same data.
    This is a cross-implementation consistency check — we verify our ema()
    helper produces the same result as the pandas expression it wraps.
    """
    reference = linear_df["close"].ewm(span=20, min_periods=20, adjust=False).mean()
    result = ema(linear_df, 20)
    np.testing.assert_allclose(result.values, reference.values, rtol=1e-9)


@pytest.mark.unit
def test_ema_returns_series_same_length(linear_df: pd.DataFrame) -> None:
    result = ema(linear_df, 20)
    assert len(result) == len(linear_df)


@pytest.mark.unit
def test_ema_custom_column(linear_df: pd.DataFrame) -> None:
    result_close = ema(linear_df, 10, column="close")
    result_open = ema(linear_df, 10, column="open")
    # Open is close - 0.10, so valid EMA(open) should be EMA(close) - 0.10.
    valid_close = result_close.dropna()
    valid_open = result_open.dropna()
    np.testing.assert_allclose(valid_open.values, valid_close.values - 0.10, rtol=1e-9)


@pytest.mark.unit
def test_ema_min_periods_returns_nan_on_insufficient_bars() -> None:
    """EMA-50 on a 30-bar series must return all NaNs (warmup not met)."""
    df = pd.DataFrame({"close": range(30)})
    result = ema(df, 50)
    assert len(result) == 30
    assert result.isna().all()


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rsi_bounds_volatile(volatile_df: pd.DataFrame) -> None:
    """All non-NaN RSI values must be in [0, 100]."""
    result = rsi(volatile_df, 14)
    valid = result.dropna()
    assert (valid >= 0.0).all() and (valid <= 100.0).all()


@pytest.mark.unit
def test_rsi_first_value_nan(volatile_df: pd.DataFrame) -> None:
    """First RSI value is always NaN (no previous close for diff)."""
    result = rsi(volatile_df, 14)
    assert math.isnan(result.iloc[0])


@pytest.mark.unit
def test_rsi_step_up_approaches_100(step_df: pd.DataFrame) -> None:
    """After 30 pure-gain bars, RSI-14 should be very close to 100."""
    result = rsi(step_df, 14)
    # The last value should be > 95 (pure gains, Wilder smoothing converges to 100).
    last_valid = result.dropna().iloc[-1]
    assert last_valid > 95.0, f"Expected RSI > 95 on step-up series, got {last_valid}"


@pytest.mark.unit
def test_rsi_returns_series_same_length(flat_df: pd.DataFrame) -> None:
    result = rsi(flat_df, 14)
    assert len(result) == len(flat_df)


@pytest.mark.unit
def test_rsi_known_value_reference(linear_df: pd.DataFrame) -> None:
    """RSI on linear_df must match a reference computed via the Wilder formula
    directly.
    """
    delta = linear_df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    reference = 100.0 - (100.0 / (1.0 + rs))
    reference = reference.where(avg_loss != 0.0, other=100.0)
    reference.iloc[0] = float("nan")

    result = rsi(linear_df, 14)
    valid_idx = ~reference.isna()
    np.testing.assert_allclose(
        result[valid_idx].values,
        reference[valid_idx].values,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_atr_positive_volatile(volatile_df: pd.DataFrame) -> None:
    """ATR values must be > 0 on a series with non-zero true range."""
    result = atr(volatile_df, 14)
    valid = result.dropna()
    assert (valid > 0.0).all()


@pytest.mark.unit
def test_atr_returns_series_same_length(volatile_df: pd.DataFrame) -> None:
    result = atr(volatile_df, 14)
    assert len(result) == len(volatile_df)


@pytest.mark.unit
def test_atr_known_value_reference(volatile_df: pd.DataFrame) -> None:
    """ATR must match a reference computed via the TR formula directly."""
    prev_close = volatile_df["close"].shift(1)
    tr = pd.concat(
        [
            volatile_df["high"] - volatile_df["low"],
            (volatile_df["high"] - prev_close).abs(),
            (volatile_df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    reference = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()

    result = atr(volatile_df, 14)
    valid_idx = ~reference.isna()
    np.testing.assert_allclose(
        result[valid_idx].values, reference[valid_idx].values, rtol=1e-9
    )


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_macd_columns(linear_df: pd.DataFrame) -> None:
    """MACD result must have exactly columns: macd, signal, histogram."""
    result = macd(linear_df)
    assert set(result.columns) == {"macd", "signal", "histogram"}


@pytest.mark.unit
def test_macd_histogram_consistency(linear_df: pd.DataFrame) -> None:
    """histogram == macd - signal for all non-NaN rows."""
    result = macd(linear_df)
    valid = result.dropna()
    np.testing.assert_allclose(
        valid["histogram"].values,
        (valid["macd"] - valid["signal"]).values,
        atol=1e-10,
    )


@pytest.mark.unit
def test_macd_returns_same_length(linear_df: pd.DataFrame) -> None:
    result = macd(linear_df)
    assert len(result) == len(linear_df)


@pytest.mark.unit
def test_macd_known_value_reference(linear_df: pd.DataFrame) -> None:
    """MACD must match reference computed via ewm directly."""
    ema_fast = linear_df["close"].ewm(span=12, min_periods=12, adjust=False).mean()
    ema_slow = linear_df["close"].ewm(span=26, min_periods=26, adjust=False).mean()
    ref_macd = ema_fast - ema_slow
    ref_signal = ref_macd.ewm(span=9, min_periods=9, adjust=False).mean()
    ref_histogram = ref_macd - ref_signal

    result = macd(linear_df)
    valid_idx = ~ref_histogram.isna()
    np.testing.assert_allclose(
        result["macd"][valid_idx].values, ref_macd[valid_idx].values, rtol=1e-9
    )
    np.testing.assert_allclose(
        result["signal"][valid_idx].values, ref_signal[valid_idx].values, rtol=1e-9
    )
    np.testing.assert_allclose(
        result["histogram"][valid_idx].values,
        ref_histogram[valid_idx].values,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# build_candidate_snapshot tests
# ---------------------------------------------------------------------------


def _make_ohlcv_row(
    date: datetime.date,
    close: float,
    source: str = "yfinance",
) -> MagicMock:
    """Create a mock OhlcvDaily row."""
    row = MagicMock()
    row.open = Decimal(str(close - 0.10))
    row.high = Decimal(str(close + 0.30))
    row.low = Decimal(str(close - 0.30))
    row.close = Decimal(str(close))
    row.volume = 1_000_000
    row.date = date
    row.source = source
    return row


def _make_bars(n: int, base_close: float = 50.0) -> list[MagicMock]:
    """Create n mock OhlcvDaily rows with a linearly increasing close."""
    today = datetime.date(2024, 1, 1)
    return [
        _make_ohlcv_row(
            today + datetime.timedelta(days=i),
            base_close + i * 0.5,
        )
        for i in range(n)
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_symbol_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SymbolNotCachedError is raised when get_by_ticker returns None."""
    session = AsyncMock()

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=None)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )

    with pytest.raises(SymbolNotCachedError) as exc_info:
        await build_candidate_snapshot(
            "AAPL",
            datetime.date(2024, 1, 31),
            session,
        )

    assert "AAPL" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_insufficient_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """< 2 bars → Candidate returned with all indicators None."""
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 42

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=_make_bars(1))

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    candidate = await build_candidate_snapshot(
        "AAPL",
        datetime.date(2024, 1, 31),
        session,
    )

    assert candidate.bars_available == 1
    assert candidate.rsi_14 is None
    assert candidate.ema_20 is None
    assert candidate.ema_50 is None
    assert candidate.atr_14 is None
    assert candidate.macd_line is None
    assert candidate.macd_signal is None
    assert candidate.macd_histogram is None
    assert candidate.ema_9 is None
    assert candidate.ema_200 is None
    assert candidate.ema_slope_20 is None
    assert candidate.ema_alignment is None
    assert candidate.adx_14 is None
    assert candidate.roc_10 is None
    assert candidate.bb_upper_20 is None
    assert candidate.atr_percentile_252 is None
    assert candidate.rvol_20 is None
    assert candidate.market_structure is None
    assert candidate.week_52_high is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """60-bar fixture: Candidate fields match indicator function outputs."""
    session = AsyncMock()
    bars = _make_bars(60, base_close=100.0)

    symbol_row = MagicMock()
    symbol_row.id = 7

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=bars)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    as_of = datetime.date(2024, 3, 1)
    candidate = await build_candidate_snapshot(
        "AAPL",
        as_of,
        session,
        source="yfinance",
    )

    assert candidate.symbol == "AAPL"
    assert candidate.as_of_date == as_of
    assert candidate.source == "yfinance"
    assert candidate.bars_available == 60

    # Reconstruct the DataFrame identically to snapshot.py.
    df = pd.DataFrame(
        [
            {
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    )
    last_close = float(bars[-1].close)

    # EMA-20
    expected_ema20 = float(ema(df, 20).iloc[-1])
    assert candidate.ema_20 is not None
    assert abs(float(candidate.ema_20) - expected_ema20) < 1e-4

    # EMA-50
    expected_ema50 = float(ema(df, 50).iloc[-1])
    assert candidate.ema_50 is not None
    assert abs(float(candidate.ema_50) - expected_ema50) < 1e-4

    # RSI-14
    expected_rsi14 = rsi(df, 14).dropna().iloc[-1]
    assert candidate.rsi_14 is not None
    assert abs(float(candidate.rsi_14) - expected_rsi14) < 1e-4

    # ATR-14
    expected_atr14 = atr(df, 14).dropna().iloc[-1]
    assert candidate.atr_14 is not None
    assert abs(float(candidate.atr_14) - expected_atr14) < 1e-4

    # MACD
    macd_df = macd(df)
    assert candidate.macd_line is not None
    assert abs(float(candidate.macd_line) - float(macd_df["macd"].iloc[-1])) < 1e-4
    assert candidate.macd_signal is not None
    assert abs(float(candidate.macd_signal) - float(macd_df["signal"].iloc[-1])) < 1e-4
    assert candidate.macd_histogram is not None
    assert (
        abs(float(candidate.macd_histogram) - float(macd_df["histogram"].iloc[-1]))
        < 1e-4
    )

    # Close price
    assert abs(float(candidate.close) - last_close) < 1e-4

    # fetch_range was called with source="yfinance" explicitly
    ohlcv_repo_mock.fetch_range.assert_called_once()
    call_kwargs = ohlcv_repo_mock.fetch_range.call_args
    assert call_kwargs.kwargs.get("source") == "yfinance"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_exchange_autoderived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exchange is auto-derived from the ticker suffix when not passed explicitly."""
    session = AsyncMock()
    bars = _make_bars(60)

    symbol_row = MagicMock()
    symbol_row.id = 3

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=bars)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    # ASML.AS should auto-derive to EURONEXT_AMSTERDAM
    await build_candidate_snapshot("ASML.AS", datetime.date(2024, 3, 1), session)

    symbol_repo_mock.get_by_ticker.assert_called_once_with(
        "ASML.AS", "EURONEXT_AMSTERDAM"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_partial_warmup_30_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """30 bars fixture: ema_50 must be None (< 50 bars), while ema_20 is present."""
    session = AsyncMock()
    bars = _make_bars(30, base_close=100.0)

    symbol_row = MagicMock()
    symbol_row.id = 10

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=bars)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    as_of = datetime.date(2024, 2, 1)
    candidate = await build_candidate_snapshot(
        "AAPL",
        as_of,
        session,
        source="yfinance",
    )

    assert candidate.symbol == "AAPL"
    assert candidate.bars_available == 30
    assert candidate.ema_20 is not None
    assert candidate.ema_50 is None  # < 50 bars: must be None!
    assert candidate.rsi_14 is not None
    assert candidate.atr_14 is not None
    assert candidate.macd_line is not None  # 30 >= 26
    assert candidate.macd_signal is None  # 30 < 34 bars for signal warmup
    assert candidate.macd_histogram is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_m1_indicators_full_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """300-bar fixture: all M1 indicator fields are populated and valid."""
    session = AsyncMock()
    bars = _make_bars(300, base_close=100.0)

    symbol_row = MagicMock()
    symbol_row.id = 99

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=bars)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    candidate = await build_candidate_snapshot(
        "NVDA",
        datetime.date(2024, 3, 1),
        session,
        source="yfinance",
    )

    assert candidate.symbol == "NVDA"
    assert candidate.bars_available == 300

    # Trend extensions
    assert candidate.ema_9 is not None
    assert candidate.ema_20 is not None
    assert candidate.ema_50 is not None
    assert candidate.ema_200 is not None
    assert candidate.ema_slope_20 is not None
    assert candidate.ema_alignment == "bullish"

    # Momentum
    assert candidate.adx_14 is not None
    assert candidate.plus_di_14 is not None
    assert candidate.minus_di_14 is not None
    assert candidate.roc_10 is not None

    # Volatility
    assert candidate.bb_upper_20 is not None
    assert candidate.bb_middle_20 is not None
    assert candidate.bb_lower_20 is not None
    assert candidate.bb_bandwidth_20 is not None
    assert candidate.atr_percentile_252 is not None
    assert candidate.atr_expansion_ratio is not None

    # Volume
    assert candidate.rvol_20 is not None
    assert candidate.dollar_volume is not None
    assert candidate.mfi_14 is not None
    assert candidate.obv is not None

    # Market structure
    assert candidate.week_52_high is not None
    assert candidate.week_52_low is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_candidate_snapshot_m1_indicators_underwarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """30-bar fixture: short-window indicators populate,
    long-window indicators are None.
    """
    session = AsyncMock()
    bars = _make_bars(30, base_close=100.0)

    symbol_row = MagicMock()
    symbol_row.id = 100

    symbol_repo_mock = AsyncMock()
    symbol_repo_mock.get_by_ticker = AsyncMock(return_value=symbol_row)

    ohlcv_repo_mock = AsyncMock()
    ohlcv_repo_mock.fetch_range = AsyncMock(return_value=bars)

    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.SymbolRepository",
        lambda _session: symbol_repo_mock,
    )
    monkeypatch.setattr(
        "mcp_finance.indicators.snapshot.OhlcvRepository",
        lambda _session: ohlcv_repo_mock,
    )

    candidate = await build_candidate_snapshot(
        "NVDA",
        datetime.date(2024, 3, 1),
        session,
        source="yfinance",
    )

    assert candidate.bars_available == 30

    # 30 bars is enough for EMA-9, EMA-20
    assert candidate.ema_9 is not None
    assert candidate.ema_20 is not None

    # 30 bars is not enough for EMA-50, EMA-200
    assert candidate.ema_50 is None
    assert candidate.ema_200 is None
    assert candidate.ema_alignment == "mixed"

    # 30 bars is enough for ROC-10
    assert candidate.roc_10 is not None

    # 30 bars is not enough for 252-day ATR percentile or 50-day expansion ratio
    assert candidate.atr_expansion_ratio is None
    assert candidate.atr_percentile_252 is None


# ---------------------------------------------------------------------------
# M1.2 — Trend-extension tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ema_slope_sign_and_magnitude_on_linear_series(
    linear_df: pd.DataFrame,
) -> None:
    """ema_slope on a strictly linear series: last value must be positive
    and match the hand-computed total pct change over lookback_bars=5.

    linear_df: close[i] = 100 + i*0.5  (i = 0..59)
    EMA-20 is strictly increasing, so slope over any 5-bar window is > 0.
    We verify the exact value against an independently computed reference.
    """
    ema_series = ema(linear_df, 20)
    slope = ema_slope(ema_series, lookback_bars=5)

    # First lookback_bars values (indices 0..4) must be NaN; first valid EMA
    # is at index 19 (period-1), so slope first valid at index 19+5=24.
    assert slope.iloc[:24].isna().all(), "Expected NaN for under-warmed slope"

    # Reference: (EMA[t] - EMA[t-5]) / EMA[t-5] for the last bar.
    last_ema = ema_series.dropna()
    ref = (last_ema.iloc[-1] - last_ema.iloc[-6]) / last_ema.iloc[-6]
    np.testing.assert_allclose(slope.iloc[-1], ref, rtol=1e-9)

    # The slope must be strictly positive on a monotonically rising EMA.
    assert slope.dropna().gt(0).all(), "Expected all positive slopes on rising EMA"


@pytest.mark.unit
def test_ema_slope_lookback_less_than_1_raises() -> None:
    """lookback_bars < 1 must raise ValueError immediately."""
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="lookback_bars must be at least 1"):
        ema_slope(s, lookback_bars=0)


@pytest.mark.unit
def test_ema_slope_negative_reference_returns_nan() -> None:
    """If prev value is <= 0, slope must be NaN (no division by zero/negative)."""
    s = pd.Series([-1.0, -0.5, -0.1, 0.0, 1.0, 2.0])
    result = ema_slope(s, lookback_bars=3)
    # indices 0..2 are NaN (warmup); index 3: ref is s[0]=-1 -> NaN; same for 4
    assert result.iloc[:5].isna().all()


@pytest.mark.unit
def test_ema_alignment_bullish() -> None:
    """ema9 > ema20 > ema50 > ema200 -> 'bullish'."""
    assert ema_alignment(90.0, 80.0, 70.0, 60.0) == "bullish"


@pytest.mark.unit
def test_ema_alignment_bearish() -> None:
    """ema9 < ema20 < ema50 < ema200 -> 'bearish'."""
    assert ema_alignment(60.0, 70.0, 80.0, 90.0) == "bearish"


@pytest.mark.unit
def test_ema_alignment_mixed_partial_order() -> None:
    """Any ordering that is neither strict bull nor strict bear -> 'mixed'."""
    assert ema_alignment(90.0, 80.0, 85.0, 60.0) == "mixed"


@pytest.mark.unit
def test_ema_alignment_equal_values_is_mixed() -> None:
    """Equal EMAs (not strictly ordered) must return 'mixed', not 'bullish'."""
    assert ema_alignment(80.0, 80.0, 70.0, 60.0) == "mixed"


@pytest.mark.unit
def test_ema_alignment_any_none_is_mixed() -> None:
    """Any None argument must immediately return 'mixed'."""
    assert ema_alignment(None, 80.0, 70.0, 60.0) == "mixed"
    assert ema_alignment(90.0, None, 70.0, 60.0) == "mixed"
    assert ema_alignment(90.0, 80.0, None, 60.0) == "mixed"
    assert ema_alignment(90.0, 80.0, 70.0, None) == "mixed"


@pytest.mark.unit
def test_price_vs_ema_distance_pct_formula() -> None:
    """Verify (price - ema) / ema formula with exact reference values."""
    price = Decimal("110")
    ema_val = Decimal("100")
    pct_dist, atr_dist = price_vs_ema_distance(price, ema_val)
    assert pct_dist == Decimal("0.1")  # (110 - 100) / 100 = 0.10
    assert atr_dist is None  # no ATR passed


@pytest.mark.unit
def test_price_vs_ema_distance_atr_normalized() -> None:
    """ATR-normalized distance = (price - ema) / atr."""
    price = Decimal("105")
    ema_val = Decimal("100")
    atr_val = Decimal("2.5")
    pct_dist, atr_dist = price_vs_ema_distance(price, ema_val, atr_val)
    assert pct_dist == Decimal("0.05")  # (105 - 100) / 100
    assert atr_dist == Decimal("2")  # (105 - 100) / 2.5


@pytest.mark.unit
def test_price_vs_ema_distance_negative_ema_raises() -> None:
    """ema_val <= 0 must raise ValueError."""
    with pytest.raises(ValueError, match="ema_val must be strictly positive"):
        price_vs_ema_distance(Decimal("100"), Decimal("0"))
    with pytest.raises(ValueError, match="ema_val must be strictly positive"):
        price_vs_ema_distance(Decimal("100"), Decimal("-5"))


@pytest.mark.unit
def test_price_vs_ema_distance_zero_atr_skips_atr_dist() -> None:
    """atr_val of zero must not cause division by zero — atr_dist = None."""
    _, atr_dist = price_vs_ema_distance(
        Decimal("100"), Decimal("90"), atr_val=Decimal("0")
    )
    assert atr_dist is None


# ---------------------------------------------------------------------------
# M1.3 — Momentum-extension tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_adx_warmup_boundary_returns_nan_until_2period_minus_1() -> None:
    """ADX on a series of length exactly 2*period - 2 must be all NaN.

    This is the mirror of test_ema_min_periods_returns_nan_on_insufficient_bars
    from Phase 5. ADX needs 2*period - 1 bars: period bars to warm DI, then
    another period bars to smooth DX into ADX. With only 2*period - 2 bars,
    the final ADX value must still be NaN.
    """
    period = 14
    n_under = 2 * period - 2  # exactly one bar short
    close = np.linspace(100.0, 115.0, n_under)
    high = close + 1.0
    low = close - 1.0
    df_short = pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "open": close - 0.5,
            "volume": np.full(n_under, 1_000_000),
        }
    )
    result = adx(df_short, period=period)
    # ADX column must be entirely NaN for under-warmed series
    assert result["adx"].isna().all(), (
        f"Expected all-NaN ADX on {n_under}-bar series (2*period-2), "
        f"got last={result['adx'].iloc[-1]}"
    )


@pytest.mark.unit
def test_adx_strong_trend_returns_high_value(linear_df: pd.DataFrame) -> None:
    """ADX on a strongly trending series must produce a high positive value.

    A 60-bar linearly rising series is a strong trend; ADX (once warmed up)
    should converge well above 25 (conventional strong-trend threshold).
    Plus-DI must exceed Minus-DI in an up-trend.
    """
    result = adx(linear_df, period=14)
    last_adx = result["adx"].dropna().iloc[-1]
    last_plus_di = result["plus_di"].dropna().iloc[-1]
    last_minus_di = result["minus_di"].dropna().iloc[-1]

    assert last_adx > 25.0, f"Expected ADX > 25 on strong up-trend, got {last_adx:.2f}"
    assert last_plus_di > last_minus_di, (
        f"Expected +DI > -DI in up-trend; +DI={last_plus_di:.2f}, "
        f"-DI={last_minus_di:.2f}"
    )


@pytest.mark.unit
def test_adx_all_values_in_range(linear_df: pd.DataFrame) -> None:
    """All non-NaN ADX, +DI, and -DI values must be in [0, 100]."""
    result = adx(linear_df, period=14)
    for col in ["adx", "plus_di", "minus_di"]:
        valid = result[col].dropna()
        assert (valid >= 0.0).all() and (valid <= 100.0).all(), (
            f"{col} out of [0, 100] range"
        )


@pytest.mark.unit
def test_roc_known_value_on_linear_series(linear_df: pd.DataFrame) -> None:
    """ROC-10 on close = 100 + i*0.5.

    Reference = (close[i] - close[i-10]) / close[i-10].
    """
    result = roc(linear_df, period=10)
    # First 10 values must be NaN.
    assert result.iloc[:10].isna().all()
    # Last value: close[59]=129.5, close[49]=124.5
    ref = (129.5 - 124.5) / 124.5
    np.testing.assert_allclose(result.iloc[-1], ref, rtol=1e-9)


@pytest.mark.unit
def test_roc_period_less_than_1_raises() -> None:
    """period < 1 must raise ValueError."""
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="period must be at least 1"):
        roc(df, period=0)


# ---------------------------------------------------------------------------
# M1.4 — Volatility-extension tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bollinger_bands_zero_std_on_constant_series() -> None:
    """Bollinger Bands on a constant series: upper = lower = middle = constant."""
    n = 25
    df = pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "open": np.full(n, 100.0),
            "volume": np.full(n, 1_000_000),
        }
    )
    bands = bollinger_bands(df, period=20, num_std=2.0)
    valid = bands.dropna()
    assert len(valid) == 6  # rows 19..24
    np.testing.assert_allclose(valid["middle"].values, 100.0, rtol=1e-9)
    np.testing.assert_allclose(valid["upper"].values, 100.0, rtol=1e-9)
    np.testing.assert_allclose(valid["lower"].values, 100.0, rtol=1e-9)


@pytest.mark.unit
def test_bollinger_bands_spread_on_volatile(volatile_df: pd.DataFrame) -> None:
    """Upper band must be > middle > lower for a series with genuine variance."""
    bands = bollinger_bands(volatile_df, period=20, num_std=2.0)
    valid = bands.dropna()
    assert (valid["upper"] > valid["middle"]).all()
    assert (valid["middle"] > valid["lower"]).all()


@pytest.mark.unit
def test_bollinger_bandwidth_formula(volatile_df: pd.DataFrame) -> None:
    """Bandwidth = (upper - lower) / middle; verify against hand-computed reference."""
    bands = bollinger_bands(volatile_df, period=20, num_std=2.0)
    bw = bollinger_bandwidth(bands)
    ref = (bands["upper"] - bands["lower"]) / bands["middle"]
    valid_bw = bw.dropna()
    valid_ref = ref.dropna()
    np.testing.assert_allclose(valid_bw.values, valid_ref.values, rtol=1e-9)


@pytest.mark.unit
def test_rolling_percentile_known_rank() -> None:
    """rolling_percentile on a known sequence: at index 3 rank is 0.0, index 4 is 1/3.

    Series:  [10, 8, 6, 4, 5, 7, 9] with window=4.
    At index 3: window=[10,8,6,4]. current=4, past=[10,8,6]. rank = 0/3 = 0.0.
    At index 4: window=[8,6,4,5]. current=5, past=[8,6,4]. rank = 1/3.
    """
    s = pd.Series([10.0, 8.0, 6.0, 4.0, 5.0, 7.0, 9.0])
    result = rolling_percentile(s, window=4)
    # First 3 values NaN (window-1).
    assert result.iloc[:3].isna().all()
    np.testing.assert_allclose(result.iloc[3], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.iloc[4], 1.0 / 3.0, rtol=1e-9)


@pytest.mark.unit
def test_rolling_percentile_window_less_than_2_raises() -> None:
    """window < 2 must raise ValueError."""
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="window must be at least 2"):
        rolling_percentile(s, window=1)


@pytest.mark.unit
def test_atr_expansion_ratio_unity_on_constant_atr() -> None:
    """If ATR is constant, the expansion ratio must be 1.0 for all valid bars."""
    atr_const = pd.Series(np.full(60, 2.0))
    ratio = atr_expansion_ratio(atr_const, avg_period=10)
    valid = ratio.dropna()
    np.testing.assert_allclose(valid.values, 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# M1.5 — Volume-extension tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_obv_hand_trace() -> None:
    """OBV on a hand-constructed up/down/flat 4-bar sequence.

    Bar 0: base (no prev_close) -> direction=0, OBV=0
    Bar 1: close UP   -> OBV += vol_1  -> OBV = 1000
    Bar 2: close DOWN -> OBV -= vol_2  -> OBV = 1000 - 500 = 500
    Bar 3: close FLAT -> OBV unchanged -> OBV = 500
    """
    df = pd.DataFrame(
        {
            "open": [10.0, 10.0, 11.0, 10.5],
            "high": [10.5, 11.5, 11.5, 11.0],
            "low": [9.5, 10.5, 10.0, 10.0],
            "close": [10.0, 11.0, 10.5, 10.5],
            "volume": [500, 1000, 500, 200],
        }
    )
    result = obv(df)
    expected = [0, 1000, 500, 500]
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.unit
def test_dollar_volume_formula(flat_df: pd.DataFrame) -> None:
    """dollar_volume = close * volume; verify on the flat fixture."""
    dv = dollar_volume(flat_df)
    expected = flat_df["close"] * flat_df["volume"]
    np.testing.assert_allclose(dv.values, expected.values, rtol=1e-9)
    # Spot-check: 50.0 * 1_000_000 = 50_000_000
    assert dv.iloc[0] == pytest.approx(50_000_000.0)


@pytest.mark.unit
def test_relative_volume_unity_when_volume_equals_avg(flat_df: pd.DataFrame) -> None:
    """RVOL on a constant-volume series = 1.0 for all valid bars."""
    rvol = relative_volume(flat_df, period=20)
    valid = rvol.dropna()
    np.testing.assert_allclose(valid.values, 1.0, rtol=1e-9)


@pytest.mark.unit
def test_relative_volume_warmup_returns_nan(flat_df: pd.DataFrame) -> None:
    """RVOL first (period - 1) values must be NaN."""
    rvol = relative_volume(flat_df, period=20)
    assert rvol.iloc[:19].isna().all()


@pytest.mark.unit
def test_mfi_reference_formula_cross_check() -> None:
    """MFI on a hand-constructed 20-bar series against the reference formula.

    Replicates the same style as test_rsi_known_value_reference (Phase 5):
    compute an independent reference via the same formula steps and assert
    the function matches to floating-point precision.
    """
    rng = np.random.default_rng(42)
    n = 20
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    result = mfi(df, period=14)

    # Independent reference computation
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    prev_tp = np.concatenate([[np.nan], tp[:-1]])
    pos_mf = np.where(tp > prev_tp, raw_mf, 0.0)
    neg_mf = np.where(tp < prev_tp, raw_mf, 0.0)

    pos_s = pd.Series(pos_mf).rolling(window=14, min_periods=14).sum()
    neg_s = pd.Series(neg_mf).rolling(window=14, min_periods=14).sum()
    ratio = pos_s / neg_s
    ref = 100.0 - (100.0 / (1.0 + ratio))
    ref = ref.where(neg_s != 0.0, other=100.0)

    valid_result = result.dropna()
    valid_ref = ref.dropna()
    assert len(valid_result) == len(valid_ref)
    np.testing.assert_allclose(valid_result.values, valid_ref.values, rtol=1e-9)
    # MFI values must be in [0, 100]
    assert (valid_result >= 0.0).all() and (valid_result <= 100.0).all()

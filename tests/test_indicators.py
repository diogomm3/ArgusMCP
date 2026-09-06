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

from mcp_finance.indicators.functions import atr, ema, macd, rsi
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
    """EMA on a constant series should equal the constant (after first value)."""
    result = ema(flat_df, 20)
    # ewm(adjust=False) initialises to the first value, so the entire series
    # should be 50.0 (flat series, first obs = 50.0, all subsequent = 50.0).
    assert result.dropna().shape[0] == 60
    np.testing.assert_allclose(result.values, 50.0, rtol=1e-6)


@pytest.mark.unit
def test_ema_linear_known_value(linear_df: pd.DataFrame) -> None:
    """EMA-20 last value on linear_df must match a reference computed independently.

    Reference: pd.Series.ewm(span=20, adjust=False).mean() on the same data.
    This is a cross-implementation consistency check — we verify our ema()
    helper produces the same result as the pandas expression it wraps.
    """
    reference = linear_df["close"].ewm(span=20, adjust=False).mean()
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
    # Open is close - 0.10, so EMA(open) should be EMA(close) - 0.10.
    np.testing.assert_allclose(
        result_open.values, result_close.values - 0.10, rtol=1e-9
    )


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
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
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
    reference = tr.ewm(alpha=1.0 / 14, adjust=False).mean()

    result = atr(volatile_df, 14)
    np.testing.assert_allclose(result.values, reference.values, rtol=1e-9)


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
    ema_fast = linear_df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = linear_df["close"].ewm(span=26, adjust=False).mean()
    ref_macd = ema_fast - ema_slow
    ref_signal = ref_macd.ewm(span=9, adjust=False).mean()
    ref_histogram = ref_macd - ref_signal

    result = macd(linear_df)
    np.testing.assert_allclose(result["macd"].values, ref_macd.values, rtol=1e-9)
    np.testing.assert_allclose(result["signal"].values, ref_signal.values, rtol=1e-9)
    np.testing.assert_allclose(
        result["histogram"].values, ref_histogram.values, rtol=1e-9
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

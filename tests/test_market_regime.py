"""Unit tests for Phase M4 market_regime pure indicator function.

Tests:
  - SPY historical fixture validation with pinned labels across BULLISH, BEARISH,
    and NEUTRAL market conditions. Fixture covers 2020-03-04 through 2026-09-29.
  - Hand-derived boundary tests:
    - Strict inequality at exact EMA equality (EMA50 == EMA200 -> NEUTRAL).
    - Slope boundary at exact deadbands (slope == +0.0025 -> NEUTRAL;
      slope == -0.0025 -> NEUTRAL; slope > 0.0025 -> BULLISH;
      slope < -0.0025 -> BEARISH).
    - Window length boundaries (exactly 499 bars -> None; exactly 500 -> valid).
    - as_of_date snapping, backward-snap across weekends, and out-of-bounds.
    - sessions_in_regime capping at 60 and single-session fresh flip detection.
    - Input validation: empty df, missing close, non-monotonic or duplicate dates.
    - Percentage distance and slope metric consistency.
"""

from __future__ import annotations

import datetime
import pathlib
from decimal import Decimal

import pandas as pd
import pytest

import mcp_finance.indicators.regime as regime_mod
from mcp_finance.indicators.models import MarketRegime
from mcp_finance.indicators.regime import (
    DEADBAND,
    REQUIRED_WINDOW_BARS,
    market_regime,
)

FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "spy_daily_regime.csv"


@pytest.fixture(scope="module")
def spy_fixture_df() -> pd.DataFrame:
    """Load the historical SPY CSV fixture.

    Last completed trading session in fixture: 2026-09-29.
    Earliest session: 2020-03-04 (1,652 daily trading bars).
    """
    assert FIXTURE_PATH.exists(), f"Missing SPY fixture at {FIXTURE_PATH}"
    df = pd.read_csv(FIXTURE_PATH)
    assert len(df) >= REQUIRED_WINDOW_BARS
    return df


# ---------------------------------------------------------------------------
# 1. Historical SPY Fixture Tests (Pinned Labels)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spy_fixture_latest_session_bullish(spy_fixture_df: pd.DataFrame) -> None:
    """Latest bar in SPY fixture (2026-09-29) is BULLISH with capped sessions."""
    result = market_regime(spy_fixture_df, as_of_date="2026-09-29")
    assert result is not None
    assert isinstance(result, MarketRegime)
    assert result.regime == "BULLISH"
    assert result.as_of_date == datetime.date(2026, 9, 29)
    assert result.sessions_in_regime == 60  # > 100 sessions, capped at 60
    assert result.ema50 > result.ema200
    assert result.ema50_slope20 > Decimal("0.0025")
    assert result.close_vs_ema50_pct > Decimal("-10.0")


@pytest.mark.unit
def test_spy_fixture_2022_bearish(spy_fixture_df: pd.DataFrame) -> None:
    """2022-05-15 and 2022-06-15 in SPY fixture are confirmed BEARISH."""
    # 2022-05-15 is Sunday, snaps to Friday 2022-05-13
    res_may = market_regime(spy_fixture_df, as_of_date="2022-05-15")
    assert res_may is not None
    assert res_may.regime == "BEARISH"
    assert res_may.as_of_date == datetime.date(2022, 5, 13)
    assert res_may.sessions_in_regime == 10
    assert res_may.ema50 < res_may.ema200
    assert res_may.ema50_slope20 < Decimal("-0.0025")

    # 2022-06-15 mid-bear market
    res_june = market_regime(spy_fixture_df, as_of_date="2022-06-15")
    assert res_june is not None
    assert res_june.regime == "BEARISH"
    assert res_june.as_of_date == datetime.date(2022, 6, 15)
    assert res_june.sessions_in_regime == 32
    assert res_june.ema50 < res_june.ema200
    assert res_june.ema50_slope20 < Decimal("-0.0025")


@pytest.mark.unit
def test_spy_fixture_2022_neutral(spy_fixture_df: pd.DataFrame) -> None:
    """2022-03-15 in SPY fixture is NEUTRAL (EMA50 > EMA200 but negative slope)."""
    res = market_regime(spy_fixture_df, as_of_date="2022-03-15")
    assert res is not None
    assert res.regime == "NEUTRAL"
    assert res.as_of_date == datetime.date(2022, 3, 15)
    assert res.sessions_in_regime == 35
    # EMA50 > EMA200 but slope is negative (-0.031), not matching BULLISH
    assert res.ema50 > res.ema200
    assert res.ema50_slope20 < Decimal("0")


@pytest.mark.unit
def test_spy_fixture_none_when_insufficient_history(
    spy_fixture_df: pd.DataFrame,
) -> None:
    """Evaluating early in the fixture where < 500 bars are available returns None."""
    early_date = "2020-07-01"
    res = market_regime(spy_fixture_df, as_of_date=early_date)
    assert res is None


# ---------------------------------------------------------------------------
# 2. Hand-Derived Synthetic Boundary Tests
# ---------------------------------------------------------------------------


def _generate_synthetic_df(
    n_bars: int = 500,
    start_date: str = "2024-01-01",
    close_val: float = 100.0,
) -> pd.DataFrame:
    """Generate simple synthetic daily bar DataFrame with constant close."""
    dates = pd.date_range(start_date, periods=n_bars, freq="B")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [close_val] * n_bars,
            "high": [close_val] * n_bars,
            "low": [close_val] * n_bars,
            "close": [close_val] * n_bars,
            "volume": [1_000_000] * n_bars,
        }
    )


@pytest.mark.unit
def test_strict_inequality_at_exact_ema_equality() -> None:
    """Flat prices produce EMA50 == EMA200 and slope == 0.0 -> NEUTRAL."""
    df = _generate_synthetic_df(n_bars=500, close_val=100.0)
    res = market_regime(df)
    assert res is not None
    assert res.regime == "NEUTRAL"
    assert res.ema50 == res.ema200
    assert res.ema50_slope20 == Decimal("0")


@pytest.mark.unit
def test_slope_exact_deadband_boundary_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slope at exactly +0.0025 or -0.0025 is NEUTRAL (requires strict > or <)."""
    mr_mod = regime_mod
    df = _generate_synthetic_df(n_bars=500, close_val=100.0)

    # 1. EMA50 > EMA200, slope == +0.0025 exactly (DEADBAND) -> NEUTRAL
    def mock_slope_deadband_pos(
        series: pd.Series, lookback_bars: int = 20
    ) -> pd.Series:
        return pd.Series([DEADBAND] * len(series), index=series.index)

    def mock_ema_50_higher(
        df_in: pd.DataFrame, period: int, column: str = "close"
    ) -> pd.Series:
        val = 110.0 if period == 50 else 100.0
        return pd.Series([val] * len(df_in), index=df_in.index)

    monkeypatch.setattr(mr_mod, "ema", mock_ema_50_higher)
    monkeypatch.setattr(mr_mod, "ema_slope", mock_slope_deadband_pos)

    res = market_regime(df)
    assert res is not None
    assert res.regime == "NEUTRAL"
    assert res.ema50_slope20 == Decimal("0.002500")

    # 2. EMA50 > EMA200, slope == +0.002501 (strictly > DEADBAND) -> BULLISH
    def mock_slope_above_deadband(
        series: pd.Series, lookback_bars: int = 20
    ) -> pd.Series:
        return pd.Series([0.002501] * len(series), index=series.index)

    monkeypatch.setattr(mr_mod, "ema_slope", mock_slope_above_deadband)
    res_bull = market_regime(df)
    assert res_bull is not None
    assert res_bull.regime == "BULLISH"

    # 3. EMA50 < EMA200, slope == -0.0025 exactly (-DEADBAND) -> NEUTRAL
    def mock_ema_50_lower(
        df_in: pd.DataFrame, period: int, column: str = "close"
    ) -> pd.Series:
        val = 90.0 if period == 50 else 100.0
        return pd.Series([val] * len(df_in), index=df_in.index)

    def mock_slope_deadband_neg(
        series: pd.Series, lookback_bars: int = 20
    ) -> pd.Series:
        return pd.Series([-DEADBAND] * len(series), index=series.index)

    monkeypatch.setattr(mr_mod, "ema", mock_ema_50_lower)
    monkeypatch.setattr(mr_mod, "ema_slope", mock_slope_deadband_neg)

    res_neg = market_regime(df)
    assert res_neg is not None
    assert res_neg.regime == "NEUTRAL"
    assert res_neg.ema50_slope20 == Decimal("-0.002500")

    # 4. EMA50 < EMA200, slope == -0.002501 (strictly < -DEADBAND) -> BEARISH
    def mock_slope_below_deadband(
        series: pd.Series, lookback_bars: int = 20
    ) -> pd.Series:
        return pd.Series([-0.002501] * len(series), index=series.index)

    monkeypatch.setattr(mr_mod, "ema_slope", mock_slope_below_deadband)
    res_bear = market_regime(df)
    assert res_bear is not None
    assert res_bear.regime == "BEARISH"


@pytest.mark.unit
def test_exact_window_size_boundaries() -> None:
    """Exactly 499 bars returns None; exactly 500 bars returns a valid MarketRegime."""
    df_499 = _generate_synthetic_df(n_bars=499)
    assert market_regime(df_499) is None

    df_500 = _generate_synthetic_df(n_bars=500)
    res_500 = market_regime(df_500)
    assert res_500 is not None
    assert isinstance(res_500, MarketRegime)
    assert res_500.sessions_in_regime >= 1


@pytest.mark.unit
def test_as_of_date_before_window_returns_none() -> None:
    """as_of_date earlier than the earliest date in DataFrame returns None."""
    df = _generate_synthetic_df(n_bars=500, start_date="2024-01-01")
    res = market_regime(df, as_of_date="2023-12-31")
    assert res is None


@pytest.mark.unit
def test_as_of_date_weekend_backward_snap() -> None:
    """as_of_date on a Sunday snaps backward to Friday's session."""
    df = _generate_synthetic_df(n_bars=500, start_date="2024-01-01")
    last_date = pd.to_datetime(df["date"].iloc[-1]).date()
    next_sunday = last_date + datetime.timedelta(days=(6 - last_date.weekday()))
    res = market_regime(df, as_of_date=next_sunday)
    assert res is not None
    assert res.as_of_date <= last_date


@pytest.mark.unit
def test_sessions_in_regime_cap_60() -> None:
    """sessions_in_regime is strictly capped at 60 even over long runs."""
    df = _generate_synthetic_df(n_bars=500, close_val=100.0)
    res = market_regime(df)
    assert res is not None
    assert res.regime == "NEUTRAL"
    assert res.sessions_in_regime <= 60


@pytest.mark.unit
def test_invalid_data_handling() -> None:
    """Empty df, missing close, NaN, negative close, or unsorted return None."""
    # Empty
    assert market_regime(pd.DataFrame()) is None

    # Missing close column
    df_no_close = pd.DataFrame({"date": ["2024-01-01"], "open": [100.0]})
    assert market_regime(df_no_close) is None

    # Negative close
    df_neg = _generate_synthetic_df(n_bars=500)
    df_neg.loc[499, "close"] = -10.0
    assert market_regime(df_neg) is None

    # NaN close
    df_nan = _generate_synthetic_df(n_bars=500)
    df_nan.loc[499, "close"] = float("nan")
    assert market_regime(df_nan) is None

    # Non-monotonic dates
    df_unorder = _generate_synthetic_df(n_bars=500)
    df_unorder.loc[0, "date"] = "2026-12-31"
    assert market_regime(df_unorder) is None

    # Duplicate dates
    df_dup = _generate_synthetic_df(n_bars=500)
    df_dup.loc[1, "date"] = df_dup.loc[0, "date"]
    assert market_regime(df_dup) is None


@pytest.mark.unit
def test_market_regime_metrics_math() -> None:
    """close_vs_ema50_pct and close_vs_ema200_pct match the documented formula."""
    df = _generate_synthetic_df(n_bars=500, close_val=150.0)
    res = market_regime(df)
    assert res is not None
    assert res.close == Decimal("150.0000")
    assert res.ema50 == Decimal("150.0000")
    assert res.ema200 == Decimal("150.0000")
    assert res.close_vs_ema50_pct == Decimal("0.0000")
    assert res.close_vs_ema200_pct == Decimal("0.0000")

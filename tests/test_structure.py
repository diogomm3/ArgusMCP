"""Unit tests for price and market-structure detection (indicators/structure.py)."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from mcp_finance.indicators.structure import (
    distance_to_level,
    gap_analysis,
    gap_pct,
    market_structure,
    support_resistance_levels,
    swing_highs_lows,
    week_52_high_low,
)


@pytest.fixture()
def zigzag_df() -> pd.DataFrame:
    """Constructed 40-bar series with known swing points at lookback=3.

    Swing lows at index 3 (low=40) and index 20 (low=45).
    Swing highs at index 10 (high=60) and index 30 (high=70).
    Overall structure: higher highs (60 -> 70) and higher lows (40 -> 45).
    """
    n = 40
    # Base baseline
    prices = np.linspace(45, 65, n)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 100_000,
        }
    )

    # Deliberate swing low 1 at index 3
    df.loc[3, "low"] = 40.0
    df.loc[3, "high"] = 42.0

    # Deliberate swing high 1 at index 10
    df.loc[10, "high"] = 60.0
    df.loc[10, "low"] = 55.0

    # Deliberate swing low 2 at index 20
    df.loc[20, "low"] = 45.0
    df.loc[20, "high"] = 48.0

    # Deliberate swing high 2 at index 30
    df.loc[30, "high"] = 70.0
    df.loc[30, "low"] = 64.0

    return df


def test_swing_highs_lows_detection(zigzag_df: pd.DataFrame) -> None:
    """Confirm swing points are correctly identified at exact index positions."""
    swings = swing_highs_lows(zigzag_df, lookback=3)

    assert swings.loc[3, "is_swing_low"]
    assert swings.loc[3, "swing_low_price"] == 40.0

    assert swings.loc[10, "is_swing_high"]
    assert swings.loc[10, "swing_high_price"] == 60.0

    assert swings.loc[20, "is_swing_low"]
    assert swings.loc[20, "swing_low_price"] == 45.0

    assert swings.loc[30, "is_swing_high"]
    assert swings.loc[30, "swing_high_price"] == 70.0

    # Boundary check: the last 3 bars cannot be confirmed swing points
    for idx in [37, 38, 39]:
        assert not swings.loc[idx, "is_swing_high"]
        assert not swings.loc[idx, "is_swing_low"]


def test_market_structure_classification(zigzag_df: pd.DataFrame) -> None:
    """Test higher-highs and higher-lows detection on the zigzag fixture."""
    struct = market_structure(zigzag_df, lookback=3)
    assert struct == "higher_highs_higher_lows"

    # Construct an explicit downtrend series (40 bars)
    n = 40
    prices = np.linspace(65, 45, n)
    downtrend_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 100_000,
        }
    )
    # Swing high 1 at idx 10 (peak at 70), swing low 1 at idx 15 (dip at 52)
    downtrend_df.loc[10, "high"] = 70.0
    downtrend_df.loc[15, "low"] = 52.0

    # Swing high 2 at idx 25 (lower peak at 60), swing low 2 at idx 30 (lower dip at 42)
    downtrend_df.loc[25, "high"] = 60.0
    downtrend_df.loc[30, "low"] = 42.0

    struct_down = market_structure(downtrend_df, lookback=3)
    assert struct_down == "lower_highs_lower_lows"


def test_market_structure_insufficient_data() -> None:
    """Test that fewer than 2 swings returns 'insufficient_data'."""
    # A short, flat 5-bar series has 0 swing points with lookback=3
    df = pd.DataFrame(
        {
            "open": [50.0] * 5,
            "high": [50.0] * 5,
            "low": [50.0] * 5,
            "close": [50.0] * 5,
            "volume": [1000] * 5,
        }
    )
    assert market_structure(df, lookback=3) == "insufficient_data"


def test_support_resistance_clustering_boundaries() -> None:
    """Test tolerance_pct clustering boundary: just-inside vs just-outside 1.5%."""
    # Construct a synthetic df with pivots at known prices
    # Base price 50.00
    # Pivot A at 100.00
    # Pivot B at 101.49 (1.49% difference <= 1.5% -> should merge)
    # Pivot C at 101.51 (1.51% difference > 1.5% -> should NOT merge with 100.00)

    # Build DF with 3 swing highs: idx 5 (100.0), idx 15 (101.49), idx 25 (105.0)
    df = pd.DataFrame(
        {
            "open": [50.0] * 35,
            "high": [50.0] * 35,
            "low": [49.0] * 35,
            "close": [50.0] * 35,
            "volume": [1000] * 35,
        }
    )
    df.loc[5, "high"] = 100.0
    df.loc[15, "high"] = 101.49
    df.loc[25, "high"] = 105.0

    # With min_touches=2, tolerance=0.015 (1.5%):
    # 100.0 and 101.49 should merge into 1 cluster with touches=2
    # 105.0 is alone (touches=1) so it is excluded
    levels = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=2
    )
    assert len(levels) == 1
    assert levels[0].touches == 2
    assert levels[0].level_type == "RESISTANCE"
    assert levels[0].min_price == Decimal("100.000000")
    assert levels[0].max_price == Decimal("101.490000")

    # Now change idx 15 to 101.51 (> 1.5%):
    # they should NOT merge, resulting in 0 levels with min_touches=2
    df.loc[15, "high"] = 101.51
    levels_outside = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=2
    )
    assert len(levels_outside) == 0

    # But with min_touches=1, both should appear as separate levels
    levels_single = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=1
    )
    assert len(levels_single) == 3


def test_support_resistance_min_touches_boundary() -> None:
    """Test min_touches boundary: 1 touch (under), 2 touches (at), 3 touches (over)."""
    df = pd.DataFrame(
        {
            "open": [50.0] * 40,
            "high": [50.0] * 40,
            "low": [49.0] * 40,
            "close": [50.0] * 40,
            "volume": [1000] * 40,
        }
    )
    # Cluster 1: 3 touches around 100.0 (highs at 5, 15, 25)
    df.loc[5, "high"] = 100.0
    df.loc[15, "high"] = 100.5
    df.loc[25, "high"] = 100.2
    # Cluster 2: 1 touch at 30.0 (low at 10)
    df.loc[10, "low"] = 30.0

    # With min_touches=2: 100.x has 3 touches (included), 30.0 has 1 touch (excluded)
    levels = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=2
    )
    assert len(levels) == 1
    assert levels[0].touches == 3
    assert levels[0].level_type == "RESISTANCE"

    # With min_touches=3: 100.x is still included
    levels_3 = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=3
    )
    assert len(levels_3) == 1
    assert levels_3[0].touches == 3

    # With min_touches=4: 100.x is now excluded
    levels_4 = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=4
    )
    assert len(levels_4) == 0


def test_level_type_classification() -> None:
    """Test that clusters classify properly as SUPPORT, RESISTANCE, or PIVOT."""
    df = pd.DataFrame(
        {
            "open": [50.0] * 50,
            "high": [51.0] * 50,
            "low": [49.0] * 50,
            "close": [50.0] * 50,
            "volume": [1000] * 50,
        }
    )
    # Cluster 1: Swing lows around 20.0 -> SUPPORT
    df.loc[5, "low"] = 20.0
    df.loc[15, "low"] = 20.2

    # Cluster 2: Swing highs around 80.0 -> RESISTANCE
    df.loc[25, "high"] = 80.0
    df.loc[35, "high"] = 80.5

    # Cluster 3: Both a swing high and a swing low around 50.0 -> PIVOT
    # At idx 8: swing high at 55.0 (neighbors are 51.0)
    df.loc[8, "high"] = 55.0
    # At idx 45: surrounding bars at 60.0, dipping to 55.2 as a swing low
    for j in range(43, 48):
        df.loc[j, "low"] = 59.0
        df.loc[j, "high"] = 62.0
    df.loc[45, "low"] = 55.2
    df.loc[45, "high"] = 57.0

    levels = support_resistance_levels(
        df, lookback=2, tolerance_pct=0.015, min_touches=2
    )
    type_map = {lvl.level_type: lvl for lvl in levels}

    assert "SUPPORT" in type_map
    assert "RESISTANCE" in type_map
    assert "PIVOT" in type_map
    assert type_map["PIVOT"].touches == 2


def test_gap_pct_and_analysis() -> None:
    """Test gap percentage computation and fill detection."""
    df = pd.DataFrame(
        {
            "open": [100.0, 105.0, 95.0],
            "high": [102.0, 106.0, 98.0],
            "low": [99.0, 101.0, 94.0],  # Bar 1 low 101.0 <= prev_close 101.0 -> filled
            "close": [101.0, 104.0, 97.0],
            "volume": [1000, 1000, 1000],
        }
    )
    # Bar 1: open=105, prev_close=101. Gap = (105-101)/101 = 4/101 = ~0.039604
    gaps = gap_pct(df)
    assert np.isnan(gaps.iloc[0])
    assert pytest.approx(gaps.iloc[1], rel=1e-4) == 4.0 / 101.0

    atr_series = pd.Series([2.0, 2.0, 2.0])
    analysis = gap_analysis(df, atr_series=atr_series)

    # Bar 1 is a gap up of $4.00, ATR=2.0 -> gap_vs_atr = 2.0
    assert analysis.loc[1, "is_gap"]
    assert pytest.approx(analysis.loc[1, "gap_vs_atr"], rel=1e-4) == 2.0
    # Bar 1 low was 101.0 <= prev_close 101.0 -> filled!
    assert analysis.loc[1, "is_filled"]


def test_distance_to_level() -> None:
    """Test signed distance to level in percentage and ATR multiples."""
    price = Decimal("110.00")
    level = Decimal("100.00")
    atr_val = Decimal("5.00")

    pct_dist, atr_dist = distance_to_level(price, level, atr=atr_val)
    # pct_dist = (110 - 100) / 100 = 0.10 (+10%)
    assert pct_dist == Decimal("0.1")
    # atr_dist = (110 - 100) / 5 = +2.0 ATR
    assert atr_dist == Decimal("2")

    # Below level
    price_below = Decimal("95.00")
    pct_dist_below, atr_dist_below = distance_to_level(price_below, level, atr=atr_val)
    assert pct_dist_below == Decimal("-0.05")
    assert atr_dist_below == Decimal("-1")

    # Without ATR
    pct_no_atr, atr_no_atr = distance_to_level(price, level, atr=None)
    assert pct_no_atr == Decimal("0.1")
    assert atr_no_atr is None


def test_week_52_high_low() -> None:
    """Test 52-week high and low window bounds."""
    # 300 bars: bar 0 has high=500 (outside 252 window),
    # bar 100 has high=250 (inside 252 window).
    n = 300
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        }
    )
    # Spike at bar 10 (outside the trailing 252 bars: 300 - 252 = 48)
    df.loc[10, "high"] = 500.0
    # Spike inside the trailing 252 bars
    df.loc[100, "high"] = 250.0
    df.loc[200, "low"] = 75.0

    high_52, low_52 = week_52_high_low(df, window_bars=252)
    assert high_52 == Decimal("250.000000")  # Not 500!
    assert low_52 == Decimal("75.000000")

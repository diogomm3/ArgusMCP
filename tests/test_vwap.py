"""Tests for mcp_finance.indicators.vwap module.

Tests cover:
  1. 4-bar reference values (hand-computed expected AVWAP at each bar)
  2. Forward-snap: non-trading anchor date snaps to next trading bar
  3. Pre-anchor bars are strictly NaN
  4. Zero-volume leading bars: AVWAP is NaN for those bars, resumes once volume > 0
  5. All-zero volume from anchor onward raises ValueError
  6. Empty DataFrame raises ValueError
  7. Missing required columns raises ValueError
  8. Duplicate dates raises ValueError
  9. Anchor after latest bar raises ValueError
 10. Anchor before earliest bar raises ValueError (forward-snap is within-series only)
 11. Unsorted input: result is correctly aligned to original (unsorted) index
 12. DatetimeIndex input (no 'date' column)
 13. datetime64 column in 'date' column (pd.Timestamp objects stored in column)
 14. Invalid ISO string anchor_date raises ValueError
 15. price_vs_vwap_distance: sign, pct, atr-norm, zero-vwap raises
 16. price_vs_vwap_distance agrees with distance_to_level (delegation contract)
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from mcp_finance.indicators.structure import distance_to_level
from mcp_finance.indicators.vwap import anchored_vwap, price_vs_vwap_distance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    dates: list[str],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    use_datetime_index: bool = False,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with a 'date' column (datetime.date objects)
    or a DatetimeIndex when use_datetime_index=True."""
    parsed = [datetime.date.fromisoformat(d) for d in dates]
    df = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    if use_datetime_index:
        df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in parsed])
    else:
        df["date"] = parsed
    return df


# ---------------------------------------------------------------------------
# Test 1 — 4-bar reference: hand-computed AVWAP at each bar
# ---------------------------------------------------------------------------
# Bars:
#   date       H      L      C     Vol     TP
#              cumTPV     cumVol   AVWAP
#   2024-01-02 102 100 101 1000  101.000  101000  101000  1000  101.000
#   2024-01-03 104 102 103 2000  103.000  206000  307000  3000  102.333
#   2024-01-04 101  99 100 1500  100.000  150000  457000  4500  101.556
#   2024-01-05 106 102 104 2500  104.000  260000  717000  7000  102.429
#
# Anchor = 2024-01-02 (first bar)

FOUR_BAR_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
FOUR_BAR_H = [102.0, 104.0, 101.0, 106.0]
FOUR_BAR_L = [100.0, 102.0, 99.0, 102.0]
FOUR_BAR_C = [101.0, 103.0, 100.0, 104.0]
FOUR_BAR_V = [1000.0, 2000.0, 1500.0, 2500.0]

EXPECTED_AVWAP = [
    101_000 / 1_000,  # 101.0
    307_000 / 3_000,  # 102.333...
    457_000 / 4_500,  # 101.555...
    717_000 / 7_000,  # 102.428...
]


def test_four_bar_reference_values() -> None:
    """AVWAP at each bar matches hand-computed cumulative TP*Vol / cumVol."""
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    result = anchored_vwap(df, anchor_date="2024-01-02")

    assert len(result) == 4
    np.testing.assert_allclose(
        result.to_numpy(),
        EXPECTED_AVWAP,
        rtol=1e-6,
        err_msg="4-bar AVWAP values do not match expected",
    )


def test_four_bar_mid_anchor() -> None:
    """Anchoring at bar 3 (2024-01-04): first two bars NaN, last two recalculated."""
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    result = anchored_vwap(df, anchor_date="2024-01-04")

    assert np.isnan(result.iloc[0]), "bar 0 (pre-anchor) must be NaN"
    assert np.isnan(result.iloc[1]), "bar 1 (pre-anchor) must be NaN"

    # From 2024-01-04:
    # bar2: TP=100, V=1500 → AVWAP=100.0
    # bar3: cumTPV=100*1500 + 104*2500=150000+260000=410000, cumV=4000 → 102.5
    np.testing.assert_allclose(result.iloc[2], 100.0, rtol=1e-9)
    np.testing.assert_allclose(result.iloc[3], 410_000 / 4_000, rtol=1e-9)


# ---------------------------------------------------------------------------
# Test 2 — Forward-snap: weekend anchor date
# ---------------------------------------------------------------------------


def test_forward_snap_weekend() -> None:
    """Anchor on Saturday 2024-01-06 snaps to Monday 2024-01-08.

    The Saturday anchor is within the series range because the series includes
    a Friday 2024-01-05 bar before the weekend. Forward-snap is applicable:
    no trading happened between Friday's close and Monday's open.
    """
    # Series: Fri 2024-01-05, Mon 2024-01-08, Tue 2024-01-09, Wed 2024-01-10
    dates = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
    df = _make_df(
        dates,
        highs=[108.0, 110.0, 112.0, 111.0],
        lows=[106.0, 108.0, 110.0, 109.0],
        closes=[107.0, 109.0, 111.0, 110.0],
        volumes=[500.0, 1000.0, 2000.0, 1500.0],
    )

    # Anchor on Saturday 2024-01-06 — inside the range (Fri..Wed), snaps to Mon
    result = anchored_vwap(df, anchor_date="2024-01-06")

    assert np.isnan(result.iloc[0]), "Friday (pre-anchor) must be NaN"
    assert not np.isnan(result.iloc[1]), "Monday (snapped bar) must have valid AVWAP"
    expected_tp_mon = (110.0 + 108.0 + 109.0) / 3.0
    np.testing.assert_allclose(
        result.iloc[1],
        expected_tp_mon,
        rtol=1e-9,
        err_msg="Single-bar AVWAP at snap = TP of that bar",
    )


def test_forward_snap_holiday_mid_series() -> None:
    """Anchor on 2024-01-03 with no bar on that date: snaps to 2024-01-04."""
    dates = ["2024-01-02", "2024-01-04", "2024-01-05"]
    df = _make_df(
        dates,
        highs=[102.0, 104.0, 106.0],
        lows=[100.0, 102.0, 103.0],
        closes=[101.0, 103.0, 104.0],
        volumes=[1000.0, 2000.0, 1500.0],
    )
    result = anchored_vwap(df, anchor_date="2024-01-03")

    assert np.isnan(result.iloc[0]), "Bar before the snap date must be NaN"
    assert not np.isnan(result.iloc[1]), "Snapped bar must be valid"
    assert not np.isnan(result.iloc[2]), "Post-snap bar must be valid"


# ---------------------------------------------------------------------------
# Test 3 — Pre-anchor bars are strictly NaN
# ---------------------------------------------------------------------------


def test_pre_anchor_bars_are_nan() -> None:
    """All bars before the anchor trading bar must return NaN."""
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    result = anchored_vwap(df, anchor_date="2024-01-05")

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])
    assert not np.isnan(result.iloc[3])


# ---------------------------------------------------------------------------
# Test 4 — Zero-volume leading bars: NaN until first positive volume
# ---------------------------------------------------------------------------


def test_zero_volume_leading_bars_are_nan() -> None:
    """When the first bars from anchor onward have zero volume, AVWAP is NaN
    for those bars (never inf); once positive volume appears AVWAP resumes."""
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    df = _make_df(
        dates,
        highs=[100.0, 100.0, 102.0],
        lows=[98.0, 98.0, 100.0],
        closes=[99.0, 99.0, 101.0],
        volumes=[0.0, 0.0, 1000.0],  # first two bars have zero volume
    )
    result = anchored_vwap(df, anchor_date="2024-01-02")

    assert np.isnan(result.iloc[0]), "Zero-volume bar must yield NaN, not inf"
    assert np.isnan(result.iloc[1]), "Zero-volume bar must yield NaN, not inf"
    assert not np.isnan(result.iloc[2]), "Bar with positive volume must be valid"
    assert not np.isinf(result.iloc[0]), "AVWAP must never be inf"
    assert not np.isinf(result.iloc[1]), "AVWAP must never be inf"

    expected_tp2 = (102.0 + 100.0 + 101.0) / 3.0
    np.testing.assert_allclose(result.iloc[2], expected_tp2, rtol=1e-9)


# ---------------------------------------------------------------------------
# Test 5 — All-zero volume from anchor onward raises ValueError
# ---------------------------------------------------------------------------


def test_all_zero_volume_from_anchor_raises() -> None:
    """Total volume from anchor onward == 0 must raise ValueError."""
    dates = ["2024-01-02", "2024-01-03"]
    df = _make_df(
        dates,
        highs=[100.0, 102.0],
        lows=[98.0, 100.0],
        closes=[99.0, 101.0],
        volumes=[0.0, 0.0],
    )
    with pytest.raises(ValueError, match="[Tt]otal volume"):
        anchored_vwap(df, anchor_date="2024-01-02")


# ---------------------------------------------------------------------------
# Test 6 — Empty DataFrame raises ValueError
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises() -> None:
    df = pd.DataFrame(columns=["date", "high", "low", "close", "volume"])
    with pytest.raises(ValueError, match="[Ee]mpty"):
        anchored_vwap(df, anchor_date="2024-01-02")


# ---------------------------------------------------------------------------
# Test 7 — Missing required columns raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop_col", ["high", "low", "close", "volume"])
def test_missing_column_raises(drop_col: str) -> None:
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    df = df.drop(columns=[drop_col])
    with pytest.raises(ValueError, match="[Mm]issing"):
        anchored_vwap(df, anchor_date="2024-01-02")


# ---------------------------------------------------------------------------
# Test 8 — Duplicate dates raises ValueError
# ---------------------------------------------------------------------------


def test_duplicate_dates_raises() -> None:
    df = _make_df(
        ["2024-01-02", "2024-01-02"],
        [102.0, 102.0],
        [100.0, 100.0],
        [101.0, 101.0],
        [1000.0, 1000.0],
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        anchored_vwap(df, anchor_date="2024-01-02")


# ---------------------------------------------------------------------------
# Test 9 — Anchor after latest bar raises ValueError
# ---------------------------------------------------------------------------


def test_anchor_after_latest_raises() -> None:
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    with pytest.raises(ValueError, match="[Aa]fter"):
        anchored_vwap(df, anchor_date="2025-01-01")


# ---------------------------------------------------------------------------
# Test 10 — Anchor before earliest bar raises ValueError
# ---------------------------------------------------------------------------


def test_anchor_before_earliest_raises() -> None:
    """An anchor strictly before the earliest bar raises ValueError.

    Forward-snap is only for non-trading gaps *within* the series range
    (e.g. weekends, holidays between two recorded bars). If anchor_date is
    before the first bar, the cache doesn't hold the volume that accumulated
    between the event and bar 0 — silently snapping would return a plausible
    but incorrect AVWAP (wrong because pre-cache volume is missing).
    """
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    with pytest.raises(ValueError, match="[Bb]efore"):
        anchored_vwap(df, anchor_date="2023-01-01")


# ---------------------------------------------------------------------------
# Test 11 — Unsorted input: result is aligned to original index
# ---------------------------------------------------------------------------


def test_unsorted_input_aligns_to_original_index() -> None:
    """Reversed input is handled correctly; result index matches original row order."""
    dates_rev = list(reversed(FOUR_BAR_DATES))
    highs_rev = list(reversed(FOUR_BAR_H))
    lows_rev = list(reversed(FOUR_BAR_L))
    closes_rev = list(reversed(FOUR_BAR_C))
    vols_rev = list(reversed(FOUR_BAR_V))

    df_sorted = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    df_rev = _make_df(dates_rev, highs_rev, lows_rev, closes_rev, vols_rev)

    result_sorted = anchored_vwap(df_sorted, anchor_date="2024-01-02")
    result_rev = anchored_vwap(df_rev, anchor_date="2024-01-02")

    # The reversed result mirrors the sorted result
    # (last bar of sorted == first bar of reversed)
    np.testing.assert_allclose(
        result_rev.to_numpy(),
        result_sorted.to_numpy()[::-1],
        rtol=1e-6,
        err_msg="Unsorted input: AVWAP values must match sorted, in original row order",
    )


# ---------------------------------------------------------------------------
# Test 12 — DatetimeIndex input (no 'date' column)
# ---------------------------------------------------------------------------


def test_datetime_index_input() -> None:
    """DataFrame with DatetimeIndex (no 'date' column) is accepted."""
    df = _make_df(
        FOUR_BAR_DATES,
        FOUR_BAR_H,
        FOUR_BAR_L,
        FOUR_BAR_C,
        FOUR_BAR_V,
        use_datetime_index=True,
    )
    assert "date" not in df.columns, "Sanity check: no 'date' column in this fixture"
    result = anchored_vwap(df, anchor_date="2024-01-02")

    assert len(result) == 4
    np.testing.assert_allclose(result.to_numpy(), EXPECTED_AVWAP, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 13 — datetime64 column ('date' column holds pd.Timestamp objects)
# ---------------------------------------------------------------------------


def test_datetime64_column_input() -> None:
    """'date' column holding pd.Timestamp (datetime64) values is accepted.

    The _make_df helper stores datetime.date objects; this test explicitly stores
    pd.Timestamp objects to cover the datetime64 column path that pd.to_datetime
    must handle (e.g. data coming from a DataFrame loaded with parse_dates=True).
    """
    parsed = [pd.Timestamp(d) for d in FOUR_BAR_DATES]
    df = pd.DataFrame(
        {
            "date": parsed,  # dtype will be datetime64[ns]
            "high": FOUR_BAR_H,
            "low": FOUR_BAR_L,
            "close": FOUR_BAR_C,
            "volume": FOUR_BAR_V,
        }
    )
    result = anchored_vwap(df, anchor_date="2024-01-02")

    assert len(result) == 4
    np.testing.assert_allclose(result.to_numpy(), EXPECTED_AVWAP, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 14 — Invalid ISO string anchor_date raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_anchor_date_string_raises() -> None:
    """An unparseable anchor_date string raises ValueError."""
    df = _make_df(FOUR_BAR_DATES, FOUR_BAR_H, FOUR_BAR_L, FOUR_BAR_C, FOUR_BAR_V)
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        anchored_vwap(df, anchor_date="2024-13-45")


# ---------------------------------------------------------------------------
# Test 15 — price_vs_vwap_distance: delegation contract
# ---------------------------------------------------------------------------


def test_price_vs_vwap_distance_pct_only() -> None:
    """Without atr_val, second element is None; pct distance is correct."""
    price = Decimal("105")
    vwap = Decimal("100")
    pct, atr_d = price_vs_vwap_distance(price, vwap)

    assert atr_d is None
    np.testing.assert_allclose(float(pct), 0.05, rtol=1e-9)


def test_price_vs_vwap_distance_below_vwap() -> None:
    """Price below VWAP yields a negative pct distance."""
    price = Decimal("95")
    vwap = Decimal("100")
    pct, _ = price_vs_vwap_distance(price, vwap)
    assert pct < Decimal("0"), "Price below VWAP must yield negative distance"
    np.testing.assert_allclose(float(pct), -0.05, rtol=1e-9)


def test_price_vs_vwap_distance_atr_normalized() -> None:
    """With atr_val provided, atr_distance = (price - vwap) / atr."""
    price = Decimal("110")
    vwap = Decimal("100")
    atr = Decimal("5")
    pct, atr_d = price_vs_vwap_distance(price, vwap, atr_val=atr)

    np.testing.assert_allclose(float(pct), 0.10, rtol=1e-9)
    assert atr_d is not None
    np.testing.assert_allclose(float(atr_d), 2.0, rtol=1e-9)


def test_price_vs_vwap_distance_zero_vwap_raises() -> None:
    """vwap_val == 0 must raise ValueError (division by zero)."""
    with pytest.raises(ValueError):
        price_vs_vwap_distance(Decimal("100"), Decimal("0"))


# ---------------------------------------------------------------------------
# Test 16 — price_vs_vwap_distance agrees with distance_to_level
# ---------------------------------------------------------------------------


def test_price_vs_vwap_distance_agrees_with_distance_to_level() -> None:
    """price_vs_vwap_distance delegates to distance_to_level.

    Asserts that price_vs_vwap_distance and a direct call to distance_to_level
    return identical results for the same inputs, confirming the delegation
    contract rather than just testing the arithmetic.
    """
    price = Decimal("115")
    vwap = Decimal("100")
    atr = Decimal("8")

    vwap_pct, vwap_atr = price_vs_vwap_distance(price, vwap, atr_val=atr)
    level_pct, level_atr = distance_to_level(price, vwap, atr=atr)

    assert vwap_pct == level_pct, "pct distance must be identical"
    assert vwap_atr == level_atr, "atr distance must be identical"

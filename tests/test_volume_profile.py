"""Unit tests for Phase M2.1 Volume Profile.

Tests:
  - Input validation (empty, zero volume, parameter bounds, date order).
  - Single-bar handling (wide bar range distribution vs flat bar single-bin collapse).
  - Known peak POC identification.
  - Hand-traced 70% Value Area dual-pointer expansion (VAH, VAL, value_area_volume).
  - Bimodal HVN/LVN detection.
  - Unimodal HVN degenerate case (< 2 HVNs -> lvn = []).
  - Boundary bin exclusion (bins 0 and N-1 cannot be HVN or LVN).
  - Date window slicing (start_date, end_date).
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pandas as pd
import pytest

from mcp_finance.indicators.volume_profile import build_volume_profile


@pytest.mark.unit
def test_build_volume_profile_empty_df_raises() -> None:
    """Empty DataFrame raises ValueError immediately."""
    with pytest.raises(ValueError, match="df cannot be empty"):
        build_volume_profile(pd.DataFrame())


@pytest.mark.unit
def test_build_volume_profile_zero_volume_raises() -> None:
    """Zero total volume raises ValueError."""
    df = pd.DataFrame(
        {
            "high": [105.0, 106.0],
            "low": [100.0, 101.0],
            "close": [104.0, 103.0],
            "volume": [0, 0],
        }
    )
    with pytest.raises(
        ValueError,
        match="Total volume across the profile window must be strictly positive",
    ):
        build_volume_profile(df)


@pytest.mark.unit
def test_build_volume_profile_invalid_params_raise() -> None:
    """Out-of-bound arguments raise ValueError immediately."""
    df = pd.DataFrame(
        {
            "high": [105.0],
            "low": [100.0],
            "close": [104.0],
            "volume": [1_000_000],
        }
    )

    with pytest.raises(ValueError, match="num_bins must be at least 10"):
        build_volume_profile(df, num_bins=9)

    with pytest.raises(ValueError, match="value_area_pct must be strictly between"):
        build_volume_profile(df, value_area_pct=0.0)

    with pytest.raises(ValueError, match="value_area_pct must be strictly between"):
        build_volume_profile(df, value_area_pct=1.0)

    with pytest.raises(ValueError, match="hvn_threshold must be strictly greater than"):
        build_volume_profile(df, hvn_threshold=1.0)

    with pytest.raises(ValueError, match="lvn_threshold must be strictly between"):
        build_volume_profile(df, lvn_threshold=0.0)

    with pytest.raises(ValueError, match="start_date cannot be after end_date"):
        build_volume_profile(
            df,
            start_date=datetime.date(2024, 2, 1),
            end_date=datetime.date(2024, 1, 1),
        )


@pytest.mark.unit
def test_build_volume_profile_single_flat_bar_collapses_to_point() -> None:
    """Flat bar (high == low): single-bin collapse with poc == vah == val == close."""
    df = pd.DataFrame(
        {
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [500_000],
            "date": [datetime.date(2024, 1, 1)],
        }
    )
    profile = build_volume_profile(df, num_bins=10)

    assert profile.total_volume == 500_000
    assert profile.poc == Decimal("100.000000")
    assert profile.val == Decimal("100.000000")
    assert profile.vah == Decimal("100.000000")
    assert profile.value_area_volume == Decimal("500000.000000")
    assert profile.bins[0].is_poc is True
    assert profile.bins[0].in_value_area is True


@pytest.mark.unit
def test_build_volume_profile_single_wide_bar_range_distribution() -> None:
    """Single bar with high > low: volume is distributed equally across all bins.

    For a single bar spanning the full range across num_bins=10, each bin
    receives 1/10th of the volume. With identical volume per bin, the dual-pointer
    expansion from the POC expands symmetrically until 70% volume is enclosed.
    """
    df = pd.DataFrame(
        {
            "high": [110.0],
            "low": [100.0],
            "close": [105.0],
            "volume": [1_000_000],
            "date": [datetime.date(2024, 1, 1)],
        }
    )
    profile = build_volume_profile(df, num_bins=10)

    assert profile.total_volume == 1_000_000
    assert len(profile.bins) == 10

    # Each of the 10 bins should have received exactly 100,000 volume
    for b in profile.bins:
        assert b.volume == Decimal("100000.000000")
        assert b.pct_of_total == Decimal("0.100000")

    # In a 10-bin flat distribution, 70% requires 7 bins (700,000 volume)
    in_va = [b for b in profile.bins if b.in_value_area]
    assert len(in_va) >= 7
    assert profile.value_area_volume >= Decimal("700000.000000")
    assert profile.val < profile.poc < profile.vah


@pytest.mark.unit
def test_build_volume_profile_known_peak_poc() -> None:
    """Known volume spike at a specific price must be identified as the POC."""
    # 5 bars at different price levels; bar at $105 has dominant volume
    df = pd.DataFrame(
        {
            "high": [101.0, 103.0, 106.0, 108.0, 110.0],
            "low": [100.0, 102.0, 105.0, 107.0, 109.0],
            "close": [100.5, 102.5, 105.5, 107.5, 109.5],
            "volume": [100_000, 150_000, 1_000_000, 200_000, 100_000],
            "date": [datetime.date(2024, 1, i + 1) for i in range(5)],
        }
    )
    profile = build_volume_profile(df, num_bins=10)

    # POC must be in the bin containing $105.5
    assert Decimal("105.0") <= profile.poc <= Decimal("106.0")
    assert profile.total_volume == 1_550_000

    poc_bin = next(b for b in profile.bins if b.is_poc)
    assert poc_bin.price_level == profile.poc
    assert poc_bin.in_value_area is True


@pytest.mark.unit
def test_build_volume_profile_dual_pointer_hand_trace() -> None:
    """Hand-traced 10-bin profile with known discrete volumes:

    Let price span be [100, 110] with 10 bins of width 1.0:
      Bin 0: [100, 101] vol = 50
      Bin 1: [101, 102] vol = 100
      Bin 2: [102, 103] vol = 200
      Bin 3: [103, 104] vol = 300
      Bin 4: [104, 105] vol = 1000 (POC)
      Bin 5: [105, 106] vol = 500
      Bin 6: [106, 107] vol = 200
      Bin 7: [107, 108] vol = 100
      Bin 8: [108, 109] vol = 30
      Bin 9: [109, 110] vol = 20

    Total volume = 2500.
    70% target volume = 1750.
    Dual-pointer trace from POC (Bin 4, vol 1000):
      Step 0: VA={4}, cum=1000. Up: Bin 5 (500), Down: Bin 3 (300).
      Step 1: Vol_up (500) > Vol_down (300) -> Add Bin 5. VA={4, 5}, cum=1500.
              Next Up: Bin 6 (200), Down: Bin 3 (300).
      Step 2: Vol_down (300) > Vol_up (200) -> Add Bin 3. VA={3, 4, 5}, cum=1800.
      1800 >= 1750 target -> Expansion stops!

    Final Value Area:
      Bins = {3, 4, 5}
      VAL = lower edge of Bin 3 = 103.0
      VAH = upper edge of Bin 5 = 106.0
      VA volume = 1800.
    """
    # Construct bars where bar i spans [100.0 + i, 101.0 + i], matching bin i exactly
    lows = [100.0 + i for i in range(10)]
    highs = [101.0 + i for i in range(10)]
    closes = [100.5 + i for i in range(10)]
    vols = [50, 100, 200, 300, 1000, 500, 200, 100, 30, 20]

    df = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "date": [datetime.date(2024, 1, i + 1) for i in range(10)],
        }
    )
    profile = build_volume_profile(df, num_bins=10, value_area_pct=0.70)

    assert profile.total_volume == 2500
    assert profile.poc == Decimal("104.500000")
    assert profile.val == Decimal("103.000000")
    assert profile.vah == Decimal("106.000000")
    assert profile.value_area_volume == Decimal("1800.000000")


@pytest.mark.unit
def test_build_volume_profile_bimodal_hvn_and_lvn() -> None:
    """Bimodal volume distribution produces 2 HVNs and 1 intermediate LVN.

    Construct a profile with 11 bins:
      Peaks at Bin 3 ($103.5) and Bin 7 ($107.5).
      Deep valley at Bin 5 ($105.5).
    """
    lows = [100.0 + i for i in range(11)]
    highs = [101.0 + i for i in range(11)]
    closes = [100.5 + i for i in range(11)]
    # Bins 3 and 7 are high peaks; Bin 5 is low valley
    vols = [50, 60, 100, 1000, 150, 30, 150, 950, 100, 60, 50]

    df = pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
            "date": [datetime.date(2024, 1, i + 1) for i in range(11)],
        }
    )
    # Mean vol = 2700 / 11 = ~245.45
    # Peaks: 1000 (Bin 3) >= 1.25 * 245 = 306.8 -> HVN
    #        950 (Bin 7) >= 1.25 * 245 = 306.8 -> HVN
    # Valley: 30 (Bin 5) <= 0.75 * 245 = 184.0 -> LVN
    profile = build_volume_profile(
        df,
        num_bins=11,
        hvn_threshold=1.25,
        lvn_threshold=0.75,
    )

    assert len(profile.hvn) == 2
    assert profile.hvn[0] == Decimal("103.500000")
    assert profile.hvn[1] == Decimal("107.500000")

    assert len(profile.lvn) == 1
    assert profile.lvn[0] == Decimal("105.500000")


@pytest.mark.unit
def test_build_volume_profile_unimodal_hvn_has_empty_lvn() -> None:
    """Unimodal distribution (single dominant HVN) must produce lvn == [].

    Per market profile definition, an LVN requires two bracketing HVNs.
    """
    centers = [100.5 + i for i in range(11)]
    # Single central peak
    vols = [50, 60, 80, 120, 200, 1500, 200, 120, 80, 60, 50]

    df = pd.DataFrame(
        {
            "high": centers,
            "low": centers,
            "close": centers,
            "volume": vols,
            "date": [datetime.date(2024, 1, i + 1) for i in range(11)],
        }
    )
    profile = build_volume_profile(df, num_bins=11)

    assert len(profile.hvn) == 1
    assert profile.hvn[0] == Decimal("105.500000")
    assert profile.lvn == []


@pytest.mark.unit
def test_build_volume_profile_edge_bins_excluded_from_hvn_and_lvn() -> None:
    """Volume peaks at bin 0 or bin N-1 must NOT be identified as HVN or LVN."""
    centers = [100.5 + i for i in range(10)]
    # Extreme peaks on bin 0 and bin 9
    vols = [2000, 50, 60, 70, 80, 80, 70, 60, 50, 2000]

    df = pd.DataFrame(
        {
            "high": centers,
            "low": centers,
            "close": centers,
            "volume": vols,
            "date": [datetime.date(2024, 1, i + 1) for i in range(10)],
        }
    )
    profile = build_volume_profile(df, num_bins=10)

    # Edge bins are not valid HVNs
    assert Decimal("100.500000") not in profile.hvn
    assert Decimal("109.500000") not in profile.hvn
    assert profile.hvn == []
    assert profile.lvn == []


@pytest.mark.unit
def test_build_volume_profile_date_range_slicing() -> None:
    """Date slicing isolates the requested profile window."""
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        datetime.date(2024, 1, 3),
        datetime.date(2024, 1, 4),
        datetime.date(2024, 1, 5),
    ]
    df = pd.DataFrame(
        {
            "high": [100.0, 102.0, 115.0, 104.0, 105.0],
            "low": [99.0, 101.0, 114.0, 103.0, 104.0],
            "close": [99.5, 101.5, 114.5, 103.5, 104.5],
            "volume": [100, 100, 999_999, 100, 100],  # bar 3 is huge spike
            "date": dates,
        }
    )

    # Slice excluding the bar on Jan 3
    profile = build_volume_profile(
        df,
        num_bins=10,
        start_date=datetime.date(2024, 1, 4),
        end_date=datetime.date(2024, 1, 5),
    )

    assert profile.total_volume == 200
    assert profile.start_date == datetime.date(2024, 1, 4)
    assert profile.end_date == datetime.date(2024, 1, 5)
    assert profile.poc < Decimal("110.0")  # Jan 3 spike was excluded

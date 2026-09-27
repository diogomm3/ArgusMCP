"""Volume Profile analytical module.

Bins traded volume by price level over a specified OHLCV bar window,
computing Point of Control (POC), Value Area (VAH/VAL via standard 70%
dual-pointer expansion), and High/Low Volume Nodes (HVN/LVN).
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from mcp_finance.indicators.models import VolumeBin, VolumeProfile


def build_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
    hvn_threshold: float = 1.25,
    lvn_threshold: float = 0.75,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> VolumeProfile:
    """Build a Volume Profile from OHLCV bars.

    Algorithm:
      1. Validates inputs immediately before performing any computation.
      2. Slices df to [start_date, end_date] if provided.
      3. Discretizes total price span [min(low), max(high)] into num_bins
         equal-width bins.
      4. Proportionally distributes each bar's volume across intersecting price bins.
      5. Identifies POC (bin with maximum volume).
      6. Expands Value Area from POC using dual-pointer expansion until accumulated
         volume >= value_area_pct * total_volume.
      7. Identifies HVN (local peaks >= hvn_threshold * mean_volume, excluding
         edge bins 0 and N-1).
      8. Identifies LVN (local valleys <= lvn_threshold * mean_volume between HVNs).

    Args:
        df: DataFrame containing at least 'high', 'low', 'close', 'volume' columns.
        num_bins: Number of price bins (minimum 10, default 50).
        value_area_pct: Fraction of volume enclosed by Value Area
            (0.0 < pct < 1.0, default 0.70).
        hvn_threshold: Multiplier of mean bin volume for High Volume Nodes
            (> 1.0, default 1.25).
        lvn_threshold: Multiplier of mean bin volume for Low Volume Nodes
            (0.0 < pct < 1.0, default 0.75).
        start_date: Optional inclusive start date to slice the profile window.
        end_date: Optional inclusive end date to slice the profile window.

    Returns:
        VolumeProfile model containing POC, VAH, VAL, HVNs, LVNs, and detailed bins.

    Raises:
        ValueError: On empty df, total volume <= 0, invalid parameters, or
            inverted dates.
    """
    # 1. Immediate upfront parameter validation
    if num_bins < 10:
        raise ValueError("num_bins must be at least 10")
    if not (0.0 < value_area_pct < 1.0):
        raise ValueError(
            "value_area_pct must be strictly between 0.0 and 1.0 (exclusive)"
        )
    if hvn_threshold <= 1.0:
        raise ValueError("hvn_threshold must be strictly greater than 1.0")
    if not (0.0 < lvn_threshold < 1.0):
        raise ValueError(
            "lvn_threshold must be strictly between 0.0 and 1.0 (exclusive)"
        )
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    if df.empty:
        raise ValueError("df cannot be empty")

    # 2. Slice date window if requested
    sliced = df
    if "date" in sliced.columns:
        date_series = pd.to_datetime(sliced["date"]).dt.date
        if start_date is not None:
            sliced = sliced[date_series >= start_date]
        if end_date is not None:
            date_series = pd.to_datetime(sliced["date"]).dt.date
            sliced = sliced[date_series <= end_date]
    elif isinstance(sliced.index, pd.DatetimeIndex):
        dt_idx = pd.DatetimeIndex(sliced.index)
        if start_date is not None:
            sliced = sliced[dt_idx.date >= start_date]
            dt_idx = pd.DatetimeIndex(sliced.index)
        if end_date is not None:
            sliced = sliced[dt_idx.date <= end_date]

    if sliced.empty:
        raise ValueError("df contains no rows within the specified date range")

    total_vol_int = int(sliced["volume"].sum())
    if total_vol_int <= 0:
        raise ValueError(
            "Total volume across the profile window must be strictly positive"
        )

    # Determine window dates
    if "date" in sliced.columns:
        profile_start = pd.to_datetime(sliced["date"].iloc[0]).date()
        profile_end = pd.to_datetime(sliced["date"].iloc[-1]).date()
    elif isinstance(sliced.index, pd.DatetimeIndex):
        profile_start = pd.Timestamp(sliced.index[0]).date()
        profile_end = pd.Timestamp(sliced.index[-1]).date()
    else:
        profile_start = start_date or datetime.date.today()
        profile_end = end_date or datetime.date.today()

    min_price = float(sliced["low"].min())
    max_price = float(sliced["high"].max())

    # 3. Discretization and range volume allocation
    if max_price == min_price:
        # Flat series: all volume in a single collapsed price level
        bin_edges = np.full(num_bins + 1, min_price)
        bin_centers = np.full(num_bins, min_price)
        bin_volumes = np.zeros(num_bins, dtype=float)
        bin_volumes[0] = float(total_vol_int)
    else:
        bin_edges = np.linspace(min_price, max_price, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = (max_price - min_price) / num_bins
        bin_volumes = np.zeros(num_bins, dtype=float)

        highs = sliced["high"].to_numpy(dtype=float)
        lows = sliced["low"].to_numpy(dtype=float)
        vols = sliced["volume"].to_numpy(dtype=float)

        for high_val, low_val, vol_val in zip(highs, lows, vols, strict=True):
            if vol_val <= 0.0:
                continue
            if high_val == low_val:
                # Single price bar: assign all volume to containing bin
                b_idx = min(int((low_val - min_price) / bin_width), num_bins - 1)
                bin_volumes[b_idx] += vol_val
            else:
                bar_range = high_val - low_val
                # Distribute across overlapping bins
                for i in range(num_bins):
                    b_low = bin_edges[i]
                    b_high = bin_edges[i + 1]
                    overlap = max(0.0, min(high_val, b_high) - max(low_val, b_low))
                    if overlap > 0.0:
                        bin_volumes[i] += vol_val * (overlap / bar_range)

    # 4. Point of Control (POC)
    poc_idx = int(np.argmax(bin_volumes))
    poc_price = Decimal(f"{bin_centers[poc_idx]:.6f}")

    # 5. Value Area 70% Dual-Pointer Expansion from POC
    target_vol = value_area_pct * float(total_vol_int)
    va_indices = {poc_idx}
    cum_vol = bin_volumes[poc_idx]
    u = poc_idx + 1
    d = poc_idx - 1

    while cum_vol < target_vol and (u < num_bins or d >= 0):
        vol_up = bin_volumes[u] if u < num_bins else 0.0
        vol_down = bin_volumes[d] if d >= 0 else 0.0

        if vol_up == 0.0 and vol_down == 0.0:
            break

        if vol_up > vol_down:
            va_indices.add(u)
            cum_vol += vol_up
            u += 1
        elif vol_down > vol_up:
            va_indices.add(d)
            cum_vol += vol_down
            d -= 1
        else:
            # Tie: expand symmetrically on both sides
            if u < num_bins:
                va_indices.add(u)
                cum_vol += vol_up
                u += 1
            if d >= 0:
                va_indices.add(d)
                cum_vol += vol_down
                d -= 1

    min_va_idx = min(va_indices)
    max_va_idx = max(va_indices)
    val_price = Decimal(f"{bin_edges[min_va_idx]:.6f}")
    vah_price = Decimal(f"{bin_edges[max_va_idx + 1]:.6f}")

    # 6. High Volume Nodes (HVN) and Low Volume Nodes (LVN)
    # Exclude boundary bins index 0 and index num_bins - 1
    mean_vol = float(np.mean(bin_volumes))
    hvn_indices: list[int] = []

    for i in range(1, num_bins - 1):
        if (
            bin_volumes[i] > bin_volumes[i - 1]
            and bin_volumes[i] > bin_volumes[i + 1]
            and bin_volumes[i] >= (hvn_threshold * mean_vol)
        ):
            hvn_indices.append(i)

    lvn_indices: list[int] = []
    # An LVN is an auction void between two distinct HVNs
    if len(hvn_indices) >= 2:
        start_search = hvn_indices[0] + 1
        end_search = hvn_indices[-1]
        for i in range(start_search, end_search):
            if (
                bin_volumes[i] < bin_volumes[i - 1]
                and bin_volumes[i] < bin_volumes[i + 1]
                and bin_volumes[i] <= (lvn_threshold * mean_vol)
            ):
                lvn_indices.append(i)

    hvn_prices = [Decimal(f"{bin_centers[i]:.6f}") for i in hvn_indices]
    lvn_prices = [Decimal(f"{bin_centers[i]:.6f}") for i in lvn_indices]

    # 7. Construct VolumeBin models
    bins: list[VolumeBin] = []
    for i in range(num_bins):
        vol_dec = Decimal(f"{bin_volumes[i]:.6f}")
        pct_dec = Decimal(f"{(bin_volumes[i] / total_vol_int):.6f}")
        bins.append(
            VolumeBin(
                price_level=Decimal(f"{bin_centers[i]:.6f}"),
                price_low=Decimal(f"{bin_edges[i]:.6f}"),
                price_high=Decimal(f"{bin_edges[i + 1]:.6f}"),
                volume=vol_dec,
                pct_of_total=pct_dec,
                is_poc=(i == poc_idx),
                in_value_area=(i in va_indices),
            )
        )

    va_volume_dec = sum(
        (b.volume for b in bins if b.in_value_area),
        Decimal("0"),
    )

    return VolumeProfile(
        poc=poc_price,
        vah=vah_price,
        val=val_price,
        total_volume=total_vol_int,
        value_area_volume=va_volume_dec,
        hvn=hvn_prices,
        lvn=lvn_prices,
        bins=bins,
        start_date=profile_start,
        end_date=profile_end,
        num_bins=num_bins,
    )

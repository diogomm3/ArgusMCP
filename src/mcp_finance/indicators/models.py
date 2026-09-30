"""Pydantic model for a per-symbol technical snapshot.

Candidate is the unit of input for Phase 7's screener. It is produced by
build_candidate_snapshot() from cached OHLCV data and pure indicator functions.

All price/indicator fields are Decimal to maintain precision consistency with the
NUMERIC(18, 6) database columns. None means insufficient history to compute the
indicator (e.g. fewer than 50 bars for EMA-50).

Volume fields (volume, avg_volume_20) are computed in the same single OHLCV
DataFrame pass as the technical indicators — the screener's volume filter must
read from these fields, never issuing a second OhlcvRepository.fetch_range call.
"""

import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """Per-symbol technical snapshot for the screener.

    Fields are Decimal | None rather than float to preserve the same precision
    contract as the ohlcv_daily table (NUMERIC 18, 6). None signals that there
    were insufficient cached bars to compute the indicator — Phase 7's screener
    should treat None as a filter-disqualifying signal, not as zero.
    """

    symbol: str = Field(
        ...,
        description="Ticker symbol (yfinance format, e.g. 'ASML.AS')",
    )
    as_of_date: datetime.date = Field(
        ...,
        description="Date for which the snapshot was computed",
    )
    source: str = Field(
        ...,
        description=(
            "OHLCV data source used to compute this snapshot (yfinance, bitstamp, etc.)"
        ),
    )

    # Price as of as_of_date
    close: Decimal = Field(
        ...,
        description="Closing price on as_of_date",
    )

    # Momentum
    rsi_14: Decimal | None = Field(
        default=None,
        description="RSI-14 (Wilder). None if < 14 bars available.",
    )

    # Trend
    ema_20: Decimal | None = Field(
        default=None,
        description="EMA-20. None if < 20 bars available.",
    )
    ema_50: Decimal | None = Field(
        default=None,
        description="EMA-50. None if < 50 bars available.",
    )

    # Volatility
    atr_14: Decimal | None = Field(
        default=None,
        description="ATR-14 (Wilder). None if < 14 bars available.",
    )

    # MACD (12, 26, 9)
    macd_line: Decimal | None = Field(
        default=None,
        description="MACD line (EMA12 - EMA26). None if < 26 bars.",
    )
    macd_signal: Decimal | None = Field(
        default=None,
        description="MACD signal (EMA9 of MACD line). None if < 35 bars.",
    )
    macd_histogram: Decimal | None = Field(
        default=None,
        description="MACD histogram (macd - signal). None if < 35 bars.",
    )

    # Volume (computed in the same DataFrame pass as indicators)
    volume: int | None = Field(
        default=None,
        description="Volume on as_of_date (last bar in the OHLCV window)",
    )
    avg_volume_20: int | None = Field(
        default=None,
        description=(
            "Simple 20-day average volume. None if fewer than 20 bars available. "
            "Computed from the same OHLCV fetch as all technical indicators — "
            "the screener volume filter must read this field, not issue a second "
            "OhlcvRepository.fetch_range call."
        ),
    )

    # Metadata
    bars_available: int = Field(
        ...,
        description="Number of cached OHLCV bars used to compute the indicators",
    )

    # -----------------------------------------------------------------------
    # M1.2 — Trend extensions
    # -----------------------------------------------------------------------

    ema_9: Decimal | None = Field(
        default=None,
        description="EMA-9. None if < 9 bars available.",
    )
    ema_200: Decimal | None = Field(
        default=None,
        description="EMA-200. None if < 200 bars available.",
    )
    ema_slope_20: Decimal | None = Field(
        default=None,
        description=(
            "Total pct change of EMA-20 over a 5-bar window: "
            "(EMA[t] - EMA[t-5]) / EMA[t-5]. None if < 25 bars."
        ),
    )
    ema_alignment: str | None = Field(
        default=None,
        description=(
            "EMA cascade alignment across EMA-9, 20, 50, 200: "
            "'bullish', 'bearish', or 'mixed'. None if any EMA is unavailable."
        ),
    )

    # -----------------------------------------------------------------------
    # M1.3 — Momentum extensions
    # -----------------------------------------------------------------------

    adx_14: Decimal | None = Field(
        default=None,
        description="ADX-14 (Wilder). None if < 27 bars (2*14-1).",
    )
    plus_di_14: Decimal | None = Field(
        default=None,
        description="+DI-14 (Wilder directional indicator). None if < 27 bars.",
    )
    minus_di_14: Decimal | None = Field(
        default=None,
        description="-DI-14 (Wilder directional indicator). None if < 27 bars.",
    )
    roc_10: Decimal | None = Field(
        default=None,
        description="Rate of Change over 10 bars. None if < 10 bars.",
    )

    # -----------------------------------------------------------------------
    # M1.4 — Volatility extensions
    # -----------------------------------------------------------------------

    bb_upper_20: Decimal | None = Field(
        default=None,
        description="Bollinger upper band (20-period, 2 std). None if < 20 bars.",
    )
    bb_middle_20: Decimal | None = Field(
        default=None,
        description="Bollinger middle band (20-period SMA). None if < 20 bars.",
    )
    bb_lower_20: Decimal | None = Field(
        default=None,
        description="Bollinger lower band (20-period, 2 std). None if < 20 bars.",
    )
    bb_bandwidth_20: Decimal | None = Field(
        default=None,
        description=(
            "Bollinger Bandwidth: (upper - lower) / middle. None if < 20 bars."
        ),
    )
    atr_expansion_ratio: Decimal | None = Field(
        default=None,
        description=(
            "ATR-14 / 50-bar rolling average of ATR-14. "
            "None if < 64 bars (14 ATR warmup + 50 avg window)."
        ),
    )
    atr_percentile_252: Decimal | None = Field(
        default=None,
        description=(
            "Percentile rank of ATR-14 in its 252-bar trailing window (0–1). "
            "None if < 266 bars (14 ATR warmup + 252 percentile window)."
        ),
    )

    # -----------------------------------------------------------------------
    # M1.5 — Volume extensions
    # -----------------------------------------------------------------------

    rvol_20: Decimal | None = Field(
        default=None,
        description="Relative volume (today / 20-day avg). None if < 20 bars.",
    )
    dollar_volume: Decimal | None = Field(
        default=None,
        description="Dollar volume (close * volume) on as_of_date.",
    )
    obv: Decimal | None = Field(
        default=None,
        description="On-Balance Volume on as_of_date.",
    )
    mfi_14: Decimal | None = Field(
        default=None,
        description="Money Flow Index-14. None if < 14 bars.",
    )

    # -----------------------------------------------------------------------
    # M1.1 — Market structure extensions
    # -----------------------------------------------------------------------

    market_structure: str | None = Field(
        default=None,
        description=(
            "Market structure classification: 'higher_highs_higher_lows', "
            "'lower_highs_lower_lows', 'ranging', or None (insufficient history "
            "— build_candidate_snapshot maps 'insufficient_data' to None)."
        ),
    )
    week_52_high: Decimal | None = Field(
        default=None,
        description="52-week (252-bar) rolling high. None if < 252 bars.",
    )
    week_52_low: Decimal | None = Field(
        default=None,
        description="52-week (252-bar) rolling low. None if < 252 bars.",
    )


LevelType = Literal["SUPPORT", "RESISTANCE", "PIVOT"]


class Level(BaseModel):
    """Support, resistance, or polarity pivot level.

    Clusters swing points within a tolerance band into a representative price level.
    """

    price: Decimal = Field(
        ...,
        description="Representative price of the level",
    )
    level_type: LevelType = Field(
        ...,
        description=(
            "Type of level: 'SUPPORT' (swing lows only), 'RESISTANCE' "
            "(swing highs only), or 'PIVOT' (both; polarity flip)"
        ),
    )
    touches: int = Field(
        ...,
        ge=1,
        description="Number of swing touches contributing to this level",
    )
    min_price: Decimal = Field(
        ...,
        description="Lowest touch price in the cluster",
    )
    max_price: Decimal = Field(
        ...,
        description="Highest touch price in the cluster",
    )


class VolumeBin(BaseModel):
    """A single price bin in the Volume Profile."""

    price_level: Decimal = Field(
        ...,
        description="Center price of this bin",
    )
    price_low: Decimal = Field(
        ...,
        description="Lower price boundary of this bin",
    )
    price_high: Decimal = Field(
        ...,
        description="Upper price boundary of this bin",
    )
    volume: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description=(
            "Volume allocated to this bin (fractional from proportional range split)"
        ),
    )
    pct_of_total: Decimal = Field(
        ...,
        description="Fraction of total profile volume in this bin (0.0 to 1.0)",
    )
    is_poc: bool = Field(
        default=False,
        description="True if this bin is the Point of Control",
    )
    in_value_area: bool = Field(
        default=False,
        description="True if this bin is inside the Value Area (VAL <= price <= VAH)",
    )


class VolumeProfile(BaseModel):
    """Volume Profile analytical output for a specified bar window."""

    poc: Decimal = Field(
        ...,
        description="Point of Control: price level of the bin with highest volume",
    )
    vah: Decimal = Field(
        ...,
        description="Value Area High: upper price boundary of the Value Area",
    )
    val: Decimal = Field(
        ...,
        description="Value Area Low: lower price boundary of the Value Area",
    )
    total_volume: int = Field(
        ...,
        gt=0,
        description="Exact integer sum of traded volume across all bars in the profile",
    )
    value_area_volume: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total volume contained within the Value Area [VAL, VAH]",
    )
    hvn: list[Decimal] = Field(
        default_factory=list,
        description="High Volume Nodes: price levels of significant local volume peaks",
    )
    lvn: list[Decimal] = Field(
        default_factory=list,
        description=(
            "Low Volume Nodes: price levels of significant volume valleys between HVNs"
        ),
    )
    bins: list[VolumeBin] = Field(
        default_factory=list,
        description="Full distribution across all price bins",
    )
    start_date: datetime.date = Field(
        ...,
        description="Start date of the analyzed OHLCV window",
    )
    end_date: datetime.date = Field(
        ...,
        description="End date of the analyzed OHLCV window",
    )
    num_bins: int = Field(
        ...,
        description="Number of price bins used for discretization",
    )


RegimeLabel = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class MarketRegime(BaseModel):
    """Market regime classification and underlying trend metrics.

    Lagging trend-state label derived from 50-day and 200-day EMAs plus 20-session
    EMA-50 slope over a fixed 500-bar trailing window.
    """

    regime: RegimeLabel = Field(
        ...,
        description="Regime state: 'BULLISH', 'BEARISH', or 'NEUTRAL'",
    )
    sessions_in_regime: int = Field(
        ...,
        ge=1,
        le=60,
        description="Consecutive sessions current label has held, capped at 60",
    )
    close_vs_ema50_pct: Decimal = Field(
        ...,
        description=(
            "Distance from close to EMA-50 in percent: (close - ema50) / ema50 * 100"
        ),
    )
    close_vs_ema200_pct: Decimal = Field(
        ...,
        description=(
            "Distance from close to EMA-200 in percent: (close - ema200) / ema200 * 100"
        ),
    )
    ema50_slope20: Decimal = Field(
        ...,
        description=(
            "20-session percentage slope of EMA-50: "
            "(ema50[t] - ema50[t-20]) / ema50[t-20]"
        ),
    )
    ema50: Decimal = Field(
        ...,
        description="50-day Exponential Moving Average",
    )
    ema200: Decimal = Field(
        ...,
        description="200-day Exponential Moving Average",
    )
    close: Decimal = Field(
        ...,
        description="Closing price on as_of_date",
    )
    as_of_date: datetime.date = Field(
        ...,
        description="Date of the evaluated bar",
    )

"""Technical indicators and alpha engine for the MCP Finance screener.

This module provides:
  - Pure indicator functions (ema, rsi, atr, macd) — no network, no DB
  - M1 extensions: trend, momentum, volatility, volume, market-structure
  - Candidate model — the screener's per-symbol snapshot
  - build_candidate_snapshot — assembles Candidate from cached OHLCV
  - SymbolNotCachedError — raised when a symbol has no ingested data

Design boundary (enforced here):
  build_candidate_snapshot reads from OhlcvRepository.fetch_range() directly,
  never via MarketDataService. Phase 7's screener must preserve this boundary.
"""

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
from mcp_finance.indicators.models import (
    Candidate,
    Level,
    LevelType,
    MarketRegime,
    RegimeLabel,
    VolumeBin,
    VolumeProfile,
)
from mcp_finance.indicators.regime import market_regime
from mcp_finance.indicators.snapshot import (
    SymbolNotCachedError,
    build_candidate_snapshot,
)
from mcp_finance.indicators.structure import (
    distance_to_level,
    gap_analysis,
    gap_pct,
    market_structure,
    support_resistance_levels,
    swing_highs_lows,
    week_52_high_low,
)
from mcp_finance.indicators.volume_profile import build_volume_profile
from mcp_finance.indicators.vwap import anchored_vwap, price_vs_vwap_distance

__all__ = [
    # Phase 5 core
    "ema",
    "rsi",
    "atr",
    "macd",
    # M1.2 trend
    "ema_slope",
    "ema_alignment",
    "price_vs_ema_distance",
    # M1.3 momentum
    "adx",
    "roc",
    # M1.4 volatility
    "bollinger_bands",
    "bollinger_bandwidth",
    "atr_expansion_ratio",
    "rolling_percentile",
    # M1.5 volume
    "relative_volume",
    "dollar_volume",
    "obv",
    "mfi",
    # Snapshot and models
    "Candidate",
    "SymbolNotCachedError",
    "build_candidate_snapshot",
    "Level",
    "LevelType",
    "MarketRegime",
    "RegimeLabel",
    # M1.1 market structure
    "swing_highs_lows",
    "market_structure",
    "support_resistance_levels",
    "gap_pct",
    "gap_analysis",
    "distance_to_level",
    "week_52_high_low",
    # M2 Volume Profile
    "VolumeBin",
    "VolumeProfile",
    "build_volume_profile",
    # M3 Anchored VWAP
    "anchored_vwap",
    "price_vs_vwap_distance",
    # M4 Market Regime
    "market_regime",
]

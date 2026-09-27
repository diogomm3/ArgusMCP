"""Technical indicators and alpha engine for the MCP Finance screener.

This module provides:
  - Pure indicator functions (ema, rsi, atr, macd) — no network, no DB
  - Candidate model — the screener's per-symbol snapshot
  - build_candidate_snapshot — assembles Candidate from cached OHLCV
  - SymbolNotCachedError — raised when a symbol has no ingested data

Design boundary (enforced here):
  build_candidate_snapshot reads from OhlcvRepository.fetch_range() directly,
  never via MarketDataService. Phase 7's screener must preserve this boundary.
"""

from mcp_finance.indicators.functions import atr, ema, macd, rsi
from mcp_finance.indicators.models import Candidate
from mcp_finance.indicators.snapshot import (
    SymbolNotCachedError,
    build_candidate_snapshot,
)
from mcp_finance.indicators.structure import (
    Level,
    distance_to_level,
    gap_analysis,
    gap_pct,
    market_structure,
    support_resistance_levels,
    swing_highs_lows,
    week_52_high_low,
)

__all__ = [
    "ema",
    "rsi",
    "atr",
    "macd",
    "Candidate",
    "SymbolNotCachedError",
    "build_candidate_snapshot",
    "Level",
    "swing_highs_lows",
    "market_structure",
    "support_resistance_levels",
    "gap_pct",
    "gap_analysis",
    "distance_to_level",
    "week_52_high_low",
]

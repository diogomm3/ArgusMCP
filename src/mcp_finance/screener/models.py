"""Pydantic models for the Phase 7 stock screening engine.

Model hierarchy:
  StrategyConfig   — user-configurable filter thresholds with documented defaults.
  FilterResult     — per-filter outcome: passed bool + human-readable reason.
  ScreenedStock    — per-symbol result merging technical + fundamental snapshot
                     with overall pass/fail and the list of filters that rejected it.
  ScreeningReport  — top-level tool response: summary counts + symbol lists.
  ScreenStocksInput — MCP tool input model.
"""

import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    """User-configurable screening strategy.

    All thresholds have sensible defaults for a momentum/trend strategy on
    mid-to-large-cap liquid equities. Pass a customised instance to tighten or
    loosen individual filters without rebuilding the engine.
    """

    name: str = Field(
        default="default_momentum_trend",
        description="Human-readable name shown in ScreeningReport.",
    )

    # ── Universe ──────────────────────────────────────────────────────────────
    min_market_cap: int | None = Field(
        default=2_000_000_000,
        description=(
            "Minimum market capitalisation in USD. Excludes micro/small caps. "
            "Set to None to disable the market-cap filter."
        ),
    )
    min_price: Decimal | None = Field(
        default=Decimal("5.00"),
        description=(
            "Minimum closing price in the ticker's native currency. "
            "Filters out penny stocks with excessive bid/ask spread. "
            "Set to None to disable."
        ),
    )
    max_pe_ratio: Decimal | None = Field(
        default=None,
        description=(
            "Maximum trailing P/E ratio. Stocks with P/E above this or with "
            "negative P/E (loss-making) are excluded when set. "
            "None disables the P/E filter."
        ),
    )

    # ── Liquidity ─────────────────────────────────────────────────────────────
    min_avg_volume_20: int | None = Field(
        default=100_000,
        description=(
            "Minimum 20-day average daily volume in shares. "
            "Prevents entering illiquid names where full position sizing cannot "
            "be achieved without significant market impact. "
            "Set to None to disable."
        ),
    )

    # ── Trend ─────────────────────────────────────────────────────────────────
    require_price_above_ema_20: bool = Field(
        default=True,
        description="Require close > EMA-20 (short-term trend confirmation).",
    )
    require_price_above_ema_50: bool = Field(
        default=True,
        description="Require close > EMA-50 (medium-term trend confirmation).",
    )
    require_ema_20_above_ema_50: bool = Field(
        default=True,
        description=(
            "Require EMA-20 > EMA-50 (golden cross / trend alignment). "
            "When both EMA conditions are true the stock is in a confirmed uptrend."
        ),
    )

    # ── Momentum ──────────────────────────────────────────────────────────────
    rsi_min: Decimal | None = Field(
        default=Decimal("40"),
        description=(
            "Minimum RSI-14 (Wilder). Values below this threshold indicate "
            "weak or oversold momentum. Set to None to disable the lower bound."
        ),
    )
    rsi_max: Decimal | None = Field(
        default=Decimal("70"),
        description=(
            "Maximum RSI-14. Values above this threshold indicate overbought "
            "conditions. Set to None to disable the upper bound."
        ),
    )
    require_macd_histogram_positive: bool = Field(
        default=True,
        description=(
            "Require MACD histogram > 0 (accelerating positive momentum). "
            "A positive histogram means the MACD line is above and diverging "
            "from the signal line."
        ),
    )
    require_macd_line_above_signal: bool = Field(
        default=True,
        description="Require MACD line > signal line (bullish crossover region).",
    )


class FilterResult(BaseModel):
    """Outcome of a single screening filter for one symbol."""

    filter_name: str = Field(..., description="e.g. 'universe', 'volume', 'trend'")
    passed: bool = Field(..., description="True if the symbol satisfied this filter.")
    reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation when passed=False, "
            "e.g. 'market_cap=None (fundamentals unavailable)'. "
            "None when passed=True."
        ),
    )


class ScreenedStock(BaseModel):
    """Per-symbol screening result merging technical and fundamental data.

    NOTE: An empty passed list (passed_count == 0 in ScreeningReport) is a
    valid and expected outcome for a selective trend filter. The default strategy
    requires seven-plus simultaneous conditions. In sideways, choppy, or bearish
    market regimes, zero symbols clearing every filter is correct behaviour, not
    a bug or configuration error.
    """

    symbol: str = Field(..., description="Ticker symbol.")
    as_of_date: datetime.date = Field(..., description="Screening date.")

    # Pass/fail summary
    passed: bool = Field(..., description="True if all active filters were satisfied.")
    failed_filters: list[str] = Field(
        default_factory=list,
        description="Names of filters the symbol did not satisfy.",
    )
    failure_reasons: list[str] = Field(
        default_factory=list,
        description="Corresponding human-readable failure messages.",
    )

    # Technical snapshot (from Candidate)
    close: Decimal | None = Field(default=None, description="Close price.")
    ema_20: Decimal | None = Field(default=None, description="EMA-20.")
    ema_50: Decimal | None = Field(default=None, description="EMA-50.")
    rsi_14: Decimal | None = Field(default=None, description="RSI-14.")
    macd_line: Decimal | None = Field(default=None, description="MACD line.")
    macd_signal: Decimal | None = Field(default=None, description="MACD signal.")
    macd_histogram: Decimal | None = Field(default=None, description="MACD histogram.")
    avg_volume_20: int | None = Field(
        default=None, description="20-day average volume."
    )
    bars_available: int = Field(
        default=0, description="OHLCV bars used to compute indicators."
    )

    # Fundamental snapshot (from CompanyFundamentals, may be None)
    company_name: str | None = Field(default=None)
    sector: str | None = Field(default=None)
    market_cap: int | None = Field(default=None)
    pe_ratio: Decimal | None = Field(default=None)
    fundamentals_source: str | None = Field(
        default=None,
        description="'fmp' when fundamentals are available, None otherwise.",
    )


class ScreeningReport(BaseModel):
    """Top-level output from the screen_stocks MCP tool.

    An empty passed_candidates list is a valid result — see ScreenedStock docstring.
    """

    as_of_date: datetime.date = Field(..., description="Date on which screening ran.")
    strategy_name: str = Field(..., description="Name from StrategyConfig.")
    total_screened: int = Field(..., description="Total number of symbols evaluated.")
    passed_count: int = Field(
        ..., description="Number of symbols that passed all active filters."
    )
    passed_candidates: list[ScreenedStock] = Field(
        default_factory=list,
        description=(
            "Symbols that passed every active filter. "
            "An empty list is a valid, expected result for a tight strategy."
        ),
    )
    failed_candidates: list[ScreenedStock] = Field(
        default_factory=list,
        description=(
            "Symbols that failed at least one filter. "
            "Only populated when include_failed=True is passed to the tool."
        ),
    )
    errors: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Symbols that could not be evaluated, keyed by symbol, "
            "with error message as value (e.g. OHLCV not cached)."
        ),
    )


class ScreenStocksInput(BaseModel):
    """Input parameters for the screen_stocks MCP tool."""

    symbols: list[str] | None = Field(
        default=None,
        description=(
            "List of ticker symbols to screen. "
            "Defaults to the project DEFAULT_WATCHLIST when None."
        ),
    )
    as_of_date: str | None = Field(
        default=None,
        description=(
            "ISO-8601 date string (YYYY-MM-DD) to screen as of. "
            "Defaults to today when None."
        ),
    )
    allow_live_fundamentals: bool = Field(
        default=True,
        description=(
            "When True (default), a stale or missing fundamentals cache triggers "
            "live FMP API calls (consuming quota). When False, only locally cached "
            "fundamentals are used; symbols with no cached data have fundamentals=None "
            "and any fundamental filters are skipped for those symbols."
        ),
    )
    include_failed: bool = Field(
        default=False,
        description=(
            "When True, ScreeningReport.failed_candidates is populated. "
            "Useful for debugging which filter each symbol failed."
        ),
    )

"""Pydantic model for a per-symbol technical snapshot.

Candidate is the unit of input for Phase 7's screener. It is produced by
build_candidate_snapshot() from cached OHLCV data and pure indicator functions.

All price/indicator fields are Decimal to maintain precision consistency with the
NUMERIC(18, 6) database columns. None means insufficient history to compute the
indicator (e.g. fewer than 50 bars for EMA-50).
"""

import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """Per-symbol technical snapshot for the screener.

    Fields are Decimal | None rather than float to preserve the same precision
    contract as the ohlcv_daily table (NUMERIC 18, 6). None signals that there
    were insufficient cached bars to compute the indicator — Phase 7's screener
    should treat None as a filter-disqualifying signal, not as zero.
    """

    symbol: str = Field(
        ..., description="Ticker symbol (yfinance format, e.g. 'ASML.AS')"
    )
    as_of_date: datetime.date = Field(
        ..., description="Date for which the snapshot was computed"
    )
    source: str = Field(default="yfinance", description="OHLCV data source used")

    # Price as of as_of_date
    close: Decimal = Field(..., description="Closing price on as_of_date")

    # Momentum
    rsi_14: Decimal | None = Field(
        default=None, description="RSI-14 (Wilder). None if < 14 bars available."
    )

    # Trend
    ema_20: Decimal | None = Field(
        default=None, description="EMA-20. None if < 20 bars available."
    )
    ema_50: Decimal | None = Field(
        default=None, description="EMA-50. None if < 50 bars available."
    )

    # Volatility
    atr_14: Decimal | None = Field(
        default=None, description="ATR-14 (Wilder). None if < 14 bars available."
    )

    # MACD (12, 26, 9)
    macd_line: Decimal | None = Field(
        default=None, description="MACD line (EMA12 - EMA26). None if < 26 bars."
    )
    macd_signal: Decimal | None = Field(
        default=None, description="MACD signal (EMA9 of MACD line). None if < 35 bars."
    )
    macd_histogram: Decimal | None = Field(
        default=None, description="MACD histogram (macd - signal). None if < 35 bars."
    )

    # Metadata
    bars_available: int = Field(
        ...,
        description="Number of cached OHLCV bars used to compute the indicators",
    )

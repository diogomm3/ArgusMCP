"""Shared Pydantic data models for market data operations."""

import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PriceQuote(BaseModel):
    """Current price snapshot for a symbol.

    INFORMATIONAL ONLY: This quote is sourced from market data providers (e.g.
    yfinance) and may be delayed or approximated. It must NEVER be treated as an
    execution price. Use broker tools (e.g. Trading212 get_positions) for actual
    execution-grade valuations.
    """

    symbol: str = Field(..., description="Ticker symbol")
    price: Decimal = Field(..., description="Latest market price")
    currency: str = Field(default="USD", description="Quoted currency")
    source: str = Field(default="yfinance", description="Data source provider")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        description="Timestamp when the quote was fetched (UTC)",
    )
    is_execution_price: bool = Field(
        default=False,
        description="Explicit flag confirming this is NOT an execution price",
    )


class OhlcvRecord(BaseModel):
    """Single daily OHLCV bar representation."""

    date: datetime.date = Field(..., description="Trading day")
    open: Decimal = Field(..., description="Opening price")
    high: Decimal = Field(..., description="Highest price of the day")
    low: Decimal = Field(..., description="Lowest price of the day")
    close: Decimal = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    source: str = Field(default="yfinance", description="Data source provider")


class BatchIngestResult(BaseModel):
    """Summary result of a batch market data ingestion run."""

    total_symbols: int = Field(..., description="Total symbols attempted")
    succeeded: int = Field(..., description="Symbols successfully ingested")
    failed: int = Field(..., description="Symbols that encountered errors")
    bars_upserted: int = Field(..., description="Total OHLCV bars written to database")
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Error messages keyed by symbol for failed tickers",
    )

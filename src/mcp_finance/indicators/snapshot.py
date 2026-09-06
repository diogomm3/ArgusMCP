"""build_candidate_snapshot — assembles a Candidate from cached OHLCV.

Design boundary (must not be violated):
  This function reads from OhlcvRepository.fetch_range() DIRECTLY with
  source passed explicitly. It must NEVER call MarketDataService.get_history()
  or make any network request. The screener (Phase 7) is designed on the
  assumption that snapshot building is a pure DB read.
"""

import datetime
import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import OhlcvRepository, SymbolRepository
from mcp_finance.indicators.functions import atr, ema, macd, rsi
from mcp_finance.indicators.models import Candidate
from mcp_finance.market_data.utils import derive_exchange


class SymbolNotCachedError(Exception):
    """Raised when a symbol has no row in the symbols table.

    This means the batch ingestion job has never run for this ticker/exchange.
    The screener must not treat this as a filter-failed symbol — it should
    surface immediately so the operator knows the watchlist is out of sync with
    the ingestion job.
    """

    def __init__(self, ticker: str, exchange: str) -> None:
        self.ticker = ticker
        self.exchange = exchange
        super().__init__(
            f"Symbol '{ticker}' on exchange '{exchange}' has never been ingested. "
            "Run the batch ingestion job first."
        )


def _to_decimal(value: float, places: int = 6) -> Decimal:
    """Round a float to `places` decimal places and return as Decimal.

    Matches NUMERIC(18, 6) DB precision.
    """
    quantize_str = Decimal(10) ** -places
    return Decimal(str(value)).quantize(quantize_str, rounding=ROUND_HALF_UP)


def _last_valid(series: "pd.Series[float]") -> float | None:
    """Return the last non-NaN, non-inf value of a Series, or None."""
    valid = series.dropna()
    valid = valid[valid.apply(lambda x: math.isfinite(x))]
    if valid.empty:
        return None
    return float(valid.iloc[-1])


async def build_candidate_snapshot(
    symbol: str,
    as_of_date: datetime.date,
    session: AsyncSession,
    *,
    exchange: str | None = None,
    source: str = "yfinance",
    lookback_days: int = 365,
) -> Candidate:
    """Build a Candidate technical snapshot for a symbol from cached OHLCV.

    Follows the same optional-exchange-with-auto-derivation convention as
    MarketDataService.ingest_ohlcv() and .get_history(), so callers don't need
    to learn a second symbol-identification pattern.

    Args:
        symbol:        Ticker in yfinance format (e.g. "AAPL", "ASML.AS").
        as_of_date:    Snapshot date. The most recent bar on or before this date
                       is used for the close price; indicators use bars from
                       [as_of_date - lookback_days, as_of_date].
        session:       Async SQLAlchemy session (caller owns commit/rollback).
        exchange:      Optional explicit exchange code. If None, derived
                       automatically via derive_exchange(symbol).
        source:        OHLCV source to read. Defaults to "yfinance". Passed
                       EXPLICITLY to fetch_range — never left implicit.
        lookback_days: How many calendar days of history to load (default 365,
                       ~252 trading days — enough for EMA-200 with margin).

    Returns:
        Candidate with computed indicators, or indicators=None if insufficient data.

    Raises:
        SymbolNotCachedError: If the symbol has no row in the symbols table
            (batch job has never ingested it). Do not catch silently.
    """
    effective_exchange = derive_exchange(symbol, exchange)

    symbol_repo = SymbolRepository(session)
    symbol_row = await symbol_repo.get_by_ticker(symbol, effective_exchange)
    if symbol_row is None:
        raise SymbolNotCachedError(symbol, effective_exchange)

    start = as_of_date - datetime.timedelta(days=lookback_days)
    ohlcv_repo = OhlcvRepository(session)
    bars = await ohlcv_repo.fetch_range(
        symbol_row.id,
        start,
        as_of_date,
        source=source,  # EXPLICIT — never leave implicit
    )

    bars_available = len(bars)

    if bars_available < 2:
        # Not enough data to compute anything meaningful.
        return Candidate(
            symbol=symbol,
            as_of_date=as_of_date,
            source=source,
            close=Decimal(str(bars[-1].close)) if bars_available == 1 else Decimal("0"),
            bars_available=bars_available,
        )

    # Build normalized DataFrame (float dtype for indicator math).
    df = pd.DataFrame(
        [
            {
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    )

    # Compute indicators — each returns None if the last value is NaN/inf.
    ema_20_val = _last_valid(ema(df, 20))
    ema_50_val = _last_valid(ema(df, 50))
    rsi_14_val = _last_valid(rsi(df, 14))
    atr_14_val = _last_valid(atr(df, 14))
    macd_df = macd(df)
    macd_line_val = _last_valid(macd_df["macd"])
    macd_signal_val = _last_valid(macd_df["signal"])
    macd_hist_val = _last_valid(macd_df["histogram"])

    last_close = float(bars[-1].close)

    return Candidate(
        symbol=symbol,
        as_of_date=as_of_date,
        source=source,
        close=_to_decimal(last_close),
        rsi_14=_to_decimal(rsi_14_val) if rsi_14_val is not None else None,
        ema_20=_to_decimal(ema_20_val) if ema_20_val is not None else None,
        ema_50=_to_decimal(ema_50_val) if ema_50_val is not None else None,
        atr_14=_to_decimal(atr_14_val) if atr_14_val is not None else None,
        macd_line=_to_decimal(macd_line_val) if macd_line_val is not None else None,
        macd_signal=(
            _to_decimal(macd_signal_val) if macd_signal_val is not None else None
        ),
        macd_histogram=(
            _to_decimal(macd_hist_val) if macd_hist_val is not None else None
        ),
        bars_available=bars_available,
    )

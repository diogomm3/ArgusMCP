"""Batch ingestion job for pre-populating historical OHLCV data into Postgres.

This script or entrypoint is used to pre-populate the database cache on a schedule
so that Phase 7's stock screener never has to make live API calls.
"""

import argparse
import asyncio
import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.engine import get_session
from mcp_finance.logger import configure_logging, get_logger
from mcp_finance.market_data.models import BatchIngestResult
from mcp_finance.market_data.service import MarketDataService

logger = get_logger(__name__)

DEFAULT_WATCHLIST: list[str] = [
    # US Tech / Large Cap
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    # European Leaders (demonstrating suffix-exchange mapping)
    "ASML.AS",
    "SAP.DE",
    "MC.PA",
    "BP.L",
    "AZN.L",
]

# ---------------------------------------------------------------------------
# M4 — Benchmark & sector ETF symbols
# These are kept SEPARATE from DEFAULT_WATCHLIST intentionally:
#   - Watchlist users should not receive ETF data by default.
#   - Relative-strength and regime calculations require these to be ingested
#     with sufficient history (≥252 bars) independently of stock ingestion.
# ---------------------------------------------------------------------------

BENCHMARK_SYMBOLS: list[str] = [
    "SPY",  # S&P 500 — primary broad-market benchmark
    "QQQ",  # Nasdaq-100 — tech/growth benchmark
]

# SPDR Sector ETF map: sector label → ETF ticker
# Covers all 11 FMP profile sector labels (as returned by FMP API) plus standard
# GICS aliases for cross-provider compatibility.
# Observed from live FMP API profile calls:
#   - "Technology" -> XLK
#   - "Consumer Cyclical" -> XLY (e.g. AMZN)
#   - "Communication Services" -> XLC (e.g. GOOGL)
# Assumed standard FMP profile sector names (canonical in FMP profile endpoint):
#   - "Financial Services", "Consumer Defensive", "Healthcare", "Energy",
#     "Industrials", "Basic Materials", "Real Estate", "Utilities"
SECTOR_ETF_MAP: dict[str, str] = {
    # 11 FMP profile sector labels
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
    # GICS standard names / aliases
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Materials": "XLB",
}


def get_sector_etf(sector: str | None) -> str | None:
    """Return the SPDR ETF ticker for a given sector name, or None if unmapped."""
    if not sector:
        return None
    return SECTOR_ETF_MAP.get(sector)


# ---------------------------------------------------------------------------
# Default ingestion watchlist: deduplicated union of DEFAULT_WATCHLIST,
# BENCHMARK_SYMBOLS, and SECTOR_ETF_MAP unique values (23 tickers total).
# Order: stock watchlist first, then benchmarks, then sector ETFs.
# ---------------------------------------------------------------------------
_seen_symbols: set[str] = set()
INGESTION_WATCHLIST: list[str] = []
for _sym in (
    DEFAULT_WATCHLIST + BENCHMARK_SYMBOLS + list(dict.fromkeys(SECTOR_ETF_MAP.values()))
):
    if _sym not in _seen_symbols:
        _seen_symbols.add(_sym)
        INGESTION_WATCHLIST.append(_sym)


async def run_batch_ingest(
    symbols: list[str] | None = None,
    days: int = 400,
    delay_seconds: float = 0.5,
    session: AsyncSession | None = None,
) -> BatchIngestResult:
    """Ingest historical bars for a list of symbols into Postgres.

    A deliberate delay is added between symbols to respect unauthenticated yfinance rate
    limits and avoid triggering throttling or silent empty responses.
    """
    target_symbols = symbols if symbols is not None else INGESTION_WATCHLIST
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days)

    total = len(target_symbols)
    succeeded = 0
    failed = 0
    total_bars = 0
    errors: dict[str, str] = {}

    logger.info(
        "Starting batch OHLCV ingestion",
        total_symbols=total,
        days=days,
        start_date=str(start_date),
        end_date=str(end_date),
        delay_seconds=delay_seconds,
    )

    async def _ingest_with_session(sess: AsyncSession) -> BatchIngestResult:
        nonlocal succeeded, failed, total_bars
        service = MarketDataService(session=sess)

        for i, sym in enumerate(target_symbols):
            try:
                bars = await service.ingest_ohlcv(
                    ticker=sym,
                    start=start_date,
                    end=end_date,
                )
                succeeded += 1
                total_bars += bars
                logger.info(
                    "Batch symbol ingested",
                    symbol=sym,
                    bars=bars,
                    progress=f"{i + 1}/{total}",
                )
            except Exception as exc:
                failed += 1
                errors[sym] = str(exc)
                logger.error(
                    "Batch ingestion failed for symbol",
                    symbol=sym,
                    error=str(exc),
                    progress=f"{i + 1}/{total}",
                )

            # Politeness delay to prevent rate limiting / throttling
            if i < total - 1 and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        return BatchIngestResult(
            total_symbols=total,
            succeeded=succeeded,
            failed=failed,
            bars_upserted=total_bars,
            errors=errors,
        )

    if session is not None:
        return await _ingest_with_session(session)

    async with get_session() as sess:
        return await _ingest_with_session(sess)


def main() -> None:
    """CLI entrypoint for batch OHLCV ingestion."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Pre-populate Postgres cache with OHLCV data."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help=(
            "Comma-separated list of symbols (e.g. AAPL,MSFT,SAP.DE). "
            "Defaults to INGESTION_WATCHLIST (stocks + benchmarks + sector ETFs)."
        ),
        default=None,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=400,
        help=(
            "Number of past calendar days of history to ingest (default: 400, "
            "yielding ~270 trading bars, ensuring ≥252 bars for EMA-200)."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between symbols to avoid throttling (default: 0.5s).",
    )

    args = parser.parse_args()
    symbols_list = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )

    result = asyncio.run(
        run_batch_ingest(
            symbols=symbols_list,
            days=args.days,
            delay_seconds=args.delay,
        )
    )
    logger.info("Batch ingestion completed", result=result.model_dump())


if __name__ == "__main__":
    main()

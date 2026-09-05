"""Market data ingestion and retrieval service coordinating clients and DB repos."""

import datetime
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import OhlcvBar, OhlcvRepository, SymbolRepository
from mcp_finance.logger import get_logger
from mcp_finance.market_data import MarketDataClient
from mcp_finance.market_data.models import OhlcvRecord, PriceQuote
from mcp_finance.market_data.utils import derive_exchange
from mcp_finance.market_data.yfinance import YFinanceClient

logger = get_logger(__name__)


class MarketDataService:
    """Coordinates market data fetching, repository persistence,
    and historical lookups.
    """

    def __init__(
        self,
        session: AsyncSession,
        client: MarketDataClient | None = None,
    ) -> None:
        self._session = session
        self._client: MarketDataClient = (
            client if client is not None else YFinanceClient()
        )
        self._symbol_repo = SymbolRepository(session)
        self._ohlcv_repo = OhlcvRepository(session)

    async def ingest_ohlcv(
        self,
        ticker: str,
        start: datetime.date,
        end: datetime.date,
        exchange: str | None = None,
    ) -> int:
        """Fetch historical bars from market data client and persist to database.

        Returns the number of rows inserted/updated in ohlcv_daily.
        """
        canonical_exchange = derive_exchange(ticker, exchange)
        symbol = await self._symbol_repo.upsert(
            ticker=ticker, exchange=canonical_exchange
        )

        df = await self._client.get_ohlcv(symbol=ticker, start=start, end=end)
        if df.empty:
            logger.info(
                "No bars returned to ingest",
                ticker=ticker,
                exchange=canonical_exchange,
                start=str(start),
                end=str(end),
            )
            return 0

        bars: list[OhlcvBar] = []
        for _, row in df.iterrows():
            bars.append(
                OhlcvBar(
                    symbol_id=symbol.id,
                    date=row["date"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=int(row["volume"]),
                    source="yfinance",
                )
            )

        count = await self._ohlcv_repo.bulk_upsert(bars)
        logger.info(
            "Ingested OHLCV bars",
            ticker=ticker,
            count=count,
            start=str(start),
            end=str(end),
        )
        return count

    async def get_history(
        self,
        ticker: str,
        start: datetime.date,
        end: datetime.date,
        exchange: str | None = None,
        sync_on_miss: bool = True,
    ) -> list[OhlcvRecord]:
        """Fetch historical OHLCV records for interactive tool lookups.

        IMPORTANT DESIGN NOTE:
        This method supports `sync_on_miss=True` as a convenience for ad-hoc, on-demand
        queries from LLM / MCP tool callers.
        Phase 7's screener (`screen_stocks`) must NEVER call this method; the screener
        must read directly from `OhlcvRepository.fetch_range()` to ensure zero live
        API calls during screen execution.
        """
        canonical_exchange = derive_exchange(ticker, exchange)
        symbol = await self._symbol_repo.get_by_ticker(
            ticker=ticker, exchange=canonical_exchange
        )

        bars = []
        if symbol is not None:
            bars = await self._ohlcv_repo.fetch_range(
                symbol_id=symbol.id,
                start=start,
                end=end,
                source="yfinance",
            )

        if not bars and sync_on_miss:
            logger.info("Cache miss for symbol history, syncing live", ticker=ticker)
            await self.ingest_ohlcv(
                ticker=ticker, start=start, end=end, exchange=canonical_exchange
            )
            symbol = await self._symbol_repo.get_by_ticker(
                ticker=ticker, exchange=canonical_exchange
            )
            if symbol is not None:
                bars = await self._ohlcv_repo.fetch_range(
                    symbol_id=symbol.id,
                    start=start,
                    end=end,
                    source="yfinance",
                )

        return [
            OhlcvRecord(
                date=b.date,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                source=b.source,
            )
            for b in bars
        ]

    async def get_price(self, ticker: str) -> PriceQuote:
        """Fetch current informational quote."""
        return await self._client.get_current_price(ticker)

"""Typed async repository layer for the three finance DB tables.

Each repository takes an `AsyncSession` injected from the caller (or from
`get_session()`). Repos do not manage transactions — the caller's context
manager is responsible for commit/rollback.

Design note on fetch_range():
  The `source` parameter defaults to `settings.primary_ohlcv_source` ('yfinance').
  This is intentional: since (symbol_id, date, source) is the uniqueness key, leaving
  source implicit would silently mix bars from different providers — computing an RSI
  off whichever row happened to land last. Always be explicit about provenance.
"""

import datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import CursorResult, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.models import FundamentalsCache, OhlcvDaily, Symbol
from mcp_finance.settings import settings


class SymbolRepository:
    """CRUD for the symbols master table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        ticker: str,
        exchange: str,
        *,
        isin: str | None = None,
        name: str | None = None,
        asset_class: str | None = None,
    ) -> Symbol:
        """Insert or update a symbol, returning the persisted row."""
        stmt = (
            insert(Symbol)
            .values(
                ticker=ticker,
                exchange=exchange,
                isin=isin,
                name=name,
                asset_class=asset_class,
            )
            .on_conflict_do_update(
                constraint="uq_symbols_ticker_exchange",
                set_={
                    "isin": isin,
                    "name": name,
                    "asset_class": asset_class,
                },
            )
            .returning(Symbol)
        )
        result = await self._session.execute(
            stmt.execution_options(populate_existing=True)
        )
        row = result.scalar_one()
        return row

    async def get_by_ticker(self, ticker: str, exchange: str) -> Symbol | None:
        """Fetch a symbol by (ticker, exchange), or None if not found."""
        stmt = select(Symbol).where(
            Symbol.ticker == ticker,
            Symbol.exchange == exchange,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class OhlcvBar:
    """Value object for a single OHLCV bar — used for bulk_upsert input."""

    __slots__ = (
        "symbol_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    )

    def __init__(
        self,
        symbol_id: int,
        date: datetime.date,
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: int,
        source: str,
    ) -> None:
        self.symbol_id = symbol_id
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.source = source


class OhlcvRepository:
    """Bulk-write and range-read for ohlcv_daily."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_upsert(self, bars: list[OhlcvBar]) -> int:
        """Insert or update a batch of bars. Returns the number of rows affected.

        On conflict (same symbol_id, date, source) the price/volume columns are
        overwritten — useful for corrected data from the same provider.
        """
        if not bars:
            return 0
        values = [
            {
                "symbol_id": b.symbol_id,
                "date": b.date,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "source": b.source,
            }
            for b in bars
        ]
        stmt = (
            insert(OhlcvDaily)
            .values(values)
            .on_conflict_do_update(
                constraint="uq_ohlcv_symbol_date_source",
                set_={
                    "open": insert(OhlcvDaily).excluded.open,
                    "high": insert(OhlcvDaily).excluded.high,
                    "low": insert(OhlcvDaily).excluded.low,
                    "close": insert(OhlcvDaily).excluded.close,
                    "volume": insert(OhlcvDaily).excluded.volume,
                },
            )
        )
        result = await self._session.execute(stmt)
        return cast(CursorResult[Any], result).rowcount

    async def fetch_range(
        self,
        symbol_id: int,
        start: datetime.date,
        end: datetime.date,
        source: str | None = None,
    ) -> list[OhlcvDaily]:
        """Return bars for a symbol within [start, end] for the given source.

        Defaults to `settings.primary_ohlcv_source` ('yfinance') when source is None,
        preventing silent mixing of bars from different providers.
        """
        effective_source = source or settings.primary_ohlcv_source
        stmt = (
            select(OhlcvDaily)
            .where(
                OhlcvDaily.symbol_id == symbol_id,
                OhlcvDaily.date >= start,
                OhlcvDaily.date <= end,
                OhlcvDaily.source == effective_source,
            )
            .order_by(OhlcvDaily.date)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class FundamentalsRepository:
    """Snapshot-based fundamentals cache with TTL staleness check."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        symbol_id: int,
        as_of_date: datetime.date,
        payload: dict[str, Any],
    ) -> FundamentalsCache:
        """Insert or replace the fundamentals snapshot for (symbol_id, as_of_date)."""
        stmt = (
            insert(FundamentalsCache)
            .values(
                symbol_id=symbol_id,
                as_of_date=as_of_date,
                payload=payload,
            )
            .on_conflict_do_update(
                constraint="uq_fundamentals_symbol_date",
                set_={"payload": payload},
            )
            .returning(FundamentalsCache)
        )
        result = await self._session.execute(
            stmt.execution_options(populate_existing=True)
        )
        return result.scalar_one()

    async def get_latest(self, symbol_id: int) -> FundamentalsCache | None:
        """Return the most recent snapshot for a symbol, or None."""
        stmt = (
            select(FundamentalsCache)
            .where(FundamentalsCache.symbol_id == symbol_id)
            .order_by(FundamentalsCache.as_of_date.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def is_fresh(self, symbol_id: int) -> bool:
        """Return True if the latest snapshot is within the configured TTL.

        'Fresh' means the snapshot's as_of_date is no older than
        `settings.fundamentals_cache_ttl_hours` hours ago.
        """
        latest = await self.get_latest(symbol_id)
        if latest is None:
            return False
        cutoff = datetime.date.today() - datetime.timedelta(
            hours=settings.fundamentals_cache_ttl_hours
        )
        return latest.as_of_date >= cutoff

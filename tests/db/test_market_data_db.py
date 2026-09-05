"""Testcontainers integration tests for MarketDataService and batch ingestion.

Uses real Postgres testcontainer from tests/db/conftest.py to ensure full schema,
foreign key, and decimal precision round-tripping.
"""

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.models import OhlcvDaily, Symbol
from mcp_finance.market_data.batch import run_batch_ingest
from mcp_finance.market_data.service import MarketDataService


class MockMarketClient:
    """Mock client returning predetermined DataFrames without network calls."""

    def __init__(self) -> None:
        self.call_count: int = 0

    async def get_ohlcv(
        self,
        symbol: str,
        start: datetime.date | str,
        end: datetime.date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.call_count += 1
        return pd.DataFrame(
            {
                "date": [datetime.date(2026, 9, 1), datetime.date(2026, 9, 2)],
                "open": [100.25, 102.50],
                "high": [105.00, 106.75],
                "low": [99.50, 101.00],
                "close": [103.75, 105.50],
                "volume": [5000000, 4200000],
            }
        )

    async def get_current_price(self, symbol: str) -> AsyncMock:
        return AsyncMock()


@pytest.mark.unit
async def test_market_data_service_ingest_testcontainers(
    db_session: AsyncSession,
) -> None:
    """Verify ingestion against real Postgres with foreign keys
    and Decimal precision.
    """
    mock_client = MockMarketClient()
    service = MarketDataService(session=db_session, client=mock_client)

    start = datetime.date(2026, 9, 1)
    end = datetime.date(2026, 9, 2)

    # Ingest US symbol
    bars_aapl = await service.ingest_ohlcv("AAPL", start, end)
    assert bars_aapl == 2

    # Ingest European symbol with suffix
    bars_asml = await service.ingest_ohlcv("ASML.AS", start, end)
    assert bars_asml == 2

    # Verify Symbols in DB
    result_sym = await db_session.execute(select(Symbol).order_by(Symbol.ticker))
    symbols = result_sym.scalars().all()
    assert len(symbols) == 2

    sym_aapl = symbols[0]
    assert sym_aapl.ticker == "AAPL"
    assert sym_aapl.exchange == "US"

    sym_asml = symbols[1]
    assert sym_asml.ticker == "ASML.AS"
    assert sym_asml.exchange == "EURONEXT_AMSTERDAM"

    # Verify OHLCV rows in DB
    result_bars = await db_session.execute(
        select(OhlcvDaily)
        .where(OhlcvDaily.symbol_id == sym_aapl.id)
        .order_by(OhlcvDaily.date)
    )
    db_bars = result_bars.scalars().all()
    assert len(db_bars) == 2

    first_bar = db_bars[0]
    assert first_bar.date == datetime.date(2026, 9, 1)
    # Check numeric(18,6) precision
    assert first_bar.open == Decimal("100.250000")
    assert first_bar.close == Decimal("103.750000")
    assert first_bar.volume == 5000000
    assert first_bar.source == "yfinance"

    # Re-ingestion is idempotent (no duplicate key errors)
    reingest_count = await service.ingest_ohlcv("AAPL", start, end)
    assert reingest_count == 2
    # Verify count remains 2
    result_bars_after = await db_session.execute(
        select(OhlcvDaily).where(OhlcvDaily.symbol_id == sym_aapl.id)
    )
    assert len(result_bars_after.scalars().all()) == 2


@pytest.mark.unit
async def test_market_data_service_get_history_caching(
    db_session: AsyncSession,
) -> None:
    """Verify get_history fetches on miss, then reads from DB on second query."""
    mock_client = MockMarketClient()
    service = MarketDataService(session=db_session, client=mock_client)

    start = datetime.date(2026, 9, 1)
    end = datetime.date(2026, 9, 2)

    # First call: cache miss -> calls client.get_ohlcv once
    records = await service.get_history("MSFT", start, end, sync_on_miss=True)
    assert len(records) == 2
    assert mock_client.call_count == 1
    assert records[0].close == Decimal("103.750000")

    # Second call: cache hit -> reads from DB, does not call client again
    records_cached = await service.get_history("MSFT", start, end, sync_on_miss=False)
    assert len(records_cached) == 2
    assert mock_client.call_count == 1


@pytest.mark.unit
async def test_batch_ingest_testcontainers(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify run_batch_ingest pre-populates Postgres with multiple symbols."""
    mock_client = MockMarketClient()

    # Monkeypatch YFinanceClient instantiation in MarketDataService
    monkeypatch.setattr(
        "mcp_finance.market_data.service.YFinanceClient",
        lambda: mock_client,
    )

    result = await run_batch_ingest(
        symbols=["AAPL", "SAP.DE"],
        days=5,
        delay_seconds=0.0,
        session=db_session,
    )
    assert result.total_symbols == 2
    assert result.succeeded == 2
    assert result.failed == 0
    assert result.bars_upserted == 4

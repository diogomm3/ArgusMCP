"""Repository unit tests — run against a real Postgres testcontainer.

All tests use the `db_session` fixture from conftest.py, which provides
an async session inside a savepoint that is rolled back after each test.
No external network required.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import (
    FundamentalsRepository,
    OhlcvBar,
    OhlcvRepository,
    SymbolRepository,
)

# ---------------------------------------------------------------------------
# Symbol tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_upsert_and_fetch_symbol(db_session: AsyncSession) -> None:
    """Insert a symbol with ISIN and retrieve it by (ticker, exchange)."""
    repo = SymbolRepository(db_session)

    symbol = await repo.upsert(
        "AAPL",
        "NASDAQ",
        isin="US0378331005",
        name="Apple Inc.",
        asset_class="STOCK",
    )
    assert symbol.id is not None
    assert symbol.ticker == "AAPL"
    assert symbol.isin == "US0378331005"

    fetched = await repo.get_by_ticker("AAPL", "NASDAQ")
    assert fetched is not None
    assert fetched.id == symbol.id
    assert fetched.name == "Apple Inc."


@pytest.mark.unit
async def test_upsert_symbol_is_idempotent(db_session: AsyncSession) -> None:
    """Re-upserting the same (ticker, exchange) updates fields, doesn't duplicate."""
    repo = SymbolRepository(db_session)

    s1 = await repo.upsert("MSFT", "NASDAQ", name="Microsoft Corp")
    s2 = await repo.upsert(
        "MSFT", "NASDAQ", name="Microsoft Corporation"
    )  # updated name

    assert s1.id == s2.id
    assert s2.name == "Microsoft Corporation"


# ---------------------------------------------------------------------------
# OHLCV tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_bulk_upsert_ohlcv_idempotent(db_session: AsyncSession) -> None:
    """Bulk-upserting the same bars twice should not duplicate rows."""
    sym_repo = SymbolRepository(db_session)
    sym = await sym_repo.upsert("TSLA", "NASDAQ")
    assert sym.id is not None

    ohlcv_repo = OhlcvRepository(db_session)
    today = datetime.date.today()
    bars = [
        OhlcvBar(
            symbol_id=sym.id,
            date=today - datetime.timedelta(days=i),
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("99.00"),
            close=Decimal("103.00"),
            volume=1_000_000,
            source="yfinance",
        )
        for i in range(5)
    ]

    count1 = await ohlcv_repo.bulk_upsert(bars)
    count2 = await ohlcv_repo.bulk_upsert(
        bars
    )  # same bars — should upsert, not duplicate

    assert count1 == 5
    # On conflict do update — rowcount behaviour varies by PG version
    # but data is correct
    fetched = await ohlcv_repo.fetch_range(
        sym.id, today - datetime.timedelta(days=4), today
    )
    assert len(fetched) == 5
    _ = count2  # rowcount on conflict is driver-dependent; we care about correctness


@pytest.mark.unit
async def test_ohlcv_source_isolation(db_session: AsyncSession) -> None:
    """Same (symbol, date) from two sources produces two rows, not a conflict."""
    sym_repo = SymbolRepository(db_session)
    sym = await sym_repo.upsert("NVDA", "NASDAQ")
    assert sym.id is not None

    ohlcv_repo = OhlcvRepository(db_session)
    today = datetime.date.today()

    bar_yf = OhlcvBar(
        sym.id,
        today,
        Decimal("800"),
        Decimal("810"),
        Decimal("795"),
        Decimal("805"),
        500_000,
        "yfinance",
    )
    bar_td = OhlcvBar(
        sym.id,
        today,
        Decimal("800.10"),
        Decimal("810.20"),
        Decimal("795.05"),
        Decimal("805.15"),
        501_000,
        "twelve_data",
    )

    await ohlcv_repo.bulk_upsert([bar_yf, bar_td])

    yf_bars = await ohlcv_repo.fetch_range(sym.id, today, today, source="yfinance")
    td_bars = await ohlcv_repo.fetch_range(sym.id, today, today, source="twelve_data")

    assert len(yf_bars) == 1
    assert len(td_bars) == 1
    assert yf_bars[0].close != td_bars[0].close  # providers disagree — both preserved


# ---------------------------------------------------------------------------
# Fundamentals tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_fundamentals_upsert_and_history(db_session: AsyncSession) -> None:
    """Two snapshots on different dates are both retained (history preserved)."""
    sym_repo = SymbolRepository(db_session)
    sym = await sym_repo.upsert("AMZN", "NASDAQ")
    assert sym.id is not None

    fund_repo = FundamentalsRepository(db_session)
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    snap1 = await fund_repo.upsert(sym.id, yesterday, {"pe_ratio": 40.0})
    snap2 = await fund_repo.upsert(sym.id, today, {"pe_ratio": 41.5})

    # Both snapshots exist — different as_of_date
    assert snap1.as_of_date == yesterday
    assert snap2.as_of_date == today

    latest = await fund_repo.get_latest(sym.id)
    assert latest is not None
    assert latest.as_of_date == today  # most recent wins in get_latest


@pytest.mark.unit
async def test_fundamentals_ttl_fresh(db_session: AsyncSession) -> None:
    """A snapshot taken today is within the TTL — is_fresh() returns True."""
    sym_repo = SymbolRepository(db_session)
    sym = await sym_repo.upsert("GOOGL", "NASDAQ")
    assert sym.id is not None

    fund_repo = FundamentalsRepository(db_session)
    await fund_repo.upsert(sym.id, datetime.date.today(), {"ev_ebitda": 20.0})

    assert await fund_repo.is_fresh(sym.id) is True


@pytest.mark.unit
async def test_fundamentals_ttl_stale(db_session: AsyncSession) -> None:
    """
    A snapshot older than the TTL (168h default) is stale —
    is_fresh() returns False.
    """
    sym_repo = SymbolRepository(db_session)
    sym = await sym_repo.upsert("META", "NASDAQ")
    assert sym.id is not None

    fund_repo = FundamentalsRepository(db_session)
    # 8 days ago — well beyond the 7-day (168h) default TTL
    stale_date = datetime.date.today() - datetime.timedelta(days=8)
    await fund_repo.upsert(sym.id, stale_date, {"ev_ebitda": 18.0})

    assert await fund_repo.is_fresh(sym.id) is False

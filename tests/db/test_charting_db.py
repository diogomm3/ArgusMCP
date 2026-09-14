"""Postgres testcontainers integration tests for ChartService.

Seeds symbols + 250 OHLCV bars, then exercises ChartService.get_chart() against
a real Postgres instance. Verifies PNG output is valid, EMA warmup works correctly
with a full warmup buffer, and SymbolNotCachedError is raised for unseen symbols.
"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.charting.service import ChartService
from mcp_finance.db.repository import OhlcvBar, OhlcvRepository, SymbolRepository
from mcp_finance.indicators.snapshot import SymbolNotCachedError

# ── Constants ────────────────────────────────────────────────────────────────

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TODAY = datetime.date.today()
TICKER = "AAPL"
EXCHANGE = "US"


# ── Seed helpers ─────────────────────────────────────────────────────────────


async def _seed_symbol(session: AsyncSession, ticker: str, exchange: str) -> int:
    repo = SymbolRepository(session)
    sym = await repo.upsert(ticker=ticker, exchange=exchange)
    assert sym.id is not None
    return int(sym.id)


async def _seed_ohlcv(
    session: AsyncSession,
    symbol_id: int,
    n_bars: int,
    base_price: float = 150.0,
    source: str = "yfinance",
) -> None:
    repo = OhlcvRepository(session)
    bars: list[OhlcvBar] = []
    for i in range(n_bars):
        bar_date = TODAY - datetime.timedelta(days=n_bars - 1 - i)
        price = Decimal(str(round(base_price + i * 0.1, 4)))
        bars.append(
            OhlcvBar(
                symbol_id=symbol_id,
                date=bar_date,
                open=price - Decimal("1.0"),
                high=price + Decimal("2.0"),
                low=price - Decimal("2.0"),
                close=price,
                volume=1500000,
                source=source,
            )
        )
    await repo.bulk_upsert(bars)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chart_service_returns_valid_png(db_session: AsyncSession) -> None:
    """ChartService returns bytes whose first 8 bytes are the PNG magic number."""
    symbol_id = await _seed_symbol(db_session, TICKER, EXCHANGE)
    # 250 bars comfortably warms EMA-200
    await _seed_ohlcv(db_session, symbol_id, n_bars=250)

    service = ChartService(db_session)
    png_bytes = await service.get_chart(symbol=TICKER, lookback_days=90)

    assert isinstance(png_bytes, bytes), "Return value must be bytes"
    assert png_bytes[:8] == _PNG_MAGIC, "Return value is not a valid PNG"
    assert len(png_bytes) >= 10_000, (
        f"PNG too small ({len(png_bytes)} bytes) — likely a rendering failure"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chart_service_min_lookback(db_session: AsyncSession) -> None:
    """lookback_days=10 (minimum) produces a valid chart from 250 seeded bars."""
    symbol_id = await _seed_symbol(db_session, "MSFT", EXCHANGE)
    await _seed_ohlcv(db_session, symbol_id, n_bars=250, base_price=300.0)

    service = ChartService(db_session)
    png_bytes = await service.get_chart(symbol="MSFT", lookback_days=10)
    assert png_bytes[:8] == _PNG_MAGIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chart_service_partial_history_unwarmed_emas(
    db_session: AsyncSession,
) -> None:
    """With only 30 bars, EMA-200 will be NaN for all rows but the chart still renders.

    The renderer's addplot guard (df['EMA200'].notna().any()) prevents mplfinance
    from seeing an all-NaN series, so no error is raised.
    """
    symbol_id = await _seed_symbol(db_session, "NVDA", EXCHANGE)
    # Only 30 bars — not enough to warm EMA-200
    await _seed_ohlcv(db_session, symbol_id, n_bars=30, base_price=500.0)

    service = ChartService(db_session)
    # Request 20 bars (well within the 30 seeded)
    png_bytes = await service.get_chart(symbol="NVDA", lookback_days=20)
    assert png_bytes[:8] == _PNG_MAGIC


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chart_service_unseen_symbol_raises(db_session: AsyncSession) -> None:
    """ChartService raises SymbolNotCachedError for a symbol not in the DB."""
    service = ChartService(db_session)
    with pytest.raises(SymbolNotCachedError):
        await service.get_chart(symbol="NOTREAL", lookback_days=90)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chart_service_insufficient_bars_raises(
    db_session: AsyncSession,
) -> None:
    """With < 5 bars ChartService raises ValueError (not a PNG, not a crash)."""
    symbol_id = await _seed_symbol(db_session, "TINY", EXCHANGE)
    # Seed exactly 3 bars — below the 5-bar minimum
    await _seed_ohlcv(db_session, symbol_id, n_bars=3, base_price=10.0)

    service = ChartService(db_session)
    with pytest.raises(ValueError, match="Insufficient OHLCV bars"):
        await service.get_chart(symbol="TINY", lookback_days=10)

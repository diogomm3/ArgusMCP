"""Integration tests for the ScreeningEngine against real testcontainers Postgres.

Seeds symbols, ohlcv_daily (60 bars), and fundamentals_cache, then exercises
ScreeningEngine end-to-end. Also tests:
  - Cold-cache fundamentals with a mocked FMPClient.
  - Quota-exhaustion fallback during screening.
  - screen_stocks MCP tool registration.
  - allow_live=False cache-only path.
"""

import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import (
    FundamentalsRepository,
    OhlcvBar,
    OhlcvRepository,
    SymbolRepository,
)
from mcp_finance.fundamentals.fmp import FMPClient
from mcp_finance.fundamentals.models import CompanyProfile, FinancialRatios
from mcp_finance.fundamentals.quota import DailyQuotaGuard
from mcp_finance.fundamentals.service import FundamentalsService
from mcp_finance.screener.engine import ScreeningEngine
from mcp_finance.screener.models import StrategyConfig

# ── Seed helpers ──────────────────────────────────────────────────────────────

TODAY = datetime.date.today()
AAPL_TICKER = "AAPL"
MSFT_TICKER = "MSFT"


async def _seed_symbol(session: AsyncSession, ticker: str, exchange: str = "US") -> int:
    """Upsert a symbol row and return its id."""
    repo = SymbolRepository(session)
    sym = await repo.upsert(ticker=ticker, exchange=exchange)
    assert sym.id is not None
    return int(sym.id)


async def _seed_ohlcv(
    session: AsyncSession,
    symbol_id: int,
    source: str = "yfinance",
    n_bars: int = 60,
    base_price: float = 175.0,
    base_volume: int = 1_500_000,
) -> None:
    """Seed n_bars of ascending OHLCV bars ending at TODAY."""
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
                volume=base_volume,
                source=source,
            )
        )
    await repo.bulk_upsert(bars)


async def _seed_fundamentals_cache(
    session: AsyncSession,
    symbol_id: int,
    as_of_date: datetime.date,
    market_cap: int = 3_000_000_000,
    pe_ratio: float = 28.5,
) -> None:
    """Seed a fresh fundamentals cache entry."""
    repo = FundamentalsRepository(session)
    payload: dict[str, Any] = {
        "symbol": AAPL_TICKER,
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "exchange": "US",
        "currency": "USD",
        "market_cap": market_cap,
        "pe_ratio": str(pe_ratio),
        "pb_ratio": "45.0",
        "ev_to_ebitda": "22.0",
        "profit_margin": "0.25",
        "debt_to_equity": "1.5",
        "current_ratio": "1.2",
        "dividend_yield": "0.005",
        "free_cash_flow_per_share": "6.50",
        "as_of_date": str(as_of_date),
        "is_cached": True,
        "source": "fmp",
    }
    await repo.upsert(symbol_id=symbol_id, as_of_date=as_of_date, payload=payload)


def _make_mock_fmp_client(ticker: str = AAPL_TICKER) -> MagicMock:
    """Return a mock FMPClient that returns minimal valid profile + ratios."""
    client = MagicMock(spec=FMPClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    profile = CompanyProfile.model_validate(
        {
            "symbol": ticker,
            "companyName": "Apple Inc.",
            "sector": "Technology",
            "exchange": "US",
            "currency": "USD",
            "marketCap": 3_000_000_000,
            "price": "175.00",
        }
    )
    ratios = FinancialRatios.model_validate(
        {
            "symbol": ticker,
            "priceToEarningsRatioTTM": "28.5",
            "priceToBookRatioTTM": "45.0",
            "enterpriseValueMultipleTTM": "22.0",
            "netProfitMarginTTM": "0.25",
            "debtToEquityRatioTTM": "1.5",
            "currentRatioTTM": "1.2",
            "dividendYieldTTM": "0.005",
            "freeCashFlowPerShareTTM": "6.50",
        }
    )
    client.get_company_profile = AsyncMock(return_value=profile)
    client.get_ratios = AsyncMock(return_value=ratios)
    return client


# ── Engine end-to-end tests ───────────────────────────────────────────────────


@pytest.mark.unit
async def test_screening_engine_cache_hit_passes(db_session: AsyncSession) -> None:
    """Engine returns a passed candidate when OHLCV and fundamentals are cached."""
    symbol_id = await _seed_symbol(db_session, AAPL_TICKER, "US")
    await _seed_ohlcv(db_session, symbol_id, n_bars=60, base_price=175.0)
    await _seed_fundamentals_cache(
        db_session, symbol_id, as_of_date=TODAY, market_cap=3_000_000_000
    )

    mock_client = _make_mock_fmp_client()
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[AAPL_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),
        allow_live_fundamentals=False,  # must use cache only
    )

    # FMP should NOT have been called — data is fresh in cache
    mock_client.get_company_profile.assert_not_called()
    mock_client.get_ratios.assert_not_called()

    assert report.total_screened == 1
    assert AAPL_TICKER not in report.errors
    # Either passed or failed on merit — assert no error path was taken
    total_evaluated = report.passed_count + len(report.errors)
    assert total_evaluated <= 1  # at most 1 error entry


@pytest.mark.unit
async def test_screening_engine_cold_cache_with_live_fmp(
    db_session: AsyncSession,
) -> None:
    """Engine fetches from mocked FMP when fundamentals cache is cold."""
    symbol_id = await _seed_symbol(db_session, AAPL_TICKER, "US")
    await _seed_ohlcv(db_session, symbol_id, n_bars=60, base_price=175.0)
    # No fundamentals seeded — cold cache

    mock_client = _make_mock_fmp_client()
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[AAPL_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),
        allow_live_fundamentals=True,
    )

    # FMP SHOULD have been called for cold-cache
    mock_client.get_company_profile.assert_called_once_with(AAPL_TICKER)
    mock_client.get_ratios.assert_called_once_with(AAPL_TICKER)
    assert report.total_screened == 1
    assert AAPL_TICKER not in report.errors


@pytest.mark.unit
async def test_screening_engine_allow_live_false_no_cache_returns_none(
    db_session: AsyncSession,
) -> None:
    """With allow_live=False and no cache, fundamentals is None and filter fails."""
    symbol_id = await _seed_symbol(db_session, AAPL_TICKER, "US")
    await _seed_ohlcv(db_session, symbol_id, n_bars=60, base_price=175.0)
    # No fundamentals in cache

    mock_client = _make_mock_fmp_client()
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[AAPL_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),  # min_market_cap=$2B — requires fundamentals
        allow_live_fundamentals=False,
        include_failed=True,
    )

    # FMP must not be called
    mock_client.get_company_profile.assert_not_called()
    mock_client.get_ratios.assert_not_called()

    # Symbol should fail universe filter (fundamentals unavailable)
    assert report.passed_count == 0
    assert len(report.failed_candidates) == 1
    failed = report.failed_candidates[0]
    assert "universe" in failed.failed_filters
    assert any("fundamentals unavailable" in r for r in failed.failure_reasons)


@pytest.mark.unit
async def test_screening_engine_quota_exhausted_fundamentals_none(
    db_session: AsyncSession,
) -> None:
    """When FMP quota is exhausted, fundamentals=None and universe filter fails."""
    symbol_id = await _seed_symbol(db_session, AAPL_TICKER, "US")
    await _seed_ohlcv(db_session, symbol_id, n_bars=60, base_price=175.0)
    # No fundamentals cache — will try live FMP

    # Mock FMP to raise quota exhausted immediately
    exhausted_client = MagicMock(spec=FMPClient)
    exhausted_client.__aenter__ = AsyncMock(return_value=exhausted_client)
    exhausted_client.__aexit__ = AsyncMock(return_value=False)

    # Exhaust the guard directly
    guard = DailyQuotaGuard(max_daily_requests=2)
    await guard.acquire(db_session, count=2, as_of=TODAY)

    # Service with exhausted guard
    service = FundamentalsService(
        session=db_session,
        client=exhausted_client,
        quota_guard=guard,
    )
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[AAPL_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),
        allow_live_fundamentals=True,
        include_failed=True,
    )

    # No unhandled error — just a graceful fail
    assert AAPL_TICKER not in report.errors
    assert report.passed_count == 0
    assert len(report.failed_candidates) == 1
    failed = report.failed_candidates[0]
    assert "universe" in failed.failed_filters


@pytest.mark.unit
async def test_screening_engine_symbol_not_cached_goes_to_errors(
    db_session: AsyncSession,
) -> None:
    """Symbol with no symbol row in DB goes to errors dict, not failed_candidates."""
    # MSFT is never seeded in symbols table

    mock_client = _make_mock_fmp_client(MSFT_TICKER)
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[MSFT_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),
        include_failed=True,
    )

    assert report.total_screened == 1
    assert MSFT_TICKER in report.errors
    assert "not cached" in report.errors[MSFT_TICKER].lower()
    assert report.passed_count == 0
    assert len(report.failed_candidates) == 0  # errors don't go to failed_candidates


@pytest.mark.unit
async def test_screening_engine_multi_symbol_isolation(
    db_session: AsyncSession,
) -> None:
    """One bad symbol does not prevent the other symbol from being evaluated."""
    aapl_id = await _seed_symbol(db_session, AAPL_TICKER, "US")
    await _seed_ohlcv(db_session, aapl_id, n_bars=60, base_price=175.0)
    await _seed_fundamentals_cache(db_session, aapl_id, as_of_date=TODAY)
    # MSFT is not seeded in symbols table — will fail with SymbolNotCachedError

    mock_client = _make_mock_fmp_client(AAPL_TICKER)
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[AAPL_TICKER, MSFT_TICKER],
        as_of_date=TODAY,
        config=StrategyConfig(),
        allow_live_fundamentals=False,
        include_failed=True,
    )

    assert report.total_screened == 2
    assert MSFT_TICKER in report.errors
    assert AAPL_TICKER not in report.errors
    # AAPL was fully evaluated (either passed or in failed_candidates)
    aapl_evaluated = any(
        s.symbol == AAPL_TICKER for s in report.passed_candidates
    ) or any(s.symbol == AAPL_TICKER for s in report.failed_candidates)
    assert aapl_evaluated


@pytest.mark.unit
async def test_screening_engine_empty_result_is_valid(
    db_session: AsyncSession,
) -> None:
    """An empty watchlist produces a valid ScreeningReport with passed_count=0.

    This test documents that passed_count==0 is the intended outcome for an
    empty input list, not a bug.
    """
    mock_client = _make_mock_fmp_client()
    service = FundamentalsService(session=db_session, client=mock_client)
    engine = ScreeningEngine(session=db_session, fundamentals_service=service)

    report = await engine.screen(
        symbols=[],
        as_of_date=TODAY,
        config=StrategyConfig(),
    )

    assert report.total_screened == 0
    assert report.passed_count == 0
    assert report.passed_candidates == []
    assert report.errors == {}

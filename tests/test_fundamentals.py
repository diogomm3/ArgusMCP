"""Unit tests for FMPClient, FundamentalsService, and MCP tool registration."""

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
import respx
from httpx import Response

from mcp_finance.fundamentals.fmp import (
    FMPApiError,
    FMPClient,
    FMPQuotaExceededError,
    SymbolNotFoundError,
)
from mcp_finance.fundamentals.models import (
    CompanyProfile,
    FinancialRatios,
)
from mcp_finance.fundamentals.quota import QuotaExhaustedError
from mcp_finance.fundamentals.service import FundamentalsService
from mcp_finance.server import mcp

SAMPLE_PROFILE_JSON = [
    {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "price": 230.50,
        "marketCap": 3500000000000,
        "beta": 1.15,
        "lastDividend": 1.00,
        "industry": "Consumer Electronics",
        "sector": "Technology",
        "country": "US",
        "description": "Apple designs and sells smartphones.",
        "website": "https://www.apple.com",
        "ceo": "Tim Cook",
    }
]

SAMPLE_RATIOS_JSON = [
    {
        "symbol": "AAPL",
        "priceToEarningsRatioTTM": 32.5,
        "priceToBookRatioTTM": 45.2,
        "priceToSalesRatioTTM": 8.5,
        "enterpriseValueMultipleTTM": 24.1,
        "grossProfitMarginTTM": 0.45,
        "operatingProfitMarginTTM": 0.30,
        "netProfitMarginTTM": 0.25,
        "currentRatioTTM": 1.02,
        "quickRatioTTM": 0.85,
        "debtToEquityRatioTTM": 1.45,
        "interestCoverageRatioTTM": 28.0,
        "dividendYieldTTM": 0.0055,
        "dividendPayoutRatioTTM": 0.16,
        "freeCashFlowPerShareTTM": 7.20,
    }
]


# ---------------------------------------------------------------------------
# FMPClient unit tests (mocked HTTP via respx)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_get_profile_success() -> None:
    """FMPClient.get_company_profile parses /stable/profile response."""
    route = respx.get(
        "https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(200, json=SAMPLE_PROFILE_JSON))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        profile = await client.get_company_profile("AAPL")

    assert route.called
    assert profile.symbol == "AAPL"
    assert profile.company_name == "Apple Inc."
    assert profile.exchange == "NASDAQ"
    assert profile.market_cap == 3500000000000
    assert profile.beta == Decimal("1.15")


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_get_profile_empty_raises_symbol_not_found() -> None:
    """FMPClient.get_company_profile raises SymbolNotFoundError on empty list."""
    respx.get(
        "https://financialmodelingprep.com/stable/profile?symbol=INVALID&apikey=test_key"
    ).mock(return_value=Response(200, json=[]))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        with pytest.raises(SymbolNotFoundError) as exc_info:
            await client.get_company_profile("INVALID")

    assert "INVALID" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_get_ratios_success() -> None:
    """FMPClient.get_ratios parses /stable/ratios-ttm response."""
    route = respx.get(
        "https://financialmodelingprep.com/stable/ratios-ttm?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(200, json=SAMPLE_RATIOS_JSON))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        ratios = await client.get_ratios("AAPL")

    assert route.called
    assert ratios.symbol == "AAPL"
    assert ratios.pe_ratio == Decimal("32.5")
    assert ratios.price_to_book == Decimal("45.2")
    assert ratios.dividend_yield == Decimal("0.0055")
    assert ratios.debt_to_equity == Decimal("1.45")


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_get_ratios_empty_raises_symbol_not_found() -> None:
    """FMPClient.get_ratios raises SymbolNotFoundError on empty list."""
    respx.get(
        "https://financialmodelingprep.com/stable/ratios-ttm?symbol=INVALID&apikey=test_key"
    ).mock(return_value=Response(200, json=[]))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        with pytest.raises(SymbolNotFoundError):
            await client.get_ratios("INVALID")


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_429_raises_quota_exceeded() -> None:
    """FMPClient raises FMPQuotaExceededError when receiving HTTP 429."""
    respx.get(
        "https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(429, text="Too Many Requests"))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        with pytest.raises(FMPQuotaExceededError) as exc_info:
            await client.get_company_profile("AAPL")

    assert exc_info.value.status_code == 429


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_500_raises_api_error() -> None:
    """FMPClient raises FMPApiError when receiving unexpected HTTP error."""
    respx.get(
        "https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(500, text="Internal Server Error"))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        with pytest.raises(FMPApiError):
            await client.get_company_profile("AAPL")


@pytest.mark.unit
@pytest.mark.anyio
@respx.mock
async def test_fmp_client_get_fundamentals_aggregates() -> None:
    """FMPClient.get_fundamentals calls both endpoints and aggregates."""
    respx.get(
        "https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(200, json=SAMPLE_PROFILE_JSON))
    respx.get(
        "https://financialmodelingprep.com/stable/ratios-ttm?symbol=AAPL&apikey=test_key"
    ).mock(return_value=Response(200, json=SAMPLE_RATIOS_JSON))

    async with FMPClient(
        api_key="test_key", base_url="https://financialmodelingprep.com/stable"
    ) as client:
        fund = await client.get_fundamentals("AAPL")

    assert fund.symbol == "AAPL"
    assert fund.company_name == "Apple Inc."
    assert fund.pe_ratio == Decimal("32.5")
    assert fund.is_cached is False
    assert fund.source == "fmp"


# ---------------------------------------------------------------------------
# FundamentalsService unit tests (mocked DB & dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_cache_hit_does_not_call_fmp_or_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache hit: fresh snapshot is returned with is_cached=True.

    Zero quota and zero FMP calls should be dispatched.
    """
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 42

    cached_snap = MagicMock()
    cached_snap.as_of_date = datetime.date(2024, 3, 1)
    cached_snap.payload = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "market_cap": 3500000000000,
        "pe_ratio": "32.5",
        "as_of_date": "2024-03-01",
        "source": "fmp",
    }

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=True)
    mock_fund_repo.get_latest = AsyncMock(return_value=cached_snap)

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    mock_quota_guard = AsyncMock()

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )
    result = await service.get_fundamentals("AAPL")

    assert result.symbol == "AAPL"
    assert result.is_cached is True
    assert result.pe_ratio == Decimal("32.5")

    # Zero quota acquired, zero FMP calls
    mock_quota_guard.acquire.assert_not_called()
    mock_client.get_company_profile.assert_not_called()
    mock_client.get_ratios.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_cache_hit_even_when_quota_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if quota is 0, cache hits are served normally."""
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 15

    cached_snap = MagicMock()
    cached_snap.as_of_date = datetime.date(2024, 3, 1)
    cached_snap.payload = {
        "symbol": "MSFT",
        "company_name": "Microsoft Corp.",
        "exchange": "NASDAQ",
        "currency": "USD",
        "pe_ratio": "35.0",
        "as_of_date": "2024-03-01",
        "source": "fmp",
    }

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=True)
    mock_fund_repo.get_latest = AsyncMock(return_value=cached_snap)

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    mock_quota_guard = AsyncMock()
    # If acquire were called, it would raise, but it must NOT be called on cache hit
    mock_quota_guard.acquire.side_effect = QuotaExhaustedError("Quota is 0")

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )
    result = await service.get_fundamentals("MSFT")

    assert result.symbol == "MSFT"
    assert result.is_cached is True
    mock_quota_guard.acquire.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_cache_miss_acquires_quota_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache miss: acquires 2 quota upfront, fetches profile + ratios.

    Persists snapshot to database only after both endpoints succeed.
    """
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 7

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=False)
    mock_fund_repo.upsert = AsyncMock()

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    profile = CompanyProfile.model_validate(SAMPLE_PROFILE_JSON[0])
    ratios = FinancialRatios.model_validate(SAMPLE_RATIOS_JSON[0])
    mock_client.get_company_profile = AsyncMock(return_value=profile)
    mock_client.get_ratios = AsyncMock(return_value=ratios)

    mock_quota_guard = AsyncMock()
    mock_quota_guard.acquire = AsyncMock()

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )
    result = await service.get_fundamentals("AAPL")

    assert result.symbol == "AAPL"
    assert result.is_cached is False
    assert result.pe_ratio == Decimal("32.5")

    # Atomic 2-count upfront reservation
    mock_quota_guard.acquire.assert_called_once_with(session, count=2)
    mock_client.get_company_profile.assert_called_once_with("AAPL")
    mock_client.get_ratios.assert_called_once_with("AAPL")
    mock_fund_repo.upsert.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_quota_exhausted_aborts_without_fmp_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When quota is exhausted on cache miss, raises QuotaExhaustedError and aborts."""
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 12

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=False)

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    mock_quota_guard = AsyncMock()
    mock_quota_guard.acquire = AsyncMock(
        side_effect=QuotaExhaustedError("Quota reached")
    )

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )

    with pytest.raises(QuotaExhaustedError):
        await service.get_fundamentals("AAPL")

    # Zero HTTP calls dispatched
    mock_client.get_company_profile.assert_not_called()
    mock_client.get_ratios.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_half_fetch_failure_does_not_persist_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If second endpoint fails, cache is NOT updated.

    Prevents corrupted or incomplete 7-day cache entries.
    """
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 7

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=False)
    mock_fund_repo.upsert = AsyncMock()

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    profile = CompanyProfile.model_validate(SAMPLE_PROFILE_JSON[0])
    mock_client.get_company_profile = AsyncMock(return_value=profile)
    mock_client.get_ratios = AsyncMock(
        side_effect=FMPApiError("500 Internal Error", status_code=500)
    )

    mock_quota_guard = AsyncMock()

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )

    with pytest.raises(FMPApiError):
        await service.get_fundamentals("AAPL")

    # DB upsert was NEVER called!
    mock_fund_repo.upsert.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_429_marks_quota_exhausted_authoritatively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If FMP returns 429, marks quota exhausted and raises QuotaExhaustedError."""
    session = AsyncMock()

    symbol_row = MagicMock()
    symbol_row.id = 7

    mock_sym_repo = AsyncMock()
    mock_sym_repo.upsert = AsyncMock(return_value=symbol_row)

    mock_fund_repo = AsyncMock()
    mock_fund_repo.is_fresh = AsyncMock(return_value=False)

    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.SymbolRepository",
        lambda _s: mock_sym_repo,
    )
    monkeypatch.setattr(
        "mcp_finance.fundamentals.service.FundamentalsRepository",
        lambda _s: mock_fund_repo,
    )

    mock_client = AsyncMock()
    mock_client.get_company_profile = AsyncMock(
        side_effect=FMPQuotaExceededError("429 Too Many Requests")
    )

    mock_quota_guard = AsyncMock()
    mock_quota_guard.max_daily_requests = 250

    service = FundamentalsService(
        session=session, client=mock_client, quota_guard=mock_quota_guard
    )

    with pytest.raises(QuotaExhaustedError):
        await service.get_fundamentals("AAPL")

    # Authoritative mark_exhausted was invoked
    mock_quota_guard.mark_exhausted.assert_called_once()


# ---------------------------------------------------------------------------
# MCP Tool Registration test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fundamentals_tool_registered() -> None:
    """Verify get_stock_fundamentals is registered on MCPServer."""
    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "get_stock_fundamentals" in tool_names

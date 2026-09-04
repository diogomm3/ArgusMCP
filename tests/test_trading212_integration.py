"""Live integration tests for Trading212Client against the real Trading212 Demo API.

Requires valid credentials in .env (or environment variables).
Run with:
    pytest -m integration
"""

from decimal import Decimal

import httpx
import pytest

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.brokers.trading212 import Trading212Client
from mcp_finance.settings import settings

# Skip test if credentials are dummy or empty placeholders
_HAS_VALID_CREDENTIALS = bool(
    settings.trading212_api_key
    and settings.trading212_api_secret
    and "your_" not in settings.trading212_api_key
    and "test" not in settings.trading212_api_key.lower()
)


@pytest.mark.integration
@pytest.mark.anyio
async def test_trading212_live_account_summary() -> None:
    """Fetch live account summary from Trading212 Demo API."""
    if not _HAS_VALID_CREDENTIALS:
        pytest.skip("Valid Trading212 credentials not configured in .env")

    try:
        async with Trading212Client() as client:
            account = await client.get_account()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            pytest.fail(
                "Trading 212 returned 403 Forbidden on /equity/account/cash. "
                "Ensure 'Account data' permission is checked for this "
                "API key in Trading 212 Settings > API."
            )
        raise

    assert isinstance(account, AccountSummary)
    assert isinstance(account.cash, Decimal)
    assert isinstance(account.total, Decimal)
    assert isinstance(account.invested, Decimal)
    assert isinstance(account.result, Decimal)


@pytest.mark.integration
@pytest.mark.anyio
async def test_trading212_live_positions() -> None:
    """Fetch live open positions list from Trading212 Demo API."""
    if not _HAS_VALID_CREDENTIALS:
        pytest.skip("Valid Trading212 credentials not configured in .env")

    async with Trading212Client() as client:
        positions = await client.get_positions()

    assert isinstance(positions, list)
    for pos in positions:
        assert isinstance(pos, Position)
        assert pos.ticker
        assert isinstance(pos.quantity, Decimal)
        assert isinstance(pos.average_price, Decimal)

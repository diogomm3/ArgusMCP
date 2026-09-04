"""Unit tests for Trading212Client — all HTTP calls are mocked with respx."""

from decimal import Decimal

import pytest
import respx
from httpx import Response

from mcp_finance.brokers.trading212 import Trading212Client

POSITIONS_PAYLOAD = [
    {
        "ticker": "AAPL_US_EQ",
        "quantity": "10.0",
        "averagePrice": "175.50",
        "currentPrice": "182.30",
        "ppl": "68.00",
        "frontendType": "STOCK",
    },
    {
        "ticker": "TSLA_US_EQ",
        "quantity": "5.0",
        "averagePrice": "240.00",
        "currentPrice": "230.00",
        "ppl": "-50.00",
        "frontendType": "STOCK",
    },
]

ACCOUNT_PAYLOAD = {
    "free": "1500.00",
    "invested": "2950.00",
    "result": "18.00",
    "total": "4468.00",
}


@pytest.mark.anyio
@respx.mock
async def test_get_positions_parses_response() -> None:
    """Trading212Client.get_positions() maps the API payload to Position models."""
    respx.get("https://demo.trading212.com/api/v0/equity/portfolio").mock(
        return_value=Response(200, json=POSITIONS_PAYLOAD)
    )

    async with Trading212Client(api_key="test_key", env="demo") as client:
        positions = await client.get_positions()

    assert len(positions) == 2
    assert positions[0].ticker == "AAPL_US_EQ"
    assert positions[0].quantity == Decimal("10.0")
    assert positions[0].ppl == Decimal("68.00")
    assert positions[1].ppl == Decimal("-50.00")


@pytest.mark.anyio
@respx.mock
async def test_get_account_parses_response() -> None:
    """Trading212Client.get_account() maps the API payload to AccountSummary."""
    respx.get("https://demo.trading212.com/api/v0/equity/account/cash").mock(
        return_value=Response(200, json=ACCOUNT_PAYLOAD)
    )

    async with Trading212Client(api_key="test_key", env="demo") as client:
        account = await client.get_account()

    assert account.cash == Decimal("1500.00")
    assert account.invested == Decimal("2950.00")
    assert account.result == Decimal("18.00")
    assert account.total == Decimal("4468.00")


@pytest.mark.anyio
@respx.mock
async def test_get_positions_raises_on_401() -> None:
    """Trading212Client.get_positions() raises when the broker returns 401."""
    respx.get("https://demo.trading212.com/api/v0/equity/portfolio").mock(
        return_value=Response(401, json={"message": "Unauthorized"})
    )

    with pytest.raises(Exception):
        async with Trading212Client(api_key="bad_key", env="demo") as client:
            await client.get_positions()

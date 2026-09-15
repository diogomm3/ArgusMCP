"""BrokerClient Protocol — every broker adapter must satisfy this interface."""

from decimal import Decimal
from typing import Protocol

from mcp_finance.brokers.models import AccountSummary, OrderResult, Position


class BrokerClient(Protocol):
    """Interface all broker adapters must implement."""

    async def get_positions(self) -> list[Position]:
        """Return all open positions."""
        ...

    async def get_account(self) -> AccountSummary:
        """Return account-level summary (cash, equity, P&L)."""
        ...

    async def place_order(
        self,
        ticker: str,
        quantity: Decimal,
        order_type: str = "MARKET",
        limit_price: Decimal | None = None,
    ) -> OrderResult:
        """Submit an order to the broker (demo environment only in Phase 9)."""
        ...

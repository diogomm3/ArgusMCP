"""BrokerClient Protocol — every broker adapter must satisfy this interface."""

from typing import Protocol

from mcp_finance.brokers.models import AccountSummary, Position


class BrokerClient(Protocol):
    """Minimal read interface all broker adapters must implement."""

    async def get_positions(self) -> list[Position]:
        """Return all open positions."""
        ...

    async def get_account(self) -> AccountSummary:
        """Return account-level summary (cash, equity, P&L)."""
        ...

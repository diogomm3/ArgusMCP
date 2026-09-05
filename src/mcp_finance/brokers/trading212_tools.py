"""MCP tools for Trading212 broker operations."""

from mcp.server.mcpserver import MCPServer

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.brokers.utils import get_trading212_client


def register_trading212_tools(mcp: MCPServer) -> None:
    """Register all Trading212 MCP tools with the server."""

    @mcp.tool()
    async def get_positions() -> list[Position]:
        """Return all open equity positions from the connected broker account."""
        async with get_trading212_client() as client:
            return await client.get_positions()

    @mcp.tool()
    async def get_account() -> AccountSummary:
        """Return the current account cash/equity/P&L snapshot."""
        async with get_trading212_client() as client:
            return await client.get_account()

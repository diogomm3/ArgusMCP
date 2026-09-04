"""MCP server — tools are registered here and wired to broker adapters."""

from mcp.server.mcpserver import MCPServer

from mcp_finance.auth import AuthMiddleware
from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.brokers.trading212 import Trading212Client
from mcp_finance.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

mcp = MCPServer("mcp_finance")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _broker() -> Trading212Client:
    """Return a fresh Trading212Client per call.

    For Phase 2 this is fine; Phase 9 will switch to a lifespan-managed
    singleton to reuse the underlying httpx connection pool.
    """
    return Trading212Client()


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


@mcp.tool()
def ping() -> str:
    """Ping the server to check if it is alive."""
    return "pong"


@mcp.tool()
async def get_positions() -> list[Position]:
    """Return all open equity positions from the connected broker account."""
    async with _broker() as client:
        return await client.get_positions()


@mcp.tool()
async def get_account() -> AccountSummary:
    """Return the current account cash/equity/P&L snapshot."""
    async with _broker() as client:
        return await client.get_account()


app = AuthMiddleware(mcp.sse_app())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

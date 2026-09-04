"""MCP server — tools are registered here and wired to broker adapters."""

from mcp.server.mcpserver import MCPServer

from mcp_finance.auth import AuthMiddleware
from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.logger import configure_logging, get_logger
from mcp_finance.utils import get_trading212_client

configure_logging()
logger = get_logger(__name__)

mcp = MCPServer("mcp_finance")


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
    async with get_trading212_client() as client:
        return await client.get_positions()


@mcp.tool()
async def get_account() -> AccountSummary:
    """Return the current account cash/equity/P&L snapshot."""
    async with get_trading212_client() as client:
        return await client.get_account()


app = AuthMiddleware(mcp.sse_app())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

"""MCP server — tools are registered here and wired to broker adapters."""

from mcp.server.mcpserver import MCPServer

from mcp_finance.auth import AuthMiddleware
from mcp_finance.brokers.trading212_tools import register_trading212_tools
from mcp_finance.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

mcp = MCPServer("mcp_finance")


@mcp.tool()
def ping() -> str:
    """Ping the server to check if it is alive."""
    return "pong"


# Register domain toolsets
register_trading212_tools(mcp)

app = AuthMiddleware(mcp.streamable_http_app(host="0.0.0.0"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

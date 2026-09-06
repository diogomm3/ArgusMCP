"""FastMCP tool registrations for company fundamentals operations."""

from mcp.server.mcpserver import MCPServer

from mcp_finance.db.engine import get_session
from mcp_finance.fundamentals.fmp import FMPClient
from mcp_finance.fundamentals.models import (
    CompanyFundamentals,
    GetStockFundamentalsInput,
)
from mcp_finance.fundamentals.service import FundamentalsService


def register_FMP_tools(mcp: MCPServer) -> None:
    """Register FMP fundamentals MCP tools with the server."""

    @mcp.tool()
    async def get_stock_fundamentals(
        input: GetStockFundamentalsInput,
    ) -> CompanyFundamentals:
        """Return fundamental valuation, financial health, and company profile data.

        Reads from local cache if a fresh snapshot is available (<= 168h TTL).
        If missing or stale, queries FMP stable endpoints and caches the result.
        Returns key metrics including market cap, P/E ratio, P/B ratio, EV/EBITDA,
        profit margins, debt-to-equity, current ratio, dividend yield, and FCF
        per share.
        """
        async with get_session() as session:
            async with FMPClient() as client:
                service = FundamentalsService(session=session, client=client)
                return await service.get_fundamentals(
                    symbol=input.symbol,
                    exchange=input.exchange,
                )

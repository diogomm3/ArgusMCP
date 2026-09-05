"""FastMCP tool registrations for market data operations."""

import datetime

from mcp.server.mcpserver import MCPServer

from mcp_finance.db.engine import get_session
from mcp_finance.market_data.models import OhlcvRecord, PriceQuote
from mcp_finance.market_data.service import MarketDataService
from mcp_finance.market_data.yfinance_client import YFinanceClient


def register_market_data_tools(mcp: MCPServer) -> None:
    """Register market data MCP tools with the server."""

    @mcp.tool()
    async def get_symbol_price(symbol: str) -> PriceQuote:
        """Return the latest informational market price quote for a symbol.

        INFORMATIONAL ONLY — NOT AN EXECUTION PRICE.
        This quote is sourced from market data providers (yfinance) and may be delayed
        or approximate. Do not use this price for trade execution, order sizing, or
        portfolio valuation. For trade-related valuations, use Trading212 broker tools
        (`get_positions`, `get_account`).
        """
        client = YFinanceClient()
        return await client.get_current_price(symbol)

    @mcp.tool()
    async def get_history(
        symbol: str,
        start_date: str,
        end_date: str,
        exchange: str | None = None,
    ) -> list[OhlcvRecord]:
        """Return historical daily OHLCV bars for a symbol between dates.

        Dates must be formatted as YYYY-MM-DD (start_date and end_date).
        Results are read from the database cache, or fetched live on cache miss
        and persisted. For international symbols, specify the standard ticker
        suffix (e.g., 'ASML.AS', 'SAP.DE') or pass an explicit exchange.
        """
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        async with get_session() as session:
            service = MarketDataService(session=session)
            return await service.get_history(
                ticker=symbol,
                start=start,
                end=end,
                exchange=exchange,
                sync_on_miss=True,
            )

"""FastMCP tool registrations for market data operations."""

import datetime

from mcp.server.mcpserver import MCPServer

from mcp_finance.db.engine import get_session
from mcp_finance.market_data.models import (
    GetHistoryInput,
    GetSymbolPriceInput,
    OhlcvRecord,
    PriceQuote,
)
from mcp_finance.market_data.service import MarketDataService
from mcp_finance.market_data.yfinance import YFinanceClient


def register_yfinance_tools(mcp: MCPServer) -> None:
    """Register yfinance MCP tools with the server."""

    @mcp.tool()
    async def get_symbol_price(input: GetSymbolPriceInput) -> PriceQuote:
        """Return the latest informational market price quote for a symbol.

        INFORMATIONAL ONLY — NOT AN EXECUTION PRICE.
        This quote is sourced from market data providers (yfinance) and may be delayed
        or approximate. Do not use this price for trade execution, order sizing, or
        portfolio valuation. For trade-related valuations, use Trading212 broker tools
        (`get_positions`, `get_account`).
        """
        client = YFinanceClient()
        return await client.get_current_price(input.symbol)

    @mcp.tool()
    async def get_history(input: GetHistoryInput) -> list[OhlcvRecord]:
        """Return historical daily OHLCV bars for a symbol between dates.

        Dates must be formatted as YYYY-MM-DD (start_date and end_date).
        Results are read from the database cache, or fetched live on cache miss
        and persisted. For international symbols, specify the standard ticker
        suffix (e.g., 'ASML.AS', 'SAP.DE') or pass an explicit exchange.
        """
        start = datetime.date.fromisoformat(input.start_date)
        end = datetime.date.fromisoformat(input.end_date)

        async with get_session() as session:
            service = MarketDataService(session=session)
            return await service.get_history(
                ticker=input.symbol,
                start=start,
                end=end,
                exchange=input.exchange,
                sync_on_miss=True,
            )

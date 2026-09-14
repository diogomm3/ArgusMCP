"""FastMCP tool registrations for stock charting."""

import base64

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

from mcp_finance.charting.models import GetStockChartInput
from mcp_finance.charting.service import ChartService
from mcp_finance.db.engine import get_session


def register_charting_tools(mcp: MCPServer) -> None:
    """Register charting MCP tools with the server."""

    @mcp.tool()
    async def get_stock_chart(input: GetStockChartInput) -> ImageContent:
        """Render a technical candlestick chart with EMA overlays for a symbol.

        Generates a publication-quality candlestick chart (with volume subplot and
        EMA-20, EMA-50, and EMA-200 overlays) for visual analysis by the Vision
        Language Model (VLM).

        Data is sourced exclusively from the local Postgres OHLCV cache. If data
        is missing, an explicit error is returned instructing the caller to ingest
        market data first.

        Args:
            symbol: Ticker symbol in standard yfinance format (e.g. AAPL, MSFT, SAP.DE).
            lookback_days: Number of trading days to display on chart
                (min 10, max 365, default 90).

        Returns:
            ImageContent with base64-encoded PNG image data.
        """
        async with get_session() as session:
            service = ChartService(session)
            png_bytes = await service.get_chart(
                symbol=input.symbol,
                lookback_days=input.lookback_days,
            )
            b64_data = base64.b64encode(png_bytes).decode("ascii")
            return ImageContent(
                type="image",
                data=b64_data,
                mimeType="image/png",
            )

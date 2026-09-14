"""Pydantic models for stock charting."""

from pydantic import BaseModel, Field


class GetStockChartInput(BaseModel):
    """Input parameters for the get_stock_chart MCP tool."""

    symbol: str = Field(
        ...,
        description=(
            "Ticker symbol in standard yfinance format (e.g. AAPL, MSFT, SAP.DE)."
        ),
    )
    lookback_days: int = Field(
        default=90,
        ge=10,
        le=365,
        description=(
            "Number of trading days to display on chart (min 10, max 365, default 90)."
        ),
    )

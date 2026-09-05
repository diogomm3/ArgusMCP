"""MarketDataClient Protocol and package exports."""

import datetime
from typing import Protocol

import pandas as pd

from mcp_finance.market_data.models import (
    BatchIngestResult,
    GetHistoryInput,
    GetSymbolPriceInput,
    OhlcvRecord,
    PriceQuote,
)


class MarketDataClient(Protocol):
    """Vendor-agnostic read interface for historical and quote market data."""

    async def get_ohlcv(
        self,
        symbol: str,
        start: datetime.date | str,
        end: datetime.date | str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return daily OHLCV bars as a DataFrame.

        The DataFrame must contain columns:
        'date', 'open', 'high', 'low', 'close', 'volume'.
        If no data is found, an empty DataFrame with these columns is returned.
        """
        ...

    async def get_current_price(self, symbol: str) -> PriceQuote:
        """Return latest informational price quote for a symbol."""
        ...


__all__ = [
    "BatchIngestResult",
    "GetHistoryInput",
    "GetSymbolPriceInput",
    "MarketDataClient",
    "OhlcvRecord",
    "PriceQuote",
]

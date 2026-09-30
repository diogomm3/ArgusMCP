"""YFinance market data client with non-blocking threads and tenacity retries."""

import asyncio
import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mcp_finance.logger import get_logger
from mcp_finance.market_data.models import PriceQuote
from mcp_finance.market_data.utils import (
    _clean_ohlcv_dataframe,
    _is_transient_network_error,
)

logger = get_logger(__name__)


class YFinanceClient:
    """MarketDataClient implementation backed by yfinance."""

    @retry(
        retry=retry_if_exception(_is_transient_network_error),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_history_sync(
        self,
        symbol: str,
        start_str: str,
        end_str: str,
        interval: str,
    ) -> pd.DataFrame:
        """Synchronous fetch wrapped with tenacity retry on transient errors."""
        ticker = yf.Ticker(symbol)
        raw_df = ticker.history(
            start=start_str, end=end_str, interval=interval, auto_adjust=False
        )
        return _clean_ohlcv_dataframe(raw_df, symbol)

    @retry(
        retry=retry_if_exception(_is_transient_network_error),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_quote_sync(self, symbol: str) -> PriceQuote:
        """Synchronous quote fetch wrapped with tenacity retry."""
        ticker = yf.Ticker(symbol)
        price: Decimal | None = None
        currency = "USD"

        try:
            fast_info: Any = ticker.fast_info
            last_price = getattr(fast_info, "last_price", None)
            if last_price is None and hasattr(fast_info, "get"):
                last_price = fast_info.get("last_price") or fast_info.get("lastPrice")
            if last_price is not None and str(last_price) != "nan":
                price = Decimal(str(round(float(last_price), 4)))
            cur = getattr(fast_info, "currency", None)
            if cur:
                currency = str(cur).upper()
        except Exception as exc:
            logger.debug(
                "fast_info quote lookup failed, falling back to history",
                symbol=symbol,
                error=str(exc),
            )

        if price is None:
            # Fallback to latest bar from 1d history
            hist = ticker.history(period="1d", auto_adjust=False)
            if not hist.empty and "Close" in hist.columns:
                last_val = hist["Close"].iloc[-1]
                if pd.notna(last_val):
                    price = Decimal(str(round(float(last_val), 4)))

        if price is None:
            raise ValueError(
                f"No price data available for symbol '{symbol}' from yfinance"
            )

        return PriceQuote(
            symbol=symbol,
            price=price,
            currency=currency,
            source="yfinance",
            is_execution_price=False,
        )

    async def get_ohlcv(
        self,
        symbol: str,
        start: datetime.date | str,
        end: datetime.date | str,
        interval: str = "1d",
        _now: datetime.datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars in [start, end] inclusive.

        The adapter treats ``end`` as an *inclusive* boundary.  It passes
        ``end + 1 day`` to yfinance (which uses an exclusive upper bound),
        capped at ``last_settled_session_date(_now) + 1 day``.  Any bar whose
        date exceeds ``last_settled_session_date`` is dropped as a second
        defence against partial in-progress bars being written to the cache.

        ``_now`` is injectable for testing (default: current system time).
        """
        from mcp_finance.market_data.utils import last_settled_session_date

        start_date = (
            datetime.date.fromisoformat(start) if isinstance(start, str) else start
        )
        end_date = datetime.date.fromisoformat(end) if isinstance(end, str) else end

        last_settled = last_settled_session_date(_now)
        # Cap inclusive end at last settled, then +1 for exclusive yfinance arg
        effective_end_exclusive = min(end_date, last_settled) + datetime.timedelta(
            days=1
        )
        start_str = start_date.isoformat()
        end_str = effective_end_exclusive.isoformat()

        df = await asyncio.to_thread(
            self._fetch_history_sync,
            symbol,
            start_str,
            end_str,
            interval,
        )

        # Second defence: drop any row dated after last_settled (partial bar guard)
        if not df.empty and "date" in df.columns:
            df = df[df["date"] <= last_settled].reset_index(drop=True)

        return df

    async def get_current_price(self, symbol: str) -> PriceQuote:
        """Asynchronously fetch informational quote in a worker thread."""
        return await asyncio.to_thread(
            self._fetch_quote_sync,
            symbol,
        )

    async def aclose(self) -> None:
        """Close any allocated resources."""
        pass

    async def __aenter__(self) -> "YFinanceClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

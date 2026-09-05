"""YFinance market data client with non-blocking threads and tenacity retries."""

import asyncio
import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mcp_finance.logger import get_logger
from mcp_finance.market_data.models import PriceQuote

logger = get_logger(__name__)


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True if exception is a transient network/HTTP error suitable for retry."""
    if isinstance(
        exc, (requests.RequestException, ConnectionError, TimeoutError, OSError)
    ):
        return True
    msg = str(exc).lower()
    return (
        "too many requests" in msg
        or "rate limit" in msg
        or "429" in msg
        or "503" in msg
    )


def _clean_ohlcv_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize yfinance DataFrame into standard columns with Date objects."""
    expected_cols = ["date", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        logger.warning("yfinance returned no OHLCV data for symbol", symbol=symbol)
        return pd.DataFrame(columns=expected_cols)

    # Reset index if Date is the index
    if "Date" in df.columns or "Datetime" in df.columns:
        work_df = df.copy()
    else:
        work_df = df.reset_index()

    # Normalize column names to lowercase
    rename_map: dict[str, str] = {}
    for col in work_df.columns:
        col_str = str(col).lower()
        if col_str in ("date", "datetime"):
            rename_map[str(col)] = "date"
        elif (
            col_str == "index"
            and "date" not in work_df.columns
            and "Date" not in work_df.columns
        ):
            rename_map[str(col)] = "date"
        elif col_str == "open":
            rename_map[str(col)] = "open"
        elif col_str == "high":
            rename_map[str(col)] = "high"
        elif col_str == "low":
            rename_map[str(col)] = "low"
        elif col_str == "close":
            rename_map[str(col)] = "close"
        elif col_str == "volume":
            rename_map[str(col)] = "volume"

    work_df = work_df.rename(columns=rename_map)

    missing = [col for col in expected_cols if col not in work_df.columns]
    if missing:
        logger.warning(
            "yfinance dataframe missing expected columns",
            symbol=symbol,
            missing=missing,
            columns=list(work_df.columns),
        )
        return pd.DataFrame(columns=expected_cols)

    work_df = work_df[expected_cols].dropna(subset=["open", "high", "low", "close"])

    # Convert timestamps to datetime.date
    if not work_df.empty:
        work_df["date"] = pd.to_datetime(work_df["date"]).dt.date

    return work_df.reset_index(drop=True)


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
    ) -> pd.DataFrame:
        """Asynchronously fetch OHLCV bars in a worker thread."""
        start_str = (
            start.isoformat() if isinstance(start, datetime.date) else str(start)
        )
        end_str = end.isoformat() if isinstance(end, datetime.date) else str(end)

        return await asyncio.to_thread(
            self._fetch_history_sync,
            symbol,
            start_str,
            end_str,
            interval,
        )

    async def get_current_price(self, symbol: str) -> PriceQuote:
        """Asynchronously fetch informational quote in a worker thread."""
        return await asyncio.to_thread(
            self._fetch_quote_sync,
            symbol,
        )

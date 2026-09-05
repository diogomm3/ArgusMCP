"""Utilities for market data formatting, exchange derivation, and symbol parsing."""

from typing import TYPE_CHECKING

import pandas as pd
import requests

from mcp_finance.logger import get_logger

if TYPE_CHECKING:
    from mcp_finance.market_data.yfinance import YFinanceClient

logger = get_logger(__name__)

_SUFFIX_EXCHANGE_MAP: dict[str, str] = {
    ".L": "LSE",
    ".IL": "LSE",
    ".DE": "XETRA",
    ".PA": "EURONEXT_PARIS",
    ".AS": "EURONEXT_AMSTERDAM",
    ".BR": "EURONEXT_BRUSSELS",
    ".MI": "BORSA_ITALIANA",
    ".MC": "BME",
    ".TO": "TSX",
    ".V": "TSX",
    ".SW": "SIX",
    ".ST": "OMXS",
    ".CO": "OMXC",
    ".HE": "OMXH",
    ".OL": "OSLO",
    ".HK": "HKEX",
}


def derive_exchange(ticker: str, explicit_exchange: str | None = None) -> str:
    """Derive canonical exchange code from a ticker or explicit parameter.

    yfinance encodes international listings via ticker suffixes
    (e.g., 'ASML.AS', 'SAP.DE'). If explicit_exchange is provided, it takes
    precedence. Otherwise, the exchange is inferred from known suffixes,
    defaulting to 'US' for standard tickers without suffix.
    """
    if explicit_exchange:
        return explicit_exchange.upper()

    upper_ticker = ticker.upper()
    for suffix, exchange in _SUFFIX_EXCHANGE_MAP.items():
        if upper_ticker.endswith(suffix):
            return exchange

    return "US"


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


def get_yfinance_client() -> "YFinanceClient":
    """Return a fresh YFinanceClient per call."""
    from mcp_finance.market_data.yfinance import YFinanceClient

    return YFinanceClient()

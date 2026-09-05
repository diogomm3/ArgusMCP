"""Utilities for market data formatting, exchange derivation, and symbol parsing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_finance.market_data.yfinance import YFinanceClient

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


def get_yfinance_client() -> "YFinanceClient":
    """Return a fresh YFinanceClient per call."""
    from mcp_finance.market_data.yfinance import YFinanceClient

    return YFinanceClient()

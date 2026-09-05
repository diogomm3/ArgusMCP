"""Unit tests for market data utils, YFinanceClient, and MCP tools (mocked)."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from mcp_finance.market_data.models import PriceQuote
from mcp_finance.market_data.utils import derive_exchange
from mcp_finance.market_data.yfinance import (
    YFinanceClient,
    _clean_ohlcv_dataframe,
    _is_transient_network_error,
)


@pytest.mark.unit
def test_derive_exchange() -> None:
    """Test canonical exchange derivation from ticker suffixes."""
    assert derive_exchange("AAPL") == "US"
    assert derive_exchange("MSFT") == "US"
    assert derive_exchange("ASML.AS") == "EURONEXT_AMSTERDAM"
    assert derive_exchange("SAP.DE") == "XETRA"
    assert derive_exchange("BP.L") == "LSE"
    assert derive_exchange("MC.PA") == "EURONEXT_PARIS"
    assert derive_exchange("ENI.MI") == "BORSA_ITALIANA"
    assert derive_exchange("IBE.MC") == "BME"
    assert derive_exchange("SHOP.TO") == "TSX"

    # Explicit override takes precedence
    assert derive_exchange("AAPL", "NASDAQ") == "NASDAQ"
    assert derive_exchange("ASML.AS", "CUSTOM") == "CUSTOM"


@pytest.mark.unit
def test_is_transient_network_error() -> None:
    """Test retry predicate for transient network and rate limit errors."""
    assert _is_transient_network_error(
        requests.exceptions.ConnectionError("Connection dropped")
    )
    assert _is_transient_network_error(requests.exceptions.Timeout("Read timeout"))
    assert _is_transient_network_error(Exception("429 Too Many Requests"))
    assert _is_transient_network_error(Exception("Rate limit reached"))
    assert not _is_transient_network_error(ValueError("Invalid format"))
    assert not _is_transient_network_error(KeyError("missing_col"))


@pytest.mark.unit
def test_clean_ohlcv_dataframe() -> None:
    """Test DataFrame sanitization and normalization."""
    raw_df = pd.DataFrame(
        {
            "Open": [150.0, 152.0],
            "High": [155.0, 156.0],
            "Low": [149.0, 151.0],
            "Close": [154.0, 153.0],
            "Volume": [1000000, 1200000],
        },
        index=pd.to_datetime(["2026-09-01", "2026-09-02"]),
    )
    cleaned = _clean_ohlcv_dataframe(raw_df, "AAPL")
    assert list(cleaned.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(cleaned) == 2
    assert cleaned["date"].iloc[0] == datetime.date(2026, 9, 1)
    assert cleaned["open"].iloc[0] == 150.0


@pytest.mark.unit
def test_clean_ohlcv_dataframe_empty() -> None:
    """Empty or None DataFrame returns empty DataFrame with expected columns."""
    cleaned = _clean_ohlcv_dataframe(pd.DataFrame(), "INVALID")
    assert list(cleaned.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert cleaned.empty


@pytest.mark.unit
@pytest.mark.asyncio
async def test_yfinance_client_get_ohlcv_success() -> None:
    """Test get_ohlcv succeeds with sanitized DataFrame."""
    client = YFinanceClient()
    sample_df = pd.DataFrame(
        {
            "Open": [100.5],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Volume": [500000],
        },
        index=pd.to_datetime(["2026-09-01"]),
    )

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_instance

        df = await client.get_ohlcv("AAPL", "2026-09-01", "2026-09-02")
        assert not df.empty
        assert len(df) == 1
        assert df["close"].iloc[0] == 104.0
        assert df["date"].iloc[0] == datetime.date(2026, 9, 1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_yfinance_client_empty_data_not_retried() -> None:
    """Test empty DataFrame does not raise and is not retried as an error."""
    client = YFinanceClient()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        df = await client.get_ohlcv("UNKNOWN", "2026-09-01", "2026-09-02")
        assert df.empty
        assert mock_instance.history.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_yfinance_client_retry_on_network_error() -> None:
    """Test transient network error triggers retry with tenacity."""
    client = YFinanceClient()
    sample_df = pd.DataFrame(
        {
            "Open": [200.0],
            "High": [205.0],
            "Low": [199.0],
            "Close": [202.0],
            "Volume": [100000],
        },
        index=pd.to_datetime(["2026-09-01"]),
    )

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.history.side_effect = [
            requests.exceptions.ConnectionError("Network dropped"),
            sample_df,
        ]
        mock_ticker_cls.return_value = mock_instance

        df = await client.get_ohlcv("AAPL", "2026-09-01", "2026-09-02")
        assert not df.empty
        assert mock_instance.history.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_yfinance_client_get_current_price_fast_info() -> None:
    """Test get_current_price extracts price and currency from fast_info."""
    client = YFinanceClient()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 224.50
        mock_fast_info.currency = "USD"
        mock_instance.fast_info = mock_fast_info
        mock_ticker_cls.return_value = mock_instance

        quote = await client.get_current_price("AAPL")
        assert isinstance(quote, PriceQuote)
        assert quote.symbol == "AAPL"
        assert quote.price == Decimal("224.5")
        assert quote.currency == "USD"
        assert quote.is_execution_price is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_yfinance_client_get_current_price_history_fallback() -> None:
    """Test get_current_price falls back to history when fast_info is unavailable."""
    client = YFinanceClient()

    with patch("yfinance.Ticker") as mock_ticker_cls:
        mock_instance = MagicMock()
        mock_instance.fast_info = None
        mock_instance.history.return_value = pd.DataFrame(
            {"Close": [180.25]},
            index=pd.to_datetime(["2026-09-01"]),
        )
        mock_ticker_cls.return_value = mock_instance

        quote = await client.get_current_price("AAPL")
        assert quote.price == Decimal("180.25")
        assert quote.symbol == "AAPL"


@pytest.mark.unit
def test_market_data_tools_registered() -> None:
    """Test that market data tools are properly registered on FastMCP."""
    from mcp_finance.server import mcp

    # Access FastMCP internal tool list
    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "get_symbol_price" in tool_names
    assert "get_history" in tool_names
    assert "get_positions" in tool_names

"""Unit tests for Phase 8 charting: renderer, service helpers, and MCP tool.

All tests are pure (no database, no network). Synthetic OHLCV DataFrames are
built inline. Marked pytest.mark.unit so they run in CI without any infra.
"""

import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mcp_finance.charting.models import GetStockChartInput
from mcp_finance.charting.renderer import render_chart
from mcp_finance.charting.service import prepare_chart_dataframe

# Force Agg backend before anything else (mirrors production code)
matplotlib.use("Agg")

# ── Helpers ──────────────────────────────────────────────────────────────────

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_ohlcv_df(n: int = 120, base_price: float = 150.0) -> pd.DataFrame:
    """Build a minimal lowercase OHLCV DataFrame suitable for charting tests."""
    today = datetime.date.today()
    dates = [today - datetime.timedelta(days=n - 1 - i) for i in range(n)]
    close = np.linspace(base_price, base_price + n * 0.5, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.full(n, 1_500_000, dtype=float),
        }
    )


def _make_plot_df(n: int = 120) -> pd.DataFrame:
    """Build a capitalized mplfinance-ready DataFrame (post prepare_chart_dataframe)."""
    raw = _make_ohlcv_df(n)
    return prepare_chart_dataframe(raw)


# ── prepare_chart_dataframe ──────────────────────────────────────────────────


@pytest.mark.unit
def test_prepare_chart_dataframe_columns() -> None:
    """Capitalized OHLCV columns are present after normalization."""
    raw = _make_ohlcv_df(30)
    plot_df = prepare_chart_dataframe(raw)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        assert col in plot_df.columns, f"Missing column: {col}"


@pytest.mark.unit
def test_prepare_chart_dataframe_index_is_datetime() -> None:
    """The index is a DatetimeIndex sorted ascending."""
    raw = _make_ohlcv_df(30)
    plot_df = prepare_chart_dataframe(raw)
    assert isinstance(plot_df.index, pd.DatetimeIndex)
    assert plot_df.index.is_monotonic_increasing


@pytest.mark.unit
def test_prepare_chart_dataframe_no_date_column() -> None:
    """After normalization the lowercase 'date' column is gone from the frame."""
    raw = _make_ohlcv_df(30)
    plot_df = prepare_chart_dataframe(raw)
    assert "date" not in plot_df.columns


# ── render_chart ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_render_chart_returns_png_bytes() -> None:
    """render_chart returns bytes with a valid PNG header."""
    plot_df = _make_plot_df(60)
    result = render_chart(plot_df, title="Test Chart")
    assert isinstance(result, bytes)
    assert result[:8] == _PNG_MAGIC, "Return value is not a valid PNG"


@pytest.mark.unit
def test_render_chart_size_reasonable() -> None:
    """PNG output is at least 10 KB — a blank render would be much smaller."""
    plot_df = _make_plot_df(90)
    result = render_chart(plot_df, title="AAPL (US) - Daily")
    assert len(result) >= 10_000, f"PNG suspiciously small: {len(result)} bytes"


@pytest.mark.unit
def test_render_chart_no_title() -> None:
    """render_chart with no title still produces a valid PNG (not a crash)."""
    plot_df = _make_plot_df(30)
    result = render_chart(plot_df)
    assert result[:8] == _PNG_MAGIC


@pytest.mark.unit
def test_render_chart_with_ema_overlays() -> None:
    """EMA columns present in the DataFrame are included without error."""
    plot_df = _make_plot_df(120)
    # Compute simple EMAs via pandas — same formula used in service.py
    plot_df["EMA20"] = plot_df["Close"].ewm(span=20, adjust=False).mean()
    plot_df["EMA50"] = plot_df["Close"].ewm(span=50, adjust=False).mean()
    plot_df["EMA200"] = plot_df["Close"].ewm(span=200, adjust=False).mean()
    result = render_chart(plot_df, title="With EMAs")
    assert result[:8] == _PNG_MAGIC


@pytest.mark.unit
def test_render_chart_partial_ema_no_crash() -> None:
    """Only EMA20 present (EMA50/200 absent) — renderer handles partial overlays."""
    plot_df = _make_plot_df(30)
    plot_df["EMA20"] = plot_df["Close"].ewm(span=20, adjust=False).mean()
    result = render_chart(plot_df, title="Partial EMAs")
    assert result[:8] == _PNG_MAGIC


# ── Memory leak check ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_render_chart_no_figure_leak_20_iterations() -> None:
    """After 20 chart renders, zero matplotlib figure handles remain open.

    This directly tests the plt.close('all') inside render_chart's finally block.
    A regression here (missing close) would accumulate one handle per call,
    degrading the process over thousands of requests in production.
    """
    plot_df = _make_plot_df(60)
    for _ in range(20):
        render_chart(plot_df, title="Leak check")
    open_figs = len(plt.get_fignums())
    assert open_figs == 0, (
        f"Figure leak detected: {open_figs} figure(s) still open "
        "after 20 render_chart calls"
    )


# ── GetStockChartInput validation ────────────────────────────────────────────


@pytest.mark.unit
def test_input_model_defaults() -> None:
    """Default lookback_days is 90."""
    inp = GetStockChartInput(symbol="AAPL")
    assert inp.lookback_days == 90
    assert inp.symbol == "AAPL"


@pytest.mark.unit
def test_input_model_bounds_ge() -> None:
    """lookback_days < 10 is rejected."""
    with pytest.raises(Exception):
        GetStockChartInput(symbol="AAPL", lookback_days=9)


@pytest.mark.unit
def test_input_model_bounds_le() -> None:
    """lookback_days > 365 is rejected."""
    with pytest.raises(Exception):
        GetStockChartInput(symbol="AAPL", lookback_days=366)


@pytest.mark.unit
def test_input_model_boundary_values() -> None:
    """Boundary values 10 and 365 are accepted."""
    inp_min = GetStockChartInput(symbol="AAPL", lookback_days=10)
    inp_max = GetStockChartInput(symbol="AAPL", lookback_days=365)
    assert inp_min.lookback_days == 10
    assert inp_max.lookback_days == 365


# ── Tool registration ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_register_charting_tools_registers_get_stock_chart() -> None:
    """register_charting_tools adds 'get_stock_chart' to the MCP server's tools."""
    from typing import Callable
    from unittest.mock import MagicMock

    from mcp_finance.charting.charting_tools import register_charting_tools

    mcp = MagicMock()
    # mcp.tool() is used as a decorator — capture the registered functions
    registered_names: list[str] = []

    def fake_tool() -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorator(fn: Callable[..., object]) -> Callable[..., object]:
            registered_names.append(fn.__name__)
            return fn

        return decorator

    mcp.tool = fake_tool
    register_charting_tools(mcp)
    assert "get_stock_chart" in registered_names

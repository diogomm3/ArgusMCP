"""Unit tests for the Phase 7 screener filter functions.

All tests use synthetic Candidate and CompanyFundamentals fixtures — no database,
no network. Each test is a pure function call against evaluate_universe,
evaluate_volume, evaluate_trend, evaluate_momentum, and evaluate_candidate.

Marked with pytest.mark.unit so they run in CI without any infra.
"""

import datetime
from decimal import Decimal

import pytest

from mcp_finance.fundamentals.models import CompanyFundamentals
from mcp_finance.indicators.models import Candidate
from mcp_finance.screener.filters import (
    evaluate_candidate,
    evaluate_momentum,
    evaluate_trend,
    evaluate_universe,
    evaluate_volume,
)
from mcp_finance.screener.models import StrategyConfig

# ── Shared fixtures ──────────────────────────────────────────────────────────

TODAY = datetime.date.today()


def _make_candidate(**overrides: object) -> Candidate:
    """Build a Candidate that passes all default filters, unless overridden."""
    defaults: dict[str, object] = {
        "symbol": "AAPL",
        "as_of_date": TODAY,
        "source": "yfinance",
        "close": Decimal("175.00"),
        "ema_20": Decimal("165.00"),
        "ema_50": Decimal("155.00"),
        "rsi_14": Decimal("55.00"),
        "macd_line": Decimal("2.50"),
        "macd_signal": Decimal("1.80"),
        "macd_histogram": Decimal("0.70"),
        "volume": 1_500_000,
        "avg_volume_20": 1_200_000,
        "bars_available": 60,
    }
    defaults.update(overrides)
    return Candidate(**defaults)


def _make_fundamentals(**overrides: object) -> CompanyFundamentals:
    """Build CompanyFundamentals that passes all universe filters."""
    defaults: dict[str, object] = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "exchange": "NASDAQ",
        "currency": "USD",
        "market_cap": 3_000_000_000,  # $3B — above $2B default
        "pe_ratio": Decimal("28.5"),
        "pb_ratio": Decimal("45.0"),
        "ev_to_ebitda": Decimal("22.0"),
        "profit_margin": Decimal("0.25"),
        "debt_to_equity": Decimal("1.5"),
        "current_ratio": Decimal("1.2"),
        "dividend_yield": Decimal("0.005"),
        "free_cash_flow_per_share": Decimal("6.50"),
        "as_of_date": TODAY,
        "is_cached": True,
        "source": "fmp",
    }
    defaults.update(overrides)
    return CompanyFundamentals(**defaults)


# ── Universe filter ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_universe_filter_market_cap_pass() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(market_cap=3_000_000_000)
    result = evaluate_universe(candidate, fund, StrategyConfig())
    assert result.passed


@pytest.mark.unit
def test_universe_filter_market_cap_fail_below_threshold() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(market_cap=500_000_000)  # $500M < $2B
    result = evaluate_universe(candidate, fund, StrategyConfig())
    assert not result.passed
    assert "market_cap" in (result.reason or "")
    assert "min_market_cap" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_market_cap_fail_fundamentals_none() -> None:
    candidate = _make_candidate()
    result = evaluate_universe(candidate, None, StrategyConfig())
    assert not result.passed
    assert "fundamentals unavailable" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_market_cap_none_not_reported() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(market_cap=None)
    result = evaluate_universe(candidate, fund, StrategyConfig())
    assert not result.passed
    assert "not reported" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_min_price_pass() -> None:
    candidate = _make_candidate(close=Decimal("10.00"))
    fund = _make_fundamentals()
    config = StrategyConfig(min_price=Decimal("5.00"))
    result = evaluate_universe(candidate, fund, config)
    assert result.passed


@pytest.mark.unit
def test_universe_filter_min_price_fail() -> None:
    candidate = _make_candidate(close=Decimal("3.50"))
    fund = _make_fundamentals()
    config = StrategyConfig(min_price=Decimal("5.00"))
    result = evaluate_universe(candidate, fund, config)
    assert not result.passed
    assert "close" in (result.reason or "")
    assert "min_price" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_pe_ratio_pass() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(pe_ratio=Decimal("30.0"))
    config = StrategyConfig(max_pe_ratio=Decimal("40.0"))
    assert evaluate_universe(candidate, fund, config).passed


@pytest.mark.unit
def test_universe_filter_pe_ratio_fail_exceeds_max() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(pe_ratio=Decimal("55.0"))
    config = StrategyConfig(max_pe_ratio=Decimal("40.0"))
    result = evaluate_universe(candidate, fund, config)
    assert not result.passed
    assert "pe_ratio" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_pe_ratio_fail_negative() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals(pe_ratio=Decimal("-5.0"))
    config = StrategyConfig(max_pe_ratio=Decimal("40.0"))
    result = evaluate_universe(candidate, fund, config)
    assert not result.passed
    assert "loss-making" in (result.reason or "")


@pytest.mark.unit
def test_universe_filter_pe_disabled_passes_without_fundamentals() -> None:
    candidate = _make_candidate()
    # min_market_cap and max_pe both disabled — no fundamentals required
    config = StrategyConfig(min_market_cap=None, max_pe_ratio=None)
    result = evaluate_universe(candidate, None, config)
    assert result.passed


# ── Volume / Liquidity filter ─────────────────────────────────────────────────


@pytest.mark.unit
def test_volume_filter_pass() -> None:
    candidate = _make_candidate(avg_volume_20=200_000)
    assert evaluate_volume(candidate, StrategyConfig()).passed


@pytest.mark.unit
def test_volume_filter_fail_below_threshold() -> None:
    candidate = _make_candidate(avg_volume_20=50_000)
    result = evaluate_volume(candidate, StrategyConfig())
    assert not result.passed
    assert "avg_volume_20" in (result.reason or "")


@pytest.mark.unit
def test_volume_filter_fail_none_insufficient_bars() -> None:
    candidate = _make_candidate(avg_volume_20=None)
    result = evaluate_volume(candidate, StrategyConfig())
    assert not result.passed
    assert "None" in (result.reason or "")
    assert "20 bars" in (result.reason or "")


@pytest.mark.unit
def test_volume_filter_disabled() -> None:
    candidate = _make_candidate(avg_volume_20=0)
    result = evaluate_volume(candidate, StrategyConfig(min_avg_volume_20=None))
    assert result.passed


# ── Trend filter ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_trend_filter_golden_alignment_pass() -> None:
    candidate = _make_candidate(
        close=Decimal("175"),
        ema_20=Decimal("165"),
        ema_50=Decimal("155"),
    )
    assert evaluate_trend(candidate, StrategyConfig()).passed


@pytest.mark.unit
def test_trend_filter_fail_price_below_ema20() -> None:
    candidate = _make_candidate(
        close=Decimal("160"),
        ema_20=Decimal("165"),
        ema_50=Decimal("155"),
    )
    result = evaluate_trend(candidate, StrategyConfig())
    assert not result.passed
    assert "ema_20" in (result.reason or "")
    assert "short-term trend" in (result.reason or "")


@pytest.mark.unit
def test_trend_filter_fail_price_below_ema50() -> None:
    candidate = _make_candidate(
        close=Decimal("170"),
        ema_20=Decimal("175"),
        ema_50=Decimal("180"),
    )
    result = evaluate_trend(candidate, StrategyConfig())
    # price > EMA20 but close <= EMA50 should fail
    assert not result.passed


@pytest.mark.unit
def test_trend_filter_fail_ema20_below_ema50() -> None:
    candidate = _make_candidate(
        close=Decimal("190"),
        ema_20=Decimal("170"),
        ema_50=Decimal("185"),
    )
    result = evaluate_trend(candidate, StrategyConfig())
    assert not result.passed
    assert "golden cross" in (result.reason or "")


@pytest.mark.unit
def test_trend_filter_none_ema_fails() -> None:
    candidate = _make_candidate(ema_50=None)
    result = evaluate_trend(candidate, StrategyConfig())
    assert not result.passed
    assert "None" in (result.reason or "")


@pytest.mark.unit
def test_trend_filter_all_disabled() -> None:
    candidate = _make_candidate(ema_20=None, ema_50=None)
    config = StrategyConfig(
        require_price_above_ema_20=False,
        require_price_above_ema_50=False,
        require_ema_20_above_ema_50=False,
    )
    assert evaluate_trend(candidate, config).passed


# ── Momentum filter ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_momentum_filter_rsi_corridor_pass() -> None:
    candidate = _make_candidate(rsi_14=Decimal("55"))
    assert evaluate_momentum(candidate, StrategyConfig()).passed


@pytest.mark.unit
def test_momentum_filter_rsi_oversold_fail() -> None:
    candidate = _make_candidate(rsi_14=Decimal("30"))
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "rsi_min" in (result.reason or "")
    assert "oversold" in (result.reason or "")


@pytest.mark.unit
def test_momentum_filter_rsi_overbought_fail() -> None:
    candidate = _make_candidate(rsi_14=Decimal("75"))
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "rsi_max" in (result.reason or "")
    assert "overbought" in (result.reason or "")


@pytest.mark.unit
def test_momentum_filter_rsi_none_fails() -> None:
    candidate = _make_candidate(rsi_14=None)
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "rsi_14=None" in (result.reason or "")


@pytest.mark.unit
def test_momentum_filter_macd_histogram_positive_pass() -> None:
    candidate = _make_candidate(macd_histogram=Decimal("0.01"))
    assert evaluate_momentum(candidate, StrategyConfig()).passed


@pytest.mark.unit
def test_momentum_filter_macd_histogram_zero_fails() -> None:
    candidate = _make_candidate(macd_histogram=Decimal("0"))
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "macd_histogram" in (result.reason or "")


@pytest.mark.unit
def test_momentum_filter_macd_histogram_negative_fails() -> None:
    candidate = _make_candidate(macd_histogram=Decimal("-0.5"))
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed


@pytest.mark.unit
def test_momentum_filter_macd_line_below_signal_fails() -> None:
    candidate = _make_candidate(
        macd_line=Decimal("1.0"),
        macd_signal=Decimal("1.5"),
        macd_histogram=Decimal("0.5"),  # histogram positive but line < signal
    )
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "bearish MACD" in (result.reason or "")


@pytest.mark.unit
def test_momentum_filter_macd_none_fails() -> None:
    candidate = _make_candidate(macd_histogram=None, macd_line=None, macd_signal=None)
    result = evaluate_momentum(candidate, StrategyConfig())
    assert not result.passed
    assert "None" in (result.reason or "")


# ── evaluate_candidate aggregate ─────────────────────────────────────────────


@pytest.mark.unit
def test_evaluate_candidate_all_pass() -> None:
    candidate = _make_candidate()
    fund = _make_fundamentals()
    passed, failed_filters, reasons = evaluate_candidate(
        candidate, fund, StrategyConfig()
    )
    assert passed
    assert failed_filters == []
    assert reasons == []


@pytest.mark.unit
def test_evaluate_candidate_reports_all_failures() -> None:
    """All four layers fail — all four should appear in failed_filters."""
    candidate = _make_candidate(
        close=Decimal("2.00"),  # fails price filter (universe)
        avg_volume_20=1_000,  # fails volume
        ema_20=Decimal("10.00"),  # price > ema_20 but ema_50 is higher
        ema_50=Decimal("50.00"),  # fails trend (price below ema_50)
        rsi_14=Decimal("80"),  # fails momentum (overbought)
        macd_histogram=Decimal("-1"),
    )
    fund = _make_fundamentals()
    passed, failed_filters, reasons = evaluate_candidate(
        candidate, fund, StrategyConfig()
    )
    assert not passed
    # All four layers should be represented
    assert "universe" in failed_filters
    assert "volume" in failed_filters
    assert "trend" in failed_filters
    assert "momentum" in failed_filters
    assert len(reasons) == 4


@pytest.mark.unit
def test_missing_indicator_disqualifies_with_reason() -> None:
    """None EMA-50 disqualifies with an explicit non-empty reason."""
    candidate = _make_candidate(ema_50=None)
    fund = _make_fundamentals()
    passed, failed_filters, reasons = evaluate_candidate(
        candidate, fund, StrategyConfig()
    )
    assert not passed
    assert "trend" in failed_filters
    assert any("ema_50" in r or "None" in r for r in reasons)


@pytest.mark.unit
def test_strategy_config_custom_overrides() -> None:
    """Disabling filters lets a tiny illiquid stock pass."""
    candidate = _make_candidate(
        close=Decimal("1.00"),
        avg_volume_20=500,
    )
    config = StrategyConfig(
        min_market_cap=None,
        min_price=None,
        min_avg_volume_20=None,
    )
    passed, failed_filters, _ = evaluate_candidate(candidate, None, config)
    # universe and volume now disabled; only trend and momentum matter
    assert "universe" not in failed_filters
    assert "volume" not in failed_filters


@pytest.mark.unit
async def test_screener_tool_registered() -> None:
    """screen_stocks is registered on the MCP server."""
    from mcp_finance.server import mcp

    tool_names = [tool.name for tool in mcp._tool_manager.list_tools()]
    assert "screen_stocks" in tool_names

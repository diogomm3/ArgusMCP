"""Unit tests for the evaluate_trade FastMCP tool and handler.

Tests cover:
  - Tool registration on the MCPServer instance.
  - Happy-path approved trade with automated position sizing.
  - Explicit quantity handling.
  - Multi-rule rejection scenarios (inverted stop loss, duplicate position, SELL side).
  - Dry-run guarantee: ZERO database writes (no AuditService, no commits, no inserts).
  - Sector auto-population from cached fundamentals vs omission.
  - Earnings blackout omission (skipped) vs supplied (enforced).
  - Zero broker order submissions (place_order is never invoked).
"""

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.risk.models import (
    OrderSide,
    OrderType,
    RiskDecision,
)
from mcp_finance.risk.risk_tools import (
    EvaluateTradeInput,
    evaluate_trade_handler,
    register_risk_tools,
)

# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _mock_account(
    total: Decimal = Decimal("100000"),
    cash: Decimal = Decimal("50000"),
    invested: Decimal = Decimal("50000"),
) -> AccountSummary:
    return AccountSummary(
        cash=cash,
        invested=invested,
        result=Decimal("0"),
        total=total,
        currency="USD",
    )


def _mock_position(ticker: str, quantity: str, price: str) -> Position:
    return Position(
        ticker=ticker,
        quantity=Decimal(quantity),
        average_price=Decimal(price),
        current_price=Decimal(price),
        ppl=Decimal("0"),
        frontend_type="STOCK",
    )


def _mock_broker_client(
    account: AccountSummary | None = None,
    positions: list[Position] | None = None,
) -> AsyncMock:
    client = AsyncMock()
    client.get_account = AsyncMock(return_value=account or _mock_account())
    client.get_positions = AsyncMock(return_value=positions or [])
    client.place_order = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# 1. Tool Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_risk_tools_registers_evaluate_trade() -> None:
    """register_risk_tools registers evaluate_trade with the MCPServer."""
    mcp = MagicMock()
    registered: list[str] = []

    def fake_tool():  # type: ignore[no-untyped-def]
        def decorator(fn):  # type: ignore[no-untyped-def]
            registered.append(fn.__name__)
            return fn

        return decorator

    mcp.tool = fake_tool
    register_risk_tools(mcp)

    assert "evaluate_trade" in registered


# ---------------------------------------------------------------------------
# 2. Happy Path & Sizing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_approved_with_auto_sizing() -> None:
    """Valid trade with plenty of headroom is approved and sized automatically."""
    broker = _mock_broker_client(
        account=_mock_account(total=Decimal("100000"), cash=Decimal("40000")),
        positions=[],
    )

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),  # $10 risk/share (6.67% distance)
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert isinstance(decision, RiskDecision)
    assert decision.approved is True
    assert decision.symbol == "AAPL"
    assert decision.side == OrderSide.BUY
    assert decision.order_type == OrderType.MARKET
    assert decision.quantity > Decimal("0")
    # Risk budget = 1% of $100k = $1000; at $10 risk/share -> 100 shares.
    # Position cap = 5% of $100k = $5000 -> floor(5000 / 150) = 33 shares (binding).
    assert decision.quantity == Decimal("33")
    assert decision.estimated_cost == Decimal("4950")  # 33 * 150
    assert decision.risk_amount == Decimal("330")  # 33 * 10
    assert len(decision.rejection_reasons) == 0


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_explicit_quantity() -> None:
    """When quantity is explicitly provided, sizing uses the explicit count."""
    broker = _mock_broker_client(
        account=_mock_account(total=Decimal("100000"), cash=Decimal("40000")),
        positions=[],
    )

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        quantity=Decimal("10"),
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert decision.approved is True
    assert decision.quantity == Decimal("10")
    assert decision.estimated_cost == Decimal("1500")
    assert decision.risk_amount == Decimal("100")


# ---------------------------------------------------------------------------
# 3. Multi-rule Rejections
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_rejection_inverted_stop_loss() -> None:
    """Stop-loss at or above entry price fails Rule 1."""
    broker = _mock_broker_client()

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("155"),  # invalid
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert decision.approved is False
    assert any("strictly below entry price" in r for r in decision.rejection_reasons)


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_rejection_duplicate_position() -> None:
    """Holding an existing position in the proposed ticker fails Rule 2."""
    broker = _mock_broker_client(
        positions=[_mock_position("AAPL_US_EQ", "20", "150")],
    )

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert decision.approved is False
    assert any("already exists" in r for r in decision.rejection_reasons)


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_rejection_sell_order() -> None:
    """SELL orders are explicitly rejected in Phase 9 by Rule 0."""
    broker = _mock_broker_client()

    inp = EvaluateTradeInput(
        symbol="AAPL",
        side=OrderSide.SELL,
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert decision.approved is False
    assert any(
        "OrderSide.SELL is not supported" in r for r in decision.rejection_reasons
    )


# ---------------------------------------------------------------------------
# 4. Zero Database Writes Guarantee (Requirement 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_zero_database_writes_guarantee() -> None:
    """evaluate_trade must never perform database writes or audit logging."""
    broker = _mock_broker_client()

    # Create a mock session to inspect all database operations
    mock_session = AsyncMock()
    # execute returns scalar_one_or_none = None for cache lookups
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
    )

    # Patch AuditService to assert it is never even instantiated
    with patch("mcp_finance.risk.audit.AuditService") as mock_audit_cls:
        decision = await evaluate_trade_handler(
            inp,
            broker_client=broker,
            session=mock_session,
        )

        assert decision.approved is True
        # 1. AuditService must never be called or instantiated
        mock_audit_cls.assert_not_called()

        # 2. Session write methods must NEVER be called
        mock_session.commit.assert_not_called()
        mock_session.rollback.assert_not_called()
        mock_session.flush.assert_not_called()

        # 3. Broker place_order must NEVER be called
        broker.place_order.assert_not_called()


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_zero_writes_on_rejection() -> None:
    """Rejected trades must NOT write audit records during evaluate_trade (dry run)."""
    broker = _mock_broker_client()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    inp = EvaluateTradeInput(
        symbol="AAPL",
        side=OrderSide.SELL,  # Rejected by Rule 0
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
    )

    with patch("mcp_finance.risk.audit.AuditService") as mock_audit_cls:
        decision = await evaluate_trade_handler(
            inp,
            broker_client=broker,
            session=mock_session,
        )

        assert decision.approved is False
        mock_audit_cls.assert_not_called()
        mock_session.commit.assert_not_called()
        broker.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Sector Auto-Population & Omission (Requirement 1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_auto_populates_sector_from_cache() -> None:
    """When sector is omitted, evaluate_trade queries local fundamentals cache."""
    broker = _mock_broker_client(
        account=_mock_account(total=Decimal("100000"), cash=Decimal("50000")),
        positions=[],
    )

    # Mock session returning a cached payload with profile.sector = "Technology"
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = {
        "profile": {"sector": "Technology", "industry": "Consumer Electronics"}
    }
    mock_session.execute = AsyncMock(return_value=mock_result)

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        sector=None,  # omitted!
    )

    decision = await evaluate_trade_handler(
        inp,
        broker_client=broker,
        session=mock_session,
    )

    assert decision.approved is True
    # Verify Rule 5 was evaluated with the auto-populated sector
    r5 = next(
        r for r in decision.rule_results if r.rule_name == "check_max_sector_exposure"
    )
    assert r5.passed is True
    assert r5.details.get("sector") == "Technology"


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_sector_omitted_cache_miss_skips_check() -> None:
    """When sector is omitted and no cache exists, sector check skips gracefully."""
    broker = _mock_broker_client()
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        sector=None,
    )

    decision = await evaluate_trade_handler(
        inp,
        broker_client=broker,
        session=mock_session,
    )

    assert decision.approved is True
    r5 = next(
        r for r in decision.rule_results if r.rule_name == "check_max_sector_exposure"
    )
    assert r5.passed is True
    assert "sector-exposure check skipped" in r5.reason


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_explicit_sector_overrides_cache() -> None:
    """Explicit sector is used directly without querying cache."""
    broker = _mock_broker_client()
    mock_session = AsyncMock()

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        sector="Healthcare",  # caller explicit
    )

    decision = await evaluate_trade_handler(
        inp,
        broker_client=broker,
        session=mock_session,
    )

    assert decision.approved is True
    r5 = next(
        r for r in decision.rule_results if r.rule_name == "check_max_sector_exposure"
    )
    assert r5.details.get("sector") == "Healthcare"


# ---------------------------------------------------------------------------
# 6. Earnings Blackout Omission vs Enforced
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_earnings_omitted_skips_blackout() -> None:
    """When next_earnings_date is omitted, earnings blackout rule skips cleanly."""
    broker = _mock_broker_client()

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        next_earnings_date=None,
    )

    decision = await evaluate_trade_handler(inp, broker_client=broker)

    assert decision.approved is True
    r_eb = next(
        r for r in decision.rule_results if r.rule_name == "check_earnings_blackout"
    )
    assert r_eb.passed is True
    assert "No earnings date provided" in r_eb.reason


@pytest.mark.unit
@pytest.mark.anyio
async def test_evaluate_trade_earnings_within_blackout_rejects() -> None:
    """When next_earnings_date is within blackout window, trade is rejected."""
    broker = _mock_broker_client()
    ref_today = datetime.date(2026, 9, 15)

    inp = EvaluateTradeInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        next_earnings_date=datetime.date(2026, 9, 17),  # 2 days away <= 3 days blackout
    )

    decision = await evaluate_trade_handler(
        inp,
        broker_client=broker,
        today=ref_today,
    )

    assert decision.approved is False
    assert any("blackout" in r.lower() for r in decision.rejection_reasons)

"""Unit tests for the place_order FastMCP tool and execution gate.

Tests cover:
  - Tool registration on MCPServer.
  - Rejection path: durable REJECTED audit record, zero broker calls.
  - Happy path: SUBMITTING -> broker call -> ACCEPTED audit update.
  - Broker failure durability: SUBMITTING -> broker exception -> FAILED audit update,
    with exception surfaced to caller (not swallowed).
  - Defense-in-depth invariant: defeating the invariant (corrupted decision with
    quantity=0, non-empty rejection reasons, or SELL side) raises RuntimeError
    and halts execution before any broker dispatch.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_finance.brokers.models import AccountSummary, OrderResult
from mcp_finance.risk.models import (
    AuditStatus,
    OrderSide,
    OrderType,
    PlaceOrderInput,
    PlaceOrderOutput,
    RiskDecision,
)
from mcp_finance.risk.risk_tools import (
    assert_trade_approved_invariant,
    place_order_handler,
    register_risk_tools,
)

# ---------------------------------------------------------------------------
# Fixtures & Helpers
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


def _mock_broker(
    order_id: str = "ord-12345",
    raises_on_place: Exception | None = None,
) -> AsyncMock:
    broker = AsyncMock()
    broker.get_account = AsyncMock(return_value=_mock_account())
    broker.get_positions = AsyncMock(return_value=[])
    if raises_on_place is not None:
        broker.place_order = AsyncMock(side_effect=raises_on_place)
    else:
        broker.place_order = AsyncMock(
            return_value=OrderResult(
                id=order_id,
                status="ACCEPTED",
                ticker="AAPL_US_EQ",
                quantity=Decimal("33"),
                order_type="MARKET",
            )
        )
    return broker


def _mock_audit_service() -> AsyncMock:
    audit = AsyncMock()
    audit.log_rejection = AsyncMock(return_value=101)
    audit.log_pre_dispatch = AsyncMock(return_value=102)
    audit.log_post_dispatch = AsyncMock(return_value=None)
    return audit


# ---------------------------------------------------------------------------
# 1. Tool Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_risk_tools_registers_place_order() -> None:
    """register_risk_tools registers both evaluate_trade and place_order."""
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
    assert "place_order" in registered


# ---------------------------------------------------------------------------
# 2. Rejection Path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_place_order_rejected_by_risk_engine() -> None:
    """When a trade fails risk checks, REJECTED is audited; broker is never called."""
    broker = _mock_broker()
    audit = _mock_audit_service()

    inp = PlaceOrderInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("155"),  # Invalid: stop loss >= entry
    )

    output = await place_order_handler(
        inp,
        broker_client=broker,
        audit_service=audit,
    )

    assert isinstance(output, PlaceOrderOutput)
    assert output.success is False
    assert output.decision.approved is False
    assert output.audit_id == 101
    assert output.broker_order_id is None
    assert output.order_result is None
    assert "rejected by risk engine" in (output.error_message or "")

    # Durable audit logged
    audit.log_rejection.assert_awaited_once()
    audit.log_pre_dispatch.assert_not_called()
    audit.log_post_dispatch.assert_not_called()

    # Zero broker dispatch
    broker.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Happy Path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_place_order_approved_full_lifecycle() -> None:
    """Approved order logs SUBMITTING, dispatches to broker, updates to ACCEPTED."""
    broker = _mock_broker(order_id="ord-abc-999")
    audit = _mock_audit_service()

    inp = PlaceOrderInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        quantity=Decimal("10"),
    )

    output = await place_order_handler(
        inp,
        broker_client=broker,
        audit_service=audit,
    )

    assert output.success is True
    assert output.decision.approved is True
    assert output.audit_id == 102
    assert output.broker_order_id == "ord-abc-999"
    assert output.order_result is not None
    assert output.order_result.id == "ord-abc-999"

    # Pre-dispatch logged with SUBMITTING
    audit.log_pre_dispatch.assert_awaited_once_with(output.decision)

    # Broker called with resolved ticker
    broker.place_order.assert_awaited_once_with(
        ticker="AAPL_US_EQ",
        quantity=Decimal("10"),
        order_type="MARKET",
        limit_price=None,
    )

    # Post-dispatch logged with ACCEPTED
    audit.log_post_dispatch.assert_awaited_once_with(
        audit_id=102,
        status=AuditStatus.ACCEPTED,
        broker_order_id="ord-abc-999",
        raw_response=output.order_result.model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# 4. Broker Failure Durability (Requirement 1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.anyio
async def test_place_order_broker_failure_updates_audit_to_failed_and_raises() -> None:
    """Under broker failure: SUBMITTING logged, FAILED updated, exception surfaced."""
    broker_err = RuntimeError("Trading212 gateway timeout (HTTP 504)")
    broker = _mock_broker(raises_on_place=broker_err)
    audit = _mock_audit_service()

    inp = PlaceOrderInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        quantity=Decimal("10"),
    )

    # Exception must NOT be swallowed — caller must see the failure
    with pytest.raises(RuntimeError, match="Trading212 gateway timeout"):
        await place_order_handler(
            inp,
            broker_client=broker,
            audit_service=audit,
        )

    # 1. Pre-dispatch SUBMITTING was logged
    audit.log_pre_dispatch.assert_awaited_once()

    # 2. Broker was called
    broker.place_order.assert_awaited_once()

    # 3. Post-dispatch was updated to FAILED with error payload
    audit.log_post_dispatch.assert_awaited_once_with(
        audit_id=102,
        status=AuditStatus.FAILED,
        broker_order_id=None,
        raw_response={
            "error": "Trading212 gateway timeout (HTTP 504)",
            "error_type": "RuntimeError",
        },
    )


# ---------------------------------------------------------------------------
# 5. Defense-in-Depth Invariant Tests (Requirement 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assert_trade_approved_invariant_rejects_unapproved() -> None:
    """assert_trade_approved_invariant refuses approved=False."""
    decision = RiskDecision(
        approved=False,
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal("10"),
        estimated_cost=Decimal("1000"),
        risk_amount=Decimal("50"),
        rule_results=[],
        rejection_reasons=["Stop loss violated"],
    )
    with pytest.raises(RuntimeError, match="attempt to place an unapproved order"):
        assert_trade_approved_invariant(decision)


@pytest.mark.unit
def test_assert_trade_approved_invariant_rejects_zero_quantity() -> None:
    """assert_trade_approved_invariant refuses quantity <= 0 even if approved=True."""
    decision = RiskDecision(
        approved=True,
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal("0"),  # Corrupted zero quantity!
        estimated_cost=Decimal("0"),
        risk_amount=Decimal("0"),
        rule_results=[],
        rejection_reasons=[],
    )
    with pytest.raises(RuntimeError, match="non-positive execution quantity"):
        assert_trade_approved_invariant(decision)


@pytest.mark.unit
def test_assert_trade_approved_invariant_rejects_non_empty_rejection_reasons() -> None:
    """assert_trade_approved_invariant refuses non-empty rejections even if approved."""

    decision = RiskDecision(
        approved=True,  # Inconsistent flag
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal("10"),
        estimated_cost=Decimal("1000"),
        risk_amount=Decimal("50"),
        rule_results=[],
        rejection_reasons=["Leftover rejection from corrupt rule"],
    )
    with pytest.raises(RuntimeError, match="order carries rejection reasons"):
        assert_trade_approved_invariant(decision)


@pytest.mark.unit
def test_assert_trade_approved_invariant_rejects_sell_side() -> None:
    """assert_trade_approved_invariant refuses SELL side in Phase 9."""
    decision = RiskDecision(
        approved=True,
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal("10"),
        estimated_cost=Decimal("1000"),
        risk_amount=Decimal("50"),
        rule_results=[],
        rejection_reasons=[],
    )
    with pytest.raises(RuntimeError, match="unsupported order side"):
        assert_trade_approved_invariant(decision)


@pytest.mark.unit
@pytest.mark.anyio
async def test_place_order_handler_refuses_corrupted_decision_before_broker() -> None:
    """Corrupted decision from evaluate_trade is caught by invariant before dispatch."""
    broker = _mock_broker()
    audit = _mock_audit_service()

    # Corrupted decision: approved=True, but quantity=0
    corrupted_decision = RiskDecision(
        approved=True,
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
        quantity=Decimal("0"),  # Defeats evaluation
        estimated_cost=Decimal("0"),
        risk_amount=Decimal("0"),
        rule_results=[],
        rejection_reasons=[],
    )

    inp = PlaceOrderInput(
        symbol="AAPL",
        entry_price=Decimal("150"),
        stop_loss_price=Decimal("140"),
    )

    # Mock evaluate_trade_handler to return this corrupted decision
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "mcp_finance.risk.risk_tools.evaluate_trade_handler",
            AsyncMock(return_value=corrupted_decision),
        )

        with pytest.raises(RuntimeError, match="non-positive execution quantity"):
            await place_order_handler(
                inp,
                broker_client=broker,
                audit_service=audit,
            )

        # Invariant halted execution: NO pre-dispatch and NO broker call
        audit.log_pre_dispatch.assert_not_called()
        broker.place_order.assert_not_called()

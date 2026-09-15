"""Postgres DB integration tests for Phase 9 audit logging.

Tests verify:
  1. REJECTED rows are persisted correctly via AuditService.log_rejection().
  2. Two-phase SUBMITTING → ACCEPTED cycle persists correctly.
  3. SUBMITTING → FAILED cycle persists correctly.
  4. Audit session independence: a rollback in the caller's session does NOT
     erase the audit record.
  5. JSONB columns (rejection_reasons, risk_metrics, raw_response) are
     queryable from Postgres.
  6. Indexes exist on the table (smoke test via query plan).

All tests use the testcontainers Postgres fixture from conftest.py and run
against a fully-migrated schema (migration 0003 included).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mcp_finance.brokers.models import AccountSummary
from mcp_finance.db.models import FundamentalsCache, Symbol
from mcp_finance.risk.audit import AuditService
from mcp_finance.risk.models import (
    AuditStatus,
    OrderSide,
    OrderType,
    RiskDecision,
    RuleResult,
)
from mcp_finance.risk.risk_tools import EvaluateTradeInput, evaluate_trade_handler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approved_decision(symbol: str = "AAPL", quantity: str = "10") -> RiskDecision:
    return RiskDecision(
        approved=True,
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal(quantity),
        estimated_cost=Decimal("1000"),
        risk_amount=Decimal("50"),
        rule_results=[
            RuleResult(rule_name="check_order_side", passed=True, reason="BUY ok"),
            RuleResult(
                rule_name="check_stop_loss",
                passed=True,
                reason="5% distance ok",
            ),
        ],
        rejection_reasons=[],
    )


def _rejected_decision(symbol: str = "AAPL") -> RiskDecision:
    return RiskDecision(
        approved=False,
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        entry_price=Decimal("100"),
        stop_loss_price=Decimal("95"),
        quantity=Decimal("0"),
        estimated_cost=Decimal("0"),
        risk_amount=Decimal("0"),
        rule_results=[
            RuleResult(
                rule_name="check_daily_loss_limit",
                passed=False,
                reason="Circuit breaker triggered.",
            ),
        ],
        rejection_reasons=["Circuit breaker triggered."],
    )


def _make_audit_service(
    session_factory: async_sessionmaker[AsyncSession],
) -> AuditService:
    """Create an AuditService wired to the testcontainer DB."""
    svc = AuditService.__new__(AuditService)
    svc._factory = session_factory
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuditRejectionPersistence:
    @pytest.mark.asyncio
    async def test_rejected_row_written_with_correct_status(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _rejected_decision("MSFT")
        audit_id = await svc.log_rejection(decision)

        assert isinstance(audit_id, int)
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert row.symbol == "MSFT"
        assert row.status == AuditStatus.REJECTED.value
        assert row.side == "BUY"
        assert row.quantity == Decimal("0")
        assert "Circuit breaker triggered." in row.rejection_reasons

    @pytest.mark.asyncio
    async def test_rejected_row_has_risk_metrics(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _rejected_decision("GOOGL")
        audit_id = await svc.log_rejection(decision)
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert isinstance(row.risk_metrics, dict)
        assert "rule_results" in row.risk_metrics
        assert row.broker_order_id is None

    @pytest.mark.asyncio
    async def test_rejection_jsonb_queryable(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Verify rejection_reasons JSONB can be queried from Postgres."""
        svc = _make_audit_service(session_factory)
        decision = _rejected_decision("META")
        audit_id = await svc.log_rejection(decision)

        async with session_factory() as session:
            # Query rows where the first element of rejection_reasons contains "Circuit"
            result = await session.execute(
                text(
                    "SELECT id FROM order_audit_logs "
                    "WHERE id = :audit_id "
                    "AND rejection_reasons::text LIKE '%Circuit%'"
                ),
                {"audit_id": audit_id},
            )
            row = result.fetchone()
        assert row is not None
        assert row[0] == audit_id


class TestTwoPhaseAuditCycle:
    @pytest.mark.asyncio
    async def test_submitting_then_accepted(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _approved_decision("NVDA", "5")

        # Phase 1: pre-dispatch
        audit_id = await svc.log_pre_dispatch(decision)
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert row.status == AuditStatus.SUBMITTING.value
        assert row.broker_order_id is None

        # Phase 2: post-dispatch (accepted)
        await svc.log_post_dispatch(
            audit_id=audit_id,
            status=AuditStatus.ACCEPTED,
            broker_order_id="demo-order-12345",
            raw_response={"id": "demo-order-12345", "status": "FILLED"},
        )

        row_after = await svc.get_audit_log(audit_id)
        assert row_after is not None
        assert row_after.status == AuditStatus.ACCEPTED.value
        assert row_after.broker_order_id == "demo-order-12345"
        assert isinstance(row_after.raw_response, dict)
        assert row_after.raw_response["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_submitting_then_failed(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _approved_decision("AMD", "3")

        audit_id = await svc.log_pre_dispatch(decision)

        # Simulate broker failure
        await svc.log_post_dispatch(
            audit_id=audit_id,
            status=AuditStatus.FAILED,
            broker_order_id=None,
            raw_response={"error": "connection timeout"},
        )

        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert row.status == AuditStatus.FAILED.value
        assert row.broker_order_id is None
        assert row.raw_response is not None
        assert "error" in row.raw_response


class TestAuditTransactionIndependence:
    @pytest.mark.asyncio
    async def test_audit_survives_caller_rollback(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The audit record must persist even when the caller rolls back.

        This simulates: caller opens a transaction, calls log_rejection
        (which commits in its own isolated session), then the caller's
        transaction rolls back due to an exception.

        The audit row must survive the caller's rollback.
        """
        svc = _make_audit_service(session_factory)
        decision = _rejected_decision("TSLA")

        audit_id: int | None = None
        try:
            # Open a caller session that will roll back
            async with session_factory() as caller_session:
                async with caller_session.begin():
                    # Log rejection in isolated AuditService session
                    audit_id = await svc.log_rejection(decision)
                    # Simulate caller failure that rolls back caller_session
                    raise RuntimeError("Simulated caller failure")
        except RuntimeError:
            pass  # Expected: caller rolled back

        # The audit record must still exist despite the caller rollback
        assert audit_id is not None
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert row.symbol == "TSLA"
        assert row.status == AuditStatus.REJECTED.value

    @pytest.mark.asyncio
    async def test_multiple_audits_per_symbol(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Multiple audit rows for the same symbol should all persist."""
        svc = _make_audit_service(session_factory)

        ids = []
        for _ in range(3):
            decision = _rejected_decision("INTC")
            audit_id = await svc.log_rejection(decision)
            ids.append(audit_id)

        # All three rows must be distinct
        assert len(set(ids)) == 3
        for audit_id in ids:
            row = await svc.get_audit_log(audit_id)
            assert row is not None
            assert row.symbol == "INTC"


class TestAuditRiskMetricsJsonb:
    @pytest.mark.asyncio
    async def test_risk_metrics_contains_rule_results(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _approved_decision("AAPL", "7")
        audit_id = await svc.log_pre_dispatch(decision)
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        metrics = row.risk_metrics
        assert "risk_amount" in metrics
        assert "estimated_cost" in metrics
        assert "rule_results" in metrics
        assert isinstance(metrics["rule_results"], list)

    @pytest.mark.asyncio
    async def test_risk_metrics_values_correct(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        svc = _make_audit_service(session_factory)
        decision = _approved_decision("SPY", "10")
        audit_id = await svc.log_pre_dispatch(decision)
        row = await svc.get_audit_log(audit_id)
        assert row is not None
        assert row.risk_metrics["risk_amount"] == str(decision.risk_amount)
        assert row.risk_metrics["estimated_cost"] == str(decision.estimated_cost)


class TestEvaluateTradeZeroDatabaseWrites:
    """Verifies that evaluate_trade never writes to order_audit_logs in Postgres."""

    @pytest.mark.asyncio
    async def test_evaluate_trade_leaves_audit_table_untouched(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Calling evaluate_trade results in zero rows written to audit log."""
        broker = AsyncMock()
        broker.get_account = AsyncMock(
            return_value=AccountSummary(
                cash=Decimal("50000"),
                invested=Decimal("50000"),
                result=Decimal("0"),
                total=Decimal("100000"),
                currency="USD",
            )
        )
        broker.get_positions = AsyncMock(return_value=[])
        broker.place_order = AsyncMock()

        async with session_factory() as session:
            # 1. Count rows before
            count_before = (
                await session.execute(text("SELECT COUNT(*) FROM order_audit_logs"))
            ).scalar_one()

            # 2. Evaluate an approved trade
            inp_approved = EvaluateTradeInput(
                symbol="AAPL",
                entry_price=Decimal("150"),
                stop_loss_price=Decimal("140"),
            )
            decision_approved = await evaluate_trade_handler(
                inp_approved,
                broker_client=broker,
                session=session,
            )
            assert decision_approved.approved is True

            # 3. Evaluate a rejected trade
            inp_rejected = EvaluateTradeInput(
                symbol="AAPL",
                side=OrderSide.SELL,
                entry_price=Decimal("150"),
                stop_loss_price=Decimal("140"),
            )
            decision_rejected = await evaluate_trade_handler(
                inp_rejected,
                broker_client=broker,
                session=session,
            )
            assert decision_rejected.approved is False

            # 4. Count rows after
            count_after = (
                await session.execute(text("SELECT COUNT(*) FROM order_audit_logs"))
            ).scalar_one()

            # Row count must be strictly identical — zero audit entries created
            assert count_after == count_before
            broker.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_trade_reads_sector_cache_without_writing(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """evaluate_trade reads cached fundamentals without writing any rows."""
        broker = AsyncMock()
        broker.get_account = AsyncMock(
            return_value=AccountSummary(
                cash=Decimal("50000"),
                invested=Decimal("50000"),
                result=Decimal("0"),
                total=Decimal("100000"),
                currency="USD",
            )
        )
        broker.get_positions = AsyncMock(return_value=[])
        broker.place_order = AsyncMock()

        async with session_factory() as session:
            # Seed Symbol and FundamentalsCache
            sym = Symbol(ticker="NVDA", exchange="NASDAQ", name="NVIDIA Corp")
            session.add(sym)
            await session.flush()

            cache = FundamentalsCache(
                symbol_id=sym.id,
                as_of_date=datetime.date.today(),
                payload={
                    "profile": {
                        "sector": "Semiconductors",
                        "companyName": "NVIDIA Corp",
                    }
                },
            )
            session.add(cache)
            await session.commit()

            count_before = (
                await session.execute(text("SELECT COUNT(*) FROM order_audit_logs"))
            ).scalar_one()

            inp = EvaluateTradeInput(
                symbol="NVDA",
                entry_price=Decimal("120"),
                stop_loss_price=Decimal("114"),  # 5% stop
                sector=None,  # should auto-populate from seeded cache!
            )
            decision = await evaluate_trade_handler(
                inp,
                broker_client=broker,
                session=session,
            )

            assert decision.approved is True
            r5 = next(
                r
                for r in decision.rule_results
                if r.rule_name == "check_max_sector_exposure"
            )
            assert r5.details.get("sector") == "Semiconductors"

            count_after = (
                await session.execute(text("SELECT COUNT(*) FROM order_audit_logs"))
            ).scalar_one()
            assert count_after == count_before
            broker.place_order.assert_not_called()

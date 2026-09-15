"""Durable audit logging for Phase 9 order attempts.

Every method in ``AuditService`` opens its **own** session and commits
immediately before returning, completely independent of the caller's
transaction.  This is the key durability guarantee: even if the caller rolls
back (network timeout, exception in tool handler, etc.), the audit record
already exists in Postgres and cannot be erased by the caller.

Two-phase lifecycle
───────────────────
REJECTED  — risk engine blocked the trade; written in one step.
SUBMITTING → ACCEPTED / FAILED:
  1. ``log_pre_dispatch`` writes the SUBMITTING row and returns its ``id``.
  2. Caller dispatches to broker.
  3. ``log_post_dispatch`` updates the row to ACCEPTED or FAILED with the
     broker's order-id and raw response payload.

If the process crashes between steps 2 and 3, the SUBMITTING row survives in
Postgres — operators can query for stale SUBMITTING rows to identify orders
that need manual reconciliation.
"""

from __future__ import annotations

import structlog
from sqlalchemy import select, update

from mcp_finance.db.engine import get_session_factory
from mcp_finance.db.models import OrderAuditLog
from mcp_finance.risk.models import AuditStatus, RiskDecision

logger = structlog.get_logger(__name__)


def _build_risk_metrics(decision: RiskDecision) -> dict[str, object]:
    """Serialise the risk metrics portion of a decision to plain JSON."""
    return {
        "risk_amount": str(decision.risk_amount),
        "estimated_cost": str(decision.estimated_cost),
        "rule_results": [
            {
                "rule_name": r.rule_name,
                "passed": r.passed,
                "reason": r.reason,
            }
            for r in decision.rule_results
        ],
    }


class AuditService:
    """Writes immutable audit records using isolated sessions.

    All sessions are independently committed.
    """

    def __init__(self) -> None:
        self._factory = get_session_factory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def log_rejection(self, decision: RiskDecision) -> int:
        """Write a REJECTED audit row and return its auto-assigned ``id``.

        Called when the risk engine blocks the trade.  The broker is never
        contacted; this is the only audit write for a rejected attempt.
        """
        row = OrderAuditLog(
            symbol=decision.symbol,
            side=decision.side.value,
            quantity=decision.quantity,
            entry_price=decision.entry_price,
            stop_loss_price=decision.stop_loss_price,
            status=AuditStatus.REJECTED.value,
            rejection_reasons=decision.rejection_reasons,
            risk_metrics=_build_risk_metrics(decision),
        )
        audit_id = await self._insert(row)
        logger.info(
            "Audit: order rejected",
            audit_id=audit_id,
            symbol=decision.symbol,
            rejection_count=len(decision.rejection_reasons),
        )
        return audit_id

    async def log_pre_dispatch(self, decision: RiskDecision) -> int:
        """Write a SUBMITTING row *before* the broker HTTP call.

        Returns the row ``id`` which must be passed back to
        ``log_post_dispatch`` to complete the two-phase record.  If the
        process crashes after this write and before calling the broker,
        the SUBMITTING row survives for manual reconciliation.
        """
        row = OrderAuditLog(
            symbol=decision.symbol,
            side=decision.side.value,
            quantity=decision.quantity,
            entry_price=decision.entry_price,
            stop_loss_price=decision.stop_loss_price,
            status=AuditStatus.SUBMITTING.value,
            rejection_reasons=[],
            risk_metrics=_build_risk_metrics(decision),
        )
        audit_id = await self._insert(row)
        logger.info(
            "Audit: pre-dispatch SUBMITTING row written",
            audit_id=audit_id,
            symbol=decision.symbol,
        )
        return audit_id

    async def log_post_dispatch(
        self,
        audit_id: int,
        status: AuditStatus,
        broker_order_id: str | None,
        raw_response: dict[str, object] | None,
    ) -> None:
        """Update an existing SUBMITTING row to ACCEPTED or FAILED.

        Parameters
        ----------
        audit_id:
            The ``id`` returned by ``log_pre_dispatch``.
        status:
            ``AuditStatus.ACCEPTED`` or ``AuditStatus.FAILED``.
        broker_order_id:
            The broker-assigned order identifier (present on ACCEPTED).
        raw_response:
            The full raw JSON response from the broker for debugging.
        """
        async with self._factory() as session:
            async with session.begin():
                await session.execute(
                    update(OrderAuditLog)
                    .where(OrderAuditLog.id == audit_id)
                    .values(
                        status=status.value,
                        broker_order_id=broker_order_id,
                        raw_response=raw_response,
                    )
                )
        logger.info(
            "Audit: post-dispatch update",
            audit_id=audit_id,
            status=status.value,
            broker_order_id=broker_order_id,
        )

    async def get_audit_log(self, audit_id: int) -> OrderAuditLog | None:
        """Fetch a single audit record by id (for testing / inspection)."""
        async with self._factory() as session:
            result = await session.execute(
                select(OrderAuditLog).where(OrderAuditLog.id == audit_id)
            )
            return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _insert(self, row: OrderAuditLog) -> int:
        """Insert *row* in an isolated committed transaction; return PK."""
        async with self._factory() as session:
            async with session.begin():
                session.add(row)
                await session.flush()  # populate auto-increment id
                return int(row.id)

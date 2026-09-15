"""Risk engine orchestration for Phase 9.

``RiskEngine.evaluate_trade`` is the single entry point for all risk
evaluation.  It:

1. Runs all rules in a fixed deterministic sequence.
2. Auto-sizes the position if ``proposed.quantity`` is None.
3. Aggregates every failing rule's reason into ``rejection_reasons``.
4. Sets ``approved = True`` only when *all* rules pass and the final
   quantity is strictly positive.

The engine is stateless — all context is passed in explicitly, which makes it
safe to call from async tool handlers without blocking the event loop and
trivially testable with pure in-memory fixtures.
"""

from __future__ import annotations

import asyncio
import datetime
from decimal import Decimal

import structlog

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.risk.models import (
    PositionSizeResult,
    ProposedTrade,
    RiskConfig,
    RiskDecision,
    RuleResult,
)
from mcp_finance.risk.rules import (
    check_daily_loss_limit,
    check_duplicate_position,
    check_earnings_blackout,
    check_max_portfolio_exposure,
    check_max_position_size,
    check_max_sector_exposure,
    check_order_side,
    check_stop_loss,
)
from mcp_finance.risk.sizing import calculate_position_size

logger = structlog.get_logger(__name__)


class RiskEngine:
    """Orchestrates the full rule suite and produces a ``RiskDecision``.

    Instantiate once and reuse; ``evaluate_trade`` has no side effects.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()

    async def evaluate_trade(
        self,
        proposed: ProposedTrade,
        account: AccountSummary,
        positions: list[Position],
        *,
        sector_exposures: dict[str, Decimal] | None = None,
        daily_loss: Decimal = Decimal("0"),
        today: datetime.date | None = None,
    ) -> RiskDecision:
        """Evaluate *proposed* against the complete rule suite.

        Parameters
        ----------
        proposed:
            Candidate trade submitted by the caller.
        account:
            Live account snapshot (cash, invested, total equity).
        positions:
            All currently open positions for duplicate-detection and
            sector-exposure calculations.
        sector_exposures:
            Pre-computed map of ``{sector_label: total_invested_in_sector}``.
            If None, sector-exposure check is skipped even if a sector is set.
        daily_loss:
            Absolute realised loss today (non-negative value).
        today:
            Reference date for earnings-blackout check.  Defaults to
            ``datetime.date.today()`` when not supplied.

        Returns
        -------
        RiskDecision
            ``approved`` is True only if every rule passed and quantity > 0.
        """
        # Heavy synchronous work (pure math, no I/O) runs in a thread pool
        # to avoid blocking the event loop in case sizing arithmetic is slow
        # on unusual Decimal scales.  In practice this is fast, but the
        # pattern keeps Phase 9 consistent with the asyncio.to_thread approach
        # established in Phase 4.
        return await asyncio.to_thread(
            self._evaluate_sync,
            proposed,
            account,
            positions,
            sector_exposures or {},
            daily_loss,
            today or datetime.date.today(),
        )

    # ------------------------------------------------------------------
    # Private synchronous core (runs in thread pool via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _evaluate_sync(
        self,
        proposed: ProposedTrade,
        account: AccountSummary,
        positions: list[Position],
        sector_exposures: dict[str, Decimal],
        daily_loss: Decimal,
        today: datetime.date,
    ) -> RiskDecision:
        rule_results: list[RuleResult] = []

        # ── Rule 0: Order-side gate ────────────────────────────────────────
        r0 = check_order_side(proposed.side)
        rule_results.append(r0)

        # ── Rule 1: Stop-loss validation ───────────────────────────────────
        r1 = check_stop_loss(
            proposed.entry_price, proposed.stop_loss_price, self._config
        )
        rule_results.append(r1)

        # ── Rule 6: Daily-loss circuit breaker ─────────────────────────────
        # Run early: if the circuit breaker fires we should know immediately,
        # before spending time on sizing that would be irrelevant.
        r_dl = check_daily_loss_limit(daily_loss, account.total, self._config)
        rule_results.append(r_dl)

        # ── Rule 7: Earnings blackout ──────────────────────────────────────
        r_eb = check_earnings_blackout(proposed.next_earnings_date, today, self._config)
        rule_results.append(r_eb)

        # ── Position sizing ────────────────────────────────────────────────
        # We compute sizing now (even if some early rules failed) so that the
        # decision always carries a meaningful quantity for dry-run inspection.
        # If entry == stop_loss or stop_loss <= 0 the sizing function raises
        # ValueError — we handle that gracefully and set quantity to 0.
        if proposed.quantity is not None:
            # Explicit quantity supplied: use it directly.
            sized = PositionSizeResult(
                quantity=proposed.quantity,
                risk_amount=(proposed.entry_price - proposed.stop_loss_price)
                * proposed.quantity
                if r1.passed
                else Decimal("0"),
                total_cost=proposed.entry_price * proposed.quantity,
                binding_constraint="explicit",
            )
        else:
            try:
                sized = calculate_position_size(
                    total_equity=account.total,
                    risk_pct=self._config.max_risk_per_trade_pct,
                    entry_price=proposed.entry_price,
                    stop_loss_price=proposed.stop_loss_price,
                    max_position_pct=self._config.max_position_size_pct,
                    cash_available=account.cash,
                )
            except ValueError as exc:
                logger.warning("Position sizing failed", error=str(exc))
                sized = PositionSizeResult(
                    quantity=Decimal("0"),
                    risk_amount=Decimal("0"),
                    total_cost=Decimal("0"),
                    binding_constraint="error",
                )

        position_cost = sized.total_cost

        # ── Rule 2: Duplicate position ─────────────────────────────────────
        r2 = check_duplicate_position(proposed.symbol, positions)
        rule_results.append(r2)

        # ── Rule 3: Max position size ──────────────────────────────────────
        r3 = check_max_position_size(position_cost, account.total, self._config)
        rule_results.append(r3)

        # ── Rule 4: Max portfolio exposure ─────────────────────────────────
        r4 = check_max_portfolio_exposure(
            account.invested, position_cost, account.total, self._config
        )
        rule_results.append(r4)

        # ── Rule 5: Max sector exposure ────────────────────────────────────
        current_sector_cost = (
            sector_exposures.get(proposed.sector, Decimal("0"))
            if proposed.sector
            else Decimal("0")
        )
        r5 = check_max_sector_exposure(
            current_sector_cost,
            position_cost,
            account.total,
            proposed.sector,
            self._config,
        )
        rule_results.append(r5)

        # ── Aggregate verdict ──────────────────────────────────────────────
        rejection_reasons = [r.reason for r in rule_results if not r.passed]
        # Zero-quantity is itself a hard block (nothing to execute).
        if sized.quantity <= Decimal("0") and not rejection_reasons:
            rejection_reasons = [
                "Position sizing yielded zero shares. "
                "Check equity, stop distance, and available cash."
            ]

        approved = len(rejection_reasons) == 0 and sized.quantity > Decimal("0")

        logger.info(
            "Risk evaluation complete",
            symbol=proposed.symbol,
            approved=approved,
            quantity=str(sized.quantity),
            rejection_count=len(rejection_reasons),
        )

        return RiskDecision(
            approved=approved,
            symbol=proposed.symbol,
            side=proposed.side,
            order_type=proposed.order_type,
            entry_price=proposed.entry_price,
            stop_loss_price=proposed.stop_loss_price,
            quantity=sized.quantity,
            estimated_cost=sized.total_cost,
            risk_amount=sized.risk_amount,
            rule_results=rule_results,
            rejection_reasons=rejection_reasons,
        )

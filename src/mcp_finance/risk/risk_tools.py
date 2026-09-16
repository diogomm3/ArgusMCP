"""FastMCP tool registrations for risk management and order evaluation.

Phase 9 exposes two tools:
  - ``evaluate_trade``: Pure dry-run risk evaluation with zero database writes
    and zero broker order submissions.
  - ``place_order``: Full-lifecycle execution gate (built in Checkpoint 6).
"""

import datetime
from decimal import Decimal

from mcp.server.mcpserver import MCPServer
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.brokers import BrokerClient
from mcp_finance.brokers.models import OrderResult, Position
from mcp_finance.brokers.utils import get_trading212_client
from mcp_finance.db.engine import get_session
from mcp_finance.db.models import FundamentalsCache, Symbol
from mcp_finance.logger import get_logger
from mcp_finance.risk.audit import AuditService
from mcp_finance.risk.engine import RiskEngine
from mcp_finance.risk.models import (
    AuditStatus,
    EvaluateTradeInput,
    OrderSide,
    PlaceOrderInput,
    PlaceOrderOutput,
    ProposedTrade,
    RiskConfig,
    RiskDecision,
)
from mcp_finance.risk.rules import canonical_ticker_symbol

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Read-only Helpers
# ---------------------------------------------------------------------------


async def _lookup_cached_sector(
    symbol: str,
    session: AsyncSession,
) -> str | None:
    """Read-only lookup of economic sector from local fundamentals cache.

    Performs a pure SELECT query without modifying or committing session state.
    Returns None if no cached snapshot is found.
    """
    canon = canonical_ticker_symbol(symbol)
    stmt = (
        select(FundamentalsCache.payload)
        .join(Symbol, FundamentalsCache.symbol_id == Symbol.id)
        .where(
            or_(
                Symbol.ticker == canon,
                Symbol.ticker == symbol,
            )
        )
        .order_by(FundamentalsCache.as_of_date.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    payload = result.scalar_one_or_none()
    if payload and isinstance(payload, dict):
        profile = payload.get("profile")
        if isinstance(profile, dict):
            sector = profile.get("sector")
            if sector:
                return str(sector)
    return None


async def _build_sector_exposures(
    positions: list[Position],
    session: AsyncSession,
) -> dict[str, Decimal]:
    """Calculate total invested amount per sector from open positions."""
    exposures: dict[str, Decimal] = {}
    for pos in positions:
        if pos.quantity <= Decimal("0"):
            continue
        try:
            sector = await _lookup_cached_sector(pos.ticker, session)
            if sector:
                invested = pos.quantity * pos.current_price
                exposures[sector] = exposures.get(sector, Decimal("0")) + invested
        except Exception as exc:
            logger.debug(
                "Could not lookup sector for position",
                ticker=pos.ticker,
                error=str(exc),
            )
    return exposures


# ---------------------------------------------------------------------------
# Tool Handler Logic
# ---------------------------------------------------------------------------


async def evaluate_trade_handler(
    input: EvaluateTradeInput,
    *,
    broker_client: BrokerClient | None = None,
    session: AsyncSession | None = None,
    engine: RiskEngine | None = None,
    config: RiskConfig | None = None,
    today: datetime.date | None = None,
) -> RiskDecision:
    """Core logic for evaluate_trade.

    Guarantees:
      1. Zero database writes: only read-only SELECT queries are executed to
         read cached fundamentals; no audit rows are written and no commits occur.
      2. Zero broker orders: never calls place_order on the broker adapter.
      3. Sector auto-population: if sector is omitted, attempts to auto-populate
         from cached fundamentals so the sector concentration rule can evaluate.
      4. Earnings blackout: caller-supplied only; skipped if omitted.
      5. Sizing: if quantity is None, auto-sizes via fixed-fractional budgeting.
    """
    resolved_engine = engine or RiskEngine(config=config or RiskConfig())

    # 1. Fetch live portfolio state from broker
    if broker_client is not None:
        account = await broker_client.get_account()
        positions = await broker_client.get_positions()
    else:
        async with get_trading212_client() as client:
            account = await client.get_account()
            positions = await client.get_positions()

    # 2. Sector auto-population & sector exposure resolution
    resolved_sector = input.sector
    sector_exposures: dict[str, Decimal] = {}

    if session is not None:
        if resolved_sector is None:
            resolved_sector = await _lookup_cached_sector(input.symbol, session)
            if resolved_sector:
                logger.info(
                    "Auto-populated sector from cache",
                    symbol=input.symbol,
                    sector=resolved_sector,
                )
        sector_exposures = await _build_sector_exposures(positions, session)
    else:
        try:
            async with get_session() as auto_session:
                if resolved_sector is None:
                    resolved_sector = await _lookup_cached_sector(
                        input.symbol, auto_session
                    )
                    if resolved_sector:
                        logger.info(
                            "Auto-populated sector from cache",
                            symbol=input.symbol,
                            sector=resolved_sector,
                        )
                sector_exposures = await _build_sector_exposures(
                    positions, auto_session
                )
        except Exception as exc:
            logger.debug(
                "Fundamentals cache lookup skipped or unavailable",
                error=str(exc),
            )

    proposed = ProposedTrade(
        symbol=input.symbol,
        side=input.side,
        order_type=input.order_type,
        entry_price=input.entry_price,
        stop_loss_price=input.stop_loss_price,
        quantity=input.quantity,
        limit_price=input.limit_price,
        sector=resolved_sector,
        next_earnings_date=input.next_earnings_date,
    )

    # 3. Synchronous, pure evaluation through RiskEngine
    decision = resolved_engine.evaluate_trade(
        proposed=proposed,
        account=account,
        positions=positions,
        sector_exposures=sector_exposures,
        today=today,
    )

    logger.info(
        "Trade evaluation completed",
        symbol=decision.symbol,
        approved=decision.approved,
        quantity=str(decision.quantity),
        estimated_cost=str(decision.estimated_cost),
        risk_amount=str(decision.risk_amount),
        rejections=decision.rejection_reasons,
    )

    return decision


# ---------------------------------------------------------------------------
# Order Placement Execution Logic
# ---------------------------------------------------------------------------


def assert_trade_approved_invariant(decision: RiskDecision) -> None:
    """Re-assert hard execution invariants before any order dispatch.

    Defense-in-depth safety: must hold true even if evaluate_trade was somehow
    bypassed or corrupted. Raises RuntimeError if violated.
    """
    if not decision.approved:
        raise RuntimeError(
            "Defense-in-depth violation: attempt to place an unapproved order."
        )
    if decision.quantity <= Decimal("0"):
        raise RuntimeError(
            f"Defense-in-depth violation: non-positive execution quantity "
            f"({decision.quantity})."
        )
    if decision.rejection_reasons:
        raise RuntimeError(
            f"Defense-in-depth violation: order carries rejection reasons: "
            f"{decision.rejection_reasons}"
        )
    if decision.side != OrderSide.BUY:
        raise RuntimeError(
            f"Defense-in-depth violation: unsupported order side {decision.side}."
        )


def to_broker_ticker(symbol: str) -> str:
    """Resolve symbol to broker ticker format (defaulting to _US_EQ)."""
    s = symbol.strip().upper()
    if "_" in s:
        return s
    dotted_suffixes: dict[str, str] = {
        ".US": "_US_EQ",
        ".DE": "_DE_EQ",
        ".UK": "_UK_EQ",
        ".CA": "_CA_EQ",
        ".FR": "_FR_EQ",
        ".NL": "_NL_EQ",
        ".L": "_UK_EQ",
    }
    for dot, broker_suff in dotted_suffixes.items():
        if s.endswith(dot):
            return f"{s[: -len(dot)]}{broker_suff}"
    return f"{s}_US_EQ"


async def place_order_handler(
    input: PlaceOrderInput,
    *,
    broker_client: BrokerClient | None = None,
    session: AsyncSession | None = None,
    audit_service: AuditService | None = None,
    engine: RiskEngine | None = None,
    config: RiskConfig | None = None,
    today: datetime.date | None = None,
) -> PlaceOrderOutput:
    """Submit a real BUY order through the risk engine execution gate.

    Lifecycle:
      1. Run full RiskEngine evaluation via evaluate_trade_handler (dry-run rules).
      2. If REJECTED:
         - Persist a REJECTED audit row to Postgres via AuditService.
         - Return PlaceOrderOutput(success=False, ...) immediately.
         - Never contact broker.
      3. If APPROVED:
         - Re-assert defense-in-depth invariant (assert_trade_approved_invariant).
         - Persist pre-dispatch SUBMITTING audit row to Postgres via AuditService.
         - Call broker place_order.
         - If broker fails: update audit row to FAILED with error payload, re-raise.
         - If broker succeeds: update audit row to ACCEPTED with broker_order_id.
         - Return PlaceOrderOutput(success=True, ...).
    """
    resolved_audit = audit_service or AuditService()

    eval_input = EvaluateTradeInput(
        symbol=input.symbol,
        entry_price=input.entry_price,
        stop_loss_price=input.stop_loss_price,
        quantity=input.quantity,
        side=OrderSide.BUY,
        order_type=input.order_type,
        limit_price=input.limit_price,
        sector=input.sector,
        next_earnings_date=input.next_earnings_date,
    )

    decision = await evaluate_trade_handler(
        eval_input,
        broker_client=broker_client,
        session=session,
        engine=engine,
        config=config,
        today=today,
    )

    # Rejection gate
    if not decision.approved:
        audit_id = await resolved_audit.log_rejection(decision)
        logger.warning(
            "Order rejected by risk engine",
            symbol=decision.symbol,
            audit_id=audit_id,
            rejections=decision.rejection_reasons,
        )
        return PlaceOrderOutput(
            success=False,
            decision=decision,
            audit_id=audit_id,
            broker_order_id=None,
            order_result=None,
            error_message=(
                f"Order rejected by risk engine: "
                f"{'; '.join(decision.rejection_reasons)}"
            ),
        )

    # Invariant assertion
    assert_trade_approved_invariant(decision)

    # Phase 1: Pre-dispatch audit logging (SUBMITTING)
    audit_id = await resolved_audit.log_pre_dispatch(decision)

    # Phase 2: Broker dispatch
    broker_ticker = to_broker_ticker(input.symbol)
    order_result: OrderResult | None = None
    try:
        if broker_client is not None:
            order_result = await broker_client.place_order(
                ticker=broker_ticker,
                quantity=decision.quantity,
                order_type=decision.order_type.value,
                limit_price=input.limit_price,
            )
        else:
            async with get_trading212_client() as client:
                order_result = await client.place_order(
                    ticker=broker_ticker,
                    quantity=decision.quantity,
                    order_type=decision.order_type.value,
                    limit_price=input.limit_price,
                )
    except Exception as exc:
        logger.error(
            "Broker order placement failed; recording FAILED audit status",
            symbol=broker_ticker,
            audit_id=audit_id,
            error=str(exc),
        )
        await resolved_audit.log_post_dispatch(
            audit_id=audit_id,
            status=AuditStatus.FAILED,
            broker_order_id=None,
            raw_response={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise

    # Phase 3: Post-dispatch success update (ACCEPTED)
    await resolved_audit.log_post_dispatch(
        audit_id=audit_id,
        status=AuditStatus.ACCEPTED,
        broker_order_id=order_result.id,
        raw_response=order_result.model_dump(mode="json"),
    )

    logger.info(
        "Order successfully placed and accepted by broker",
        symbol=broker_ticker,
        audit_id=audit_id,
        broker_order_id=order_result.id,
        quantity=str(decision.quantity),
    )

    return PlaceOrderOutput(
        success=True,
        decision=decision,
        audit_id=audit_id,
        broker_order_id=order_result.id,
        order_result=order_result,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------


def register_risk_tools(mcp: MCPServer) -> None:
    """Register risk management MCP tools with the server."""

    @mcp.tool()
    async def evaluate_trade(input: EvaluateTradeInput) -> RiskDecision:
        """Perform a dry-run pre-trade risk evaluation for a proposed order.

        Evaluates the trade against all Phase 9 portfolio risk rules:
          - Rule 0: Order-side check (BUY supported; SELL deferred).
          - Rule 1: Stop-loss validation (positive, below entry, within permitted
            envelope).
          - Rule 2: Duplicate position detection (canonical matching & fail-closed
            fallback).
          - Rule 3: Max position size (ceiling % of total equity).
          - Rule 4: Max portfolio exposure (ceiling % total invested).
          - Rule 5: Max sector concentration (ceiling % sector exposure).
          - Rule 6: Daily-loss circuit breaker.
          - Rule 7: Earnings blackout window.

        Dry-Run Guarantee:
          This tool performs zero database writes and submits zero orders to
          the broker. It is completely safe for analysis, screening, and planning.

        Auto-population Policy:
          - sector: If omitted by caller, the tool queries the local read-only
            fundamentals cache for the symbol's economic sector. If found, it
            is auto-populated so the sector concentration rule can evaluate.
            If missing from cache, sector concentration is skipped.
          - next_earnings_date: Caller-supplied only. When omitted, earnings
            blackout validation is skipped.

        Args:
            input: Candidate trade details (symbol, entry price, stop-loss price,
                   optional quantity, limit price, sector, next earnings date).

        Returns:
            RiskDecision containing approval status, sized quantity, cost/risk
            breakdowns, individual rule results, and rejection reasons.
        """
        return await evaluate_trade_handler(input)

    @mcp.tool()
    async def place_order(input: PlaceOrderInput) -> PlaceOrderOutput:
        """Submit a BUY order to the broker, strictly gated by the risk engine.

        Full execution lifecycle:
          1. Evaluates all Phase 9 portfolio risk rules and sizes position.
          2. If rejected: logs REJECTED audit row, returns failure, NEVER touches
             the broker.
          3. If approved:
             - Re-asserts hard defense-in-depth invariant.
             - Logs pre-dispatch SUBMITTING audit row.
             - Dispatches BUY order to Trading212 (demo environment only).
             - On broker failure: logs FAILED audit row and surfaces error.
             - On broker success: logs ACCEPTED audit row with broker_order_id.

        Args:
            input: Order candidate details (symbol, entry price, stop-loss price,
                   optional quantity, limit price, sector, next earnings date).

        Returns:
            PlaceOrderOutput with success status, RiskDecision, audit_id, and
            broker_order_id.
        """
        return await place_order_handler(input)

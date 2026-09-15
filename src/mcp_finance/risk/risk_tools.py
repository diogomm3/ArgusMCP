"""FastMCP tool registrations for risk management and order evaluation.

Phase 9 exposes two tools:
  - ``evaluate_trade``: Pure dry-run risk evaluation with zero database writes
    and zero broker order submissions.
  - ``place_order``: Full-lifecycle execution gate (built in Checkpoint 6).
"""

import datetime
from decimal import Decimal

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.brokers import BrokerClient
from mcp_finance.brokers.models import Position
from mcp_finance.brokers.utils import get_trading212_client
from mcp_finance.db.engine import get_session
from mcp_finance.db.models import FundamentalsCache, Symbol
from mcp_finance.logger import get_logger
from mcp_finance.risk.engine import RiskEngine
from mcp_finance.risk.models import (
    OrderSide,
    OrderType,
    ProposedTrade,
    RiskConfig,
    RiskDecision,
)
from mcp_finance.risk.rules import canonical_ticker_symbol

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------


class EvaluateTradeInput(BaseModel):
    """Input payload for candidate trade dry-run evaluation."""

    symbol: str = Field(
        ...,
        description=(
            "Canonical or broker-formatted ticker symbol "
            "(e.g. 'AAPL', 'AAPL_US_EQ', 'SAP_DE_EQ')."
        ),
    )
    entry_price: Decimal = Field(
        ...,
        description="Intended entry price per share.",
    )
    stop_loss_price: Decimal = Field(
        ...,
        description="Maximum acceptable loss price per share (stop-loss trigger).",
    )
    quantity: Decimal | None = Field(
        default=None,
        description=(
            "Explicit share count. If omitted or None, the engine auto-sizes "
            "the position via fixed-fractional risk budgeting."
        ),
    )
    side: OrderSide = Field(
        default=OrderSide.BUY,
        description="Order direction (Phase 9 only supports BUY).",
    )
    order_type: OrderType = Field(
        default=OrderType.MARKET,
        description="Execution type ('MARKET' or 'LIMIT').",
    )
    limit_price: Decimal | None = Field(
        default=None,
        description="Limit price per share (required when order_type is 'LIMIT').",
    )
    sector: str | None = Field(
        default=None,
        description=(
            "GICS/ICB sector label. If omitted, the tool attempts to auto-populate "
            "from the local fundamentals cache if available; if not cached, "
            "the sector concentration check is skipped."
        ),
    )
    next_earnings_date: datetime.date | None = Field(
        default=None,
        description=(
            "Next scheduled earnings release date (YYYY-MM-DD). If omitted, the "
            "earnings blackout check is skipped (earnings calendar lookup is "
            "caller-supplied only in Phase 9)."
        ),
    )


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

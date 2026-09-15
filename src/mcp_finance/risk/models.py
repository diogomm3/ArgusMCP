"""Data models for the Phase 9 risk engine.

All monetary values use ``Decimal`` for exact arithmetic. ``RiskConfig``
carries all tuneable parameters with safe defaults. ``RiskDecision`` is the
immutable outcome returned by the engine — it is the single artifact that
controls whether ``place_order`` dispatches to the broker.
"""

import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    """Direction of the proposed order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Execution type for the proposed order."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class AuditStatus(str, Enum):
    """Lifecycle status values written to ``order_audit_logs``."""

    REJECTED = "REJECTED"
    SUBMITTING = "SUBMITTING"
    ACCEPTED = "ACCEPTED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RiskConfig(BaseModel):
    """All tuneable risk-management parameters in one place.

    Defaults reflect conservative starting values. Callers may override any
    field at construction time without touching global state.
    """

    # Position-size ceiling: no single position may exceed this fraction of
    # total portfolio equity.
    max_position_size_pct: Decimal = Field(
        default=Decimal("0.05"),
        description="Position-size ceiling as fraction of total portfolio equity.",
    )

    # Portfolio-exposure ceiling: at most this fraction of equity may be
    # invested at any one time (preserving a 10 % cash buffer).
    max_portfolio_exposure_pct: Decimal = Field(
        default=Decimal("0.90"),
        description="Portfolio-exposure ceiling as fraction of total portfolio equity.",
    )

    # Sector-concentration ceiling.
    max_sector_exposure_pct: Decimal = Field(
        default=Decimal("0.25"),
        description="Sector-concentration ceiling.",
    )

    # Fixed-fractional risk budget per trade.
    max_risk_per_trade_pct: Decimal = Field(
        default=Decimal("0.01"),
        description="Fixed-fractional risk budget per trade.",
    )

    # Hard daily-loss circuit breaker.  Unlike exposure ceilings (≤), this
    # uses a strict less-than (<): losing 100 % of the daily allowance
    # *immediately* halts trading so no further losers can stack on top.
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.03"),
        description=(
            "Hard daily-loss circuit breaker as fraction of total portfolio equity."
        ),
    )

    # Stop-loss distance envelope (as fraction of entry price).
    min_stop_loss_distance_pct: Decimal = Field(
        default=Decimal("0.005"),
        description="Minimum stop-loss distance as fraction of entry price.",
    )
    max_stop_loss_distance_pct: Decimal = Field(
        default=Decimal("0.15"),
        description="Maximum stop-loss distance as fraction of entry price.",
    )

    # Earnings blackout: block all new buys if earnings are this many
    # calendar days away or fewer.
    earnings_blackout_days: int = Field(
        default=3,
        description="Earnings blackout window in calendar days.",
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class ProposedTrade(BaseModel):
    """A candidate trade submitted to the risk engine for evaluation."""

    symbol: str = Field(
        ...,
        description="Canonical or broker-formatted ticker symbol.",
    )
    side: OrderSide = Field(
        default=OrderSide.BUY,
        description="Order direction.",
    )
    order_type: OrderType = Field(
        default=OrderType.MARKET,
        description="Execution type.",
    )
    entry_price: Decimal = Field(
        ...,
        description="Intended entry price per share.",
    )
    stop_loss_price: Decimal = Field(
        ...,
        description="Maximum acceptable loss price per share.",
    )
    quantity: Decimal | None = Field(
        default=None,
        description=(
            "Explicit share count. If None the engine auto-sizes via "
            "fixed-fractional sizing."
        ),
    )
    limit_price: Decimal | None = Field(
        default=None,
        description="Limit price (required when order_type == LIMIT).",
    )
    sector: str | None = Field(
        default=None,
        description="GICS/ICB sector label. Used for sector-exposure checks.",
    )
    next_earnings_date: datetime.date | None = Field(
        default=None,
        description="Next scheduled earnings release date. Absent → blackout skipped.",
    )


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
# Outputs
# ---------------------------------------------------------------------------


class RuleResult(BaseModel):
    """Structured outcome from a single pure rule function."""

    rule_name: str = Field(
        ...,
        description="Identifier of the rule function that produced this result.",
    )
    passed: bool = Field(
        ...,
        description="True if the candidate trade satisfies this risk rule.",
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of why the rule passed or failed.",
    )
    details: dict[str, object] = Field(
        default_factory=dict,
        description="Contextual key-value metrics from the evaluation.",
    )


class PositionSizeResult(BaseModel):
    """Output of the position-sizing calculator."""

    quantity: Decimal = Field(
        ...,
        description="Recommended share count.",
    )
    risk_amount: Decimal = Field(
        ...,
        description="Monetary amount at risk if stop loss is hit.",
    )
    total_cost: Decimal = Field(
        ...,
        description="Total monetary cost to purchase the recommended shares.",
    )
    binding_constraint: str = Field(
        ...,
        description=(
            "Which cap was the binding constraint: "
            "'risk_budget', 'position_limit', or 'cash'."
        ),
    )


class RiskDecision(BaseModel):
    """Final, immutable verdict from the risk engine.

    ``approved`` is True if and only if every rule passed *and* the sized
    quantity is strictly positive. ``place_order`` may dispatch to the broker
    only after re-asserting this invariant internally.
    """

    approved: bool = Field(
        ...,
        description="True if and only if every rule passed and quantity > 0.",
    )
    symbol: str = Field(
        ...,
        description="Canonical ticker symbol of the evaluated instrument.",
    )
    side: OrderSide = Field(
        ...,
        description="Order direction.",
    )
    order_type: OrderType = Field(
        ...,
        description="Execution type.",
    )
    entry_price: Decimal = Field(
        ...,
        description="Intended entry price per share.",
    )
    stop_loss_price: Decimal = Field(
        ...,
        description="Maximum acceptable loss price per share.",
    )
    quantity: Decimal = Field(
        ...,
        description="Approved whole share quantity.",
    )
    estimated_cost: Decimal = Field(
        ...,
        description="Estimated total cost (quantity * entry_price).",
    )
    risk_amount: Decimal = Field(
        ...,
        description="Estimated monetary risk (quantity * (entry - stop)).",
    )
    rule_results: list[RuleResult] = Field(
        default_factory=list,
        description="Results from every evaluated risk rule.",
    )
    rejection_reasons: list[str] = Field(
        default_factory=list,
        description="List of failure reasons (empty if approved).",
    )

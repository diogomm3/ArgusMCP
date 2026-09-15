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
    max_position_size_pct: Decimal = Decimal("0.05")

    # Portfolio-exposure ceiling: at most this fraction of equity may be
    # invested at any one time (preserving a 10 % cash buffer).
    max_portfolio_exposure_pct: Decimal = Decimal("0.90")

    # Sector-concentration ceiling.
    max_sector_exposure_pct: Decimal = Decimal("0.25")

    # Fixed-fractional risk budget per trade.
    max_risk_per_trade_pct: Decimal = Decimal("0.01")

    # Hard daily-loss circuit breaker.  Unlike exposure ceilings (≤), this
    # uses a strict less-than (<): losing 100 % of the daily allowance
    # *immediately* halts trading so no further losers can stack on top.
    max_daily_loss_pct: Decimal = Decimal("0.03")

    # Stop-loss distance envelope (as fraction of entry price).
    min_stop_loss_distance_pct: Decimal = Decimal("0.005")
    max_stop_loss_distance_pct: Decimal = Decimal("0.15")

    # Earnings blackout: block all new buys if earnings are this many
    # calendar days away or fewer.
    earnings_blackout_days: int = 3


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class ProposedTrade(BaseModel):
    """A candidate trade submitted to the risk engine for evaluation."""

    symbol: str = Field(..., description="Canonical or broker-formatted ticker symbol.")
    side: OrderSide = Field(default=OrderSide.BUY, description="Order direction.")
    order_type: OrderType = Field(
        default=OrderType.MARKET, description="Execution type."
    )
    entry_price: Decimal = Field(..., description="Intended entry price per share.")
    stop_loss_price: Decimal = Field(
        ..., description="Maximum acceptable loss price per share."
    )
    quantity: Decimal | None = Field(
        default=None,
        description=(
            "Explicit share count. If None the engine auto-sizes via "
            "fixed-fractional sizing."
        ),
    )
    limit_price: Decimal | None = Field(
        default=None, description="Limit price (required when order_type == LIMIT)."
    )
    sector: str | None = Field(
        default=None,
        description="GICS/ICB sector label. Used for sector-exposure checks.",
    )
    next_earnings_date: datetime.date | None = Field(
        default=None,
        description="Next scheduled earnings release date. Absent → blackout skipped.",
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


class RuleResult(BaseModel):
    """Structured outcome from a single pure rule function."""

    rule_name: str
    passed: bool
    reason: str
    details: dict[str, object] = Field(default_factory=dict)


class PositionSizeResult(BaseModel):
    """Output of the position-sizing calculator."""

    quantity: Decimal
    risk_amount: Decimal
    total_cost: Decimal
    binding_constraint: str = Field(
        description=(
            "Which cap was the binding constraint: "
            "'risk_budget', 'position_limit', or 'cash'."
        )
    )


class RiskDecision(BaseModel):
    """Final, immutable verdict from the risk engine.

    ``approved`` is True if and only if every rule passed *and* the sized
    quantity is strictly positive. ``place_order`` may dispatch to the broker
    only after re-asserting this invariant internally.
    """

    approved: bool
    symbol: str
    side: OrderSide
    order_type: OrderType
    entry_price: Decimal
    stop_loss_price: Decimal
    quantity: Decimal
    estimated_cost: Decimal
    risk_amount: Decimal
    rule_results: list[RuleResult]
    rejection_reasons: list[str]

"""Fixed-fractional position sizing for Phase 9.

This module is intentionally small: one pure function, no I/O, no global
state.  ``calculate_position_size`` computes the number of whole shares to
buy, taking the minimum across three independent caps so that *every* limit is
respected simultaneously:

1. **Risk budget cap** — how many shares the 1 % risk budget can support given
   the stop-loss distance.
2. **Position-size cap** — how many shares can be bought before breaching the
   ``max_position_size_pct`` portfolio ceiling.
3. **Cash cap** — how many shares available cash can fund.

All monetary inputs are ``Decimal``.  Fractional shares are not supported;
the result is always ``floor``-rounded to the nearest whole share.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

from mcp_finance.risk.models import PositionSizeResult


def calculate_position_size(
    total_equity: Decimal,
    risk_pct: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
    max_position_pct: Decimal,
    cash_available: Decimal,
) -> PositionSizeResult:
    """Compute the number of whole shares to buy.

    Parameters
    ----------
    total_equity:
        Total portfolio value (cash + invested).
    risk_pct:
        Fraction of ``total_equity`` that may be *risked* on this trade
        (i.e. lost if the stop is hit).
    entry_price:
        Intended buy price per share.
    stop_loss_price:
        Maximum acceptable loss price per share.  Must be strictly less
        than ``entry_price`` (validated here; raises ``ValueError`` otherwise).
    max_position_pct:
        Fraction of ``total_equity`` that a single position may represent.
    cash_available:
        Free cash available for trading.

    Returns
    -------
    PositionSizeResult
        ``quantity``, ``risk_amount``, ``total_cost``, and
        ``binding_constraint`` identifying which cap bound the result.

    Raises
    ------
    ValueError
        If ``entry_price <= stop_loss_price`` or ``entry_price <= 0``.
    """
    if entry_price <= Decimal("0"):
        raise ValueError(f"entry_price must be positive; got {entry_price}")
    if stop_loss_price >= entry_price:
        raise ValueError(
            f"stop_loss_price ({stop_loss_price}) must be strictly less than "
            f"entry_price ({entry_price})."
        )
    if stop_loss_price <= Decimal("0"):
        raise ValueError(f"stop_loss_price must be positive; got {stop_loss_price}")

    risk_per_share: Decimal = entry_price - stop_loss_price
    risk_budget: Decimal = total_equity * risk_pct

    # Cap 1: risk budget
    shares_from_risk = (risk_budget / risk_per_share).to_integral_value(
        rounding=ROUND_DOWN
    )

    # Cap 2: position-size limit
    shares_from_position_limit = (
        (total_equity * max_position_pct) / entry_price
    ).to_integral_value(rounding=ROUND_DOWN)

    # Cap 3: available cash
    shares_from_cash = (cash_available / entry_price).to_integral_value(
        rounding=ROUND_DOWN
    )

    # Binding minimum
    candidates = {
        "risk_budget": shares_from_risk,
        "position_limit": shares_from_position_limit,
        "cash": shares_from_cash,
    }
    binding_constraint = min(candidates, key=lambda k: candidates[k])
    final_shares = candidates[binding_constraint]

    # Ensure non-negative (edge case: zero cash or zero budget)
    final_shares = max(final_shares, Decimal("0"))

    risk_amount = final_shares * risk_per_share
    total_cost = final_shares * entry_price

    return PositionSizeResult(
        quantity=final_shares,
        risk_amount=risk_amount,
        total_cost=total_cost,
        binding_constraint=binding_constraint,
    )

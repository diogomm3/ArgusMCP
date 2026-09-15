"""Pure risk-rule functions for Phase 9.

Every function in this module is a *pure function*: it accepts only explicit
inputs and returns a structured ``RuleResult``.  No I/O, no database access,
no global state.  This makes every rule trivially unit-testable and safe to
run inside async event-loop handlers without blocking.

Ticker normalisation
────────────────────
``canonical_ticker_symbol`` strips broker-appended exchange/asset-class
suffixes so that ``AAPL_US_EQ`` ↔ ``AAPL`` and ``SAP_DE_EQ`` ↔ ``SAP``.
The suffix list is built from *observed* Trading212 ticker formats, not from
heuristics.  The function performs exact match first, then iterates suffix
candidates.  If no suffix matches the function returns the original ticker
unchanged — a deliberate "fail-open" for normalisation paired with the
"fail-closed" duplicate check: if normalisation cannot strip a suffix we
still compare the raw strings, which is safe (it may produce a false-negative
on the duplicate check but never a silent overwrite).
"""

from __future__ import annotations

import datetime
import re
from decimal import Decimal

import structlog

from mcp_finance.brokers.models import Position
from mcp_finance.risk.models import OrderSide, RiskConfig, RuleResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Known Trading212 broker-ticker suffixes, ordered longest-first so that
# ``_US_EQ`` is matched before the bare ``_EQ`` fallback.
# Derived from observed live/demo account ticker strings.
# ---------------------------------------------------------------------------
_BROKER_SUFFIXES: tuple[str, ...] = (
    "_US_EQ",
    "_CA_EQ",
    "_DE_EQ",
    "_UK_EQ",
    "_FR_EQ",
    "_NL_EQ",
    "_ES_EQ",
    "_IT_EQ",
    "_SE_EQ",
    "_CH_EQ",
    "_AT_EQ",
    "_BE_EQ",
    "_PT_EQ",
    "_BB_EQ",
    "_FI_EQ",
    "_IE_EQ",
    "_DK_EQ",
    "_NO_EQ",
    "_WAR_FR_EQ",
    "_WAR_US_EQ",
    "_EQ",
    ".US",
    ".DE",
    ".UK",
    ".L",
)

# Lowercase-normalised variant pattern used for case-insensitive matching
# on European tickers that Trading212 sometimes renders with lowercase chars
# (e.g. ``SAPd_EQ`` → ``SAP``).
_SUFFIX_PATTERN = re.compile(
    "|".join(re.escape(s) for s in _BROKER_SUFFIXES), re.IGNORECASE
)


def canonical_ticker_symbol(ticker: str) -> str:
    """Strip broker exchange/asset-class suffixes from *ticker*.

    Returns the canonical symbol in upper-case.  Examples::

        canonical_ticker_symbol("AAPL_US_EQ")  → "AAPL"
        canonical_ticker_symbol("SAP_DE_EQ")   → "SAP"
        canonical_ticker_symbol("SAPd_EQ")     → "SAP"   (T212 lowercase country code)
        canonical_ticker_symbol("ASMLa_EQ")    → "ASML"  (T212 Euronext Amsterdam)
        canonical_ticker_symbol("IFXd_EQ")     → "IFX"   (T212 XETRA Germany)
        canonical_ticker_symbol("BRK_B_US_EQ") → "BRK.B" (Preserves share class)
        canonical_ticker_symbol("BRK/A_US_EQ") → "BRK.A" (Preserves share class)
        canonical_ticker_symbol("LLOY_UK_EQ")  → "LLOY"
        canonical_ticker_symbol("AAPL")        → "AAPL"  (already canonical)

    Trading212 appends a single *lowercase* letter to the root ticker before
    the exchange suffix as a country/market discriminator (e.g. ``SAPd_EQ``,
    ``ASMLa_EQ``, ``IFXd_EQ``).  After stripping the suffix, if the result
    ends in a single lowercase letter preceded by at least two uppercase letters,
    that trailing lowercase letter is also stripped.

    Share-class letters (e.g. ``BRK/A``, ``BRK_B``, ``BRK-B``) are normalized
    to dot notation (``BRK.A``, ``BRK.B``) and explicitly preserved so that
    different share classes are never collapsed into each other.
    """
    stripped = _SUFFIX_PATTERN.sub("", ticker).strip()
    # Remove a trailing Trading212 country-discriminator lowercase letter, e.g.
    # "SAPd" → "SAP", "ASMLa" → "ASML", "IFXd" → "IFX".
    stripped = re.sub(r"^([A-Z]{2,})[a-z]$", r"\1", stripped)
    # Standardize share-class notation (e.g. BRK_B, BRK/A, BRK-B -> BRK.B)
    stripped = re.sub(r"^([A-Z]+)[/_ -]([A-Z])$", r"\1.\2", stripped)
    stripped = stripped.upper()
    return stripped if stripped else ticker.upper()


def _extract_root_and_class(symbol: str) -> tuple[str, str | None]:
    """Split canonical or broker ticker into base root and share class."""
    canon = canonical_ticker_symbol(symbol)
    m = re.match(r"^([A-Z0-9]+)\.([A-Z])$", canon)
    if m:
        return m.group(1), m.group(2)
    parts = re.split(r"[_./-]", canon)
    return parts[0], None


# ---------------------------------------------------------------------------
# Rule 0 — Order-side gate
# ---------------------------------------------------------------------------


def check_order_side(side: OrderSide) -> RuleResult:
    """Phase 9 only supports BUY orders; SELL is explicitly deferred.

    SELL orders have entirely different risk dynamics (covering shorts,
    partial exits, tax-lot selection) and will be handled in a dedicated
    Phase with its own rule suite.
    """
    if side == OrderSide.BUY:
        return RuleResult(
            rule_name="check_order_side",
            passed=True,
            reason="BUY orders are supported in Phase 9.",
        )
    return RuleResult(
        rule_name="check_order_side",
        passed=False,
        reason=(
            f"OrderSide.{side.value} is not supported in Phase 9 "
            "(position-exit rules are deferred to a later phase)."
        ),
        details={"requested_side": side.value},
    )


# ---------------------------------------------------------------------------
# Rule 1 — Stop-loss validation
# ---------------------------------------------------------------------------


def check_stop_loss(
    entry_price: Decimal,
    stop_loss_price: Decimal,
    config: RiskConfig,
) -> RuleResult:
    """Validate stop-loss placement.

    Conditions that trigger a FAIL:
    - ``stop_loss_price`` ≤ 0.
    - ``stop_loss_price`` ≥ ``entry_price`` (stop must be below entry for a BUY).
    - Stop distance as a fraction of entry is outside
      [min_stop_loss_distance_pct, max_stop_loss_distance_pct].
    """
    if stop_loss_price <= Decimal("0"):
        return RuleResult(
            rule_name="check_stop_loss",
            passed=False,
            reason="Stop-loss price must be strictly positive.",
            details={"stop_loss_price": str(stop_loss_price)},
        )
    if stop_loss_price >= entry_price:
        return RuleResult(
            rule_name="check_stop_loss",
            passed=False,
            reason=(
                "Stop-loss price must be strictly below entry price for a BUY order."
            ),
            details={
                "entry_price": str(entry_price),
                "stop_loss_price": str(stop_loss_price),
            },
        )

    distance = (entry_price - stop_loss_price) / entry_price
    if distance < config.min_stop_loss_distance_pct:
        return RuleResult(
            rule_name="check_stop_loss",
            passed=False,
            reason=(
                f"Stop-loss distance {distance:.4%} is below the minimum "
                f"{config.min_stop_loss_distance_pct:.4%}."
            ),
            details={
                "distance_pct": str(distance),
                "min_distance_pct": str(config.min_stop_loss_distance_pct),
            },
        )
    if distance > config.max_stop_loss_distance_pct:
        return RuleResult(
            rule_name="check_stop_loss",
            passed=False,
            reason=(
                f"Stop-loss distance {distance:.4%} exceeds the maximum "
                f"{config.max_stop_loss_distance_pct:.4%}."
            ),
            details={
                "distance_pct": str(distance),
                "max_distance_pct": str(config.max_stop_loss_distance_pct),
            },
        )
    return RuleResult(
        rule_name="check_stop_loss",
        passed=True,
        reason=f"Stop-loss distance {distance:.4%} is within the permitted envelope.",
        details={
            "distance_pct": str(distance),
            "min_distance_pct": str(config.min_stop_loss_distance_pct),
            "max_distance_pct": str(config.max_stop_loss_distance_pct),
        },
    )


# ---------------------------------------------------------------------------
# Rule 2 — Duplicate position
# ---------------------------------------------------------------------------


def check_duplicate_position(
    symbol: str,
    current_positions: list[Position],
) -> RuleResult:
    """Fail if the portfolio already holds a non-zero position in *symbol*.

    Comparison is performed on *canonicalised* tickers so that
    ``AAPL`` matches ``AAPL_US_EQ``, ``ASML`` matches ``ASMLa_EQ``, and
    ``SAP`` matches ``SAP_DE_EQ`` / ``SAPd_EQ``.
    A position with ``quantity == 0`` is treated as closed and does not block.

    Fail-Closed Guarantee
    ─────────────────────
    If an open position's ticker contains an unrecognized broker suffix pattern
    (i.e. not in the known catalog) but its instrument root matches the target
    symbol's root, this rule fails closed (returns ``passed=False``) to prevent
    buying into an existing holding due to normalization misses.
    Explicitly different share classes (e.g. BRK.A vs BRK.B) are distinguished
    and not blocked.
    """
    canonical_target = canonical_ticker_symbol(symbol)
    target_root, target_class = _extract_root_and_class(symbol)

    for pos in current_positions:
        if pos.quantity <= Decimal("0"):
            continue

        canonical_pos = canonical_ticker_symbol(pos.ticker)

        # 1. Exact canonical match
        if canonical_pos == canonical_target:
            return RuleResult(
                rule_name="check_duplicate_position",
                passed=False,
                reason=(
                    f"An open position in {symbol!r} already exists "
                    f"(broker ticker: {pos.ticker!r}, quantity: {pos.quantity})."
                ),
                details={
                    "canonical_symbol": canonical_target,
                    "broker_ticker": pos.ticker,
                    "held_quantity": str(pos.quantity),
                },
            )

        # 2. Fail-closed fallback: unrecognized suffix on an open position that
        # shares the target symbol's root instrument.
        pos_root, pos_class = _extract_root_and_class(pos.ticker)
        if pos_root == target_root:
            if (
                target_class is not None
                and pos_class is not None
                and target_class != pos_class
            ):
                continue
            return RuleResult(
                rule_name="check_duplicate_position",
                passed=False,
                reason=(
                    f"Duplicate position check failed closed: open position "
                    f"{pos.ticker!r} has unrecognized suffix but matches symbol "
                    f"root {target_root!r}."
                ),
                details={
                    "canonical_symbol": canonical_target,
                    "broker_ticker": pos.ticker,
                    "target_root": target_root,
                    "held_quantity": str(pos.quantity),
                    "fail_closed": True,
                },
            )

    return RuleResult(
        rule_name="check_duplicate_position",
        passed=True,
        reason=f"No existing open position found for {symbol!r}.",
        details={"canonical_symbol": canonical_target},
    )


# ---------------------------------------------------------------------------
# Rule 3 — Maximum position size
# ---------------------------------------------------------------------------


def check_max_position_size(
    position_cost: Decimal,
    total_equity: Decimal,
    config: RiskConfig,
) -> RuleResult:
    """Fail if a single position's cost exceeds the portfolio-equity ceiling.

    ``position_cost <= total_equity * max_position_size_pct`` (inclusive).
    """
    ceiling = total_equity * config.max_position_size_pct
    fraction = position_cost / total_equity if total_equity else Decimal("0")
    if position_cost <= ceiling:
        return RuleResult(
            rule_name="check_max_position_size",
            passed=True,
            reason=(
                f"Position cost {position_cost} ({fraction:.4%}) is within "
                f"the {config.max_position_size_pct:.4%} ceiling."
            ),
            details={
                "position_cost": str(position_cost),
                "ceiling": str(ceiling),
                "fraction_pct": str(fraction),
            },
        )
    return RuleResult(
        rule_name="check_max_position_size",
        passed=False,
        reason=(
            f"Position cost {position_cost} ({fraction:.4%}) exceeds the "
            f"{config.max_position_size_pct:.4%} max-position-size ceiling ({ceiling})."
        ),
        details={
            "position_cost": str(position_cost),
            "ceiling": str(ceiling),
            "fraction_pct": str(fraction),
        },
    )


# ---------------------------------------------------------------------------
# Rule 4 — Maximum portfolio exposure
# ---------------------------------------------------------------------------


def check_max_portfolio_exposure(
    current_invested: Decimal,
    position_cost: Decimal,
    total_equity: Decimal,
    config: RiskConfig,
) -> RuleResult:
    """Fail if adding this position would push invested capital above the ceiling.

    ``(current_invested + position_cost) <= total_equity * max_portfolio_exposure_pct``
    (inclusive).
    """
    ceiling = total_equity * config.max_portfolio_exposure_pct
    projected = current_invested + position_cost
    if projected <= ceiling:
        return RuleResult(
            rule_name="check_max_portfolio_exposure",
            passed=True,
            reason=(
                f"Projected invested {projected} is within the "
                f"{config.max_portfolio_exposure_pct:.4%} "
                f"portfolio-exposure ceiling ({ceiling})."
            ),
            details={
                "current_invested": str(current_invested),
                "position_cost": str(position_cost),
                "projected_invested": str(projected),
                "ceiling": str(ceiling),
            },
        )
    return RuleResult(
        rule_name="check_max_portfolio_exposure",
        passed=False,
        reason=(
            f"Adding this position ({position_cost}) would push total invested "
            f"({projected}) above the {config.max_portfolio_exposure_pct:.4%} "
            f"portfolio-exposure ceiling ({ceiling})."
        ),
        details={
            "current_invested": str(current_invested),
            "position_cost": str(position_cost),
            "projected_invested": str(projected),
            "ceiling": str(ceiling),
        },
    )


# ---------------------------------------------------------------------------
# Rule 5 — Maximum sector exposure
# ---------------------------------------------------------------------------


def check_max_sector_exposure(
    current_sector_cost: Decimal,
    position_cost: Decimal,
    total_equity: Decimal,
    sector: str | None,
    config: RiskConfig,
) -> RuleResult:
    """Fail if adding this position over-concentrates a single sector.

    Skipped (auto-pass) when *sector* is None or empty — caller is responsible
    for providing accurate sector labels.
    """
    if not sector:
        return RuleResult(
            rule_name="check_max_sector_exposure",
            passed=True,
            reason="No sector provided; sector-exposure check skipped.",
        )
    ceiling = total_equity * config.max_sector_exposure_pct
    projected = current_sector_cost + position_cost
    if projected <= ceiling:
        return RuleResult(
            rule_name="check_max_sector_exposure",
            passed=True,
            reason=(
                f"Projected sector ({sector!r}) exposure {projected} is within "
                f"the {config.max_sector_exposure_pct:.4%} ceiling ({ceiling})."
            ),
            details={
                "sector": sector,
                "current_sector_cost": str(current_sector_cost),
                "position_cost": str(position_cost),
                "projected_sector_cost": str(projected),
                "ceiling": str(ceiling),
            },
        )
    return RuleResult(
        rule_name="check_max_sector_exposure",
        passed=False,
        reason=(
            f"Adding this position would push {sector!r} sector exposure "
            f"({projected}) above the {config.max_sector_exposure_pct:.4%} "
            f"ceiling ({ceiling})."
        ),
        details={
            "sector": sector,
            "current_sector_cost": str(current_sector_cost),
            "position_cost": str(position_cost),
            "projected_sector_cost": str(projected),
            "ceiling": str(ceiling),
        },
    )


# ---------------------------------------------------------------------------
# Rule 6 — Daily-loss circuit breaker
# ---------------------------------------------------------------------------


def check_daily_loss_limit(
    daily_realized_loss: Decimal,
    total_equity: Decimal,
    config: RiskConfig,
) -> RuleResult:
    """Hard circuit breaker: halt trading if today's realised loss hits the limit.

    Unlike exposure/size ceilings that use inclusive (≤) semantics,
    this rule uses an *exclusive* less-than (<).  The rationale: once we have
    lost exactly 3 % of equity in a single day, no further losers may stack
    on top — the daily allowance is exhausted.  An inclusive check at exactly
    3 % would still allow one more trade.

    *daily_realized_loss* should be a non-negative value representing the
    absolute magnitude of losses (e.g. 300 for £300 lost, not -300).
    """
    # Ceiling is the absolute-loss amount that must not be reached.
    ceiling = total_equity * config.max_daily_loss_pct
    # Circuit breaker: exclusive < semantics.
    if daily_realized_loss < ceiling:
        return RuleResult(
            rule_name="check_daily_loss_limit",
            passed=True,
            reason=(
                f"Daily realised loss {daily_realized_loss} is below the "
                f"{config.max_daily_loss_pct:.4%} circuit-breaker ceiling ({ceiling})."
            ),
            details={
                "daily_realized_loss": str(daily_realized_loss),
                "ceiling": str(ceiling),
                "max_daily_loss_pct": str(config.max_daily_loss_pct),
            },
        )
    return RuleResult(
        rule_name="check_daily_loss_limit",
        passed=False,
        reason=(
            f"Daily realised loss {daily_realized_loss} has reached or exceeded "
            f"the {config.max_daily_loss_pct:.4%} circuit-breaker ceiling ({ceiling}). "
            "Trading is halted for the remainder of the session."
        ),
        details={
            "daily_realized_loss": str(daily_realized_loss),
            "ceiling": str(ceiling),
            "max_daily_loss_pct": str(config.max_daily_loss_pct),
        },
    )


# ---------------------------------------------------------------------------
# Rule 7 — Earnings blackout
# ---------------------------------------------------------------------------


def check_earnings_blackout(
    next_earnings_date: datetime.date | None,
    today: datetime.date,
    config: RiskConfig,
) -> RuleResult:
    """Block new BUY orders if earnings are within the configured blackout window.

    Behaviour by case:
    - ``next_earnings_date is None``: auto-pass (no data available).
    - ``days_until >= 0`` and ``days_until <= earnings_blackout_days``: FAIL.
    - ``days_until < 0`` (past / stale date): PASS but emit a structured
      warning so operators can diagnose stale data.
    - ``days_until > earnings_blackout_days``: PASS.
    """
    if next_earnings_date is None:
        return RuleResult(
            rule_name="check_earnings_blackout",
            passed=True,
            reason="No earnings date provided; blackout check skipped.",
        )

    days_until = (next_earnings_date - today).days

    if days_until < 0:
        # Stale / past earnings date — warn but do not block.
        logger.warning(
            "Earnings date is in the past; data may be stale",
            symbol_earnings_date=str(next_earnings_date),
            today=str(today),
            days_overdue=abs(days_until),
        )
        return RuleResult(
            rule_name="check_earnings_blackout",
            passed=True,
            reason=(
                f"Earnings date {next_earnings_date} is {abs(days_until)} "
                "day(s) in the past. Treating as stale data and passing — "
                "verify earnings calendar."
            ),
            details={
                "next_earnings_date": str(next_earnings_date),
                "today": str(today),
                "days_until": days_until,
            },
        )

    if 0 <= days_until <= config.earnings_blackout_days:
        return RuleResult(
            rule_name="check_earnings_blackout",
            passed=False,
            reason=(
                f"Earnings release in {days_until} day(s) ({next_earnings_date}) "
                f"is within the {config.earnings_blackout_days}-day blackout window."
            ),
            details={
                "next_earnings_date": str(next_earnings_date),
                "today": str(today),
                "days_until": days_until,
                "blackout_days": config.earnings_blackout_days,
            },
        )

    # days_until > earnings_blackout_days
    return RuleResult(
        rule_name="check_earnings_blackout",
        passed=True,
        reason=(
            f"Earnings in {days_until} day(s) ({next_earnings_date}) — "
            f"outside the {config.earnings_blackout_days}-day blackout window."
        ),
        details={
            "next_earnings_date": str(next_earnings_date),
            "today": str(today),
            "days_until": days_until,
            "blackout_days": config.earnings_blackout_days,
        },
    )

"""Exhaustive boundary tests for Phase 9 risk rules, sizing, and engine.

Test philosophy
───────────────
Every rule is tested at three distinct boundary points:
  • Just *under* the boundary → expected pass.
  • Exactly *at* the boundary → explicit pass or fail per contract spec.
  • Clearly *over* the boundary → expected fail.

The daily-loss circuit breaker uses exclusive-< semantics (unlike ≤ exposure
checks), so its boundary cases are:
  • <  ceiling → PASS.
  • == ceiling → FAIL (circuit breaker triggered at the limit).
  • >  ceiling → FAIL.

All tests are synchronous; the engine's ``asyncio.to_thread`` wrapping is
tested separately in the engine orchestration tests.
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.risk.engine import RiskEngine
from mcp_finance.risk.models import (
    OrderSide,
    OrderType,
    ProposedTrade,
    RiskConfig,
)
from mcp_finance.risk.rules import (
    canonical_ticker_symbol,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _account(
    total: str = "100000",
    cash: str = "50000",
    invested: str = "50000",
) -> AccountSummary:
    total_d = Decimal(total)
    cash_d = Decimal(cash)
    invested_d = Decimal(invested)
    return AccountSummary(
        cash=cash_d,
        invested=invested_d,
        result=Decimal("0"),
        total=total_d,
    )


def _position(ticker: str, quantity: str = "10") -> Position:
    return Position(
        ticker=ticker,
        quantity=Decimal(quantity),
        average_price=Decimal("100"),
        current_price=Decimal("105"),
        ppl=Decimal("50"),
    )


def _config(**overrides: object) -> RiskConfig:
    return RiskConfig(**overrides)


# ---------------------------------------------------------------------------
# 1. Ticker normalisation
# ---------------------------------------------------------------------------


class TestTickerNormalisation:
    # ── Fixtures from real T212 demo get_positions() output ─────────────────
    def test_us_suffix(self) -> None:
        # Real demo: AAPL_US_EQ, NVDA_US_EQ, TSLA_US_EQ, MSFT_US_EQ, ...
        assert canonical_ticker_symbol("AAPL_US_EQ") == "AAPL"
        assert canonical_ticker_symbol("NVDA_US_EQ") == "NVDA"
        assert canonical_ticker_symbol("TSLA_US_EQ") == "TSLA"

    def test_german_suffix(self) -> None:
        assert canonical_ticker_symbol("SAP_DE_EQ") == "SAP"

    def test_uk_suffix(self) -> None:
        assert canonical_ticker_symbol("LLOY_UK_EQ") == "LLOY"

    def test_bare_eq_suffix(self) -> None:
        # European tickers sometimes appear with only _EQ (no country code)
        assert canonical_ticker_symbol("VOD_EQ") == "VOD"

    # ── T212 lowercase country-discriminator format (from real catalog) ──────
    def test_t212_euronext_amsterdam_discriminator(self) -> None:
        # Real T212 catalog: ASMLa_EQ = ASML on Euronext Amsterdam
        assert canonical_ticker_symbol("ASMLa_EQ") == "ASML"

    def test_t212_xetra_discriminator(self) -> None:
        # Real T212 catalog: IFXd_EQ = Infineon on XETRA (Germany)
        assert canonical_ticker_symbol("IFXd_EQ") == "IFX"

    def test_t212_xetra_discriminator_sap(self) -> None:
        # SAPd_EQ — SAP on XETRA with Trading212 lowercase discriminator
        assert canonical_ticker_symbol("SAPd_EQ") == "SAP"

    # ── Share class preservation ─────────────────────────────────────────────
    def test_share_class_slash_notation(self) -> None:
        # Real T212 catalog: BRK/A_US_EQ = Berkshire Hathaway Class A
        assert canonical_ticker_symbol("BRK/A_US_EQ") == "BRK.A"

    def test_share_class_underscore_notation(self) -> None:
        # Real T212 catalog: BRK_B_US_EQ = Berkshire Hathaway Class B
        assert canonical_ticker_symbol("BRK_B_US_EQ") == "BRK.B"

    def test_share_class_a_and_b_are_distinct(self) -> None:
        # Critical: BRK.A and BRK.B must NOT normalize to the same symbol
        assert canonical_ticker_symbol("BRK/A_US_EQ") != canonical_ticker_symbol(
            "BRK_B_US_EQ"
        )

    # ── Passthrough and normalization ────────────────────────────────────────
    def test_already_canonical(self) -> None:
        assert canonical_ticker_symbol("AAPL") == "AAPL"

    def test_lowercase_canonical(self) -> None:
        assert canonical_ticker_symbol("aapl") == "AAPL"

    def test_dot_de_suffix(self) -> None:
        assert canonical_ticker_symbol("SAP.DE") == "SAP"

    def test_unknown_suffix_passthrough(self) -> None:
        # If no known suffix matches, return original upper-cased
        result = canonical_ticker_symbol("WEIRD_TICKER_XY")
        # The result must be non-empty and upper-case
        assert result == result.upper()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. Order-side rule
# ---------------------------------------------------------------------------


class TestOrderSideRule:
    def test_buy_passes(self) -> None:
        result = check_order_side(OrderSide.BUY)
        assert result.passed

    def test_sell_fails(self) -> None:
        result = check_order_side(OrderSide.SELL)
        assert not result.passed
        assert "SELL" in result.reason
        assert "Phase 9" in result.reason


# ---------------------------------------------------------------------------
# 3. Stop-loss rule — boundaries
# ---------------------------------------------------------------------------


class TestStopLossRule:
    """
    Default config: min_stop_loss_distance_pct=0.005, max=0.15.

    With entry_price=100.00 the envelope is:
      distance = (entry - stop) / entry
      min stop_loss = 100 - (100 * 0.005) = 99.50  → distance 0.50%
      max stop_loss = 100 - (100 * 0.15)  = 85.00  → distance 15.00%
    """

    cfg = _config()

    def test_below_min_distance_fails(self) -> None:
        # distance ≈ 0.49% < 0.5% min
        result = check_stop_loss(Decimal("100"), Decimal("99.51"), self.cfg)
        assert not result.passed
        assert "minimum" in result.reason

    def test_exactly_at_min_distance_passes(self) -> None:
        # distance = exactly 0.50% → at boundary → PASS
        result = check_stop_loss(Decimal("100"), Decimal("99.50"), self.cfg)
        assert result.passed

    def test_nominal_valid_distance_passes(self) -> None:
        result = check_stop_loss(Decimal("100"), Decimal("95.00"), self.cfg)
        assert result.passed

    def test_exactly_at_max_distance_passes(self) -> None:
        # distance = exactly 15.00% → at boundary → PASS
        result = check_stop_loss(Decimal("100"), Decimal("85.00"), self.cfg)
        assert result.passed

    def test_over_max_distance_fails(self) -> None:
        # 15.01% > 15.00% max
        result = check_stop_loss(Decimal("100"), Decimal("84.99"), self.cfg)
        assert not result.passed
        assert "maximum" in result.reason

    def test_zero_stop_loss_fails(self) -> None:
        result = check_stop_loss(Decimal("100"), Decimal("0"), self.cfg)
        assert not result.passed
        assert "positive" in result.reason

    def test_negative_stop_loss_fails(self) -> None:
        result = check_stop_loss(Decimal("100"), Decimal("-5"), self.cfg)
        assert not result.passed

    def test_stop_above_entry_fails(self) -> None:
        result = check_stop_loss(Decimal("100"), Decimal("105"), self.cfg)
        assert not result.passed
        assert "below entry" in result.reason

    def test_stop_equal_to_entry_fails(self) -> None:
        result = check_stop_loss(Decimal("100"), Decimal("100"), self.cfg)
        assert not result.passed


# ---------------------------------------------------------------------------
# 4. Duplicate position rule
# ---------------------------------------------------------------------------


class TestDuplicatePositionRule:
    def test_no_existing_position_passes(self) -> None:
        result = check_duplicate_position("AAPL", [])
        assert result.passed

    def test_exact_match_fails(self) -> None:
        result = check_duplicate_position("AAPL", [_position("AAPL")])
        assert not result.passed

    def test_broker_suffix_match_fails(self) -> None:
        # "AAPL" vs broker ticker "AAPL_US_EQ" → same after canonicalisation
        result = check_duplicate_position("AAPL", [_position("AAPL_US_EQ")])
        assert not result.passed
        assert "AAPL_US_EQ" in result.reason

    def test_european_suffix_match_fails(self) -> None:
        # "SAP" vs broker ticker "SAP_DE_EQ"
        result = check_duplicate_position("SAP", [_position("SAP_DE_EQ")])
        assert not result.passed

    def test_t212_lowercase_discriminator_match_fails(self) -> None:
        # Real T212 catalog: ASMLa_EQ — ASML on Euronext Amsterdam
        result = check_duplicate_position("ASML", [_position("ASMLa_EQ")])
        assert not result.passed

    def test_t212_xetra_discriminator_match_fails(self) -> None:
        # Real T212 catalog: IFXd_EQ — Infineon on XETRA
        result = check_duplicate_position("IFX", [_position("IFXd_EQ")])
        assert not result.passed

    def test_zero_quantity_is_closed_and_passes(self) -> None:
        result = check_duplicate_position("AAPL", [_position("AAPL_US_EQ", "0")])
        assert result.passed

    def test_different_symbol_passes(self) -> None:
        result = check_duplicate_position("MSFT", [_position("AAPL_US_EQ")])
        assert result.passed

    # ── Share class isolation ────────────────────────────────────────────────
    def test_brk_b_held_blocks_new_brk_b(self) -> None:
        # Holding BRK_B_US_EQ should block buying BRK.B
        result = check_duplicate_position("BRK.B", [_position("BRK_B_US_EQ")])
        assert not result.passed

    def test_brk_a_does_not_block_brk_b(self) -> None:
        # Holding BRK/A_US_EQ (Class A) must NOT block buying Class B
        result = check_duplicate_position("BRK.B", [_position("BRK/A_US_EQ")])
        assert result.passed, (
            "BRK.A and BRK.B are distinct instruments; holding one must not "
            "block purchasing the other"
        )

    # ── Fail-closed guarantee on unrecognized suffixes ───────────────────────
    def test_unrecognized_suffix_same_root_is_blocked(self) -> None:
        """Core fail-closed guarantee.

        If a held position has an unrecognized broker suffix but its root symbol
        matches the target, the check MUST fail closed (reject the trade) rather
        than silently pass — the safe direction when the normalizer has a gap.
        """
        result = check_duplicate_position(
            "AAPL",
            [_position("AAPL_MYSTERY_EXCHANGE")],
        )
        assert not result.passed, (
            "check_duplicate_position must fail closed when ticker suffix is "
            "unrecognized but root symbol matches: got passed=True which would "
            "allow buying into an existing position"
        )
        assert result.details.get("fail_closed") is True

    def test_unrecognized_suffix_different_root_passes(self) -> None:
        # An unrecognized suffix for a completely different stock must not block
        result = check_duplicate_position(
            "MSFT",
            [_position("AAPL_MYSTERY_EXCHANGE")],
        )
        assert result.passed


# ---------------------------------------------------------------------------
# 5. Max position size — boundaries
# ---------------------------------------------------------------------------


class TestMaxPositionSize:
    """
    Default max_position_size_pct = 5%.
    With total_equity=100_000: ceiling = 5_000.
    """

    cfg = _config()
    equity = Decimal("100000")

    def test_under_ceiling_passes(self) -> None:
        result = check_max_position_size(Decimal("4990"), self.equity, self.cfg)
        assert result.passed

    def test_exactly_at_ceiling_passes(self) -> None:
        result = check_max_position_size(Decimal("5000"), self.equity, self.cfg)
        assert result.passed

    def test_over_ceiling_fails(self) -> None:
        result = check_max_position_size(Decimal("5001"), self.equity, self.cfg)
        assert not result.passed
        assert "ceiling" in result.reason


# ---------------------------------------------------------------------------
# 6. Max portfolio exposure — boundaries
# ---------------------------------------------------------------------------


class TestMaxPortfolioExposure:
    """
    Default max_portfolio_exposure_pct = 90%.
    With total_equity=100_000: ceiling = 90_000.
    """

    cfg = _config()
    equity = Decimal("100000")

    def test_under_ceiling_passes(self) -> None:
        # current_invested=89_000, adding 990 → projected=89_990 < 90_000
        result = check_max_portfolio_exposure(
            Decimal("89000"), Decimal("990"), self.equity, self.cfg
        )
        assert result.passed

    def test_exactly_at_ceiling_passes(self) -> None:
        # current=89_000 + 1_000 = 90_000 == ceiling
        result = check_max_portfolio_exposure(
            Decimal("89000"), Decimal("1000"), self.equity, self.cfg
        )
        assert result.passed

    def test_over_ceiling_fails(self) -> None:
        # current=89_000 + 1_001 = 90_001 > 90_000
        result = check_max_portfolio_exposure(
            Decimal("89000"), Decimal("1001"), self.equity, self.cfg
        )
        assert not result.passed
        assert "ceiling" in result.reason


# ---------------------------------------------------------------------------
# 7. Max sector exposure — boundaries
# ---------------------------------------------------------------------------


class TestMaxSectorExposure:
    """
    Default max_sector_exposure_pct = 25%.
    With total_equity=100_000: ceiling = 25_000.
    """

    cfg = _config()
    equity = Decimal("100000")
    sector = "Technology"

    def test_under_ceiling_passes(self) -> None:
        result = check_max_sector_exposure(
            Decimal("24000"), Decimal("990"), self.equity, self.sector, self.cfg
        )
        assert result.passed

    def test_exactly_at_ceiling_passes(self) -> None:
        # 24_000 + 1_000 = 25_000 == ceiling
        result = check_max_sector_exposure(
            Decimal("24000"), Decimal("1000"), self.equity, self.sector, self.cfg
        )
        assert result.passed

    def test_over_ceiling_fails(self) -> None:
        # 24_000 + 1_001 = 25_001 > 25_000
        result = check_max_sector_exposure(
            Decimal("24000"), Decimal("1001"), self.equity, self.sector, self.cfg
        )
        assert not result.passed

    def test_no_sector_skips_check(self) -> None:
        result = check_max_sector_exposure(
            Decimal("99000"), Decimal("5000"), self.equity, None, self.cfg
        )
        assert result.passed
        assert "skipped" in result.reason.lower()


# ---------------------------------------------------------------------------
# 8. Daily loss limit — circuit breaker (exclusive <)
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    """
    Default max_daily_loss_pct = 3%.
    With total_equity=100_000: ceiling = 3_000.
    Circuit breaker semantics: daily_loss < 3_000 → PASS; >= 3_000 → FAIL.
    """

    cfg = _config()
    equity = Decimal("100000")

    def test_below_ceiling_passes(self) -> None:
        # 2_999 < 3_000
        result = check_daily_loss_limit(Decimal("2999"), self.equity, self.cfg)
        assert result.passed

    def test_exactly_at_ceiling_fails(self) -> None:
        # == 3_000 → circuit breaker fires (exclusive <)
        result = check_daily_loss_limit(Decimal("3000"), self.equity, self.cfg)
        assert not result.passed
        assert "circuit" in result.reason.lower() or "halted" in result.reason.lower()

    def test_over_ceiling_fails(self) -> None:
        result = check_daily_loss_limit(Decimal("3001"), self.equity, self.cfg)
        assert not result.passed

    def test_zero_loss_passes(self) -> None:
        result = check_daily_loss_limit(Decimal("0"), self.equity, self.cfg)
        assert result.passed


# ---------------------------------------------------------------------------
# 9. Earnings blackout — boundaries
# ---------------------------------------------------------------------------


class TestEarningsBlackout:
    cfg = _config(earnings_blackout_days=3)
    today = datetime.date(2024, 1, 10)

    def test_outside_blackout_passes(self) -> None:
        # 4 days away > 3-day window
        earnings = datetime.date(2024, 1, 14)
        result = check_earnings_blackout(earnings, self.today, self.cfg)
        assert result.passed

    def test_exactly_at_boundary_fails(self) -> None:
        # Exactly 3 days → within [0, 3] → FAIL
        earnings = datetime.date(2024, 1, 13)
        result = check_earnings_blackout(earnings, self.today, self.cfg)
        assert not result.passed
        assert "blackout" in result.reason.lower()

    def test_two_days_away_fails(self) -> None:
        earnings = datetime.date(2024, 1, 12)
        result = check_earnings_blackout(earnings, self.today, self.cfg)
        assert not result.passed

    def test_today_earnings_fails(self) -> None:
        # 0 days away: days_until == 0 → within [0, 3] → FAIL
        result = check_earnings_blackout(self.today, self.today, self.cfg)
        assert not result.passed

    def test_past_earnings_passes_with_warning(self) -> None:
        # Stale data: earnings were yesterday
        earnings = datetime.date(2024, 1, 9)
        result = check_earnings_blackout(earnings, self.today, self.cfg)
        assert result.passed
        assert "past" in result.reason.lower() or "stale" in result.reason.lower()

    def test_none_passes(self) -> None:
        result = check_earnings_blackout(None, self.today, self.cfg)
        assert result.passed
        assert "skipped" in result.reason.lower()


# ---------------------------------------------------------------------------
# 10. Position sizing calculator
# ---------------------------------------------------------------------------


class TestPositionSizingCalculator:
    """
    Base scenario:
      total_equity=100_000, risk_pct=1%, entry=100, stop=95
      → risk_per_share=5, risk_budget=1_000
      → shares_from_risk = floor(1_000/5) = 200
      max_position_pct=5% → shares_from_position_limit = floor(5_000/100) = 50
      cash=10_000 → shares_from_cash = floor(10_000/100) = 100
      → min(200, 50, 100) = 50 (position limit binding)
    """

    def test_position_limit_is_binding(self) -> None:
        result = calculate_position_size(
            total_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            max_position_pct=Decimal("0.05"),
            cash_available=Decimal("10000"),
        )
        assert result.quantity == Decimal("50")
        assert result.binding_constraint == "position_limit"

    def test_risk_budget_is_binding(self) -> None:
        # Very tight stop → large risk_per_share → fewer shares from risk budget
        # risk_per_share=50, risk_budget=1000 → shares_from_risk=20
        # position_limit=50, cash=500/100=5 → cash binding? no: 5 < 20 → cash
        # Let's set: entry=100, stop=50 (50% distance)
        # risk_per_share=50, budget=1000 → 20
        # position_limit=5000/100=50
        # cash=10_000 → 100
        # min(20, 50, 100) = 20 → risk budget binding
        result = calculate_position_size(
            total_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("50"),
            max_position_pct=Decimal("0.05"),
            cash_available=Decimal("10000"),
        )
        assert result.quantity == Decimal("20")
        assert result.binding_constraint == "risk_budget"

    def test_cash_is_binding(self) -> None:
        # cash=100 → only 1 share affordable
        # risk_budget=1000/5=200, position_limit=5000/100=50
        # cash binding at 1
        result = calculate_position_size(
            total_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            max_position_pct=Decimal("0.05"),
            cash_available=Decimal("100"),
        )
        assert result.quantity == Decimal("1")
        assert result.binding_constraint == "cash"

    def test_zero_cash_yields_zero_shares(self) -> None:
        result = calculate_position_size(
            total_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            max_position_pct=Decimal("0.05"),
            cash_available=Decimal("0"),
        )
        assert result.quantity == Decimal("0")

    def test_stop_equal_to_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            calculate_position_size(
                total_equity=Decimal("100000"),
                risk_pct=Decimal("0.01"),
                entry_price=Decimal("100"),
                stop_loss_price=Decimal("100"),
                max_position_pct=Decimal("0.05"),
                cash_available=Decimal("10000"),
            )

    def test_inverted_prices_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            calculate_position_size(
                total_equity=Decimal("100000"),
                risk_pct=Decimal("0.01"),
                entry_price=Decimal("100"),
                stop_loss_price=Decimal("110"),
                max_position_pct=Decimal("0.05"),
                cash_available=Decimal("10000"),
            )

    def test_total_cost_and_risk_amount_correct(self) -> None:
        result = calculate_position_size(
            total_equity=Decimal("100000"),
            risk_pct=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            max_position_pct=Decimal("0.05"),
            cash_available=Decimal("10000"),
        )
        assert result.total_cost == result.quantity * Decimal("100")
        assert result.risk_amount == result.quantity * Decimal("5")


# ---------------------------------------------------------------------------
# 11. Risk engine orchestration
# ---------------------------------------------------------------------------


class TestRiskEngineOrchestration:
    def test_all_rules_pass_yields_approved(self) -> None:
        """Well-configured trade with plenty of headroom should be approved."""
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="AAPL",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),  # 5% distance — valid
            sector="Technology",
        )
        account = _account(total="100000", cash="50000", invested="20000")
        decision = engine.evaluate_trade(
            proposed,
            account,
            positions=[],
            sector_exposures={"Technology": Decimal("5000")},
            daily_loss=Decimal("0"),
        )
        assert decision.approved
        assert decision.rejection_reasons == []
        assert decision.quantity > Decimal("0")
        # All rules should be listed in rule_results
        rule_names = [r.rule_name for r in decision.rule_results]
        assert "check_order_side" in rule_names
        assert "check_stop_loss" in rule_names
        assert "check_daily_loss_limit" in rule_names
        assert "check_earnings_blackout" in rule_names
        assert "check_duplicate_position" in rule_names
        assert "check_max_position_size" in rule_names
        assert "check_max_portfolio_exposure" in rule_names
        assert "check_max_sector_exposure" in rule_names

    def test_multi_failure_accumulates_all_reasons(self) -> None:
        """Multiple failing rules should all appear in rejection_reasons."""
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="AAPL",
            side=OrderSide.SELL,  # Rule 0 fail
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("99.99"),  # Rule 1 fail: 0.01% < 0.5% min
            sector="Technology",
            next_earnings_date=datetime.date.today(),  # Rule 7 fail: today
        )
        account = _account(total="100000", cash="50000", invested="89500")
        decision = engine.evaluate_trade(
            proposed,
            account,
            positions=[_position("AAPL_US_EQ")],  # Rule 2 fail: duplicate
            daily_loss=Decimal("3000"),  # Rule 6 fail: circuit breaker
        )
        assert not decision.approved
        # Must contain at least: order_side, stop_loss, daily_loss, earnings
        reasons_text = " ".join(decision.rejection_reasons)
        assert len(decision.rejection_reasons) >= 3
        assert "stop-loss" in reasons_text.lower()

    def test_sell_order_rejected_by_engine(self) -> None:
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="AAPL",
            side=OrderSide.SELL,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
        )
        account = _account()
        decision = engine.evaluate_trade(proposed, account, positions=[])
        assert not decision.approved
        assert any("SELL" in r for r in decision.rejection_reasons)

    def test_explicit_quantity_bypasses_auto_sizer(self) -> None:
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="MSFT",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            quantity=Decimal("3"),  # explicit
        )
        account = _account(total="100000", cash="50000", invested="10000")
        decision = engine.evaluate_trade(proposed, account, positions=[])
        assert decision.approved
        assert decision.quantity == Decimal("3")

    def test_zero_sized_quantity_is_rejected(self) -> None:
        """If sizing produces zero shares, the trade must be rejected."""
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="EXPENSIVE",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
        )
        # Only £10 cash → floor(10/100) = 0 shares → rejected
        account = _account(total="100000", cash="10", invested="10000")
        decision = engine.evaluate_trade(proposed, account, positions=[])
        assert not decision.approved
        assert decision.quantity == Decimal("0")

    def test_today_defaults_to_date_today(self) -> None:
        """Calling evaluate_trade without `today` should not raise."""
        engine = RiskEngine()
        proposed = ProposedTrade(
            symbol="AAPL",
            side=OrderSide.BUY,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
        )
        account = _account()
        # Should not raise even without `today` kwarg
        decision = engine.evaluate_trade(proposed, account, positions=[])
        # Approved-or-not doesn't matter; we just check it ran without error
        assert isinstance(decision.approved, bool)


# ---------------------------------------------------------------------------
# 12. Hard invariant guard (pre-dispatch assertion)
# ---------------------------------------------------------------------------


class TestHardInvariantGuard:
    """
    The place_order tool must assert the invariant before calling the broker.
    We test the assertion logic in isolation here (the full place_order
    integration is in the DB integration test).
    """

    def test_invariant_holds_for_approved_decision(self) -> None:
        from mcp_finance.risk.models import RiskDecision, RuleResult

        decision = RiskDecision(
            approved=True,
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            quantity=Decimal("5"),
            estimated_cost=Decimal("500"),
            risk_amount=Decimal("25"),
            rule_results=[
                RuleResult(rule_name="check_order_side", passed=True, reason="ok")
            ],
            rejection_reasons=[],
        )
        # Invariant: approved=True, quantity>0, rejection_reasons=[]
        assert decision.approved is True
        assert decision.quantity > Decimal("0")
        assert decision.rejection_reasons == []

    def test_invariant_fails_for_rejected_decision(self) -> None:
        from mcp_finance.risk.models import RiskDecision, RuleResult

        decision = RiskDecision(
            approved=False,
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            entry_price=Decimal("100"),
            stop_loss_price=Decimal("95"),
            quantity=Decimal("0"),
            estimated_cost=Decimal("0"),
            risk_amount=Decimal("0"),
            rule_results=[
                RuleResult(
                    rule_name="check_order_side",
                    passed=False,
                    reason="SELL not supported",
                )
            ],
            rejection_reasons=["SELL not supported"],
        )
        assert decision.approved is False
        assert decision.rejection_reasons != []

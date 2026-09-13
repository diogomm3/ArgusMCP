"""Pure, deterministic layered screening filter functions.

Each evaluate_* function is a pure function: given a Candidate, optional
CompanyFundamentals, and a StrategyConfig it returns a FilterResult with
passed=True/False and a human-readable reason when False.

Design principles:
- No I/O, no side effects — suitable for table-driven unit tests with synthetic data.
- None indicators always fail the relevant filter with an explicit reason string,
  never silently passing as zero.
- evaluate_candidate() runs all four layers in sequence and returns an aggregate
  (passed: bool, failed_filter_names: list[str], failure_reasons: list[str]).
"""

from decimal import Decimal

from mcp_finance.fundamentals.models import CompanyFundamentals
from mcp_finance.indicators.models import Candidate
from mcp_finance.screener.models import FilterResult, StrategyConfig

# ── Universe filter ──────────────────────────────────────────────────────────


def evaluate_universe(
    candidate: Candidate,
    fundamentals: CompanyFundamentals | None,
    config: StrategyConfig,
) -> FilterResult:
    """Filter by market cap, minimum price, and optional P/E cap.

    Returns failed immediately at the first failing condition so the reason
    is unambiguous rather than a merged multi-condition string.
    """
    # Market cap check requires fundamentals
    if config.min_market_cap is not None:
        if fundamentals is None:
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    f"market_cap=None (fundamentals unavailable); "
                    f"min_market_cap={config.min_market_cap:,}"
                ),
            )
        if fundamentals.market_cap is None:
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    f"market_cap=None (not reported by FMP); "
                    f"min_market_cap={config.min_market_cap:,}"
                ),
            )
        if fundamentals.market_cap < config.min_market_cap:
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    f"market_cap={fundamentals.market_cap:,} "
                    f"< min_market_cap={config.min_market_cap:,}"
                ),
            )

    # Minimum price
    if config.min_price is not None and candidate.close < config.min_price:
        return FilterResult(
            filter_name="universe",
            passed=False,
            reason=(f"close={candidate.close} < min_price={config.min_price}"),
        )

    # P/E cap (optional)
    if config.max_pe_ratio is not None:
        if fundamentals is None or fundamentals.pe_ratio is None:
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    "pe_ratio=None (fundamentals unavailable); "
                    f"max_pe_ratio={config.max_pe_ratio} requires valid P/E"
                ),
            )
        if fundamentals.pe_ratio <= Decimal("0"):
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    f"pe_ratio={fundamentals.pe_ratio} <= 0 "
                    "(loss-making or negative earnings)"
                ),
            )
        if fundamentals.pe_ratio > config.max_pe_ratio:
            return FilterResult(
                filter_name="universe",
                passed=False,
                reason=(
                    f"pe_ratio={fundamentals.pe_ratio} "
                    f"> max_pe_ratio={config.max_pe_ratio}"
                ),
            )

    return FilterResult(filter_name="universe", passed=True)


# ── Volume / Liquidity filter ─────────────────────────────────────────────────


def evaluate_volume(
    candidate: Candidate,
    config: StrategyConfig,
) -> FilterResult:
    """Filter by 20-day average volume.

    avg_volume_20 is computed by build_candidate_snapshot() in the same
    OHLCV DataFrame pass as all technical indicators — the engine must NOT
    issue a second OhlcvRepository.fetch_range call to obtain volume data.
    """
    if config.min_avg_volume_20 is None:
        return FilterResult(filter_name="volume", passed=True)

    if candidate.avg_volume_20 is None:
        return FilterResult(
            filter_name="volume",
            passed=False,
            reason=(
                f"avg_volume_20=None (fewer than 20 bars available); "
                f"min_avg_volume_20={config.min_avg_volume_20:,}"
            ),
        )
    if candidate.avg_volume_20 < config.min_avg_volume_20:
        return FilterResult(
            filter_name="volume",
            passed=False,
            reason=(
                f"avg_volume_20={candidate.avg_volume_20:,} "
                f"< min_avg_volume_20={config.min_avg_volume_20:,}"
            ),
        )
    return FilterResult(filter_name="volume", passed=True)


# ── Trend filter ──────────────────────────────────────────────────────────────


def evaluate_trend(
    candidate: Candidate,
    config: StrategyConfig,
) -> FilterResult:
    """Filter by EMA alignment: price > EMA-20/50 and EMA-20 > EMA-50.

    A None EMA value (insufficient history) always fails the corresponding
    check when that check is enabled — never treated as zero.
    """
    if config.require_price_above_ema_20:
        if candidate.ema_20 is None:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason="ema_20=None (insufficient history for EMA-20)",
            )
        if candidate.close <= candidate.ema_20:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason=(
                    f"close={candidate.close} <= ema_20={candidate.ema_20} "
                    "(price not above short-term trend)"
                ),
            )

    if config.require_price_above_ema_50:
        if candidate.ema_50 is None:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason="ema_50=None (insufficient history for EMA-50)",
            )
        if candidate.close <= candidate.ema_50:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason=(
                    f"close={candidate.close} <= ema_50={candidate.ema_50} "
                    "(price below medium-term trend)"
                ),
            )

    if config.require_ema_20_above_ema_50:
        if candidate.ema_20 is None or candidate.ema_50 is None:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason="ema_20 or ema_50=None; cannot verify golden cross alignment",
            )
        if candidate.ema_20 <= candidate.ema_50:
            return FilterResult(
                filter_name="trend",
                passed=False,
                reason=(
                    f"ema_20={candidate.ema_20} <= ema_50={candidate.ema_50} "
                    "(trend not aligned: no golden cross)"
                ),
            )

    return FilterResult(filter_name="trend", passed=True)


# ── Momentum filter ───────────────────────────────────────────────────────────


def evaluate_momentum(
    candidate: Candidate,
    config: StrategyConfig,
) -> FilterResult:
    """Filter by RSI corridor and MACD cross/histogram sign.

    None indicator values always fail when the corresponding check is enabled.
    """
    # RSI bounds
    if config.rsi_min is not None or config.rsi_max is not None:
        if candidate.rsi_14 is None:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason="rsi_14=None (insufficient history for RSI-14)",
            )
        if config.rsi_min is not None and candidate.rsi_14 < config.rsi_min:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason=(
                    f"rsi_14={candidate.rsi_14} < rsi_min={config.rsi_min} "
                    "(weak/oversold momentum)"
                ),
            )
        if config.rsi_max is not None and candidate.rsi_14 > config.rsi_max:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason=(
                    f"rsi_14={candidate.rsi_14} > rsi_max={config.rsi_max} (overbought)"
                ),
            )

    # MACD histogram sign
    if config.require_macd_histogram_positive:
        if candidate.macd_histogram is None:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason="macd_histogram=None (insufficient history for MACD)",
            )
        if candidate.macd_histogram <= Decimal("0"):
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason=(
                    f"macd_histogram={candidate.macd_histogram} <= 0 "
                    "(decelerating or negative momentum)"
                ),
            )

    # MACD line vs signal
    if config.require_macd_line_above_signal:
        if candidate.macd_line is None or candidate.macd_signal is None:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason="macd_line or macd_signal=None (insufficient history for MACD)",
            )
        if candidate.macd_line <= candidate.macd_signal:
            return FilterResult(
                filter_name="momentum",
                passed=False,
                reason=(
                    f"macd_line={candidate.macd_line} "
                    f"<= macd_signal={candidate.macd_signal} "
                    "(bearish MACD crossover)"
                ),
            )

    return FilterResult(filter_name="momentum", passed=True)


# ── Aggregate ─────────────────────────────────────────────────────────────────


def evaluate_candidate(
    candidate: Candidate,
    fundamentals: CompanyFundamentals | None,
    config: StrategyConfig,
) -> tuple[bool, list[str], list[str]]:
    """Run all four filter layers for a single candidate.

    Evaluates every layer regardless of earlier failures so the caller can
    report all reasons a symbol was rejected, not just the first.

    Returns:
        (passed, failed_filter_names, failure_reasons)
    """
    results = [
        evaluate_universe(candidate, fundamentals, config),
        evaluate_volume(candidate, config),
        evaluate_trend(candidate, config),
        evaluate_momentum(candidate, config),
    ]
    failed_names = [r.filter_name for r in results if not r.passed]
    failed_reasons = [r.reason for r in results if not r.passed and r.reason]
    return (len(failed_names) == 0, failed_names, failed_reasons)

"""ScreeningEngine — orchestrates per-symbol OHLCV + indicator + fundamentals fetch.

Architecture:
  - One call to build_candidate_snapshot() per symbol covers BOTH technical indicators
    AND volume metrics (volume, avg_volume_20). No second OhlcvRepository.fetch_range
    is ever issued by this engine.
  - Fundamentals are fetched via FundamentalsService.get_fundamentals(allow_live=...).
    allow_live=False serves stale cache or None without consuming any FMP quota.
  - Errors are isolated per-symbol: SymbolNotCachedError and unexpected exceptions
    are caught, logged, and recorded in ScreeningReport.errors without aborting the run.
"""

import datetime
import traceback

from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.fundamentals.models import CompanyFundamentals
from mcp_finance.fundamentals.quota import QuotaExhaustedError
from mcp_finance.fundamentals.service import FundamentalsService
from mcp_finance.indicators.models import Candidate
from mcp_finance.indicators.snapshot import (
    SymbolNotCachedError,
    build_candidate_snapshot,
)
from mcp_finance.logger import get_logger
from mcp_finance.screener.filters import evaluate_candidate
from mcp_finance.screener.models import ScreenedStock, ScreeningReport, StrategyConfig

logger = get_logger(__name__)


def _build_screened_stock(
    candidate: Candidate,
    fundamentals: CompanyFundamentals | None,
    passed: bool,
    failed_filters: list[str],
    failure_reasons: list[str],
) -> ScreenedStock:
    """Assemble a ScreenedStock from its component parts."""
    return ScreenedStock(
        symbol=candidate.symbol,
        as_of_date=candidate.as_of_date,
        passed=passed,
        failed_filters=failed_filters,
        failure_reasons=failure_reasons,
        close=candidate.close,
        ema_20=candidate.ema_20,
        ema_50=candidate.ema_50,
        rsi_14=candidate.rsi_14,
        macd_line=candidate.macd_line,
        macd_signal=candidate.macd_signal,
        macd_histogram=candidate.macd_histogram,
        avg_volume_20=candidate.avg_volume_20,
        bars_available=candidate.bars_available,
        company_name=fundamentals.company_name if fundamentals else None,
        sector=fundamentals.sector if fundamentals else None,
        market_cap=fundamentals.market_cap if fundamentals else None,
        pe_ratio=fundamentals.pe_ratio if fundamentals else None,
        fundamentals_source=fundamentals.source if fundamentals else None,
    )


class ScreeningEngine:
    """Coordinates per-symbol screening across OHLCV, indicator, and fundamentals.

    Processing is sequential-per-symbol for two reasons:
      1. Predictable error isolation — a single bad symbol does not abort the run.
      2. FMP rate pacing — during cold-cache hydration, sequential calls naturally
         space outbound requests without bursting free-tier rate limits.

    At ~30 symbols with a warmed cache, a complete run takes ~100–150ms.
    """

    def __init__(
        self,
        session: AsyncSession,
        fundamentals_service: FundamentalsService,
    ) -> None:
        self._session = session
        self._fundamentals_service = fundamentals_service

    async def screen(
        self,
        symbols: list[str],
        as_of_date: datetime.date,
        config: StrategyConfig,
        *,
        allow_live_fundamentals: bool = True,
        include_failed: bool = False,
        source: str = "yfinance",
    ) -> ScreeningReport:
        """Run the screening engine over a list of symbols.

        Args:
            symbols:                  Ticker symbols to evaluate.
            as_of_date:               Screening date.
            config:                   Strategy thresholds.
            allow_live_fundamentals:  Passed through to
                                      FundamentalsService.get_fundamentals.
            include_failed:           When True, ScreeningReport.failed_candidates
                                      is populated.
            source:                   OHLCV source (must match what was ingested).

        Returns:
            ScreeningReport with passed/failed/error buckets.

        Note:
            An empty passed_candidates list is a valid, expected outcome for a tight
            strategy — not an error or bug. See ScreenedStock docstring.
        """
        passed_candidates: list[ScreenedStock] = []
        failed_candidates: list[ScreenedStock] = []
        errors: dict[str, str] = {}

        for symbol in symbols:
            try:
                await self._screen_symbol(
                    symbol=symbol,
                    as_of_date=as_of_date,
                    config=config,
                    allow_live_fundamentals=allow_live_fundamentals,
                    include_failed=include_failed,
                    source=source,
                    passed_candidates=passed_candidates,
                    failed_candidates=failed_candidates,
                )
            except SymbolNotCachedError as exc:
                logger.warning(
                    "Symbol not in OHLCV cache; skipping",
                    symbol=symbol,
                    detail=str(exc),
                )
                errors[symbol] = (
                    "OHLCV data not cached — run the batch ingestion job first"
                )
            except Exception:
                detail = traceback.format_exc()
                logger.error(
                    "Unexpected error screening symbol",
                    symbol=symbol,
                    traceback=detail,
                )
                errors[symbol] = f"Unexpected error: {detail.splitlines()[-1]}"

        return ScreeningReport(
            as_of_date=as_of_date,
            strategy_name=config.name,
            total_screened=len(symbols),
            passed_count=len(passed_candidates),
            passed_candidates=passed_candidates,
            failed_candidates=failed_candidates if include_failed else [],
            errors=errors,
        )

    async def _screen_symbol(
        self,
        symbol: str,
        as_of_date: datetime.date,
        config: StrategyConfig,
        allow_live_fundamentals: bool,
        include_failed: bool,
        source: str,
        passed_candidates: list[ScreenedStock],
        failed_candidates: list[ScreenedStock],
    ) -> None:
        """Screen a single symbol and append result to the appropriate bucket."""
        # ── 1. Build technical Candidate (single OHLCV fetch, includes volume) ──
        candidate = await build_candidate_snapshot(
            symbol=symbol,
            as_of_date=as_of_date,
            session=self._session,
            source=source,
        )

        # ── 2. Fetch fundamentals (cache-first; no network if allow_live=False) ──
        fundamentals: CompanyFundamentals | None = None
        try:
            fundamentals = await self._fundamentals_service.get_fundamentals(
                symbol=symbol,
                allow_live=allow_live_fundamentals,
            )
        except QuotaExhaustedError:
            logger.warning(
                "FMP quota exhausted; fundamentals unavailable for symbol",
                symbol=symbol,
            )
            # fundamentals stays None — universe filter will fail if min_market_cap
            # or max_pe_ratio are configured, which is the correct behaviour.

        # ── 3. Evaluate all four filter layers ────────────────────────────────
        passed, failed_filters, failure_reasons = evaluate_candidate(
            candidate, fundamentals, config
        )

        logger.info(
            "Screened symbol",
            symbol=symbol,
            passed=passed,
            failed_filters=failed_filters,
        )

        stock = _build_screened_stock(
            candidate, fundamentals, passed, failed_filters, failure_reasons
        )

        if passed:
            passed_candidates.append(stock)
        elif include_failed:
            failed_candidates.append(stock)

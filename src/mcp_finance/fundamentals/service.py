"""FundamentalsService — orchestrates cache-first retrieval and quota management."""

import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import FundamentalsRepository, SymbolRepository
from mcp_finance.fundamentals.fmp import FMPClient, FMPQuotaExceededError
from mcp_finance.fundamentals.models import CompanyFundamentals
from mcp_finance.fundamentals.quota import DailyQuotaGuard, QuotaExhaustedError
from mcp_finance.logger import get_logger
from mcp_finance.market_data.utils import derive_exchange

logger = get_logger(__name__)


class FundamentalsService:
    """Coordinates fundamentals caching, quota management, and FMP fetching.

    Cache-first policy:
    1. Checks if a snapshot exists within the configured TTL (168h default).
    2. If fresh, serves directly from Postgres (zero network calls,
       zero quota consumed). Fresh cache is served even if today's quota is exhausted.
    3. If missing or stale, atomically acquires 2 quota units upfront before
       dispatching HTTP calls to /stable/profile and /stable/ratios-ttm.
    4. Persists to Postgres only after both endpoints return valid data, preventing
       partial or corrupted 7-day cache entries.
    5. If FMP returns 429, marks local quota as exhausted authoritatively.
    """

    def __init__(
        self,
        session: AsyncSession,
        client: FMPClient | None = None,
        quota_guard: DailyQuotaGuard | None = None,
    ) -> None:
        self._session = session
        self._client = client or FMPClient()
        self._quota_guard = quota_guard or DailyQuotaGuard()

    async def get_fundamentals(
        self,
        symbol: str,
        exchange: str | None = None,
    ) -> CompanyFundamentals:
        """Retrieve company fundamentals cache-first."""
        canonical_exchange = derive_exchange(symbol, exchange)
        clean_symbol = symbol.strip().upper()

        symbol_repo = SymbolRepository(self._session)
        sym = await symbol_repo.upsert(ticker=clean_symbol, exchange=canonical_exchange)
        assert sym.id is not None

        fund_repo = FundamentalsRepository(self._session)

        # 1. Check cache first
        if await fund_repo.is_fresh(sym.id):
            latest = await fund_repo.get_latest(sym.id)
            if latest is not None:
                logger.info(
                    "Fundamentals cache hit",
                    symbol=clean_symbol,
                    as_of_date=str(latest.as_of_date),
                )
                payload = dict(latest.payload)
                payload["is_cached"] = True
                payload["as_of_date"] = latest.as_of_date
                return CompanyFundamentals.model_validate(payload)

        # 2. Cache miss or stale: reserve 2 quota units upfront
        logger.info(
            "Fundamentals cache miss or stale; acquiring FMP quota",
            symbol=clean_symbol,
        )
        await self._quota_guard.acquire(self._session, count=2)

        # 3. Fetch both profile and ratios-ttm
        try:
            profile = await self._client.get_company_profile(clean_symbol)
            ratios = await self._client.get_ratios(clean_symbol)
        except FMPQuotaExceededError:
            today_utc = datetime.datetime.now(datetime.timezone.utc).date()
            await self._quota_guard.mark_exhausted(self._session, as_of=today_utc)
            raise QuotaExhaustedError(
                f"FMP daily quota exhausted for {today_utc} (HTTP 429 received)",
                used=self._quota_guard.max_daily_requests,
                max_daily=self._quota_guard.max_daily_requests,
                attempted=2,
            )

        today = datetime.datetime.now(datetime.timezone.utc).date()
        fundamentals = CompanyFundamentals.from_api_data(
            profile=profile,
            ratios=ratios,
            as_of_date=today,
            is_cached=False,
        )

        # 4. Only persist if BOTH endpoints succeeded
        payload = fundamentals.model_dump(mode="json")
        await fund_repo.upsert(sym.id, today, payload)
        await self._session.commit()

        logger.info(
            "Fundamentals successfully fetched and cached",
            symbol=clean_symbol,
            as_of_date=str(today),
        )
        return fundamentals

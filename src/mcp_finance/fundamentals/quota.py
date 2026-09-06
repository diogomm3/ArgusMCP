"""Postgres-persisted daily quota guard for FMP API calls."""

import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.models import FmpQuotaUsage
from mcp_finance.logger import get_logger
from mcp_finance.settings import settings

logger = get_logger(__name__)


class QuotaExhaustedError(Exception):
    """Raised when the daily request quota for FMP is reached."""

    def __init__(
        self,
        message: str,
        used: int = 0,
        max_daily: int = 250,
        attempted: int = 1,
    ) -> None:
        super().__init__(message)
        self.used = used
        self.max_daily = max_daily
        self.attempted = attempted


class DailyQuotaGuard:
    """Tracks and limits daily FMP API calls via PostgreSQL.

    Persisted in Postgres table `fmp_quota_usage` to survive container rebuilds
    and process restarts. Uses a single atomic statement with WHERE clause to
    eliminate check-then-act race conditions.
    """

    def __init__(self, max_daily_requests: int | None = None) -> None:
        self.max_daily_requests = (
            max_daily_requests
            if max_daily_requests is not None
            else settings.fmp_daily_quota
        )

    async def acquire(
        self,
        session: AsyncSession,
        count: int = 1,
        as_of: datetime.date | None = None,
    ) -> int:
        """Atomically reserve `count` quota units for the day.

        Executes a single atomic INSERT ... ON CONFLICT DO UPDATE ... WHERE
        statement. If the new total exceeds `max_daily_requests`, PostgreSQL
        returns no rows and this method raises QuotaExhaustedError.

        Returns:
            The new cumulative used count for the day.

        Raises:
            QuotaExhaustedError: if requested count cannot be accommodated.
        """
        target_date = as_of or datetime.datetime.now(datetime.timezone.utc).date()

        if count > self.max_daily_requests:
            raise QuotaExhaustedError(
                f"Requested count {count} exceeds "
                f"maximum daily quota {self.max_daily_requests}",
                used=0,
                max_daily=self.max_daily_requests,
                attempted=count,
            )

        stmt = (
            insert(FmpQuotaUsage)
            .values(
                date=target_date,
                request_count=count,
            )
            .on_conflict_do_update(
                index_elements=["date"],
                set_={
                    "request_count": FmpQuotaUsage.request_count + count,
                    "updated_at": func.now(),
                },
                where=(FmpQuotaUsage.request_count + count <= self.max_daily_requests),
            )
            .returning(FmpQuotaUsage.request_count)
        )

        result = await session.execute(stmt)
        new_count = result.scalar_one_or_none()

        if new_count is None:
            # Query did not update because WHERE condition was violated (quota full)
            check_stmt = select(FmpQuotaUsage.request_count).where(
                FmpQuotaUsage.date == target_date
            )
            current_used = (
                await session.execute(check_stmt)
            ).scalar_one_or_none() or self.max_daily_requests
            logger.warning(
                "FMP daily quota exhausted",
                date=str(target_date),
                used=current_used,
                requested=count,
                max_daily=self.max_daily_requests,
            )
            raise QuotaExhaustedError(
                f"FMP daily quota exhausted for {target_date}: "
                f"{current_used}/{self.max_daily_requests} used. "
                f"Cannot acquire {count} additional requests.",
                used=current_used,
                max_daily=self.max_daily_requests,
                attempted=count,
            )

        await session.flush()
        logger.info(
            "FMP quota acquired",
            date=str(target_date),
            acquired=count,
            total_used=new_count,
            remaining=max(0, self.max_daily_requests - new_count),
        )
        return new_count

    async def get_usage(
        self,
        session: AsyncSession,
        as_of: datetime.date | None = None,
    ) -> tuple[int, int]:
        """Return (used, remaining) quota for the target date."""
        target_date = as_of or datetime.datetime.now(datetime.timezone.utc).date()
        stmt = select(FmpQuotaUsage.request_count).where(
            FmpQuotaUsage.date == target_date
        )
        used = (await session.execute(stmt)).scalar_one_or_none() or 0
        remaining = max(0, self.max_daily_requests - used)
        return (used, remaining)

    async def mark_exhausted(
        self,
        session: AsyncSession,
        as_of: datetime.date | None = None,
    ) -> None:
        """Mark today's quota as exhausted immediately (authoritative 429 handler).

        Called when FMP returns an HTTP 429 Too Many Requests response to snap
        local tracking to reality.
        """
        target_date = as_of or datetime.datetime.now(datetime.timezone.utc).date()
        stmt = (
            insert(FmpQuotaUsage)
            .values(
                date=target_date,
                request_count=self.max_daily_requests,
            )
            .on_conflict_do_update(
                index_elements=["date"],
                set_={
                    "request_count": self.max_daily_requests,
                    "updated_at": func.now(),
                },
            )
        )
        await session.execute(stmt)
        await session.flush()
        logger.warning(
            "FMP quota marked exhausted following authoritative 429 response",
            date=str(target_date),
            count=self.max_daily_requests,
        )

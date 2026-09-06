"""Integration tests for DailyQuotaGuard against real testcontainers Postgres."""

import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.fundamentals.quota import DailyQuotaGuard, QuotaExhaustedError


@pytest.mark.unit
async def test_quota_guard_acquire_and_increment_postgres(
    db_session: AsyncSession,
) -> None:
    """Atomic acquisition, increment, and quota overflow rejection in Postgres."""
    guard = DailyQuotaGuard(max_daily_requests=5)
    today = datetime.date(2025, 6, 1)

    # First acquisition: creates row with count 2
    count = await guard.acquire(db_session, count=2, as_of=today)
    assert count == 2

    # Second acquisition: increments to 4
    count = await guard.acquire(db_session, count=2, as_of=today)
    assert count == 4

    # Third acquisition: requesting 2 with only 1 remaining must
    # raise QuotaExhaustedError
    with pytest.raises(QuotaExhaustedError) as exc_info:
        await guard.acquire(db_session, count=2, as_of=today)

    assert exc_info.value.used == 4
    assert exc_info.value.max_daily == 5
    assert exc_info.value.attempted == 2

    # Remaining 1 can still be acquired
    count = await guard.acquire(db_session, count=1, as_of=today)
    assert count == 5

    # Completely exhausted
    with pytest.raises(QuotaExhaustedError):
        await guard.acquire(db_session, count=1, as_of=today)

    used, remaining = await guard.get_usage(db_session, as_of=today)
    assert used == 5
    assert remaining == 0


@pytest.mark.unit
async def test_quota_guard_day_rollover_postgres(
    db_session: AsyncSession,
) -> None:
    """Different dates maintain independent quota records in Postgres."""
    guard = DailyQuotaGuard(max_daily_requests=10)
    day1 = datetime.date(2025, 6, 1)
    day2 = datetime.date(2025, 6, 2)

    await guard.acquire(db_session, count=7, as_of=day1)
    used1, rem1 = await guard.get_usage(db_session, as_of=day1)
    assert used1 == 7
    assert rem1 == 3

    # day2 starts completely clean at 0
    used2, rem2 = await guard.get_usage(db_session, as_of=day2)
    assert used2 == 0
    assert rem2 == 10

    await guard.acquire(db_session, count=4, as_of=day2)
    used2_after, rem2_after = await guard.get_usage(db_session, as_of=day2)
    assert used2_after == 4
    assert rem2_after == 6


@pytest.mark.unit
async def test_quota_guard_mark_exhausted_postgres(
    db_session: AsyncSession,
) -> None:
    """mark_exhausted immediately caps today's usage to max_daily_requests."""
    guard = DailyQuotaGuard(max_daily_requests=250)
    today = datetime.date(2025, 6, 3)

    await guard.acquire(db_session, count=10, as_of=today)
    await guard.mark_exhausted(db_session, as_of=today)

    used, remaining = await guard.get_usage(db_session, as_of=today)
    assert used == 250
    assert remaining == 0

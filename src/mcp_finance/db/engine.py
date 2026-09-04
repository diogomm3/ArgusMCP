"""Async SQLAlchemy engine and session factory.

The engine is lazily initialized on first access via `get_engine()` so that
tests can swap out `settings.database_url` before connecting. Module-level
eager initialization would bind to whatever DATABASE_URL is in .env at
import time, making it impossible to point tests at a testcontainer.

Import `get_session_factory()` to obtain a sessionmaker, or use the
`get_session()` async context manager for scoped sessions in tools and repos.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mcp_finance.settings import settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return (and cache) the async engine for the configured database URL.

    lru_cache ensures a single engine is reused across calls, while still
    allowing tests to bypass this by constructing their own engine directly.
    """
    return create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return an async session factory bound to the current engine."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a transactional async session, rolling back on error."""
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            yield session

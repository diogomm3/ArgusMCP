"""Pytest fixtures for database repository tests.

Uses testcontainers to spin up a real postgres:16 container per test session,
then runs Alembic migrations against it before yielding a session factory.
No docker-compose stack required — fully self-contained.

The `db_session` fixture yields a fresh AsyncSession per test inside a
savepoint (nested transaction), which is rolled back after each test so
tests are isolated without re-running migrations on every test.
"""

from collections.abc import AsyncGenerator, Generator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str, None, None]:
    """Start a postgres:16 testcontainer for the full test session."""
    with PostgresContainer("postgres:16") as pg:
        # testcontainers returns a psycopg2-style URL; normalise to asyncpg
        sync_url = pg.get_connection_url()
        async_url = sync_url.replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        ).replace("postgresql://", "postgresql+asyncpg://")
        yield async_url


@pytest.fixture(scope="session")
def apply_migrations(postgres_url: str) -> None:
    """Run `alembic upgrade head` against the testcontainer.

    Patches settings.database_url so alembic/env.py picks up the testcontainer
    URL instead of the .env value. Also clears the get_engine() lru_cache so
    any lazy engine created during migrations uses the patched URL.
    """
    import mcp_finance.settings as _settings_module
    from mcp_finance.db.engine import get_engine

    original_url = _settings_module.settings.database_url
    # Bypass pydantic frozen model via object.__setattr__
    object.__setattr__(_settings_module.settings, "database_url", postgres_url)
    get_engine.cache_clear()
    try:
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
    finally:
        object.__setattr__(_settings_module.settings, "database_url", original_url)
        get_engine.cache_clear()


@pytest.fixture()
def session_factory(
    postgres_url: str, apply_migrations: None
) -> async_sessionmaker[AsyncSession]:
    """Return an async session factory wired to the testcontainer DB.

    Constructed directly from the testcontainer URL — does NOT go through
    get_engine() to avoid interfering with the production engine cache.
    Uses NullPool to avoid cross-loop connection pooling issues in pytest.
    """
    engine = create_async_engine(postgres_url, echo=False, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture()
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a per-test async session inside a rolled-back savepoint.

    Each test gets a clean slate without re-running migrations.
    """
    async with session_factory() as session:
        async with session.begin():
            nested = await session.begin_nested()
            yield session
            await nested.rollback()

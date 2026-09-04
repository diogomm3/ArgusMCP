"""Alembic env.py — async-compatible using asyncio + AsyncEngine.

The database URL is always read from settings (environment / .env file),
never from alembic.ini directly, so no credentials live in config files.
`target_metadata` points at our ORM Base so `alembic revision --autogenerate`
can diff against the live schema — though the first migration is hand-written.
"""

import asyncio
from logging.config import fileConfig

from alembic import context

from mcp_finance.db.engine import get_engine

# Import our ORM base so autogenerate has metadata to work from
from mcp_finance.db.models import Base
from mcp_finance.settings import settings

# Alembic Config object — gives access to alembic.ini values
config = context.config

# Wire up Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection).

    Generates SQL script output suitable for review or manual application.
    """
    url = settings.database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations inside a sync connection wrapper."""
    connectable = get_engine()
    async with connectable.connect() as connection:
        await connection.run_sync(_run_migrations_sync)
    # We do not dispose the engine here because it might be cached
    # and used by the app or tests.


def _run_migrations_sync(connection: object) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)  # type: ignore[arg-type]
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Entry point for online migrations — delegates to the async runner."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

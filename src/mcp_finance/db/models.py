"""SQLAlchemy 2.0 ORM models for the finance MCP persistence layer.

Four tables:
  - symbols            — master instrument list (ticker + exchange unique)
  - ohlcv_daily        — daily OHLCV bars per symbol, one row per (symbol, date, source)
  - fundamentals_cache — historical snapshots of fundamentals JSON,
                         keyed on (symbol, date)
  - order_audit_logs   — immutable audit trail for every order attempt
                         (two-phase: SUBMITTING → ACCEPTED / FAILED / REJECTED)
"""

import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Symbol(Base):
    """Master instrument list.

    Uniqueness is on (ticker, exchange) because the same ticker string can
    represent different companies on different exchanges (e.g. SAP on XETRA vs SAP
    as some unrelated US OTC name). ISIN is the globally-unique identifier — populated
    from Trading212 metadata — but stored as nullable until backfilled.
    """

    __tablename__ = "symbols"
    __table_args__ = (
        UniqueConstraint("ticker", "exchange", name="uq_symbols_ticker_exchange"),
        Index(
            "ix_symbols_isin", "isin"
        ),  # non-unique: same ISIN can list on multiple exchanges
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)
    isin: Mapped[str | None] = mapped_column(String(12), nullable=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    asset_class: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="e.g. 'STOCK', 'ETF'"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    ohlcv_bars: Mapped[list["OhlcvDaily"]] = relationship(
        "OhlcvDaily", back_populates="symbol", cascade="all, delete-orphan"
    )
    fundamentals: Mapped[list["FundamentalsCache"]] = relationship(
        "FundamentalsCache", back_populates="symbol", cascade="all, delete-orphan"
    )


class OhlcvDaily(Base):
    """Daily OHLCV bars.

    Unique on (symbol_id, date, source) so multiple providers can coexist without
    conflict and we know exactly where each row came from. fetch_range() defaults
    to settings.primary_ohlcv_source ('yfinance') to avoid silently mixing bars.
    NUMERIC(18, 6) mirrors the Decimal types used in broker Pydantic models.
    """

    __tablename__ = "ohlcv_daily"
    __table_args__ = (
        UniqueConstraint(
            "symbol_id", "date", "source", name="uq_ohlcv_symbol_date_source"
        ),
        Index("ix_ohlcv_symbol_date_desc", "symbol_id", "date"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str] = mapped_column(
        String(30), nullable=False, comment="e.g. 'yfinance', 'twelve_data'"
    )

    symbol: Mapped["Symbol"] = relationship("Symbol", back_populates="ohlcv_bars")


class FundamentalsCache(Base):
    """Historical fundamentals snapshots.

    Keyed on (symbol_id, as_of_date) — one snapshot per symbol per calendar day.
    Re-fetching on the same day is idempotent (upsert). History is preserved so
    backtests can ask "what did we know on date X?" via ORDER BY as_of_date DESC.
    Staleness is checked against settings.fundamentals_cache_ttl_hours (default 168h).
    """

    __tablename__ = "fundamentals_cache"
    __table_args__ = (
        UniqueConstraint("symbol_id", "as_of_date", name="uq_fundamentals_symbol_date"),
        Index("ix_fundamentals_symbol_date_desc", "symbol_id", "as_of_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    as_of_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)  # type: ignore[type-arg]
    fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    symbol: Mapped["Symbol"] = relationship("Symbol", back_populates="fundamentals")


class OrderAuditLog(Base):
    """Immutable audit record for every order attempt.

    Two-phase lifecycle:
      1. Written with status=SUBMITTING *before* the broker HTTP call.
      2. Updated to ACCEPTED (with ``broker_order_id``) or FAILED on return.
    Rejections write a single REJECTED row; they never reach the broker.

    The session that writes this record is *independent* of the caller's
    transaction so that a rollback in the caller cannot erase the audit entry.
    """

    __tablename__ = "order_audit_logs"
    __table_args__ = (
        Index("ix_order_audit_symbol", "symbol"),
        Index("ix_order_audit_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(30), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False, comment="BUY or SELL")
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    stop_loss_price: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="REJECTED | SUBMITTING | ACCEPTED | FAILED",
    )
    rejection_reasons: Mapped[list] = mapped_column(  # type: ignore[type-arg]
        JSONB, nullable=False, server_default="'[]'"
    )
    risk_metrics: Mapped[dict] = mapped_column(  # type: ignore[type-arg]
        JSONB,
        nullable=False,
        server_default="'{}'",
        comment="risk_amount, estimated_cost, binding_constraint, rule_details",
    )
    broker_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # type: ignore[type-arg]


class FmpQuotaUsage(Base):
    """Daily FMP request quota tracking.

    Persisted in Postgres to survive container restarts and ensure multiple
    workers/processes share an accurate view of today's quota usage.
    Keyed on date.
    """

    __tablename__ = "fmp_quota_usage"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

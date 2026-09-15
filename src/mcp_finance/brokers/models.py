"""Shared Pydantic data models for broker responses."""

from decimal import Decimal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """A single open position in the portfolio."""

    ticker: str = Field(
        ...,
        description="Ticker symbol",
    )
    quantity: Decimal = Field(
        ...,
        description="Number of shares held",
    )
    average_price: Decimal = Field(
        ...,
        description="Average price per share",
    )
    current_price: Decimal = Field(
        ...,
        description="The current market price per share",
    )
    ppl: Decimal = Field(
        ...,
        description="Profit/loss in account currency",
    )
    frontend_type: str = Field(
        default="STOCK",
        description="Instrument type as reported by the broker.",
    )


class AccountSummary(BaseModel):
    """High-level account snapshot."""

    cash: Decimal = Field(
        ...,
        description="Free cash available to trade",
    )
    invested: Decimal = Field(
        ...,
        description="Total amount currently invested",
    )
    result: Decimal = Field(
        ...,
        description="Unrealised profit/loss",
    )
    total: Decimal = Field(
        ...,
        description="Total account value (cash + invested + result)",
    )
    currency: str = Field(
        default="GBP",
        description="Account base currency",
    )


class OrderResult(BaseModel):
    """Response from a successful broker order submission."""

    id: str = Field(..., description="Broker-assigned order identifier.")
    status: str = Field(..., description="Order status as returned by broker.")
    ticker: str = Field(..., description="Instrument ticker symbol.")
    quantity: Decimal = Field(..., description="Ordered quantity.")
    filled_quantity: Decimal = Field(
        default=Decimal("0"),
        description="Quantity filled so far (may be 0 for pending market orders).",
    )
    order_type: str = Field(..., description="MARKET or LIMIT.")
    created_at: str | None = Field(
        default=None, description="ISO-8601 creation timestamp."
    )
    limit_price: Decimal | None = Field(
        default=None, description="Limit price (for LIMIT orders)."
    )

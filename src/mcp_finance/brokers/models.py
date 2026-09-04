"""Shared Pydantic data models for broker responses."""

from decimal import Decimal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """A single open position in the portfolio."""

    ticker: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    ppl: Decimal = Field(description="Profit/loss in account currency")
    frontend_type: str = Field(
        default="STOCK",
        description="Instrument type as reported by the broker.",
    )


class AccountSummary(BaseModel):
    """High-level account snapshot."""

    cash: Decimal = Field(description="Free cash available to trade")
    invested: Decimal = Field(description="Total amount currently invested")
    result: Decimal = Field(description="Unrealised profit/loss")
    total: Decimal = Field(description="Total account value (cash + invested + result)")
    currency: str = Field(default="GBP", description="Account base currency")

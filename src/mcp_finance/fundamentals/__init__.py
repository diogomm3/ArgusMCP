"""Fundamentals package exports and client Protocol."""

from typing import Protocol

from mcp_finance.fundamentals.fmp import (
    FMPApiError,
    FMPClient,
    FMPQuotaExceededError,
    SymbolNotFoundError,
)
from mcp_finance.fundamentals.models import (
    CompanyFundamentals,
    CompanyProfile,
    FinancialRatios,
    GetStockFundamentalsInput,
)
from mcp_finance.fundamentals.quota import DailyQuotaGuard, QuotaExhaustedError
from mcp_finance.fundamentals.service import FundamentalsService


class FundamentalsClient(Protocol):
    """Vendor-agnostic interface for company fundamentals data."""

    async def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Fetch general company profile and metadata."""
        ...

    async def get_ratios(self, symbol: str) -> FinancialRatios:
        """Fetch trailing twelve month (TTM) financial ratios."""
        ...

    async def get_fundamentals(self, symbol: str) -> CompanyFundamentals:
        """Fetch consolidated company fundamentals (profile + ratios)."""
        ...


__all__ = [
    "CompanyFundamentals",
    "CompanyProfile",
    "DailyQuotaGuard",
    "FMPApiError",
    "FMPClient",
    "FMPQuotaExceededError",
    "FinancialRatios",
    "FundamentalsClient",
    "FundamentalsService",
    "GetStockFundamentalsInput",
    "QuotaExhaustedError",
    "SymbolNotFoundError",
]

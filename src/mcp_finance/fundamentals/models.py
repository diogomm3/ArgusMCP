"""Pydantic models for company fundamentals from FMP."""

import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class CompanyProfile(BaseModel):
    """Company profile data from FMP /stable/profile endpoint."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(
        ...,
        description="Ticker symbol",
    )
    company_name: str = Field(
        ...,
        alias="companyName",
        description="Company legal name",
    )
    exchange: str = Field(
        ...,
        alias="exchange",
        description="Exchange identifier",
    )
    currency: str = Field(
        ...,
        alias="currency",
        description="Reporting currency",
    )
    price: Decimal | None = Field(
        default=None,
        description="Latest price",
    )
    market_cap: int | None = Field(
        default=None,
        alias="marketCap",
        description="Market capitalization",
    )
    beta: Decimal | None = Field(
        default=None,
        description="Beta vs benchmark",
    )
    last_dividend: Decimal | None = Field(
        default=None,
        alias="lastDividend",
        description="Last dividend",
    )
    industry: str | None = Field(
        default=None,
        description="Industry group",
    )
    sector: str | None = Field(
        default=None,
        description="Economic sector",
    )
    country: str | None = Field(
        default=None,
        description="Country of domicile",
    )
    description: str | None = Field(
        default=None,
        description="Company description",
    )
    website: str | None = Field(
        default=None,
        description="Company website",
    )
    ceo: str | None = Field(
        default=None,
        description="CEO name",
    )


class FinancialRatios(BaseModel):
    """Trailing twelve month (TTM) financial ratios from FMP /stable/ratios-ttm."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(
        ...,
        description="Ticker symbol",
    )
    pe_ratio: Decimal | None = Field(
        default=None,
        alias="priceToEarningsRatioTTM",
        description="Price/Earnings (TTM)",
    )
    price_to_book: Decimal | None = Field(
        default=None,
        alias="priceToBookRatioTTM",
        description="Price/Book (TTM)",
    )
    price_to_sales: Decimal | None = Field(
        default=None,
        alias="priceToSalesRatioTTM",
        description="Price/Sales (TTM)",
    )
    enterprise_value_multiple: Decimal | None = Field(
        default=None,
        alias="enterpriseValueMultipleTTM",
        description="Enterprise Value/EBITDA (TTM)",
    )
    gross_profit_margin: Decimal | None = Field(
        default=None,
        alias="grossProfitMarginTTM",
        description="Gross profit margin (TTM)",
    )
    operating_margin: Decimal | None = Field(
        default=None,
        alias="operatingProfitMarginTTM",
        description="Operating profit margin (TTM)",
    )
    net_profit_margin: Decimal | None = Field(
        default=None,
        alias="netProfitMarginTTM",
        description="Net profit margin (TTM)",
    )
    current_ratio: Decimal | None = Field(
        default=None,
        alias="currentRatioTTM",
        description="Current ratio (TTM)",
    )
    quick_ratio: Decimal | None = Field(
        default=None,
        alias="quickRatioTTM",
        description="Quick ratio (TTM)",
    )
    debt_to_equity: Decimal | None = Field(
        default=None,
        alias="debtToEquityRatioTTM",
        description="Debt-to-equity ratio (TTM)",
    )
    interest_coverage: Decimal | None = Field(
        default=None,
        alias="interestCoverageRatioTTM",
        description="Interest coverage ratio (TTM)",
    )
    dividend_yield: Decimal | None = Field(
        default=None,
        alias="dividendYieldTTM",
        description="Dividend yield (TTM)",
    )
    payout_ratio: Decimal | None = Field(
        default=None,
        alias="dividendPayoutRatioTTM",
        description="Dividend payout ratio (TTM)",
    )
    free_cash_flow_per_share: Decimal | None = Field(
        default=None,
        alias="freeCashFlowPerShareTTM",
        description="Free cash flow per share (TTM)",
    )


class CompanyFundamentals(BaseModel):
    """Consolidated company fundamentals snapshot."""

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(
        ...,
        description="Ticker symbol",
    )
    company_name: str = Field(
        ...,
        description="Company legal name",
    )
    exchange: str = Field(
        ...,
        description="Exchange identifier",
    )
    currency: str = Field(
        ...,
        description="Reporting currency",
    )
    sector: str | None = Field(
        default=None,
        description="Economic sector",
    )
    industry: str | None = Field(
        default=None,
        description="Industry group",
    )
    country: str | None = Field(
        default=None,
        description="Country of domicile",
    )

    # Valuation & Market Size
    market_cap: int | None = Field(
        default=None,
        description="Market capitalization",
    )
    price: Decimal | None = Field(
        default=None,
        description="Latest price",
    )
    beta: Decimal | None = Field(
        default=None,
        description="Beta vs benchmark",
    )
    pe_ratio: Decimal | None = Field(
        default=None,
        description="Trailing P/E ratio",
    )
    price_to_book: Decimal | None = Field(
        default=None,
        description="Price-to-Book ratio",
    )
    price_to_sales: Decimal | None = Field(
        default=None,
        description="Price-to-Sales ratio",
    )
    enterprise_value_multiple: Decimal | None = Field(
        default=None,
        description="EV/EBITDA multiple",
    )

    # Profitability & Margins
    gross_profit_margin: Decimal | None = Field(
        default=None,
        description="Gross profit margin",
    )
    operating_margin: Decimal | None = Field(
        default=None,
        description="Operating profit margin",
    )
    net_profit_margin: Decimal | None = Field(
        default=None,
        description="Net profit margin",
    )
    current_ratio: Decimal | None = Field(
        default=None,
        description="Current ratio",
    )
    quick_ratio: Decimal | None = Field(
        default=None,
        description="Quick ratio",
    )
    debt_to_equity: Decimal | None = Field(
        default=None,
        description="Debt-to-Equity ratio",
    )
    interest_coverage: Decimal | None = Field(
        default=None,
        description="Interest coverage ratio",
    )

    # Dividends & Cash Flow
    dividend_yield: Decimal | None = Field(
        default=None,
        description="Dividend yield",
    )
    payout_ratio: Decimal | None = Field(
        default=None,
        description="Dividend payout ratio",
    )
    free_cash_flow_per_share: Decimal | None = Field(
        default=None,
        description="Free cash flow per share",
    )

    # Metadata & Tracking
    as_of_date: datetime.date = Field(
        ...,
        description="Date of snapshot calculation",
    )
    is_cached: bool = Field(
        default=False,
        description="Whether data was loaded from cache",
    )
    source: str = Field(
        default="fmp",
        description="Data source provider",
    )

    @classmethod
    def from_api_data(
        cls,
        profile: CompanyProfile,
        ratios: FinancialRatios,
        as_of_date: datetime.date,
        is_cached: bool = False,
    ) -> "CompanyFundamentals":
        """Assemble unified fundamentals from profile and ratios."""
        return cls(
            symbol=profile.symbol,
            company_name=profile.company_name,
            exchange=profile.exchange,
            currency=profile.currency,
            sector=profile.sector,
            industry=profile.industry,
            country=profile.country,
            market_cap=profile.market_cap,
            price=profile.price,
            beta=profile.beta,
            pe_ratio=ratios.pe_ratio,
            price_to_book=ratios.price_to_book,
            price_to_sales=ratios.price_to_sales,
            enterprise_value_multiple=ratios.enterprise_value_multiple,
            gross_profit_margin=ratios.gross_profit_margin,
            operating_margin=ratios.operating_margin,
            net_profit_margin=ratios.net_profit_margin,
            current_ratio=ratios.current_ratio,
            quick_ratio=ratios.quick_ratio,
            debt_to_equity=ratios.debt_to_equity,
            interest_coverage=ratios.interest_coverage,
            dividend_yield=ratios.dividend_yield,
            payout_ratio=ratios.payout_ratio,
            free_cash_flow_per_share=ratios.free_cash_flow_per_share,
            as_of_date=as_of_date,
            is_cached=is_cached,
            source="fmp",
        )


class GetStockFundamentalsInput(BaseModel):
    """Input parameters for get_stock_fundamentals MCP tool."""

    symbol: str = Field(
        ...,
        description="Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'ASML.AS')",
    )
    exchange: str | None = Field(
        default=None,
        description="Optional exchange name (e.g. 'NASDAQ', 'EURONEXT_AMSTERDAM')",
    )

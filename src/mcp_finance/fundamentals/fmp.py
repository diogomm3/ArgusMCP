"""FMP (Financial Modeling Prep) REST API adapter for company fundamentals."""

import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from mcp_finance.fundamentals.models import (
    CompanyFundamentals,
    CompanyProfile,
    FinancialRatios,
)
from mcp_finance.logger import get_logger
from mcp_finance.settings import settings

logger = get_logger(__name__)


class SymbolNotFoundError(Exception):
    """Raised when FMP returns an empty list for a requested ticker symbol."""


class FMPApiError(Exception):
    """Raised when FMP returns a non-200 HTTP error or unexpected payload."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FMPQuotaExceededError(FMPApiError):
    """Raised when FMP returns HTTP 429 Too Many Requests (authoritative)."""


def _is_transient_fmp_error(exc: BaseException) -> bool:
    """Retry only on low-level connection issues or transient 503 Service Unavailable.

    Never retry client errors (401/403/404), 429s, or business errors.
    """
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
        return True
    return False


class FMPClient:
    """Async client for FMP stable endpoints (/stable/profile and /stable/ratios-ttm).

    Architecture note on retries:
    Tenacity retry sits strictly *inside* the HTTP method (`_get`), retrying low-level
    transient connection errors. A network retry never re-acquires or double-counts
    against the daily quota at the service level.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or settings.fmp_api_key
        self._base_url = (base_url or settings.fmp_base_url).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        logger.info("FMPClient initialised", base_url=self._base_url)

    async def aclose(self) -> None:
        """Close the underlying HTTP client session."""
        await self._client.aclose()

    async def __aenter__(self) -> "FMPClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @retry(
        retry=retry_if_exception(_is_transient_fmp_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=5.0),
        reraise=True,
    )
    async def _get(self, endpoint: str, symbol: str) -> list[dict[str, Any]]:
        """Low-level GET request with scoped retry for transient network errors."""
        url = f"{endpoint}?symbol={symbol}&apikey={self._api_key}"
        try:
            response = await self._client.get(url)
        except httpx.HTTPError as exc:
            logger.warning(
                "FMP HTTP transport error",
                endpoint=endpoint,
                symbol=symbol,
                error=str(exc),
            )
            raise

        if response.status_code == 429:
            logger.warning(
                "FMP returned 429 Too Many Requests (authoritative quota hit)",
                endpoint=endpoint,
                symbol=symbol,
            )
            raise FMPQuotaExceededError(
                "FMP daily request quota exceeded (HTTP 429)", status_code=429
            )

        if response.status_code != 200:
            logger.error(
                "FMP error response",
                endpoint=endpoint,
                symbol=symbol,
                status_code=response.status_code,
                body=response.text[:200],
            )
            raise FMPApiError(
                f"FMP API returned status {response.status_code}: "
                f"{response.text[:150]}",
                status_code=response.status_code,
            )

        data = response.json()
        if not isinstance(data, list):
            logger.error(
                "Unexpected FMP response structure",
                endpoint=endpoint,
                symbol=symbol,
                response_type=type(data).__name__,
            )
            raise FMPApiError(
                f"Expected list response from FMP, got {type(data).__name__}"
            )

        return data

    async def get_company_profile(self, symbol: str) -> CompanyProfile:
        """Fetch general company profile from /stable/profile."""
        data = await self._get("/profile", symbol)
        if not data:
            logger.info("FMP company profile empty", symbol=symbol)
            raise SymbolNotFoundError(f"Symbol '{symbol}' not found on FMP")
        return CompanyProfile.model_validate(data[0])

    async def get_ratios(self, symbol: str) -> FinancialRatios:
        """Fetch trailing twelve month (TTM) ratios from /stable/ratios-ttm."""
        data = await self._get("/ratios-ttm", symbol)
        if not data:
            logger.info("FMP financial ratios empty", symbol=symbol)
            raise SymbolNotFoundError(
                f"Financial ratios for '{symbol}' not found on FMP"
            )
        return FinancialRatios.model_validate(data[0])

    async def get_fundamentals(self, symbol: str) -> CompanyFundamentals:
        """Fetch and consolidate company profile and financial ratios."""
        profile = await self.get_company_profile(symbol)
        ratios = await self.get_ratios(symbol)
        today = datetime.datetime.now(datetime.timezone.utc).date()
        return CompanyFundamentals.from_api_data(
            profile=profile,
            ratios=ratios,
            as_of_date=today,
            is_cached=False,
        )

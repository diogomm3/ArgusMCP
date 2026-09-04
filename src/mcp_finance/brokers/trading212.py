"""Trading212 REST API adapter (read-only for now, demo mode by default)."""

from decimal import Decimal

import httpx

from mcp_finance.brokers.models import AccountSummary, Position
from mcp_finance.logger import get_logger
from mcp_finance.settings import settings

logger = get_logger(__name__)

_BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}


class Trading212Client:
    """Thin httpx wrapper around the Trading212 REST API.

    All methods are async and use a shared httpx.AsyncClient.
    Instantiate once and reuse; call `aclose()` when done (or use as an async
    context manager).
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        env: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key or settings.trading212_api_key
        self._api_secret = api_secret or settings.trading212_api_secret
        self._env = env or settings.trading212_env
        base_url = _BASE_URLS.get(self._env, _BASE_URLS["demo"])
        self._client = httpx.AsyncClient(
            base_url=base_url,
            auth=(self._api_key, self._api_secret),
            headers={
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        logger.info(
            "Trading212Client initialised",
            env=self._env,
            base_url=base_url,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "Trading212Client":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Read-only portfolio endpoints
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch all open equity positions."""
        resp = await self._client.get("/equity/portfolio")
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        positions = []
        for item in data:
            positions.append(
                Position(
                    ticker=str(item["ticker"]),
                    quantity=Decimal(str(item["quantity"])),
                    average_price=Decimal(str(item["averagePrice"])),
                    current_price=Decimal(str(item["currentPrice"])),
                    ppl=Decimal(str(item["ppl"])),
                    frontend_type=str(item.get("frontendType", "STOCK")),
                )
            )
        logger.info("Fetched positions", count=len(positions))
        return positions

    async def get_account(self) -> AccountSummary:
        """Fetch account cash/equity snapshot."""
        resp = await self._client.get("/equity/account/cash")
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        summary = AccountSummary(
            cash=Decimal(str(data["free"])),
            invested=Decimal(str(data["invested"])),
            result=Decimal(str(data["result"])),
            total=Decimal(str(data["total"])),
        )
        logger.info("Fetched account summary", total=str(summary.total))
        return summary

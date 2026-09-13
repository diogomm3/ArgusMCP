"""FastMCP tool registrations for the stock screener."""

import datetime

from mcp.server.mcpserver import MCPServer

from mcp_finance.db.engine import get_session
from mcp_finance.fundamentals.fmp import FMPClient
from mcp_finance.fundamentals.service import FundamentalsService
from mcp_finance.market_data.batch import DEFAULT_WATCHLIST
from mcp_finance.screener.engine import ScreeningEngine
from mcp_finance.screener.models import (
    ScreeningReport,
    ScreenStocksInput,
    StrategyConfig,
)


def register_screener_tools(mcp: MCPServer) -> None:
    """Register screener MCP tools with the server."""

    @mcp.tool()
    async def screen_stocks(input: ScreenStocksInput) -> ScreeningReport:
        """Screen a list of stocks using a layered momentum/trend strategy.

        Evaluates each symbol through four sequential filter layers:
          1. Universe   — market cap >= $2B, price >= $5, optional max P/E.
          2. Liquidity  — 20-day average volume >= 100,000 shares.
          3. Trend      — price > EMA-20 > EMA-50 (golden cross alignment).
          4. Momentum   — RSI-14 in [40, 70], MACD histogram > 0, MACD > signal.

        All OHLCV data is read from the local Postgres cache (never triggers live
        yfinance syncs). Fundamentals are fetched from FMP cache; a stale or absent
        cache triggers live FMP calls only when allow_live_fundamentals=True (default).

        **An empty passed_candidates list is a valid, expected result.** The default
        strategy requires seven-plus simultaneous conditions. In sideways, choppy,
        or bearish market regimes, zero symbols clearing all filters is correct
        behaviour — not a bug or configuration error.

        Args:
            symbols:               Tickers to screen. Defaults to DEFAULT_WATCHLIST.
            as_of_date:            ISO date to screen as of. Defaults to today.
            allow_live_fundamentals: When True (default), stale/absent FMP cache
                                     triggers live API calls (consumes quota).
                                     When False, only cached fundamentals used.
            include_failed:        When True, failed symbols are included in the
                                   report (useful for debugging strategy filters).
        """
        symbols: list[str] = input.symbols or list(DEFAULT_WATCHLIST)

        as_of_date: datetime.date
        if input.as_of_date is not None:
            as_of_date = datetime.date.fromisoformat(input.as_of_date)
        else:
            as_of_date = datetime.date.today()

        config: StrategyConfig = StrategyConfig()

        async with get_session() as session:
            async with FMPClient() as client:
                fundamentals_service = FundamentalsService(
                    session=session, client=client
                )
                engine = ScreeningEngine(
                    session=session,
                    fundamentals_service=fundamentals_service,
                )
                return await engine.screen(
                    symbols=symbols,
                    as_of_date=as_of_date,
                    config=config,
                    allow_live_fundamentals=input.allow_live_fundamentals,
                    include_failed=input.include_failed,
                )

"""Service layer coordinating OHLCV retrieval, indicators, and rendering."""

import asyncio
import datetime

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.charting.renderer import render_chart
from mcp_finance.db.repository import OhlcvRepository, SymbolRepository
from mcp_finance.indicators.functions import ema
from mcp_finance.indicators.snapshot import SymbolNotCachedError
from mcp_finance.logger import get_logger
from mcp_finance.market_data.utils import derive_exchange

logger = get_logger(__name__)


def prepare_chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lowercase OHLCV DataFrame into mplfinance-compatible format.

    Maps internal column names ('date', 'open', 'high', 'low', 'close', 'volume')
    to capitalized columns with a pandas DatetimeIndex as required by mplfinance.
    """
    plot_df = df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["date"])
    plot_df.set_index("Date", inplace=True)
    plot_df.sort_index(inplace=True)
    plot_df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    # Drop the original lowercase 'date' column — it is now the index
    plot_df.drop(columns=["date"], inplace=True, errors="ignore")
    return plot_df


class ChartService:
    """Retrieves cached OHLCV data, calculates warmed indicators, and renders charts."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_chart(
        self,
        symbol: str,
        lookback_days: int = 90,
        exchange: str | None = None,
        source: str = "yfinance",
        as_of_date: datetime.date | None = None,
    ) -> bytes:
        """Render a candlestick chart with EMA overlays for a symbol.

        Args:
            symbol: Ticker symbol.
            lookback_days: Number of trading bars to display on chart (10-365).
            exchange: Optional exchange override.
            source: OHLCV source identifier (default: 'yfinance').
            as_of_date: Optional end date for the chart (defaults to today).

        Returns:
            PNG image bytes.

        Raises:
            SymbolNotCachedError: If symbol or bars are not found in the database.
            ValueError: If fewer than 5 bars exist to plot.
        """
        canonical_exchange = derive_exchange(symbol, exchange)
        clean_symbol = symbol.strip().upper()
        target_date = as_of_date or datetime.date.today()

        symbol_repo = SymbolRepository(self._session)
        sym = await symbol_repo.get_by_ticker(clean_symbol, canonical_exchange)
        if sym is None:
            raise SymbolNotCachedError(clean_symbol, canonical_exchange)

        # Warmup buffer: fetch lookback_days + 200 trading days of history
        # (~1.5 calendar days per trading day roughly, plus buffer for weekends)
        warmup_calendar_days = int(lookback_days * 1.5) + 365
        start_date = target_date - datetime.timedelta(days=warmup_calendar_days)

        ohlcv_repo = OhlcvRepository(self._session)
        bars = await ohlcv_repo.fetch_range(
            symbol_id=sym.id,
            start=start_date,
            end=target_date,
            source=source,
        )

        if not bars:
            raise SymbolNotCachedError(clean_symbol, canonical_exchange)

        if len(bars) < 5:
            raise ValueError(
                f"Insufficient OHLCV bars ({len(bars)}) for '{clean_symbol}' "
                "to render a chart."
            )

        # Build raw DataFrame in lowercase convention
        raw_df = pd.DataFrame(
            [
                {
                    "date": b.date,
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                }
                for b in bars
            ]
        )
        raw_df.sort_values("date", inplace=True)
        raw_df.reset_index(drop=True, inplace=True)

        # Compute technical indicators across full series for proper warmup
        raw_df["EMA20"] = ema(raw_df, 20)
        raw_df["EMA50"] = ema(raw_df, 50)
        raw_df["EMA200"] = ema(raw_df, 200)

        # Slice to the requested lookback_days window
        display_df = raw_df.tail(lookback_days).copy()

        # Transform to mplfinance specification: capitalized columns + DatetimeIndex
        plot_df = prepare_chart_dataframe(display_df)

        # Render in worker thread to prevent blocking asyncio event loop
        title = f"{clean_symbol} ({canonical_exchange}) - Daily"
        png_bytes = await asyncio.to_thread(render_chart, plot_df, title=title)

        logger.info(
            "Chart rendered successfully",
            symbol=clean_symbol,
            lookback_days=lookback_days,
            total_bars_fetched=len(bars),
            displayed_bars=len(display_df),
            image_size_bytes=len(png_bytes),
        )

        return png_bytes

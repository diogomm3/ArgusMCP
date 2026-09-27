"""build_candidate_snapshot — assembles a Candidate from cached OHLCV.

Design boundary (must not be violated):
  This function reads from OhlcvRepository.fetch_range() DIRECTLY with
  source passed explicitly. It must NEVER call MarketDataService.get_history()
  or make any network request. The screener (Phase 7) is designed on the
  assumption that snapshot building is a pure DB read.
"""

import datetime
import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from mcp_finance.db.repository import OhlcvRepository, SymbolRepository
from mcp_finance.indicators.functions import (
    adx,
    atr,
    atr_expansion_ratio,
    bollinger_bands,
    bollinger_bandwidth,
    dollar_volume,
    ema,
    ema_alignment,
    ema_slope,
    macd,
    mfi,
    obv,
    relative_volume,
    roc,
    rolling_percentile,
    rsi,
)
from mcp_finance.indicators.models import Candidate
from mcp_finance.indicators.structure import market_structure, week_52_high_low
from mcp_finance.market_data.utils import derive_exchange


class SymbolNotCachedError(Exception):
    """Raised when a symbol has no row in the symbols table.

    This means the batch ingestion job has never run for this ticker/exchange.
    The screener must not treat this as a filter-failed symbol — it should
    surface immediately so the operator knows the watchlist is out of sync with
    the ingestion job.
    """

    def __init__(self, ticker: str, exchange: str) -> None:
        self.ticker = ticker
        self.exchange = exchange
        super().__init__(
            f"Symbol '{ticker}' on exchange '{exchange}' has never been ingested. "
            "Run the batch ingestion job first."
        )


def _to_decimal(value: float, places: int = 6) -> Decimal:
    """Round a float to `places` decimal places and return as Decimal.

    Matches NUMERIC(18, 6) DB precision.
    """
    quantize_str = Decimal(10) ** -places
    return Decimal(str(value)).quantize(quantize_str, rounding=ROUND_HALF_UP)


def _last_valid(series: "pd.Series[float]") -> float | None:
    """Return the last value if finite, or None if empty or NaN/inf."""
    if series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val) or not math.isfinite(float(val)):
        return None
    return float(val)


async def build_candidate_snapshot(
    symbol: str,
    as_of_date: datetime.date,
    session: AsyncSession,
    *,
    exchange: str | None = None,
    source: str = "yfinance",
    lookback_days: int = 400,
) -> Candidate:
    """Build a Candidate technical snapshot for a symbol from cached OHLCV.

    Follows the same optional-exchange-with-auto-derivation convention as
    MarketDataService.ingest_ohlcv() and .get_history(), so callers don't need
    to learn a second symbol-identification pattern.

    Args:
        symbol:        Ticker in yfinance format (e.g. "AAPL", "ASML.AS").
        as_of_date:    Snapshot date. The most recent bar on or before this date
                       is used for the close price; indicators use bars from
                       [as_of_date - lookback_days, as_of_date].
        session:       Async SQLAlchemy session (caller owns commit/rollback).
        exchange:      Optional explicit exchange code. If None, derived
                       automatically via derive_exchange(symbol).
        source:        OHLCV source to read. Defaults to "yfinance". Passed
                       EXPLICITLY to fetch_range — never left implicit.
        lookback_days: How many calendar days of history to load.
                       Default raised from 365 to 400 in M1.6 to guarantee
                       ≥252 trading bars (the widest window required by
                       rolling_percentile(ATR, 252) and EMA-200). With ~252
                       trading days per year, 400 calendar days provides
                       roughly 285 trading bars — enough margin for holidays
                       and weekends without over-fetching.

    Returns:
        Candidate with computed indicators, or indicators=None if insufficient data.

    Raises:
        SymbolNotCachedError: If the symbol has no row in the symbols table
            (batch job has never ingested it). Do not catch silently.
    """
    effective_exchange = derive_exchange(symbol, exchange)

    symbol_repo = SymbolRepository(session)
    symbol_row = await symbol_repo.get_by_ticker(symbol, effective_exchange)
    if symbol_row is None:
        raise SymbolNotCachedError(symbol, effective_exchange)

    start = as_of_date - datetime.timedelta(days=lookback_days)
    ohlcv_repo = OhlcvRepository(session)
    bars = await ohlcv_repo.fetch_range(
        symbol_row.id,
        start,
        as_of_date,
        source=source,  # EXPLICIT — never leave implicit
    )

    bars_available = len(bars)

    if bars_available < 2:
        # Not enough data to compute anything meaningful.
        last_volume = int(bars[-1].volume) if bars_available == 1 else None
        return Candidate(
            symbol=symbol,
            as_of_date=as_of_date,
            source=source,
            close=Decimal(str(bars[-1].close)) if bars_available == 1 else Decimal("0"),
            volume=last_volume,
            bars_available=bars_available,
        )

    # Build normalized DataFrame (float dtype for indicator math).
    df = pd.DataFrame(
        [
            {
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": int(b.volume),
            }
            for b in bars
        ]
    )

    # ------------------------------------------------------------------
    # Single-pass indicator computation — no secondary DB queries allowed.
    # ------------------------------------------------------------------

    # Phase 5 core indicators
    ema_20_series = ema(df, 20)
    ema_50_series = ema(df, 50)
    rsi_14_val = _last_valid(rsi(df, 14))
    atr_14_series = atr(df, 14)
    atr_14_val = _last_valid(atr_14_series)
    macd_df = macd(df)
    macd_line_val = _last_valid(macd_df["macd"])
    macd_signal_val = _last_valid(macd_df["signal"])
    macd_hist_val = _last_valid(macd_df["histogram"])

    # M1.2 — Trend extensions
    ema_9_series = ema(df, 9)
    ema_200_series = ema(df, 200)
    ema_9_val = _last_valid(ema_9_series)
    ema_20_val = _last_valid(ema_20_series)
    ema_50_val = _last_valid(ema_50_series)
    ema_200_val = _last_valid(ema_200_series)
    ema_slope_20_val = _last_valid(ema_slope(ema_20_series, lookback_bars=5))
    ema_align_val = ema_alignment(ema_9_val, ema_20_val, ema_50_val, ema_200_val)

    # M1.3 — Momentum extensions
    adx_df = adx(df, period=14)
    adx_val = _last_valid(adx_df["adx"])
    plus_di_val = _last_valid(adx_df["plus_di"])
    minus_di_val = _last_valid(adx_df["minus_di"])
    roc_10_val = _last_valid(roc(df, period=10))

    # M1.4 — Volatility extensions
    bb_df = bollinger_bands(df, period=20, num_std=2.0)
    bb_upper_val = _last_valid(bb_df["upper"])
    bb_middle_val = _last_valid(bb_df["middle"])
    bb_lower_val = _last_valid(bb_df["lower"])
    bb_bw_val = _last_valid(bollinger_bandwidth(bb_df))
    atr_exp_val = _last_valid(atr_expansion_ratio(atr_14_series, avg_period=50))
    atr_pct_val = _last_valid(rolling_percentile(atr_14_series, window=252))

    # M1.5 — Volume extensions
    rvol_20_val = _last_valid(relative_volume(df, period=20))
    dv_val = _last_valid(dollar_volume(df))
    obv_val = _last_valid(obv(df))
    mfi_14_val = _last_valid(mfi(df, period=14))

    # M1.1 — Market structure extensions
    ms_raw = market_structure(df)
    # "insufficient_data" maps to None per codebase contract (None = no data).
    ms_val: str | None = None if ms_raw == "insufficient_data" else ms_raw
    # week_52_high_low returns tuple[Decimal, Decimal] — unpack directly.
    w52_high_dec, w52_low_dec = week_52_high_low(df, window_bars=252)
    w52_high_val: Decimal | None = w52_high_dec
    w52_low_val: Decimal | None = w52_low_dec

    # Volume (already in single pass above)
    last_close = float(bars[-1].close)
    last_volume = int(bars[-1].volume)
    avg_vol_20 = int(df["volume"].tail(20).mean()) if bars_available >= 20 else None

    def _d(v: float | None) -> Decimal | None:
        return _to_decimal(v) if v is not None else None

    return Candidate(
        symbol=symbol,
        as_of_date=as_of_date,
        source=source,
        close=_to_decimal(last_close),
        # Phase 5 core
        rsi_14=_d(rsi_14_val),
        ema_20=_d(ema_20_val),
        ema_50=_d(ema_50_val),
        atr_14=_d(atr_14_val),
        macd_line=_d(macd_line_val),
        macd_signal=_d(macd_signal_val),
        macd_histogram=_d(macd_hist_val),
        volume=last_volume,
        avg_volume_20=avg_vol_20,
        bars_available=bars_available,
        # M1.2 trend
        ema_9=_d(ema_9_val),
        ema_200=_d(ema_200_val),
        ema_slope_20=_d(ema_slope_20_val),
        ema_alignment=ema_align_val
        if any(v is not None for v in [ema_9_val, ema_20_val, ema_50_val, ema_200_val])
        else None,
        # M1.3 momentum
        adx_14=_d(adx_val),
        plus_di_14=_d(plus_di_val),
        minus_di_14=_d(minus_di_val),
        roc_10=_d(roc_10_val),
        # M1.4 volatility
        bb_upper_20=_d(bb_upper_val),
        bb_middle_20=_d(bb_middle_val),
        bb_lower_20=_d(bb_lower_val),
        bb_bandwidth_20=_d(bb_bw_val),
        atr_expansion_ratio=_d(atr_exp_val),
        atr_percentile_252=_d(atr_pct_val),
        # M1.5 volume
        rvol_20=_d(rvol_20_val),
        dollar_volume=_d(dv_val),
        obv=_d(obv_val),
        mfi_14=_d(mfi_14_val),
        # M1.1 market structure
        market_structure=ms_val,
        week_52_high=w52_high_val,
        week_52_low=w52_low_val,
    )

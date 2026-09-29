"""Tests for M4 relative-strength module and ETF ingestion constants.

Commit-group 1: constant invariant tests for BENCHMARK_SYMBOLS and SECTOR_ETF_MAP.
Commit-group 3: relative_return and relative_price_ratio unit tests.

No network calls, no database connections in any of these tests.
"""

import datetime

import pandas as pd
import pytest

from mcp_finance.indicators.relative_strength import (
    relative_price_ratio,
    relative_return,
)
from mcp_finance.market_data.batch import (
    BENCHMARK_SYMBOLS,
    DEFAULT_WATCHLIST,
    INGESTION_WATCHLIST,
    SECTOR_ETF_MAP,
    get_sector_etf,
)

# ---------------------------------------------------------------------------
# Helpers (group-1)
# ---------------------------------------------------------------------------

_UNIQUE_SECTOR_ETF_SYMBOLS: list[str] = list(dict.fromkeys(SECTOR_ETF_MAP.values()))
_ALL_ETF_SYMBOLS: list[str] = list(BENCHMARK_SYMBOLS) + _UNIQUE_SECTOR_ETF_SYMBOLS

# Expected 13 tickers: 2 benchmarks + 11 unique sector ETFs
_EXPECTED_COUNT = 13
_EXPECTED_BENCHMARK_COUNT = 2
_EXPECTED_UNIQUE_SECTOR_COUNT = 11
_EXPECTED_MAP_KEY_COUNT = 17  # 11 FMP labels + 6 GICS aliases

# 3 observed labels from live FMP API profile calls in this project:
#   - Technology (AAPL, MSFT, NVDA)
#   - Consumer Cyclical (AMZN)
#   - Communication Services (GOOGL)
_OBSERVED_FMP_SECTORS: dict[str, str] = {
    "Technology": "XLK",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
}

# 8 assumed standard FMP sector profile names (canonical FMP profile convention):
_ASSUMED_FMP_SECTORS: dict[str, str] = {
    "Financial Services": "XLF",
    "Consumer Defensive": "XLP",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# 6 GICS standard sector aliases supported for cross-provider compatibility:
_GICS_ALIASES: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Materials": "XLB",
}


# ---------------------------------------------------------------------------
# BENCHMARK_SYMBOLS invariants
# ---------------------------------------------------------------------------


class TestBenchmarkSymbols:
    def test_benchmark_symbols_is_list(self) -> None:
        assert isinstance(BENCHMARK_SYMBOLS, list)

    def test_benchmark_symbols_count(self) -> None:
        """SPY and QQQ — exactly 2 benchmarks."""
        assert len(BENCHMARK_SYMBOLS) == _EXPECTED_BENCHMARK_COUNT

    def test_spy_present(self) -> None:
        """SPY is the primary broad-market benchmark; must always be included."""
        assert "SPY" in BENCHMARK_SYMBOLS

    def test_qqq_present(self) -> None:
        """QQQ is the Nasdaq-100 tech/growth benchmark."""
        assert "QQQ" in BENCHMARK_SYMBOLS

    def test_benchmark_symbols_uppercase(self) -> None:
        """All tickers must be uppercase strings (yfinance convention)."""
        for sym in BENCHMARK_SYMBOLS:
            assert sym == sym.upper(), f"{sym!r} is not uppercase"

    def test_benchmark_symbols_no_empty_strings(self) -> None:
        for sym in BENCHMARK_SYMBOLS:
            assert sym.strip(), f"Empty or whitespace-only symbol found: {sym!r}"

    def test_benchmark_symbols_no_duplicates(self) -> None:
        assert len(BENCHMARK_SYMBOLS) == len(set(BENCHMARK_SYMBOLS))

    def test_benchmark_symbols_us_only(self) -> None:
        """Benchmark ETFs must be US-listed (no suffix)."""
        for sym in BENCHMARK_SYMBOLS:
            assert "." not in sym, (
                f"{sym!r} contains a dot — benchmark ETFs must be US-listed"
            )

    def test_benchmark_symbols_not_in_default_watchlist(self) -> None:
        """Benchmark symbols must be kept separate from the stock watchlist."""
        from mcp_finance.market_data.batch import DEFAULT_WATCHLIST

        overlap = set(BENCHMARK_SYMBOLS) & set(DEFAULT_WATCHLIST)
        assert overlap == set(), (
            f"Benchmark symbols leaked into DEFAULT_WATCHLIST: {overlap}"
        )


# ---------------------------------------------------------------------------
# SECTOR_ETF_MAP invariants
# ---------------------------------------------------------------------------


class TestSectorEtfMap:
    def test_sector_etf_map_is_dict(self) -> None:
        assert isinstance(SECTOR_ETF_MAP, dict)

    def test_sector_key_count(self) -> None:
        """11 FMP labels + 6 GICS aliases = 17 keys."""
        assert len(SECTOR_ETF_MAP) == _EXPECTED_MAP_KEY_COUNT

    def test_unique_etf_count_is_11(self) -> None:
        """Must cover exactly 11 unique SPDR sector ETF tickers."""
        unique_tickers = set(SECTOR_ETF_MAP.values())
        assert len(unique_tickers) == _EXPECTED_UNIQUE_SECTOR_COUNT

    @pytest.mark.parametrize("sector,expected_etf", _OBSERVED_FMP_SECTORS.items())
    def test_observed_fmp_labels_map_correctly(
        self, sector: str, expected_etf: str
    ) -> None:
        """Labels observed live from FMP profile endpoint map to correct ETF."""
        assert SECTOR_ETF_MAP[sector] == expected_etf
        assert get_sector_etf(sector) == expected_etf

    @pytest.mark.parametrize("sector,expected_etf", _ASSUMED_FMP_SECTORS.items())
    def test_assumed_fmp_labels_map_correctly(
        self, sector: str, expected_etf: str
    ) -> None:
        """Canonical assumed FMP profile sector labels map to correct ETF."""
        assert SECTOR_ETF_MAP[sector] == expected_etf
        assert get_sector_etf(sector) == expected_etf

    @pytest.mark.parametrize("sector,expected_etf", _GICS_ALIASES.items())
    def test_gics_aliases_map_correctly(self, sector: str, expected_etf: str) -> None:
        """GICS standard aliases map to correct ETF for cross-provider compatibility."""
        assert SECTOR_ETF_MAP[sector] == expected_etf
        assert get_sector_etf(sector) == expected_etf

    def test_unknown_or_none_sector_maps_to_none(self) -> None:
        """Unknown or None sector label must map to None."""
        assert get_sector_etf(None) is None
        assert get_sector_etf("") is None
        assert get_sector_etf("Unknown Sector") is None
        assert SECTOR_ETF_MAP.get("Unknown Sector") is None
        assert SECTOR_ETF_MAP.get("") is None

    def test_sector_etf_tickers_uppercase(self) -> None:
        for sector, sym in SECTOR_ETF_MAP.items():
            assert sym == sym.upper(), (
                f"Sector '{sector}': ticker {sym!r} is not uppercase"
            )

    def test_sector_etf_tickers_no_empty_strings(self) -> None:
        for sector, sym in SECTOR_ETF_MAP.items():
            assert sym.strip(), f"Sector '{sector}' maps to empty/whitespace ticker"

    def test_sector_etf_tickers_us_only(self) -> None:
        """Sector ETFs must be US-listed (no dot suffix)."""
        for sector, sym in SECTOR_ETF_MAP.items():
            assert "." not in sym, (
                f"Sector '{sector}': {sym!r} contains a dot — must be US-listed"
            )

    def test_sector_etf_not_in_default_watchlist(self) -> None:
        overlap = set(SECTOR_ETF_MAP.values()) & set(DEFAULT_WATCHLIST)
        assert overlap == set(), f"Sector ETFs leaked into DEFAULT_WATCHLIST: {overlap}"

    def test_sector_etf_not_in_benchmark_symbols(self) -> None:
        overlap = set(SECTOR_ETF_MAP.values()) & set(BENCHMARK_SYMBOLS)
        assert overlap == set(), (
            f"Sector ETFs overlap with benchmark symbols: {overlap}"
        )


# ---------------------------------------------------------------------------
# Combined: 13-ticker total count & INGESTION_WATCHLIST
# ---------------------------------------------------------------------------


class TestCombinedEtfInventory:
    def test_total_etf_count_is_13(self) -> None:
        """BENCHMARK_SYMBOLS (2) + unique SECTOR_ETF_MAP values (11) = 13 tickers."""
        total = len(BENCHMARK_SYMBOLS) + len(set(SECTOR_ETF_MAP.values()))
        assert total == _EXPECTED_COUNT, (
            f"Expected {_EXPECTED_COUNT} ETF tickers total, got {total}"
        )

    def test_all_etf_symbols_unique_across_both_sets(self) -> None:
        """No ticker should appear in both benchmarks and sector map."""
        all_syms = _ALL_ETF_SYMBOLS
        assert len(all_syms) == len(set(all_syms)), (
            "Duplicate tickers found across BENCHMARK_SYMBOLS and SECTOR_ETF_MAP"
        )

    @pytest.mark.parametrize("sym", _ALL_ETF_SYMBOLS)
    def test_each_etf_symbol_is_non_empty_uppercase_no_dot(self, sym: str) -> None:
        assert sym and sym == sym.upper() and "." not in sym

    def test_ingestion_watchlist_contains_all_components(self) -> None:
        """INGESTION_WATCHLIST is the deduplicated union of
        stocks, benchmarks, and ETFs.
        """
        assert len(INGESTION_WATCHLIST) == 23
        assert len(INGESTION_WATCHLIST) == len(set(INGESTION_WATCHLIST))
        # Stocks first
        assert INGESTION_WATCHLIST[: len(DEFAULT_WATCHLIST)] == DEFAULT_WATCHLIST
        # All benchmarks included
        for b in BENCHMARK_SYMBOLS:
            assert b in INGESTION_WATCHLIST
        # All sector ETFs included
        for s in set(SECTOR_ETF_MAP.values()):
            assert s in INGESTION_WATCHLIST

    def test_run_batch_ingest_defaults_to_ingestion_watchlist(self) -> None:
        """When symbols is None, run_batch_ingest defaults to INGESTION_WATCHLIST."""
        import inspect

        from mcp_finance.market_data.batch import run_batch_ingest

        sig = inspect.signature(run_batch_ingest)
        assert sig.parameters["symbols"].default is None


# ---------------------------------------------------------------------------
# Commit-group 3: relative_return and relative_price_ratio unit tests
# ---------------------------------------------------------------------------


def _make_df(dates: list[str], closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV-shaped DataFrame with a 'date' column."""
    return pd.DataFrame(
        {
            "date": [datetime.date.fromisoformat(d) for d in dates],
            "close": closes,
        }
    )


# 5-bar window used across most tests
_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
# Stock: starts at 100, ends at 110 → +10%
_STOCK_CLOSES = [100.0, 102.0, 104.0, 107.0, 110.0]
# Bench: starts at 200, ends at 204 → +2%
_BENCH_CLOSES = [200.0, 201.0, 202.0, 203.0, 204.0]

_STOCK_DF = _make_df(_DATES, _STOCK_CLOSES)
_BENCH_DF = _make_df(_DATES, _BENCH_CLOSES)


# ---------------------------------------------------------------------------
# relative_return — basic correctness
# ---------------------------------------------------------------------------


class TestRelativeReturn:
    def test_hand_computed_reference(self) -> None:
        """4-bar window: stock +10%, bench +2% → relative_return = +0.08."""
        result = relative_return(_STOCK_DF, _BENCH_DF, period=4)
        assert result is not None
        # stock: (110-100)/100 = 0.10, bench: (204-200)/200 = 0.02, diff = 0.08
        assert abs(result - 0.08) < 1e-9

    def test_returns_float(self) -> None:
        result = relative_return(_STOCK_DF, _BENCH_DF, period=4)
        assert isinstance(result, float)

    # --- Identity: stock == benchmark → relative_return == 0 ---

    def test_identity_same_series(self) -> None:
        """When stock and benchmark are identical, relative return must be 0."""
        df = _make_df(_DATES, _STOCK_CLOSES)
        result = relative_return(df, df.copy(), period=4)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-10)

    # --- Antisymmetry: rr(A, B) == -rr(B, A) ---

    def test_antisymmetry(self) -> None:
        ab = relative_return(_STOCK_DF, _BENCH_DF, period=4)
        ba = relative_return(_BENCH_DF, _STOCK_DF, period=4)
        assert ab is not None and ba is not None
        assert ab == pytest.approx(-ba, abs=1e-9)

    # --- as_of_date backward-snap ---

    def test_as_of_date_exact_match(self) -> None:
        """Specifying as_of_date == last bar date gives same result as default."""
        result_default = relative_return(_STOCK_DF, _BENCH_DF, period=4)
        result_explicit = relative_return(
            _STOCK_DF, _BENCH_DF, period=4, as_of_date="2024-01-08"
        )
        assert result_default == pytest.approx(result_explicit)

    def test_as_of_date_backward_snap_to_friday(self) -> None:
        """as_of_date on a weekend snaps backward to the preceding Friday."""
        # Last trading bar in fixture is 2024-01-08 (Monday). Use Sunday.
        result_sunday = relative_return(
            _STOCK_DF, _BENCH_DF, period=3, as_of_date="2024-01-07"
        )
        # Snaps to 2024-01-05 (Friday). Window: bars [0:4] → dates 02,03,04,05
        result_friday = relative_return(
            _STOCK_DF, _BENCH_DF, period=3, as_of_date="2024-01-05"
        )
        assert result_sunday == pytest.approx(result_friday)

    def test_as_of_date_before_all_data_returns_none(self) -> None:
        result = relative_return(
            _STOCK_DF, _BENCH_DF, period=4, as_of_date="2023-12-31"
        )
        assert result is None

    def test_look_ahead_not_used(self) -> None:
        """Bars after as_of_date must not affect the result."""
        # as_of = 2024-01-04 (index 2), period=2 → uses bars [0,1,2]
        result_early = relative_return(
            _STOCK_DF, _BENCH_DF, period=2, as_of_date="2024-01-04"
        )
        # Truncated DF: same first 3 bars, no future bars
        trunc_stock = _make_df(_DATES[:3], _STOCK_CLOSES[:3])
        trunc_bench = _make_df(_DATES[:3], _BENCH_CLOSES[:3])
        result_trunc = relative_return(trunc_stock, trunc_bench, period=2)
        assert result_early is not None and result_trunc is not None
        assert result_early == pytest.approx(result_trunc, abs=1e-9)

    # --- Contiguity / missing-bar ---

    def test_benchmark_missing_stock_last_bar_returns_none(self) -> None:
        """Stale benchmark missing stock's last bar returns None
        (no silent backward shift).
        """
        stale_bench = _make_df(_DATES[:-1], _BENCH_CLOSES[:-1])
        result = relative_return(_STOCK_DF, stale_bench, period=3)
        assert result is None

    def test_gap_in_stock_returns_none(self) -> None:
        """Stock missing bar 2024-01-04 causes a contiguity mismatch → None."""
        gap_dates = [_DATES[0], _DATES[1], _DATES[3], _DATES[4]]
        gap_closes = [
            _STOCK_CLOSES[0],
            _STOCK_CLOSES[1],
            _STOCK_CLOSES[3],
            _STOCK_CLOSES[4],
        ]
        gap_stock = _make_df(gap_dates, gap_closes)
        result = relative_return(gap_stock, _BENCH_DF, period=4)
        assert result is None

    def test_gap_in_benchmark_returns_none(self) -> None:
        """Benchmark missing a bar causes a contiguity mismatch → None."""
        gap_dates = [_DATES[0], _DATES[1], _DATES[3], _DATES[4]]
        gap_closes = [
            _BENCH_CLOSES[0],
            _BENCH_CLOSES[1],
            _BENCH_CLOSES[3],
            _BENCH_CLOSES[4],
        ]
        gap_bench = _make_df(gap_dates, gap_closes)
        result = relative_return(_STOCK_DF, gap_bench, period=4)
        assert result is None

    def test_insufficient_bars_returns_none(self) -> None:
        """period=10 but only 5 bars → None."""
        result = relative_return(_STOCK_DF, _BENCH_DF, period=10)
        assert result is None

    # --- Non-positive / NaN closes ---

    def test_non_positive_stock_start_returns_none(self) -> None:
        bad_closes = [0.0] + _STOCK_CLOSES[1:]
        bad_stock = _make_df(_DATES, bad_closes)
        result = relative_return(bad_stock, _BENCH_DF, period=4)
        assert result is None

    def test_non_positive_bench_start_returns_none(self) -> None:
        bad_closes = [-1.0] + _BENCH_CLOSES[1:]
        bad_bench = _make_df(_DATES, bad_closes)
        result = relative_return(_STOCK_DF, bad_bench, period=4)
        assert result is None

    def test_period_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be at least 1"):
            relative_return(_STOCK_DF, _BENCH_DF, period=0)

    def test_empty_stock_df_returns_none(self) -> None:
        empty = pd.DataFrame(columns=["date", "close"])
        result = relative_return(empty, _BENCH_DF, period=4)
        assert result is None

    def test_empty_bench_df_returns_none(self) -> None:
        empty = pd.DataFrame(columns=["date", "close"])
        result = relative_return(_STOCK_DF, empty, period=4)
        assert result is None

    def test_missing_close_column_returns_none(self) -> None:
        no_close = pd.DataFrame({"date": _DATES, "open": [1.0] * 5})
        result = relative_return(no_close, _BENCH_DF, period=4)
        assert result is None

    def test_datetime_index_input(self) -> None:
        """Both DataFrames can use DatetimeIndex instead of date column."""
        idx = pd.DatetimeIndex(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )
        s_df = pd.DataFrame({"close": _STOCK_CLOSES}, index=idx)
        b_df = pd.DataFrame({"close": _BENCH_CLOSES}, index=idx)
        result = relative_return(s_df, b_df, period=4)
        assert result is not None
        assert abs(result - 0.08) < 1e-9


# ---------------------------------------------------------------------------
# relative_price_ratio
# ---------------------------------------------------------------------------


class TestRelativePriceRatio:
    def test_hand_computed_ratio_values(self) -> None:
        """5 shared dates: ratios should be stock/bench for each date."""
        ratio = relative_price_ratio(_STOCK_DF, _BENCH_DF)
        assert ratio is not None
        assert len(ratio) == 5
        for s_c, b_c, r in zip(_STOCK_CLOSES, _BENCH_CLOSES, ratio.values):
            assert r == pytest.approx(s_c / b_c, abs=1e-9)

    def test_returns_series_of_floats(self) -> None:
        ratio = relative_price_ratio(_STOCK_DF, _BENCH_DF)
        assert isinstance(ratio, pd.Series)
        assert ratio.dtype == float

    def test_index_is_datetimeindex(self) -> None:
        ratio = relative_price_ratio(_STOCK_DF, _BENCH_DF)
        assert ratio is not None
        assert isinstance(ratio.index, pd.DatetimeIndex)

    def test_no_overlap_returns_none(self) -> None:
        """Non-overlapping date ranges → None (strict window)."""
        stock_past = _make_df(["2023-01-02", "2023-01-03"], [50.0, 51.0])
        bench_future = _make_df(["2024-01-02", "2024-01-03"], [200.0, 201.0])
        assert relative_price_ratio(stock_past, bench_future) is None

    def test_partial_overlap_missing_bar_returns_none(self) -> None:
        """Benchmark missing bars from stock window → None (no silent truncation)."""
        stock = _make_df(_DATES[:4], _STOCK_CLOSES[:4])
        bench = _make_df(_DATES[2:], _BENCH_CLOSES[2:])
        assert relative_price_ratio(stock, bench) is None

    def test_benchmark_missing_stock_last_bar_returns_none(self) -> None:
        """Benchmark missing stock's last bar returns None."""
        stale_bench = _make_df(_DATES[:-1], _BENCH_CLOSES[:-1])
        assert relative_price_ratio(_STOCK_DF, stale_bench) is None

    def test_zero_benchmark_close_returns_none(self) -> None:
        """Benchmark close = 0 → None on invalid prices."""
        bench_zero = _make_df(_DATES, [200.0, 0.0, 202.0, 203.0, 204.0])
        assert relative_price_ratio(_STOCK_DF, bench_zero) is None

    def test_negative_benchmark_close_returns_none(self) -> None:
        bench_neg = _make_df(_DATES, [200.0, -5.0, 202.0, 203.0, 204.0])
        assert relative_price_ratio(_STOCK_DF, bench_neg) is None

    def test_nan_close_returns_none(self) -> None:
        bench_nan = _make_df(_DATES, [200.0, float("nan"), 202.0, 203.0, 204.0])
        assert relative_price_ratio(_STOCK_DF, bench_nan) is None

    def test_empty_stock_df_returns_none(self) -> None:
        empty = pd.DataFrame({"date": [], "close": []})
        assert relative_price_ratio(empty, _BENCH_DF) is None

    def test_empty_bench_df_returns_none(self) -> None:
        empty = pd.DataFrame({"date": [], "close": []})
        assert relative_price_ratio(_STOCK_DF, empty) is None

    def test_missing_close_raises(self) -> None:
        no_close = pd.DataFrame({"date": _DATES, "open": [1.0] * 5})
        with pytest.raises(ValueError, match="'close' column"):
            relative_price_ratio(no_close, _BENCH_DF)

    def test_inversion_of_ratio(self) -> None:
        """ratio(A, B) * ratio(B, A) == 1 (inversion property)."""
        ab = relative_price_ratio(_STOCK_DF, _BENCH_DF)
        ba = relative_price_ratio(_BENCH_DF, _STOCK_DF)
        assert ab is not None and ba is not None
        assert len(ab) == len(ba)
        for v_ab, v_ba in zip(ab.values, ba.values):
            assert (v_ab * v_ba) == pytest.approx(1.0, rel=1e-9)

    def test_identity_ratio_is_one(self) -> None:
        """Same DataFrame vs itself → ratio is 1.0 for every bar."""
        df = _make_df(_DATES, _STOCK_CLOSES)
        ratio = relative_price_ratio(df, df.copy())
        assert ratio is not None
        assert all(v == pytest.approx(1.0, abs=1e-9) for v in ratio.values)

    def test_lookback_bars_parameter(self) -> None:
        """Specifying lookback_bars slices to exactly that many trailing bars."""
        ratio_full = relative_price_ratio(_STOCK_DF, _BENCH_DF)
        ratio_3 = relative_price_ratio(_STOCK_DF, _BENCH_DF, lookback_bars=3)
        assert ratio_full is not None and ratio_3 is not None
        assert len(ratio_3) == 3
        pd.testing.assert_series_equal(ratio_3, ratio_full.iloc[-3:])

    def test_lookback_bars_insufficient_returns_none(self) -> None:
        """lookback_bars > available bars → None."""
        assert relative_price_ratio(_STOCK_DF, _BENCH_DF, lookback_bars=10) is None

    def test_lookback_bars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback_bars must be at least 1"):
            relative_price_ratio(_STOCK_DF, _BENCH_DF, lookback_bars=0)

    def test_as_of_date_snap(self) -> None:
        """as_of_date backward-snap selects bars on or before as_of_date."""
        ratio_early = relative_price_ratio(
            _STOCK_DF, _BENCH_DF, lookback_bars=2, as_of_date="2024-01-04"
        )
        assert ratio_early is not None
        assert len(ratio_early) == 2
        assert ratio_early.index[-1] == pd.Timestamp("2024-01-04")

    def test_as_of_date_before_first_bar_returns_none(self) -> None:
        assert (
            relative_price_ratio(_STOCK_DF, _BENCH_DF, as_of_date="2023-12-31") is None
        )

    def test_duplicate_dates_returns_none(self) -> None:
        dup_dates = [_DATES[0], _DATES[0], _DATES[2], _DATES[3], _DATES[4]]
        dup_df = _make_df(dup_dates, _STOCK_CLOSES)
        assert relative_price_ratio(dup_df, _BENCH_DF) is None
        assert relative_return(dup_df, _BENCH_DF, period=2) is None

    def test_unsorted_dates_returns_none(self) -> None:
        unsorted_dates = [_DATES[1], _DATES[0], _DATES[2], _DATES[3], _DATES[4]]
        unsorted_df = _make_df(unsorted_dates, _STOCK_CLOSES)
        assert relative_price_ratio(unsorted_df, _BENCH_DF) is None
        assert relative_return(unsorted_df, _BENCH_DF, period=2) is None

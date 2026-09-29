"""Tests for M4 relative-strength module and ETF ingestion constants.

Commit-group 1 (this file): constant invariant tests for BENCHMARK_SYMBOLS
and SECTOR_ETF_MAP that verify structural properties without making any
network calls or database connections.

Commit-group 3 (functions added later): relative_return, relative_price_ratio,
and relative_strength_trend unit tests against frozen fixtures.
"""

import pytest

from mcp_finance.market_data.batch import BENCHMARK_SYMBOLS, SECTOR_ETF_MAP

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_ETF_SYMBOLS: list[str] = list(BENCHMARK_SYMBOLS) + list(SECTOR_ETF_MAP.values())

# Expected 13 tickers: 2 benchmarks + 11 sector ETFs
_EXPECTED_COUNT = 13
_EXPECTED_BENCHMARK_COUNT = 2
_EXPECTED_SECTOR_COUNT = 11

# The 11 GICS sectors that must all be present in SECTOR_ETF_MAP
_EXPECTED_SECTORS = {
    "Technology",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Industrials",
    "Materials",
    "Real Estate",
    "Utilities",
    "Communication Services",
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

    def test_sector_count(self) -> None:
        """Must cover all 11 GICS sectors."""
        assert len(SECTOR_ETF_MAP) == _EXPECTED_SECTOR_COUNT

    def test_all_expected_sectors_present(self) -> None:
        missing = _EXPECTED_SECTORS - set(SECTOR_ETF_MAP.keys())
        assert missing == set(), f"Missing GICS sectors: {missing}"

    def test_no_unexpected_sectors(self) -> None:
        extra = set(SECTOR_ETF_MAP.keys()) - _EXPECTED_SECTORS
        assert extra == set(), f"Unexpected sectors in map: {extra}"

    def test_sector_etf_tickers_uppercase(self) -> None:
        for sector, sym in SECTOR_ETF_MAP.items():
            assert sym == sym.upper(), (
                f"Sector '{sector}': ticker {sym!r} is not uppercase"
            )

    def test_sector_etf_tickers_no_empty_strings(self) -> None:
        for sector, sym in SECTOR_ETF_MAP.items():
            assert sym.strip(), f"Sector '{sector}' maps to empty/whitespace ticker"

    def test_sector_etf_tickers_no_duplicates(self) -> None:
        tickers = list(SECTOR_ETF_MAP.values())
        assert len(tickers) == len(set(tickers)), (
            "Duplicate ticker values in SECTOR_ETF_MAP"
        )

    def test_sector_etf_tickers_us_only(self) -> None:
        """Sector ETFs must be US-listed (no dot suffix)."""
        for sector, sym in SECTOR_ETF_MAP.items():
            assert "." not in sym, (
                f"Sector '{sector}': {sym!r} contains a dot — must be US-listed"
            )

    def test_xly_is_consumer_discretionary(self) -> None:
        """Spot-check: XLY maps to Consumer Discretionary."""
        assert SECTOR_ETF_MAP["Consumer Discretionary"] == "XLY"

    def test_xlk_is_technology(self) -> None:
        """Spot-check: XLK maps to Technology."""
        assert SECTOR_ETF_MAP["Technology"] == "XLK"

    def test_sector_etf_not_in_default_watchlist(self) -> None:
        """Sector ETFs must be kept separate from the stock watchlist."""
        from mcp_finance.market_data.batch import DEFAULT_WATCHLIST

        overlap = set(SECTOR_ETF_MAP.values()) & set(DEFAULT_WATCHLIST)
        assert overlap == set(), f"Sector ETFs leaked into DEFAULT_WATCHLIST: {overlap}"

    def test_sector_etf_not_in_benchmark_symbols(self) -> None:
        """Sector ETFs and benchmark symbols must be orthogonal sets."""
        overlap = set(SECTOR_ETF_MAP.values()) & set(BENCHMARK_SYMBOLS)
        assert overlap == set(), (
            f"Sector ETFs overlap with benchmark symbols: {overlap}"
        )


# ---------------------------------------------------------------------------
# Combined: 13-ticker total count
# ---------------------------------------------------------------------------


class TestCombinedEtfInventory:
    def test_total_etf_count_is_13(self) -> None:
        """BENCHMARK_SYMBOLS (2) + SECTOR_ETF_MAP values (11) = 13 tickers."""
        total = len(BENCHMARK_SYMBOLS) + len(SECTOR_ETF_MAP)
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

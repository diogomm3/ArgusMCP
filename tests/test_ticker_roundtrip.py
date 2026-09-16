"""Round-trip tests between to_broker_ticker and canonical_ticker_symbol.

The two functions serve opposite directions:
  - ``to_broker_ticker``:       caller symbol  → broker wire format
  - ``canonical_ticker_symbol``: broker format  → canonical symbol

The invariant under test is:

    canonical_ticker_symbol(to_broker_ticker(symbol)) == canonical_ticker_symbol(symbol)

That is: both functions must agree on the *canonical* form after a
round-trip through broker format. They do not need to produce the same
intermediate string — ``to_broker_ticker`` is a dispatch helper, not an
inverse of ``canonical_ticker_symbol``.

Separate from the round-trip, we also verify that ``to_broker_ticker``
does not double-suffix already-formatted tickers, and that
``canonical_ticker_symbol`` correctly handles every suffix pattern in
the known T212 catalog (``_BROKER_SUFFIXES``).
"""

import pytest

from mcp_finance.risk.risk_tools import to_broker_ticker
from mcp_finance.risk.rules import _BROKER_SUFFIXES, canonical_ticker_symbol

# ---------------------------------------------------------------------------
# 1.  to_broker_ticker: caller-input → wire format
# ---------------------------------------------------------------------------

_TO_BROKER_CASES: list[tuple[str, str]] = [
    # Plain US tickers: must get _US_EQ suffix
    ("AAPL", "AAPL_US_EQ"),
    ("MSFT", "MSFT_US_EQ"),
    ("TSLA", "TSLA_US_EQ"),
    # .US suffix → _US_EQ
    ("AAPL.US", "AAPL_US_EQ"),
    # .DE suffix → _DE_EQ
    ("SAP.DE", "SAP_DE_EQ"),
    ("BAS.DE", "BAS_DE_EQ"),
    # Already-broker-formatted tickers: must pass through unchanged (no double suffix)
    ("AAPL_US_EQ", "AAPL_US_EQ"),
    ("SAP_DE_EQ", "SAP_DE_EQ"),
    ("LLOY_UK_EQ", "LLOY_UK_EQ"),
    ("SHOP_CA_EQ", "SHOP_CA_EQ"),
    ("AIR_FR_EQ", "AIR_FR_EQ"),
    ("ASML_NL_EQ", "ASML_NL_EQ"),
    ("SIE_DE_EQ", "SIE_DE_EQ"),
    ("ITX_ES_EQ", "ITX_ES_EQ"),
    ("G_IT_EQ", "G_IT_EQ"),
    ("VOLV_B_SE_EQ", "VOLV_B_SE_EQ"),
    ("NESN_CH_EQ", "NESN_CH_EQ"),
    ("OMV_AT_EQ", "OMV_AT_EQ"),
    ("SOLB_BE_EQ", "SOLB_BE_EQ"),
    ("EDP_PT_EQ", "EDP_PT_EQ"),
    ("NOKIA_FI_EQ", "NOKIA_FI_EQ"),
    ("CRH_IE_EQ", "CRH_IE_EQ"),
    ("NOVO_B_DK_EQ", "NOVO_B_DK_EQ"),
    ("EQNR_NO_EQ", "EQNR_NO_EQ"),
    # Lowercase input: must be uppercased before processing
    ("aapl", "AAPL_US_EQ"),
    ("sap.de", "SAP_DE_EQ"),
]


@pytest.mark.unit
@pytest.mark.parametrize("symbol,expected_broker", _TO_BROKER_CASES)
def test_to_broker_ticker(symbol: str, expected_broker: str) -> None:
    """to_broker_ticker maps caller symbols to expected broker wire format."""
    assert to_broker_ticker(symbol) == expected_broker


# ---------------------------------------------------------------------------
# 2.  No double-suffixing: already-formatted tickers must pass through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "already_formatted",
    [
        "AAPL_US_EQ",
        "SAP_DE_EQ",
        "LLOY_UK_EQ",
        "AIR_FR_EQ",
        "ASML_NL_EQ",
        "EQNR_NO_EQ",
        "NOVO_B_DK_EQ",
    ],
)
def test_to_broker_ticker_no_double_suffix(already_formatted: str) -> None:
    """Passing an already-broker-formatted ticker must not append a second suffix."""
    result = to_broker_ticker(already_formatted)
    assert result == already_formatted, (
        f"to_broker_ticker({already_formatted!r}) = {result!r}; "
        f"expected pass-through unchanged"
    )


# ---------------------------------------------------------------------------
# 3.  canonical_ticker_symbol: broker format → canonical
# ---------------------------------------------------------------------------

_CANONICAL_CASES: list[tuple[str, str]] = [
    # Standard US
    ("AAPL_US_EQ", "AAPL"),
    ("MSFT_US_EQ", "MSFT"),
    ("TSLA_US_EQ", "TSLA"),
    # Standard European
    ("SAP_DE_EQ", "SAP"),
    ("LLOY_UK_EQ", "LLOY"),
    ("AIR_FR_EQ", "AIR"),
    ("ASML_NL_EQ", "ASML"),
    ("ITX_ES_EQ", "ITX"),
    ("SHOP_CA_EQ", "SHOP"),
    ("NESN_CH_EQ", "NESN"),
    ("OMV_AT_EQ", "OMV"),
    ("NOKIA_FI_EQ", "NOKIA"),
    ("CRH_IE_EQ", "CRH"),
    ("EQNR_NO_EQ", "EQNR"),
    ("EDP_PT_EQ", "EDP"),
    # T212 lowercase country-discriminator letter (e.g. SAPd_EQ)
    ("SAPd_EQ", "SAP"),
    ("ASMLa_EQ", "ASML"),
    ("IFXd_EQ", "IFX"),
    # Already canonical (no suffix)
    ("AAPL", "AAPL"),
    ("SAP", "SAP"),
    # .US / .DE dotted suffixes (from canonical_ticker_symbol suffix list)
    ("AAPL.US", "AAPL"),
    ("BAS.DE", "BAS"),
    # Share-class preservation — THE critical case
    # T212 stores BRK.B as "BRK_B_US_EQ"
    ("BRK_B_US_EQ", "BRK.B"),
    # BRK/A variant
    ("BRK/A_US_EQ", "BRK.A"),
    # Multi-class European (e.g. Novo Nordisk B shares)
    ("NOVO_B_DK_EQ", "NOVO.B"),
    # VOLV B shares on Stockholm
    ("VOLV_B_SE_EQ", "VOLV.B"),
]


@pytest.mark.unit
@pytest.mark.parametrize("broker_fmt,expected_canon", _CANONICAL_CASES)
def test_canonical_ticker_symbol(broker_fmt: str, expected_canon: str) -> None:
    """canonical_ticker_symbol strips known suffixes and returns canonical form."""
    result = canonical_ticker_symbol(broker_fmt)
    assert result == expected_canon, (
        f"canonical_ticker_symbol({broker_fmt!r}) = {result!r}; "
        f"expected {expected_canon!r}"
    )


# ---------------------------------------------------------------------------
# 4.  Round-trip invariant:
#     canonical(to_broker_ticker(symbol)) == canonical(symbol)
# ---------------------------------------------------------------------------

_ROUNDTRIP_CASES: list[tuple[str, str]] = [
    # Plain canonical symbols
    ("AAPL", "AAPL"),
    ("MSFT", "MSFT"),
    ("TSLA", "TSLA"),
    # .US suffixed input
    ("AAPL.US", "AAPL"),
    # .DE suffixed input
    ("SAP.DE", "SAP"),
    ("BAS.DE", "BAS"),
    # Already-broker-formatted — canonical must survive the pass-through
    ("AAPL_US_EQ", "AAPL"),
    ("SAP_DE_EQ", "SAP"),
    ("LLOY_UK_EQ", "LLOY"),
    ("AIR_FR_EQ", "AIR"),
    ("ASML_NL_EQ", "ASML"),
    # Share classes: to_broker_ticker passes them through unchanged (contains "_"),
    # canonical_ticker_symbol then strips the exchange suffix and normalises
    ("BRK_B_US_EQ", "BRK.B"),
    ("NOVO_B_DK_EQ", "NOVO.B"),
    ("VOLV_B_SE_EQ", "VOLV.B"),
]


@pytest.mark.unit
@pytest.mark.parametrize("symbol,expected_canon", _ROUNDTRIP_CASES)
def test_roundtrip_to_broker_then_canonical(symbol: str, expected_canon: str) -> None:
    """canonical_ticker_symbol(to_broker_ticker(symbol)) == expected canonical form."""
    broker_fmt = to_broker_ticker(symbol)
    canon = canonical_ticker_symbol(broker_fmt)
    assert canon == expected_canon, (
        f"Round-trip failed for {symbol!r}: "
        f"to_broker_ticker → {broker_fmt!r}, "
        f"canonical_ticker_symbol → {canon!r}, "
        f"expected {expected_canon!r}"
    )


# ---------------------------------------------------------------------------
# 5.  Consistency invariant:
#     canonical(to_broker(symbol)) == canonical(symbol)  for all inputs
# ---------------------------------------------------------------------------

_CONSISTENCY_CASES: list[str] = [
    "AAPL",
    "MSFT",
    "AAPL_US_EQ",
    "SAP_DE_EQ",
    "LLOY_UK_EQ",
    "SHOP_CA_EQ",
    "AIR_FR_EQ",
    "ASML_NL_EQ",
    "BRK_B_US_EQ",
    "NOVO_B_DK_EQ",
    "VOLV_B_SE_EQ",
    "AAPL.US",
    "SAP.DE",
]


@pytest.mark.unit
@pytest.mark.parametrize("symbol", _CONSISTENCY_CASES)
def test_roundtrip_consistency(symbol: str) -> None:
    """canonical(to_broker(symbol)) == canonical(symbol) for all tested inputs."""
    direct = canonical_ticker_symbol(symbol)
    via_broker = canonical_ticker_symbol(to_broker_ticker(symbol))
    assert via_broker == direct, (
        f"Consistency failure for {symbol!r}: "
        f"canonical directly = {direct!r}, "
        f"canonical via to_broker_ticker = {via_broker!r}"
    )


# ---------------------------------------------------------------------------
# 6.  Every suffix in _BROKER_SUFFIXES is handled by canonical_ticker_symbol.
#     Verifies the suffix table is not stale relative to the regex engine.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("suffix", _BROKER_SUFFIXES)
def test_all_known_suffixes_are_stripped(suffix: str) -> None:
    """Every suffix in _BROKER_SUFFIXES is correctly stripped by canonical."""
    synthetic = f"TEST{suffix}"
    result = canonical_ticker_symbol(synthetic)
    assert result == "TEST", (
        f"Suffix {suffix!r} not stripped: "
        f"canonical_ticker_symbol({synthetic!r}) = {result!r}"
    )

"""General utility and helper functions."""

from mcp_finance.brokers.trading212 import Trading212Client


def get_trading212_client() -> Trading212Client:
    """Return a fresh Trading212Client per call.

    For Phase 2 this is fine; Phase 9 will switch to a lifespan-managed
    singleton to reuse the underlying httpx connection pool.
    """
    return Trading212Client()

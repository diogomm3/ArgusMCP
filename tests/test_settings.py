import pytest
from pydantic import ValidationError

from mcp_finance.settings import Settings


@pytest.mark.unit
def test_settings_load_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings model parses valid env vars correctly."""
    monkeypatch.setenv("TRADING212_API_KEY", "test_key")
    monkeypatch.setenv("TRADING212_API_SECRET", "test_secret")
    monkeypatch.setenv("TRADING212_ENV", "demo")
    monkeypatch.setenv("FMP_API_KEY", "fmp_test")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "auth_token")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    # We must explicitly disable dotenv loading or it might pull from actual
    # .env during tests. Setting _env_file to None for this test instantiation
    # prevents it reading from .env
    s = Settings(_env_file=None)

    assert s.trading212_api_key == "test_key"
    assert s.trading212_api_secret == "test_secret"
    assert s.trading212_env == "demo"
    assert s.fmp_api_key == "fmp_test"
    assert s.mcp_auth_token == "auth_token"
    assert s.database_url == "sqlite:///:memory:"


@pytest.mark.unit
def test_settings_missing_var_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that a missing required variable raises a validation error."""
    # Ensure environment is clear of one required var
    monkeypatch.delenv("TRADING212_API_KEY", raising=False)
    monkeypatch.setenv("TRADING212_API_SECRET", "test_secret")
    monkeypatch.setenv("FMP_API_KEY", "fmp_test")
    monkeypatch.setenv("MCP_AUTH_TOKEN", "auth_token")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "trading212_api_key" in str(exc_info.value)

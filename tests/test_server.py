import pytest
from starlette.testclient import TestClient

from mcp_finance.server import app
from mcp_finance.settings import settings

# Use 127.0.0.1 as the base URL so the Host header matches the allowed host
# configured in streamable_http_app(host="0.0.0.0") transport security settings.
client = TestClient(app, base_url="http://127.0.0.1:8000")


@pytest.mark.unit
def test_missing_auth() -> None:
    """Test that requests without auth token are rejected on /mcp."""
    response = client.post("/mcp")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


@pytest.mark.unit
def test_wrong_auth() -> None:
    """Test that requests with wrong auth token are rejected on /mcp."""
    headers = {"Authorization": "Bearer wrong_token_here"}
    response = client.post("/mcp", headers=headers)
    assert response.status_code == 401


@pytest.mark.unit
def test_correct_auth_passes() -> None:
    """Test that a valid auth token passes the auth middleware on /mcp.

    The MCP layer returns 400 (Missing session ID) rather than 401,
    proving that auth passed and the request reached the MCP handler.
    """
    headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        response = c.post("/mcp", headers=headers)
    assert response.status_code != 401

from fastapi.testclient import TestClient

from mcp_finance.server import app
from mcp_finance.settings import settings

client = TestClient(app)


def test_missing_auth() -> None:
    """Test that requests without auth token are rejected."""
    response = client.get("/sse")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}

    response = client.post("/messages")
    assert response.status_code == 401


def test_wrong_auth() -> None:
    """Test that requests with wrong auth token are rejected."""
    headers = {"Authorization": "Bearer wrong_token_here"}
    response = client.get("/sse", headers=headers)
    assert response.status_code == 401

    response = client.post("/messages", headers=headers)
    assert response.status_code == 401


def test_correct_auth_passes() -> None:
    """Test that a correct auth token is accepted.

    We don't do a full SSE handshake here to keep it simple, but we verify
    the endpoint accepts the token and attempts to proceed.
    """
    headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}

    # POST to /messages with correct auth but no valid session
    # (since we didn't connect via GET /sse). It should not return 401.
    response = client.post("/messages", headers=headers)
    assert response.status_code != 401

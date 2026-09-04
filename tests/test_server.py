import pytest
from starlette.testclient import TestClient

from mcp_finance.server import app
from mcp_finance.settings import settings

client = TestClient(app)


@pytest.mark.unit
def test_missing_auth() -> None:
    """Test that requests without auth token are rejected."""
    response = client.get("/sse")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}

    response = client.post("/messages/")
    assert response.status_code == 401


@pytest.mark.unit
def test_wrong_auth() -> None:
    """Test that requests with wrong auth token are rejected."""
    headers = {"Authorization": "Bearer wrong_token_here"}
    response = client.get("/sse", headers=headers)
    assert response.status_code == 401

    response = client.post("/messages/", headers=headers)
    assert response.status_code == 401


@pytest.mark.unit
def test_correct_auth_passes() -> None:
    """Test that a correct auth token is accepted.

    We verify auth passes by checking we don't get a 401 on /messages/.
    The MCP layer will return 404 (no session), not 401.
    """
    headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"}
    # POST to /messages/ with correct auth — session won't exist, but we should
    # get a non-401 (e.g. 404) back, proving auth passed.
    response = client.post("/messages/", headers=headers)
    assert response.status_code != 401

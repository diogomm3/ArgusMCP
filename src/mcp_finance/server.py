from mcp.server.mcpserver import MCPServer
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from mcp_finance.logger import configure_logging, get_logger
from mcp_finance.settings import settings

configure_logging()
logger = get_logger(__name__)

mcp = MCPServer("mcp_finance")


@mcp.tool()  # type: ignore[misc]
def ping() -> str:
    """Ping the server to check if it is alive."""
    return "pong"


class AuthMiddleware:
    """Pure ASGI middleware enforcing Bearer token authentication."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith("/sse") or path.startswith("/messages"):
                headers = dict(scope.get("headers", []))
                auth = headers.get(b"authorization", b"").decode("latin-1")
                expected = f"Bearer {settings.mcp_auth_token}"
                if auth != expected:
                    logger.warning("Unauthorized access attempt", path=path)
                    response = JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized"},
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app = AuthMiddleware(mcp.sse_app())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

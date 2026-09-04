from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from mcp.server.mcpserver import MCPServer

from mcp_finance.logger import configure_logging, get_logger
from mcp_finance.settings import settings

configure_logging()
logger = get_logger(__name__)

mcp = MCPServer("mcp_finance")


@mcp.tool()  # type: ignore[misc]
def ping() -> str:
    """Ping the server to check if it is alive."""
    return "pong"


app = FastAPI()


@app.middleware("http")  # type: ignore[misc]
async def auth_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # Only enforce auth on the SSE and messages endpoints
    if request.url.path.startswith("/sse") or request.url.path.startswith("/messages"):
        auth_header = request.headers.get("Authorization")
        expected = f"Bearer {settings.mcp_auth_token}"
        if not auth_header or auth_header != expected:
            logger.warning("Unauthorized access attempt")
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


# Mount the MCP SSE ASGI application at the root
app.mount("/", mcp.sse_app())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

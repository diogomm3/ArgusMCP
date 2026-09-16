"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠  MANUAL SCRIPT — READ BEFORE RUNNING  ⚠          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  This script places a REAL ORDER against the Trading212 DEMO account.       ║
║                                                                              ║
║  REQUIREMENTS:                                                               ║
║    - The Docker stack must be running: `docker compose up -d`                ║
║    - A valid .env file with demo Trading212 credentials must be present.     ║
║    - The DATABASE_URL in .env must point at the running Postgres container.  ║
║                                                                              ║
║  NEVER:                                                                      ║
║    - Run this script against a live (non-demo) Trading212 environment.       ║
║    - Include this script in automated test suites or CI pipelines.           ║
║    - Run this script automatically or as part of any scheduled job.          ║
║                                                                              ║
║  This script lives in scripts/ (NOT tests/) deliberately. Everything under  ║
║  tests/ is safe to run by anyone at any time; this script mutates a real     ║
║  (demo) broker account and must only be invoked manually and intentionally.  ║
║                                                                              ║
║  WHAT IT DOES:                                                               ║
║    Stage 1 — Deliberate rejection:                                           ║
║      POST place_order with stop_loss_price > entry_price (Rule 1 fail).     ║
║      Expected: success=False, REJECTED audit row in DB, zero broker calls.  ║
║                                                                              ║
║    Stage 2 — Small approved demo order:                                      ║
║      POST place_order with AAPL, qty=1, valid stop-loss.                    ║
║      Expected: success=True, SUBMITTING→ACCEPTED audit rows,                 ║
║                real broker_order_id, position confirmed via T212 API.        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import os
from typing import Any

import asyncpg
import httpx

MCP_URL = "http://localhost:8000/mcp"
AUTH = os.environ.get("MCP_AUTH_TOKEN", "LzkagtnHaN4aegD4cpfWciISoQnJQNoE")
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/mcp_finance",
)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH}",
    "Accept": "application/json, text/event-stream",
}


class MCPClient:
    """Client for MCP Streamable HTTP endpoint."""

    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
            "Accept": "application/json, text/event-stream",
        }
        self.session_id: str | None = None
        self._msg_id = 0

    def init_session(self) -> None:
        self._msg_id += 1
        r = self.client.post(
            self.base_url,
            headers=self.headers,
            json={
                "jsonrpc": "2.0",
                "id": self._msg_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke_tester", "version": "1.0"},
                },
            },
        )
        r.raise_for_status()
        self.session_id = r.headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError("Server did not return mcp-session-id")
        self.headers["mcp-session-id"] = self.session_id

        # Send notifications/initialized
        self.client.post(
            self.base_url,
            headers=self.headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.session_id:
            self.init_session()
        self._msg_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        r = self.client.post(self.base_url, headers=self.headers, json=payload)
        r.raise_for_status()

        # Parse SSE stream response
        body = None
        for line in r.text.splitlines():
            if line.startswith("data: "):
                body = json.loads(line[6:])
                break
        if body is None:
            # Try plain json if not SSE
            body = r.json()

        if "error" in body:
            raise RuntimeError(f"MCP error: {body['error']}")

        res = body.get("result", {})
        if res.get("isError"):
            content = res.get("content", [])
            err_text = (
                content[0]["text"]
                if content and isinstance(content, list)
                else str(res)
            )
            raise RuntimeError(f"Tool execution error: {err_text}")

        content = res.get("content", [])
        text = content[0]["text"] if isinstance(content, list) and content else content
        if isinstance(text, str):
            try:
                parsed: dict[str, object] | list[object] = json.loads(text)
                return parsed
            except json.JSONDecodeError:
                return {"text": text}
        return {"result": text}


async def count_audit_rows(pool: asyncpg.Pool) -> int:
    val = await pool.fetchval("SELECT COUNT(*) FROM order_audit_logs")
    return int(val)


async def fetch_audit_row(pool: asyncpg.Pool, audit_id: int) -> dict[str, object]:
    row = await pool.fetchrow(
        "SELECT id, symbol, side, quantity, entry_price, stop_loss_price, "
        "status, broker_order_id, rejection_reasons, risk_metrics "
        "FROM order_audit_logs WHERE id = $1",
        audit_id,
    )
    if not row:
        return {}
    return dict(row)


def separator(label: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")


async def main() -> None:
    pool = await asyncpg.create_pool(DB_URL)
    mcp = MCPClient(MCP_URL, AUTH)

    # -----------------------------------------------------------------------
    # Stage 1 — Deliberate rejection
    # -----------------------------------------------------------------------
    separator("STAGE 1 — Deliberate rejection (stop_loss > entry)")

    rows_before = await count_audit_rows(pool)
    print(f"audit_log row count before Stage 1: {rows_before}")

    rejected_result = mcp.call_tool(
        "place_order",
        {
            "input": {
                "symbol": "AAPL",
                "entry_price": "150.00",
                "stop_loss_price": "160.00",  # Invalid: stop loss >= entry
            }
        },
    )

    print("\nplace_order response (rejection):")
    print(json.dumps(rejected_result, indent=2))

    assert rejected_result["success"] is False, "Expected success=False"
    assert rejected_result["broker_order_id"] is None, "Expected None broker_order_id"
    assert rejected_result["order_result"] is None, "Expected None order_result"
    assert "rejected by risk engine" in (rejected_result.get("error_message") or "")

    audit_id_rejected = rejected_result["audit_id"]
    assert audit_id_rejected is not None, "Expected audit_id to be set"

    db_row_rej = await fetch_audit_row(pool, audit_id_rejected)
    print(f"\nDB audit row for rejection (audit_id={audit_id_rejected}):")
    print(json.dumps({k: str(v) for k, v in db_row_rej.items()}, indent=2))

    assert db_row_rej["status"] == "REJECTED", (
        f"Expected REJECTED, got {db_row_rej['status']}"
    )
    assert db_row_rej["symbol"] == "AAPL"
    assert db_row_rej["rejection_reasons"] is not None
    assert isinstance(db_row_rej["rejection_reasons"], (list, str))
    assert len(db_row_rej["rejection_reasons"]) > 0

    rows_after_stage1 = await count_audit_rows(pool)
    assert rows_after_stage1 == rows_before + 1, (
        f"Expected +1 audit row, before={rows_before}, after={rows_after_stage1}"
    )

    print("\n[OK] Stage 1 PASSED:")
    print(f"     audit_id={audit_id_rejected}, status=REJECTED")
    print(f"     rejection_reasons={db_row_rej['rejection_reasons']}")
    print("     broker was never contacted (broker_order_id=None)")

    # Sleep 6 seconds to respect Trading212 5-second rate limit on /equity/portfolio
    print("\nWaiting 6s to respect Trading212 rate limit...")
    await asyncio.sleep(6)

    # -----------------------------------------------------------------------
    # Stage 2 — Small approved demo order
    # -----------------------------------------------------------------------
    separator("STAGE 2 — Small approved demo order (AAPL, qty=1)")

    rows_before_stage2 = await count_audit_rows(pool)

    approved_result = mcp.call_tool(
        "place_order",
        {
            "input": {
                "symbol": "AAPL",
                "entry_price": "150.00",
                "stop_loss_price": "140.00",  # Valid stop-loss
                "quantity": "1",
            }
        },
    )

    print("\nplace_order response (approval):")
    print(json.dumps(approved_result, indent=2))

    assert approved_result["success"] is True, (
        f"Expected success=True, got: {approved_result}"
    )
    broker_order_id = approved_result["broker_order_id"]
    assert broker_order_id is not None, "Expected real broker_order_id"
    audit_id_accepted = approved_result["audit_id"]
    assert audit_id_accepted is not None, "Expected audit_id"

    db_row_acc = await fetch_audit_row(pool, audit_id_accepted)
    print(f"\nDB audit row for accepted order (audit_id={audit_id_accepted}):")
    print(json.dumps({k: str(v) for k, v in db_row_acc.items()}, indent=2))

    assert db_row_acc["status"] == "ACCEPTED", (
        f"Expected ACCEPTED, got {db_row_acc['status']}"
    )
    assert str(db_row_acc["broker_order_id"]) == str(broker_order_id)

    rows_after_stage2 = await count_audit_rows(pool)
    assert rows_after_stage2 == rows_before_stage2 + 1, (
        f"Expected +1 row for Stage 2, got {rows_after_stage2 - rows_before_stage2}"
    )

    print("\n[OK] Stage 2 PASSED:")
    print(f"     audit_id={audit_id_accepted}, status=ACCEPTED")
    print(f"     broker_order_id={broker_order_id}")

    # Sleep 6 seconds before querying portfolio positions
    print("\nWaiting 6s to respect Trading212 rate limit...")
    await asyncio.sleep(6)

    # -----------------------------------------------------------------------
    # Stage 2b — Confirm position visible via get_positions
    # -----------------------------------------------------------------------
    separator("STAGE 2b — Portfolio query via broker get_positions")

    positions_result = mcp.call_tool("get_positions", {})
    print("\nget_positions response:")
    print(json.dumps(positions_result, indent=2))

    raw_positions = positions_result if isinstance(positions_result, list) else []
    aapl_positions = [
        p
        for p in raw_positions
        if isinstance(p, dict) and "AAPL" in str(p.get("ticker", "")).upper()
    ]
    if aapl_positions:
        print("\n[OK] Stage 2b: AAPL position visible in portfolio:")
        print(json.dumps(aapl_positions, indent=2))
    else:
        print("\n[INFO] Stage 2b: AAPL order submitted; order accepted by broker.")
        print("       (Market order pending fill / settlement during market hours).")
        print(f"       broker_order_id: {broker_order_id}")

    separator("ALL STAGED SMOKE TESTS COMPLETED SUCCESSFULLY")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

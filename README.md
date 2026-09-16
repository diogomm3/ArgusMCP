# ArgusMCP

ArgusMCP is an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that exposes financial research and trading tools — market data, technical indicators, company fundamentals, a deterministic stock screener, chart rendering, and risk-gated order execution via Trading212 — for use by any MCP-compatible LLM or agent framework (Claude, LangGraph, Google ADK, etc.). The server handles data access, caching, risk evaluation, and broker communication; it has no reasoning or decision-making of its own. An external agent decides which tools to call and in what order.

---

## Architecture

```mermaid
flowchart LR
    A["MCP Client\n(Claude / LangGraph / ADK)"] -->|"Bearer token\nHTTP POST /mcp"| B["ArgusMCP Server\n(this repo, port 8000)"]

    B --> C["Trading212 API\n(account · positions · orders)"]
    B --> D["yfinance\n(OHLCV history · price quotes)"]
    B --> E["FMP API\n(company fundamentals)"]
    B --> F[("Postgres\nlocal cache")]

    F --> B
```

OHLCV bars and fundamentals snapshots are cached locally in Postgres. The screener and all technical-indicator computations read exclusively from that cache — no live API calls are made during a screen run. `get_history` fetches from yfinance on a cache miss and writes the result; subsequent calls for the same date range are served from Postgres. Fundamentals are cached with a 168-hour TTL (7 days); a stale or absent entry triggers a live FMP call unless `allow_live_fundamentals=false`.

Trading212 is used exclusively for account/positions queries and order execution — never for price history.

**Data provider constraints:** OHLCV data comes from the yfinance unofficial API (no SLA, no auth, subject to undocumented rate limits). Fundamentals come from FMP's free tier, which allows **250 requests/day**. The live-trading path in `Trading212Client.place_order` is hard-gated: calling it with `trading212_env=live` raises `RuntimeError` immediately. The current default is `demo`.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | ≥ 3.12 | Runtime |
| `mcp` | latest | MCP server protocol and tool registration |
| `pydantic` / `pydantic-settings` | latest | Schema validation, settings from env |
| `sqlalchemy[asyncio]` | ≥ 2.0 | Async ORM, DB models |
| `asyncpg` | latest | Async Postgres driver |
| `alembic` | latest | Schema migrations |
| `httpx` | latest | Async HTTP client (Trading212, FMP) |
| `yfinance` | 1.7.0 | OHLCV history and price quotes |
| `pandas` | ≥ 2.0 | Data handling for OHLCV and screener |
| `matplotlib` | ≥ 3.9 | Chart rendering backend |
| `mplfinance` | ≥ 0.12.10b0 | Candlestick chart library |
| `tenacity` | ≥ 8.2, < 10 | Retry logic for external HTTP calls |
| `structlog` | latest | Structured JSON logging |
| `uvicorn` | latest | ASGI server |
| `ruff` | 0.16.6 | Linter and formatter |
| `mypy` | latest | Static type checker |
| `pytest` + `pytest-asyncio` | latest | Test runner |
| `testcontainers[postgres]` | latest | Disposable Postgres for DB integration tests |
| `respx` | latest | HTTP mocking for unit tests |
| `pre-commit` | latest | Git hook runner |

---

## Available MCP Tools

### Utility

#### `ping`
Checks whether the server is alive.

| | |
|---|---|
| **Parameters** | none |
| **Returns** | `"pong"` (string) |

---

### Broker (Trading212)

#### `get_positions`
Returns all open equity positions from the connected Trading212 account.

| | |
|---|---|
| **Parameters** | none |
| **Returns** | `list[Position]` — each entry has `ticker`, `quantity`, `average_price`, `current_price`, and unrealised P&L fields |
| **Notes** | Live call to Trading212. Subject to API rate limits. Uses the environment set by `TRADING212_ENV`. |

#### `get_account`
Returns the current account cash/equity/P&L snapshot.

| | |
|---|---|
| **Parameters** | none |
| **Returns** | `AccountSummary` — `cash`, `invested`, `result` (unrealised P&L), `total` |
| **Notes** | Live call to Trading212. |

---

### Market Data

#### `get_symbol_price`
Returns the latest informational price quote for a symbol.

| Parameter | Type | Required | Notes |
|---|---|---|---|
| `symbol` | `str` | ✓ | Standard ticker (e.g. `AAPL`, `SAP.DE`) |

| | |
|---|---|
| **Returns** | `PriceQuote` — `symbol`, `price`, `currency`, `timestamp` |
| **Notes** | **Informational only — not an execution price.** Sourced from yfinance; may be delayed. Do not use for trade sizing or valuation. |

#### `get_history`
Returns historical daily OHLCV bars for a symbol between two dates.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbol` | `str` | ✓ | — | Ticker in yfinance format (e.g. `AAPL`, `ASML.AS`) |
| `start_date` | `str` | ✓ | — | `YYYY-MM-DD` |
| `end_date` | `str` | ✓ | — | `YYYY-MM-DD` |
| `exchange` | `str` | ✗ | `None` | Optional exchange hint |

| | |
|---|---|
| **Returns** | `list[OhlcvRecord]` — each entry has `date`, `open`, `high`, `low`, `close`, `volume`, `source` |
| **Notes** | Reads from Postgres cache first. On cache miss, fetches from yfinance and persists. Use `exchange` for international symbols not uniquely identified by ticker alone. |

---

### Fundamentals

#### `get_stock_fundamentals`
Returns fundamental valuation, financial health, and company profile data.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbol` | `str` | ✓ | — | Ticker (e.g. `AAPL`, `ASML.AS`) |
| `exchange` | `str` | ✗ | `None` | Optional exchange name |

| | |
|---|---|
| **Returns** | `CompanyFundamentals` — `symbol`, `company_name`, market cap, P/E, P/B, EV/EBITDA, profit margins, debt-to-equity, current ratio, dividend yield, FCF per share, sector, and more |
| **Notes** | Reads from local cache if fresh (≤ 168h TTL). On cache miss or stale entry, queries FMP and caches the result — **consumes FMP daily quota**. Raises `QuotaExhaustedError` if today's 250-request limit is exhausted. |

---

### Technical Analysis

#### `screen_stocks`
Screens a list of stocks using a layered momentum/trend strategy.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbols` | `list[str]` | ✗ | `DEFAULT_WATCHLIST` | Tickers to screen |
| `as_of_date` | `str` | ✗ | today | ISO date (`YYYY-MM-DD`) |
| `allow_live_fundamentals` | `bool` | ✗ | `true` | When `false`, only cached fundamentals are used (no FMP quota consumed) |
| `include_failed` | `bool` | ✗ | `false` | Include filter-failed symbols in the report |

Filters are applied sequentially; a symbol must pass all four layers to appear in `passed_candidates`:

1. **Universe** — market cap ≥ $2B, price ≥ $5, optional max P/E
2. **Liquidity** — 20-day average volume ≥ 100,000 shares
3. **Trend** — price > EMA-20 > EMA-50 (golden-cross alignment)
4. **Momentum** — RSI-14 in [40, 70], MACD histogram > 0, MACD > signal line

| | |
|---|---|
| **Returns** | `ScreeningReport` — `as_of_date`, `strategy_name`, `total_screened`, `passed_candidates`, optionally `failed_candidates` |
| **Notes** | All OHLCV data is read from the local Postgres cache — **never triggers yfinance calls**. An empty `passed_candidates` list is a valid, expected result in sideways or bearish market regimes. |

#### `get_stock_chart`
Renders a technical candlestick chart with EMA overlays for a symbol.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbol` | `str` | ✓ | — | Ticker in yfinance format |
| `lookback_days` | `int` | ✗ | `90` | Days to display; min 10, max 365 |

| | |
|---|---|
| **Returns** | `ImageContent` — base64-encoded PNG with candlestick bars, volume subplot, and EMA-20/EMA-50/EMA-200 overlays |
| **Notes** | Data is read **exclusively from the local Postgres OHLCV cache**. If data is absent, an error is returned instructing the caller to ingest via `get_history` first. |

---

### Risk & Execution

#### `evaluate_trade`
Performs a dry-run pre-trade risk evaluation for a proposed order.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbol` | `str` | ✓ | — | Canonical or broker-formatted ticker |
| `entry_price` | `Decimal` | ✓ | — | Intended entry price per share |
| `stop_loss_price` | `Decimal` | ✓ | — | Stop-loss trigger price |
| `quantity` | `Decimal` | ✗ | `None` | Explicit share count; auto-sized by fixed-fractional risk budgeting if omitted |
| `side` | `str` | ✗ | `"BUY"` | Order direction (`"BUY"` only in current version) |
| `order_type` | `str` | ✗ | `"MARKET"` | `"MARKET"` or `"LIMIT"` |
| `limit_price` | `Decimal` | ✗ | `None` | Required when `order_type="LIMIT"` |
| `sector` | `str` | ✗ | `None` | GICS/ICB sector label; auto-populated from local fundamentals cache if available |
| `next_earnings_date` | `str` | ✗ | `None` | `YYYY-MM-DD`; earnings blackout check is skipped if omitted |

| | |
|---|---|
| **Returns** | `RiskDecision` — `approved`, `symbol`, `side`, `order_type`, `entry_price`, `stop_loss_price`, `quantity`, `estimated_cost`, `risk_amount`, `rule_results` (one per rule), `rejection_reasons` |
| **⚠ Dry-run guarantee** | Zero database writes. Zero broker calls. Safe to call freely for analysis and planning. |

**Risk rules evaluated (in order):**

| Rule | Name | Condition | Default threshold |
|---|---|---|---|
| 0 | Order-side gate | BUY supported; SELL rejected | — |
| 1 | Stop-loss validity | stop > 0, stop < entry, distance in [min, max] | min **0.5%**, max **15%** of entry |
| 2 | Duplicate position | No existing non-zero position in same instrument (canonical + root matching, fail-closed) | — |
| 3 | Max position size | `position_cost ≤ total_equity × 5%` | **5%** of total equity |
| 4 | Max portfolio exposure | `(invested + position_cost) ≤ total_equity × 90%` | **90%** of total equity |
| 5 | Max sector exposure | `(sector_cost + position_cost) ≤ total_equity × 25%` (skipped if sector unknown) | **25%** of total equity |
| 6 | Daily-loss circuit breaker | `daily_realized_loss < total_equity × 3%` (exclusive `<`) | **3%** of total equity |
| 7 | Earnings blackout | Block new BUYs within N calendar days of earnings | **3 days** |

All thresholds are defaults in `RiskConfig` (`src/mcp_finance/risk/models.py`) and can be overridden per-call.

#### `place_order`
Submits a BUY order to the broker, gated by the full risk engine.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `symbol` | `str` | ✓ | — | Canonical or broker-formatted ticker |
| `entry_price` | `Decimal` | ✓ | — | Intended entry price |
| `stop_loss_price` | `Decimal` | ✓ | — | Stop-loss trigger |
| `quantity` | `Decimal` | ✗ | `None` | Auto-sized if omitted |
| `order_type` | `str` | ✗ | `"MARKET"` | `"MARKET"` or `"LIMIT"` |
| `limit_price` | `Decimal` | ✗ | `None` | Required for LIMIT orders |
| `sector` | `str` | ✗ | `None` | Auto-populated from cache if available |
| `next_earnings_date` | `str` | ✗ | `None` | `YYYY-MM-DD`; blackout skipped if omitted |

Execution lifecycle:
1. All risk rules evaluated; position sized via fixed-fractional risk budgeting.
2. **Rejected:** writes `REJECTED` audit row, returns `success=false`. Broker never contacted.
3. **Approved:** re-asserts defense-in-depth invariant (`approved=true` and `quantity > 0`), writes `SUBMITTING` audit row, dispatches order to Trading212.
   - Broker failure → `FAILED` audit row, error surfaced in response.
   - Broker success → `ACCEPTED` audit row with `broker_order_id`.

| | |
|---|---|
| **Returns** | `PlaceOrderOutput` — `success`, `decision` (`RiskDecision`), `audit_id`, `broker_order_id`, `order_result`, `error_message` |
| **⚠ Demo only** | Hard-gated: `TRADING212_ENV=live` raises `RuntimeError` immediately before any HTTP call. Only `demo` is permitted. |
| **Notes** | Every attempt (approved or rejected) is durably written to `order_audit_logs` in an independent DB session. A caller-side rollback cannot erase the audit record. |

---

## Data & Persistence

Schema is managed by hand-written Alembic migrations (not autogenerated) in `alembic/versions/` for auditability.

| Table | Purpose |
|---|---|
| `symbols` | Master instrument list; unique on `(ticker, exchange)`. ISIN stored when available. |
| `ohlcv_daily` | Daily OHLCV bars; unique on `(symbol_id, date, source)`. Supports multiple data providers per symbol without conflict. |
| `fundamentals_cache` | Fundamentals snapshots keyed on `(symbol_id, as_of_date)`. History preserved for point-in-time queries. Staleness checked against TTL. |
| `fmp_quota_usage` | Daily FMP request counter. Persisted in Postgres so multiple workers share an accurate count and container restarts don't reset it. |
| `order_audit_logs` | Immutable record of every `place_order` attempt. Written before broker dispatch (`SUBMITTING`), updated on return (`ACCEPTED` / `FAILED`). Rejections write a single `REJECTED` row. |

Apply migrations:

```bash
# Via Docker (recommended when stack is up, no host venv needed):
docker compose exec mcp-finance alembic upgrade head

# Or directly from host virtualenv:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/mcp_finance \
  alembic upgrade head
```

---

## Environment Variables

Copy `.env.example` to `.env`. **Never commit `.env`.**

| Variable | Purpose | Default |
|---|---|---|
| `TRADING212_API_KEY` | **Secret.** Trading212 API key | — |
| `TRADING212_API_SECRET` | **Secret.** Trading212 API secret | — |
| `TRADING212_ENV` | Broker environment (`"demo"` or `"live"`) | `demo` |
| `FMP_API_KEY` | **Secret.** Financial Modeling Prep API key | — |
| `MCP_AUTH_TOKEN` | **Secret.** Bearer token required by all MCP clients | — |
| `DATABASE_URL` | Async Postgres connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/mcp_finance` |
| `FUNDAMENTALS_CACHE_TTL_HOURS` | Hours before a fundamentals snapshot is considered stale | `168` |
| `PRIMARY_OHLCV_SOURCE` | Source tag written on new OHLCV rows | `yfinance` |
| `FMP_DAILY_QUOTA` | FMP free-tier daily request ceiling | `250` |
| `FMP_BASE_URL` | FMP API base URL | `https://financialmodelingprep.com/stable` |

Where to get credentials:
- Trading212 demo API key: Settings → API (demo) in the Trading212 app.
- FMP free API key: [financialmodelingprep.com/register](https://financialmodelingprep.com/register).

---

## Setup & Running Locally

### 1. Clone

```bash
git clone git@github.com:diogomm3/ArgusMCP.git
cd ArgusMCP
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in TRADING212_API_KEY, TRADING212_API_SECRET, FMP_API_KEY,
# MCP_AUTH_TOKEN, and DATABASE_URL.
```

### 3. Start the stack

```bash
docker compose up --build -d
```

Brings up `mcp-finance` (port 8000) and `postgres` (port 5432). `mcp-finance` waits for Postgres to pass its healthcheck before starting.

Confirm both are healthy:

```bash
docker compose ps
# Both services should show (healthy).
```

The `mcp-finance` healthcheck POSTs to `/mcp` and expects HTTP `401` or `400` — meaning the server is up and enforcing auth.

### 4. Apply database migrations

Run migrations inside the running container (no local Python setup required):

```bash
docker compose exec mcp-finance alembic upgrade head
```

### 5. Connect an MCP client

The server listens on `http://localhost:8000/mcp` using the Streamable HTTP transport conforming to the Model Context Protocol specification. All requests require authentication:

```
Authorization: Bearer <MCP_AUTH_TOKEN>
```

To verify server liveness and authentication enforcement with curl:

```bash
# Returns HTTP 401 Unauthorized (verifies server is up and enforcing bearer auth):
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/mcp
```

Because MCP Streamable HTTP requires a protocol handshake (`initialize` negotiation, session ID tracking via `mcp-session-id`, and `notifications/initialized`), interactive tool testing is best done via the **MCP Inspector** below or an MCP-compliant client.

### Manual testing with MCP Inspector

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) is a browser-based tool for calling MCP tools interactively without writing a full client. Run it with:

```bash
npx @modelcontextprotocol/inspector
```

Then in the Inspector UI:
1. Set **Transport** to `Streamable HTTP`.
2. Set **URL** to `http://localhost:8000/mcp`.
3. Add a request header: `Authorization: Bearer <MCP_AUTH_TOKEN>`.
4. Click **Connect**.

Sample argument payloads for every registered tool are in `tests/mcp_tool_inputs.json`. Each entry is keyed by tool name; entries with a `_tool` field are alternative scenarios for the same tool (the tool name to select is the `_tool` value). Copy the `input` object from the relevant entry into the Inspector's **Arguments** field.

> [!CAUTION]
> The `place_order` entries in `mcp_tool_inputs.json` submit a real order to the Trading212 **demo** account. Only run these if the Docker stack is up, `.env` has valid demo credentials, and you intend to place an order.

---

## Docker

| Service | Image | Port | Healthcheck |
|---|---|---|---|
| `mcp-finance` | built from `./Dockerfile` | `8000:8000` | POST `/mcp` → expects 401 or 400 |
| `postgres` | `postgres:16` | `5432:5432` | `pg_isready -U postgres` |

Named volume `pgdata` persists Postgres data across restarts.

The `Dockerfile` is a two-stage build: dependencies installed in a builder stage, then only packages + source copied into a minimal final image. The container runs as a non-root user.

```bash
# Start (detached)
docker compose up -d

# Rebuild after code changes
docker compose up --build -d

# Follow logs
docker compose logs -f mcp-finance

# Stop
docker compose down

# Stop and delete Postgres volume (WARNING: deletes all cached data)
docker compose down -v
```

> [!WARNING]
> `docker compose down -v` deletes the `pgdata` volume and all cached OHLCV and fundamentals data. Re-populating requires re-running `get_history` and `get_stock_fundamentals` calls.

---

## Development Workflow

### Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Pre-commit hooks

```bash
pre-commit install

# Run all hooks on all files:
pre-commit run --all-files
```

Hooks: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-added-large-files`, `ruff --fix`, `ruff-format`, `mypy`.

### Linting and formatting

```bash
ruff check .            # lint check
ruff check --fix .      # lint with auto-fix
ruff format --check .   # format check (dry-run)
ruff format .           # format in-place
```

### Type checking

`mypy` requires settings to be importable; set dummy env vars:

```bash
TRADING212_API_KEY=ci-dummy \
TRADING212_API_SECRET=ci-dummy \
FMP_API_KEY=ci-dummy \
MCP_AUTH_TOKEN=ci-dummy \
DATABASE_URL=postgresql+asyncpg://ci:ci@localhost/ci \
  mypy src/
```

### Unit tests (no Docker required)

Excludes database integration tests:

```bash
TRADING212_API_KEY=ci-dummy \
TRADING212_API_SECRET=ci-dummy \
FMP_API_KEY=ci-dummy \
MCP_AUTH_TOKEN=ci-dummy \
DATABASE_URL=postgresql+asyncpg://ci:ci@localhost/ci \
  pytest --ignore=tests/db/ -m "not integration" -v
```

### Postgres integration tests

Runs all repository tests under `tests/db/` using `testcontainers`. Docker must be running locally, but the project stack does not need to be up (testcontainers manages its own isolated `postgres:16` container):

```bash
TRADING212_API_KEY=ci-dummy \
TRADING212_API_SECRET=ci-dummy \
FMP_API_KEY=ci-dummy \
MCP_AUTH_TOKEN=ci-dummy \
  pytest tests/db/ -v
```

### Full suite

Runs all unit tests and testcontainers database tests across the entire test tree:

```bash
TRADING212_API_KEY=ci-dummy \
TRADING212_API_SECRET=ci-dummy \
FMP_API_KEY=ci-dummy \
MCP_AUTH_TOKEN=ci-dummy \
DATABASE_URL=postgresql+asyncpg://ci:ci@localhost/ci \
  pytest -m "not integration" -v
```

### Live integration tests

Tests marked `@pytest.mark.integration` make real calls to the Trading212 demo API. Requires valid demo credentials in `.env`.

```bash
pytest -m integration -v
```

> [!CAUTION]
> Live integration tests contact the real Trading212 demo API. Never run them in CI or against a live account.

---

## Commit Conventions

Conventional Commits with scope:

```
<type>(<scope>): <short description>
```

Types: `feat`, `fix`, `test`, `refactor`, `style`, `chore`, `build`, `docs`, `ci`.
Scopes: `risk`, `brokers`, `market_data`, `fundamentals`, `screener`, `charting`, `db`.

One logical change per commit.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push and pull requests to `main`:

| Step | Command | Description |
|---|---|---|
| Install dependencies | `pip install ".[dev]"` | Installs project and test dependencies |
| Ruff check | `ruff check .` | Code style and lint checks |
| mypy | `mypy src/` | Type checking with dummy credentials |
| pytest | `pytest -m "not integration" -v` | Unit tests + testcontainers DB integration tests |

CI runs on GitHub Actions `ubuntu-latest` with Docker available. The test run executes both unit tests and Postgres repository integration tests via `testcontainers` (spinning up an isolated `postgres:16` container and running Alembic migrations). Only live Trading212 broker API tests (`@pytest.mark.integration`), which require real network access and live demo credentials, are excluded. CI must be green before merging.

---

## Safety Notes

- **Demo only.** `place_order` is hard-gated in `src/mcp_finance/brokers/trading212.py`: if `TRADING212_ENV` is anything other than `"demo"`, the call raises `RuntimeError` before any HTTP request is made. Live trading is not enabled.

- **Audit durability.** Every `place_order` attempt — approved or rejected — is written to `order_audit_logs` in an independent database session before any broker call. A caller-side rollback or network failure cannot erase the audit record.

- **Data provider limits.** FMP's free tier allows **250 requests/day**, tracked in `fmp_quota_usage`. yfinance is an unofficial API with no SLA or documented rate limits. Both constrain how much can be screened or fetched per day.

- **Manual smoke-test script.** `scripts/manual_place_order_smoke_test.py` runs a two-stage order flow against the Trading212 demo account: a deliberate rejection (confirming zero broker contact) and a real 1-share AAPL BUY (confirming the full `SUBMITTING→ACCEPTED` lifecycle). This script lives in `scripts/` — not `tests/` — and is excluded from pytest discovery and CI. It must only be run manually, with the Docker stack up and a valid `.env` present, and must never be pointed at a live environment.

---

## License

[MIT](LICENSE)

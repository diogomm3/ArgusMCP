# Finance MCP

A standalone finance Model Context Protocol (MCP) server. It integrates Trading212 execution, yfinance and FMP data, a deterministic screening/risk engine, and Qwen-facing MCP tools.

## Architecture
- **Stack**: Python 3.12, Docker, Postgres.
- **Providers**: Trading212 (execution), yfinance (OHLCV), FMP (fundamentals).
- **Core Loop**: Screens market data and applies risk engine before placing paper/live trades.

## Setup
1. Copy `.env.example` to `.env` and fill in credentials (e.g. FMP, Trading212).
2. Run `docker compose up --build` to start the service and Postgres db.

⚠️ **WARNING**: When running in live mode, this system will execute real trades. Treat credentials and execution settings with extreme caution.

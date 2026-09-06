from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    trading212_api_key: str
    trading212_api_secret: str
    trading212_env: Literal["demo", "live"] = "demo"
    fmp_api_key: str
    mcp_auth_token: str
    database_url: str
    # Fundamentals cache: rows older than this many hours trigger a re-fetch.
    # Default is 7 days (168h) — fundamentals move slowly and FMP quota is limited.
    fundamentals_cache_ttl_hours: int = 168
    # Primary OHLCV data source used by fetch_range when no source is specified.
    primary_ohlcv_source: str = "yfinance"
    # FMP daily request quota (250 calls/day on free plan) and stable base URL
    fmp_daily_quota: int = 250
    fmp_base_url: str = "https://financialmodelingprep.com/stable"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

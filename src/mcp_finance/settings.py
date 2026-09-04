from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    trading212_api_key: str
    trading212_api_secret: str
    trading212_env: Literal["demo", "live"] = "demo"
    fmp_api_key: str
    mcp_auth_token: str
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

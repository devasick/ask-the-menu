"""Runtime settings. Everything comes from the environment (or a local .env file)."""

from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = "postgresql+asyncpg://atm:atm@localhost:5432/ask_the_menu"

    # Server-to-server keys (MenuNow backend -> this service).
    ingest_api_key: SecretStr = SecretStr("change-me-ingest")
    admin_api_key: SecretStr = SecretStr("change-me-admin")

    # Comma-separated list in the env, e.g. "https://menunow.de,https://www.menunow.de".
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

"""Runtime settings. Everything comes from the environment (or a local .env file)."""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ModelPrice(BaseModel):
    """USD per million tokens."""

    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(default=Decimal(0), ge=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = "postgresql+asyncpg://atm:atm@localhost:5432/ask_the_menu"

    # Server-to-server keys (MenuNow backend -> this service).
    ingest_api_key: SecretStr = SecretStr("change-me-ingest")
    admin_api_key: SecretStr = SecretStr("change-me-admin")

    # Comma-separated list in the env, e.g. "https://menunow.de,https://www.menunow.de".
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- LLM ---
    # "direct" = real provider SDKs (app/llm/providers.py); "fake" = deterministic offline client
    # for demos and CI without API keys.
    llm_backend: Literal["direct", "fake"] = "direct"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    # Alias -> "<provider>:<model id>". The only place real model IDs live.
    model_menu_fast: str = "anthropic:REPLACE_WITH_FAST_MODEL_ID"
    model_menu_smart: str = "anthropic:REPLACE_WITH_SMART_MODEL_ID"
    model_menu_backup: str = "openai:REPLACE_WITH_BACKUP_MODEL_ID"
    model_menu_embed: str = "openai:REPLACE_WITH_EMBEDDING_MODEL_ID"
    # JSON in the env: {"<model id>": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}, ...}
    model_prices: dict[str, ModelPrice] = Field(default_factory=dict)
    llm_timeout_s: float = Field(default=20.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    # Escalate RAG generation from menu-fast to menu-smart for multi-part questions.
    escalate_min_question_marks: int = Field(default=2, ge=1)
    escalate_min_words: int = Field(default=30, ge=1)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

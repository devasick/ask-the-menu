import pytest
from pydantic import SecretStr

from app.config import Settings


def test_cors_origins_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example,")
    assert Settings(_env_file=None).cors_origins == ["https://a.example", "https://b.example"]


def test_secrets_are_not_shown_in_repr() -> None:
    s = Settings(_env_file=None, ingest_api_key=SecretStr("super-secret"))
    assert "super-secret" not in repr(s)

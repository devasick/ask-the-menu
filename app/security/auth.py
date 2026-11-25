"""API key checks. Constant-time comparison; keys never appear in logs or errors."""

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import SecretStr

from app.config import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _check(provided: str | None, expected: SecretStr) -> None:
    if provided is None or not hmac.compare_digest(
        provided.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")


def require_ingest_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    _check(x_api_key, settings.ingest_api_key)


def require_admin_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    _check(x_api_key, settings.admin_api_key)

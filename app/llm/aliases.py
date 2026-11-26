"""Model aliases. App code only ever names an alias; the real model lives in config."""

from dataclasses import dataclass
from typing import Literal, get_args

from app.config import Settings

Alias = Literal["menu-fast", "menu-smart", "menu-backup", "menu-embed"]
Provider = Literal["anthropic", "openai", "gemini"]

# On a provider error, retry the request once on this alias.
FALLBACKS: dict[Alias, Alias] = {"menu-fast": "menu-backup"}


@dataclass(frozen=True)
class ModelRoute:
    alias: Alias
    provider: Provider
    model: str


def resolve(settings: Settings, alias: Alias) -> ModelRoute:
    raw: str = getattr(settings, "model_" + alias.replace("-", "_"))
    provider, sep, model = raw.partition(":")
    if not sep or not model or provider not in get_args(Provider):
        raise ValueError(f"{alias}: expected '<anthropic|openai|gemini>:<model id>', got {raw!r}")
    return ModelRoute(alias=alias, provider=provider, model=model)  # type: ignore[arg-type]

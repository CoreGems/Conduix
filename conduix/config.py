import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_workspace() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "conduix" / "workspace"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONDUIX_",
        env_file=".env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8766

    default_model: str | None = None  # None → Codex's own default
    default_effort: str | None = None  # None → the model's default effort
    default_instructions: str | None = None

    codex_bin: str | None = None  # None → the SDK's bundled binary
    # Codex web_search mode when a request asks for the web_search tool:
    # "live" (fetch now) or "cached" (Codex's search cache).
    web_search_mode: str = "live"
    workspace_dir: Path = _default_workspace()

    session_idle_timeout_s: int = 30 * 60
    max_sessions: int = 100


@lru_cache
def settings() -> Settings:
    return Settings()

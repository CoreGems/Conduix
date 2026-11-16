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

    # Speech output, /v1/audio/speech (experimental; needs `conduix[voice]`).
    speech: bool = False
    speech_voice: str = "cove"  # a realtime v3 voice; OpenAI's names map to it
    # How closely a turn's transcript must match its text (0-1, normalized
    # difflib ratio) before the audio is accepted.
    speech_min_similarity: float = 0.8

    # Transcription, /v1/audio/transcriptions (experimental; same extra).
    transcription: bool = False
    # Longest upload accepted: it is played in real time, holding the only
    # voice session for as long as it lasts.
    transcription_max_seconds: int = 600


@lru_cache
def settings() -> Settings:
    return Settings()

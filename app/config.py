from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    app_name: str = "AI Document Assistant"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_provider: str = "local"
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "local"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 20
    upload_dir: Path = Path("data/uploads")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or None

    # Development should work without secrets, but real deployments can opt in
    # through env vars or simply by providing OPENAI_API_KEY.
    default_provider = "openai" if api_key else "local"

    return Settings(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", default_provider).lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_provider=os.getenv("LLM_PROVIDER", default_provider).lower(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        chunk_size=_int_from_env("CHUNK_SIZE", 900),
        chunk_overlap=_int_from_env("CHUNK_OVERLAP", 150),
        top_k=_int_from_env("TOP_K", 4),
        max_upload_mb=_int_from_env("MAX_UPLOAD_MB", 20),
        upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")),
    )

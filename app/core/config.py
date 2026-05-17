from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    llm_provider: str | None = os.getenv("LLM_PROVIDER") or None
    llm_model: str | None = os.getenv("LLM_MODEL") or None


settings = Settings()

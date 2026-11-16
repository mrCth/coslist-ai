"""Central configuration for CosList AI.

All values can be overridden with environment variables (see ``.env.example``).
The module exposes a single cached ``settings`` object so the rest of the
application never reads ``os.environ`` directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings:
    """Runtime settings resolved from the environment."""

    def __init__(self) -> None:
        self.app_name = "CosList AI"
        self.app_version = "1.0.0"
        self.prompt_version = "v1.2"

        # --- AI provider selection ---------------------------------------
        # "ollama" (default) | "gemini" | "text" (text-only fallback)
        self.ai_provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
        self.ai_model = os.getenv("AI_MODEL", "qwen3.5:4b").strip()
        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        ).rstrip("/")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        self.gemini_base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
        ).rstrip("/")

        # --- Request limits ----------------------------------------------
        self.max_images = _get_int("MAX_IMAGES", 4)
        self.max_image_bytes = _get_int("MAX_IMAGE_BYTES", 5 * 1024 * 1024)
        self.max_long_side = _get_int("MAX_LONG_SIDE", 1024)
        self.max_request_bytes = _get_int("MAX_REQUEST_BYTES", 25 * 1024 * 1024)
        self.request_timeout_s = _get_float("REQUEST_TIMEOUT_S", 240.0)

        # --- Rate limiting (per client IP) -------------------------------
        self.rate_limit_per_min = _get_int("RATE_LIMIT_PER_MIN", 20)
        self.rate_limit_window_s = _get_float("RATE_LIMIT_WINDOW_S", 60.0)

        # --- Storage ------------------------------------------------------
        self.db_path = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "coslist.db")))
        self.image_dir = Path(os.getenv("IMAGE_DIR", str(BASE_DIR / "data" / "images")))
        self.thumb_dir = Path(os.getenv("THUMB_DIR", str(BASE_DIR / "data" / "thumbs")))
        self.static_dir = BASE_DIR / "static"

        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.image_dir, self.thumb_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def provider_label(self) -> str:
        if self.ai_provider == "gemini":
            return f"gemini:{self.gemini_model}"
        if self.ai_provider == "text":
            return f"text-only:{self.ai_model}"
        return f"ollama:{self.ai_model}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

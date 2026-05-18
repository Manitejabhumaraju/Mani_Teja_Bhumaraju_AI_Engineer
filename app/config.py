"""
App config.

Pulled from env with sane defaults. One config object, imported wherever needed.
Avoid sprinkling os.environ.get throughout the codebase - it makes test setup
miserable and means defaults drift across files.

The .env file is loaded if present, but env vars set in the shell win over it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load .env early. If the file isn't there, this is a no-op.
load_dotenv()


REPO_ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # === LLM ===
    nim_api_key: str = os.environ.get("NVIDIA_API_KEY", "")
    nim_base_url: str = os.environ.get(
        "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    nim_model: str = os.environ.get("NIM_MODEL", "meta/llama-3.3-70b-instruct")

    # Per-request defaults. Override at call sites if needed.
    llm_temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    llm_max_tokens: int = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
    llm_timeout_s: float = float(os.environ.get("LLM_TIMEOUT_S", "30"))

    # === Paths ===
    db_path: Path = REPO_ROOT / "data" / "loadshare.db"
    docs_dir: Path = REPO_ROOT / "docs"

    # === Redis (semantic cache, sessions) ===
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    semantic_cache_threshold: float = float(
        os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.92")
    )
    embedding_model: str = os.environ.get(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    # === Observability ===
    langsmith_enabled: bool = _bool("LANGCHAIN_TRACING_V2", False)
    langsmith_project: str = os.environ.get(
        "LANGCHAIN_PROJECT", "loadshare-rca-agent"
    )

    # === Server ===
    api_host: str = os.environ.get("API_HOST", "127.0.0.1")
    api_port: int = int(os.environ.get("API_PORT", "8000"))

    # === Auth (mock for demo - see README) ===
    # Hardcoded demo users. Replace with real auth before any production use.
    demo_users: dict = None  # populated in __post_init__-equivalent below


# Sentinel mutable default isn't allowed in frozen dataclass; build it in a
# factory function and override the field.
def _build_demo_users() -> dict:
    """
    Mock auth credentials for the demo. Override via env if needed (rare).
    Format: USER1:PASS1,USER2:PASS2
    """
    raw = os.environ.get("DEMO_USERS", "demo:demo,analyst:analyst")
    users = {}
    for pair in raw.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            users[u.strip()] = p.strip()
    return users


# Singleton settings. Build once at import.
settings = Settings()
# Patch in the dict (frozen dataclass won't let us set it directly during init,
# so we use object.__setattr__ - this is a known idiom for one-shot post-init).
object.__setattr__(settings, "demo_users", _build_demo_users())


def assert_llm_configured() -> None:
    """Call at app boot. Fail fast if the API key isn't set."""
    if not settings.nim_api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Get a free key at "
            "https://build.nvidia.com and put it in .env"
        )

"""Settings, read once from the environment.

Same shape as `perception/config.py` on purpose: one frozen object, every
value overridable by an env var, no settings scattered through the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

load_dotenv(ROOT / ".env")


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


#: What each provider runs when ORCH_MODEL is blank.
DEFAULT_MODELS = {"openai": "gpt-6-astra", "xai": "grok-4.7"}
#: Where each provider's key is read from.
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "xai": "XAI_API_KEY"}
#: xAI serves the OpenAI protocol, Responses API included, at its own address.
XAI_BASE_URL = "https://api.x.ai/v1"

_PROVIDER = _s("ORCH_PROVIDER", "openai")
_MODEL = _s("ORCH_MODEL", "") or DEFAULT_MODELS.get(_PROVIDER, "gpt-6-astra")


def provider_of(model: str) -> str:
    """Which provider serves a model, from its name. For picking a model
    per run, as the evals do, without editing .env."""
    if model.startswith("grok"):
        return "xai"
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


@dataclass(frozen=True)
class Config:
    # 1. Model -----------------------------------------------------------
    #: openai | xai | anthropic
    provider: str = _PROVIDER
    model: str = _MODEL
    #: Must accept images. Blank reuses `model`.
    vision_model: str = _s("ORCH_VISION_MODEL", "") or _MODEL
    #: Blank disables vector search; the memory falls back to full-text.
    #: Always an OpenAI model, whichever provider runs the agent.
    embed_model: str = _s("ORCH_EMBED_MODEL", "text-embedding-3-small")

    # 2. Services --------------------------------------------------------
    perception_base: str = _s("PERCEPTION_BASE", "http://localhost:8001").rstrip("/")
    port: int = _i("PORT", 8000)
    log_level: str = _s("LOG_LEVEL", "INFO")

    # 3. Agent loop ------------------------------------------------------
    #: Hard cap on model steps in one turn; each may request several tools.
    max_steps: int = _i("ORCH_MAX_STEPS", 4)
    #: Wake the agent at most once per behaviour per this many seconds. A
    #: person standing near the duck would otherwise fire thirty times a
    #: second, and one alert that arrives beats a hundred that get dropped.
    event_cooldown_s: float = _f("ORCH_EVENT_COOLDOWN_S", 5.0)
    #: A tool call that takes longer than this is a demo that has stalled.
    tool_timeout_s: float = _f("ORCH_TOOL_TIMEOUT_S", 12.0)
    #: Frontier models stall occasionally; the frontend learned this the hard way.
    llm_timeout_s: float = _f("ORCH_LLM_TIMEOUT_S", 20.0)
    turn_timeout_s: float = _f("ORCH_TURN_TIMEOUT_S", 35.0)

    # 4. Paths -----------------------------------------------------------
    documents: Path = ROOT / "documents"
    memory_dir: Path = Path(_s("ORCH_MEMORY_DIR", str(ROOT / ".lancedb")))
    #: Built frontend, served at / when it exists. Vite's dev server is the
    #: normal path; this is for a single-process demo machine.
    static_dir: Path = REPO_ROOT / "frontend" / "dist"

    @property
    def api_key(self) -> str:
        """The key for whichever provider runs the agent."""
        return key_for(self.provider)

    @property
    def openai_key(self) -> str:
        """For the phrase memory's embeddings, which are OpenAI's even when
        the agent runs on another provider."""
        return os.environ.get("OPENAI_API_KEY", "")


def key_for(provider: str) -> str:
    return os.environ.get(KEY_ENV.get(provider, "OPENAI_API_KEY"), "")


CFG = Config()

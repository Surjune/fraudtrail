"""Settings read from the environment (see .env.example).

Nothing here has a credential default: a missing credential raises rather than silently
falling back, because the policy is explicit that missing access produces an error, never
fabricated data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]


class LlmProvider(StrEnum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"
    NONE = "none"
    """No model configured: the agent still runs and writes every answer file, with
    templated prose instead of generated prose."""


class ConfigError(RuntimeError):
    """A setting the caller needs is missing or unusable."""


@dataclass(frozen=True)
class TigerGraphSettings:
    host: str
    graph: str
    username: str
    password: str
    secret: str
    restpp_port: str
    gs_port: str
    is_cloud: bool

    @property
    def configured(self) -> bool:
        return bool(self.host) and bool(self.secret or (self.username and self.password))

    def require(self) -> None:
        if not self.configured:
            raise ConfigError(
                "TigerGraph is not configured. Copy .env.example to .env and set TG_HOST "
                "plus TG_SECRET, or TG_USERNAME and TG_PASSWORD."
            )


@dataclass(frozen=True)
class LlmSettings:
    provider: LlmProvider
    model: str
    api_key: str
    base_url: str = ""
    """Only for an OpenAI-compatible endpoint that is not OpenAI's own, which is how
    most free tiers are served. Empty means the provider's default."""

    @property
    def enabled(self) -> bool:
        return self.provider is not LlmProvider.NONE and bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    tigergraph: TigerGraphSettings
    llm: LlmSettings
    embedding_model: str
    raw_dir: Path
    processed_dir: Path

    @property
    def load_dir(self) -> Path:
        return self.processed_dir / "load"


def _path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name, default))
    return value if value.is_absolute() else REPO / value


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or REPO / ".env", override=False)
    provider_name = os.environ.get("FRAUDTRAIL_LLM_PROVIDER", "none").strip().lower()
    try:
        provider = LlmProvider(provider_name)
    except ValueError as exc:
        options = ", ".join(p.value for p in LlmProvider)
        raise ConfigError(f"FRAUDTRAIL_LLM_PROVIDER must be one of: {options}") from exc

    return Settings(
        tigergraph=TigerGraphSettings(
            host=os.environ.get("TG_HOST", "").strip(),
            graph=os.environ.get("TG_GRAPHNAME", "FraudTrail").strip(),
            username=os.environ.get("TG_USERNAME", "").strip(),
            password=os.environ.get("TG_PASSWORD", "").strip(),
            secret=os.environ.get("TG_SECRET", "").strip(),
            restpp_port=os.environ.get("TG_RESTPP_PORT", "443").strip(),
            gs_port=os.environ.get("TG_GS_PORT", "443").strip(),
            is_cloud=os.environ.get("TG_TGCLOUD", "true").strip().lower() == "true",
        ),
        llm=LlmSettings(
            provider=provider,
            model=os.environ.get("FRAUDTRAIL_LLM_MODEL", "").strip(),
            api_key=os.environ.get("FRAUDTRAIL_LLM_API_KEY", "").strip(),
            base_url=os.environ.get("FRAUDTRAIL_LLM_BASE_URL", "").strip(),
        ),
        embedding_model=os.environ.get(
            "FRAUDTRAIL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        ).strip(),
        raw_dir=_path("FRAUDTRAIL_RAW_DIR", "data/raw"),
        processed_dir=_path("FRAUDTRAIL_PROCESSED_DIR", "data/processed"),
    )

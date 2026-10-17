"""Settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("NYAYALENS_DATA_DIR", str(ROOT / "data"))))
    db_path: Path = field(default_factory=lambda: Path(_env("NYAYALENS_DB", str(ROOT / "var" / "nyayalens.db"))))

    # LLM: auto | openai | azure | bedrock | mock. auto -> openai if OPENAI_API_KEY is set, else mock (offline).
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    openai_small: str = field(default_factory=lambda: _env("OPENAI_MODEL_SMALL", "gpt-4.1-mini"))
    openai_large: str = field(default_factory=lambda: _env("OPENAI_MODEL_LARGE", "gpt-4.1"))
    azure_small: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT_SMALL", "gpt-4.1-mini"))
    azure_large: str = field(default_factory=lambda: _env("AZURE_OPENAI_DEPLOYMENT_LARGE", "gpt-4.1"))
    bedrock_small: str = field(default_factory=lambda: _env("BEDROCK_MODEL_SMALL", "amazon.nova-lite-v1:0"))
    bedrock_large: str = field(default_factory=lambda: _env("BEDROCK_MODEL_LARGE", "amazon.nova-pro-v1:0"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    llm_timeout_s: float = 45.0

    max_upload_mb: int = 15
    # Fixed "today" (YYYY-MM-DD) for reproducible tests and evaluation; empty = real date.
    today_override: str = field(default_factory=lambda: _env("NYAYALENS_TODAY", ""))

    def today(self) -> date:
        return date.fromisoformat(self.today_override) if self.today_override else date.today()

    def resolved_provider(self) -> str:
        p = self.llm_provider.lower()
        if p == "auto":
            return "openai" if _env("OPENAI_API_KEY") else "mock"
        return p

    def models(self) -> tuple[str, str]:
        p = self.resolved_provider()
        return {"openai": (self.openai_small, self.openai_large),
                "azure": (self.azure_small, self.azure_large),
                "bedrock": (self.bedrock_small, self.bedrock_large)}.get(p, ("mock", "mock"))

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.config import ROOT, Settings
from app.llm import LLMClient
from app.service import NyayaLens

SAMPLES = ROOT / "data" / "samples"
TODAY = "2026-10-17"


def sample(name: str) -> bytes:
    return (SAMPLES / f"{name}.txt").read_bytes()


@pytest.fixture(scope="session")
def engine(tmp_path_factory) -> NyayaLens:
    db = tmp_path_factory.mktemp("db") / "t.db"
    return NyayaLens(Settings(db_path=db, llm_provider="mock", today_override=TODAY), llm=LLMClient("mock", "m", "m"))


@pytest.fixture(scope="session")
def analysed(engine) -> dict:
    return {p.stem: engine.analyze(p.name, p.read_bytes()) for p in sorted(SAMPLES.glob("*.txt"))}


class ScriptedLLM(LLMClient):
    """Stands in for a real model: returns scripted JSON per step and records every prompt it receives."""

    def __init__(self, responses: dict):
        super().__init__("openai", "gpt-4.1-mini", "gpt-4.1")
        self.responses, self.calls = responses, []

    def complete_json(self, system, user, *, tier, step, usage, max_tokens=700):
        self.calls.append({"step": step, "tier": tier, "user": user})
        usage.add(step, self.model_for(tier), len(user) // 4, 80)
        r = self.responses.get(step, {})
        r = r(user) if callable(r) else r
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture()
def llm_engine(engine):
    """Same trained models and store as `engine`, with a scripted LLM swapped in."""
    def make(responses: dict):
        llm = ScriptedLLM(responses)
        e = NyayaLens.__new__(NyayaLens)
        e.settings, e.models, e.playbook, e.store, e.llm = engine.settings, engine.models, engine.playbook, engine.store, llm
        return e, llm
    return make


tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")

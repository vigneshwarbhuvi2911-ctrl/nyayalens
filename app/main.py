"""HTTP API and web page."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.config import ROOT, Settings
from app.parse import ParseError
from app.service import NotFound, NyayaLens

STATIC = Path(__file__).parent / "static"
SAMPLES = ROOT / "data" / "samples"


class TextIn(BaseModel):
    filename: str = Field(default="pasted.txt", max_length=120)
    text: str = Field(min_length=20, max_length=300_000)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class ReplyIn(BaseModel):
    instructions: str = Field(default="", max_length=2000)


def create_app(engine: NyayaLens | None = None) -> FastAPI:
    app = FastAPI(title="NyayaLens", version="1.0.0",
                  description="AI Legal Document Analysis Assistant (personal demo with synthetic documents). "
                              "Not legal advice.")
    app.state.engine = engine

    def eng() -> NyayaLens:
        if app.state.engine is None:
            app.state.engine = NyayaLens(Settings())
        return app.state.engine

    def guard(fn, *args):
        try:
            return fn(*args)
        except NotFound:
            raise HTTPException(404, "analysis not found")
        except ParseError as exc:
            raise HTTPException(422, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        e = eng()
        return {"status": "ok", "llm_provider": e.llm.provider, "clause_labels": len(e.models.clause_labels)}

    @app.get("/samples")
    def samples():
        return sorted(p.name for p in SAMPLES.glob("*.txt"))

    @app.post("/samples/{name}/analyze")
    def analyze_sample(name: str):
        if not re.fullmatch(r"[A-Za-z0-9-]+\.txt", name) or not (SAMPLES / name).is_file():
            raise HTTPException(404, "sample not found")
        return eng().analyze(name, (SAMPLES / name).read_bytes())

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        data = await file.read(eng().settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > eng().settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"file too large (max {eng().settings.max_upload_mb} MB)")
        return guard(eng().analyze, Path(file.filename or "upload.txt").name, data)

    @app.post("/analyze/text")
    def analyze_text(body: TextIn):
        return guard(eng().analyze_text, body.filename, body.text)

    @app.get("/analyses")
    def recent():
        return eng().store.recent()

    @app.get("/analyses/{rid}")
    def get(rid: str):
        return guard(eng().get, rid)

    @app.post("/analyses/{rid}/ask")
    def ask(rid: str, q: Question):
        return guard(eng().ask, rid, q.question)

    @app.post("/analyses/{rid}/reply")
    def reply(rid: str, body: ReplyIn):
        return guard(eng().reply, rid, body.instructions)

    @app.get("/analyses/{rid}/calendar.ics")
    def calendar(rid: str):
        ics = guard(eng().calendar, rid)
        return Response(ics, media_type="text/calendar",
                        headers={"Content-Disposition": f'attachment; filename="nyayalens-{rid}.ics"'})

    @app.get("/compare")
    def compare(a: str, b: str):
        return guard(eng().compare, a, b)

    return app


@lru_cache
def get_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()

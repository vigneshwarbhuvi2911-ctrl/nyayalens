"""The analysis pipeline and storage.

    read file -> split into clauses -> document type (ML) -> clause types (ML) -> fields (rules, LLM fills gaps)
              -> risk playbook -> timeline -> summary
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from datetime import date
from pathlib import Path

from app.classify import DOC_TYPE_NAMES, Classifiers
from app.config import Settings
from app.extract import FIELDS, LABELS, Field, agreement_parties, extract, timeline, to_ics
from app.features import DISCLAIMER, answer, compare, llm_fill, llm_summary, offline_summary, reply_draft
from app.guardrails import mask_pii
from app.llm import LLMClient, Usage, make_llm
from app.parse import Document, load, segment
from app.risks import Playbook

ORDER = {t: (["parties"] if t != "legal_notice" else []) + fs for t, fs in FIELDS.items()}


class NotFound(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, created REAL, filename TEXT, "
                          "doc_type TEXT, text TEXT, result TEXT)")
        self.conn.commit()

    def save(self, rid: str, filename: str, doc_type: str, text: str, result: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO analyses VALUES (?,?,?,?,?,?)",
                              (rid, time.time(), filename, doc_type, text, json.dumps(result)))
            self.conn.commit()

    def get(self, rid: str) -> tuple[str, dict]:
        with self.lock:
            row = self.conn.execute("SELECT text, result FROM analyses WHERE id=?", (rid,)).fetchone()
        if not row:
            raise NotFound(rid)
        return row[0], json.loads(row[1])

    def recent(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, created, filename, doc_type FROM analyses ORDER BY created DESC "
                                     "LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "created": r[1], "filename": r[2], "doc_type": r[3]} for r in rows]


class NyayaLens:
    def __init__(self, settings: Settings, llm: LLMClient | None = None):
        self.settings = settings
        self.models = Classifiers(settings.data_dir / "training")
        self.playbook = Playbook(settings.data_dir / "playbook.yaml")
        self.store = Store(settings.db_path)
        self.llm = llm or make_llm(settings)

    # ---------------------------------------------------------------------------------------- analyse
    def analyze(self, filename: str, data: bytes) -> dict:
        t0 = time.time()
        doc = load(filename, data)
        return self._analyze(doc, filename, t0)

    def analyze_text(self, filename: str, text: str) -> dict:
        t0 = time.time()
        doc = segment(text)
        doc.source = filename
        return self._analyze(doc, filename, t0)

    def _analyze(self, doc: Document, filename: str, t0: float) -> dict:
        usage = Usage()
        today = self.settings.today()
        doc_type, conf, dist = self.models.doc_type(doc)
        self.models.label_clauses(doc, doc_type)
        fields = extract(doc, doc_type) if doc_type != "unknown" else {}
        filled = llm_fill(self.llm, doc, doc_type, fields, usage) if doc_type != "unknown" else []
        parties = agreement_parties(doc) if doc_type not in ("legal_notice", "unknown") else []
        risks = self.playbook.run(doc, doc_type, fields, today) if doc_type != "unknown" else []
        events = timeline(doc_type, fields)
        for e in events:
            e["past"] = e["date"] < today.isoformat()
        summary = llm_summary(self.llm, doc, usage)
        summary_source = "llm" if summary else "template"
        if not summary:
            summary = offline_summary(doc_type, parties, fields) if doc_type != "unknown" else [
                "This does not look like a loan agreement, rent agreement, NDA or legal notice, so only the clause "
                "list is shown."]
        _, pii = mask_pii(doc.text)
        rid = "a-" + uuid.uuid4().hex[:12]
        order = [k for k in ORDER.get(doc_type, []) if k in fields]
        result = {
            "id": rid, "filename": filename, "doc_type": doc_type, "doc_type_name": DOC_TYPE_NAMES[doc_type],
            "doc_type_confidence": round(conf, 3), "doc_type_scores": {str(k): v for k, v in dist.items()},
            "ocr": doc.ocr, "pii_detected": pii, "parties": parties,
            "fields": {k: fields[k].public() | {"label": LABELS.get(k, k)} for k in order},
            "missing_fields": [k for k in FIELDS.get(doc_type, []) if k not in fields],
            "llm_filled": filled, "clauses": [c.public() for c in doc.clauses], "risks": risks,
            "timeline": events, "summary": summary, "summary_source": summary_source,
            "as_of": today.isoformat(), "disclaimer": DISCLAIMER, "usage": usage.as_dict(),
            "provider": self.llm.provider, "latency_ms": int((time.time() - t0) * 1000),
        }
        self.store.save(rid, filename, doc_type, doc.text, result)
        return result

    # ----------------------------------------------------------------------------------------- reload
    def _reload(self, rid: str) -> tuple[Document, dict, dict[str, Field]]:
        text, result = self.store.get(rid)
        doc = segment(text)
        labels = {c["num"]: (c["label"], c["confidence"]) for c in result["clauses"]}
        for c in doc.clauses:
            c.label, c.confidence = labels.get(c.num, ("", 0.0))
        fields = {}
        for k, f in result["fields"].items():
            val = f["value"]
            if isinstance(val, str) and len(val) == 10 and val[4] == "-" and val[7] == "-":
                val = date.fromisoformat(val)
            fields[k] = Field(k, val, f["display"], f["quote"], f["clause"], f.get("source", "rule"), f.get("extra", {}))
        return doc, result, fields

    def get(self, rid: str) -> dict:
        return self.store.get(rid)[1]

    def ask(self, rid: str, question: str) -> dict:
        doc, result, fields = self._reload(rid)
        usage = Usage()
        out = answer(question, doc, fields, self.llm, usage)
        return out | {"question": question, "usage": usage.as_dict(), "disclaimer": DISCLAIMER}

    def compare(self, rid_a: str, rid_b: str) -> dict:
        a, b = self.get(rid_a), self.get(rid_b)
        return {"a": {"id": rid_a, "filename": a["filename"]}, "b": {"id": rid_b, "filename": b["filename"]},
                "same_type": a["doc_type"] == b["doc_type"]} | compare(a, b)

    def reply(self, rid: str, instructions: str = "") -> dict:
        doc, result, fields = self._reload(rid)
        if result["doc_type"] != "legal_notice":
            raise ValueError("replies can be drafted only for legal notices")
        usage = Usage()
        return reply_draft(doc, fields, instructions, self.llm, usage, self.settings.today()) | {"usage": usage.as_dict()}

    def calendar(self, rid: str) -> str:
        result = self.get(rid)
        title = f"{result['doc_type_name']} ({result['filename']})"
        return to_ics(title, result["timeline"])

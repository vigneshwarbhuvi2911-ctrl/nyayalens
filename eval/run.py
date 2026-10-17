"""Evaluation against hand-labelled documents (eval/labels.json) and questions (eval/qa.json).

    python -m eval.run                      # offline
    python -m eval.run --provider openai    # with a real LLM (fills gaps, summaries, Q&A)
    python -m eval.run --gate --out eval/report.md

Metrics
  document type accuracy        - 12 documents
  clause type accuracy          - every numbered clause, against the hand labels
  field accuracy                - every labelled field (including "not stated" fields, which must stay empty)
  party accuracy                - parties named in agreements
  risk flags precision / recall - against the expected flag ids
  timeline accuracy             - key dates (EMI end, lock-in end, deadlines)
  Q&A accuracy                  - right clause cited and expected fact in the answer; "not in document"
                                  questions must be declined
The training data for the classifiers is generated from separate templates (scripts/make_training_data.py);
none of these documents is used for training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from app.config import ROOT, Settings
from app.extract import clean_name
from app.service import NyayaLens

GATES = {"doc_type_accuracy": 1.0, "clause_accuracy": 0.95, "field_accuracy": 0.98, "party_accuracy": 1.0,
         "risk_precision": 1.0, "risk_recall": 1.0, "timeline_accuracy": 1.0, "qa_accuracy": 0.9}


def _eq(expected, got) -> bool:
    if expected is None:
        return got is None
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(expected - got) < 0.01
    return str(expected).strip().lower() == str(got).strip().lower()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9% ]", "", s.lower().replace(",", ""))


def run(provider: str = "mock") -> dict:
    tmp = Path(tempfile.mkdtemp())
    app = NyayaLens(Settings(db_path=tmp / "e.db", llm_provider=provider, today_override="2026-10-17"))
    labels = json.loads((ROOT / "eval" / "labels.json").read_text())
    t0 = time.time()
    rows, ids = [], {}
    c_tot = c_ok = f_tot = f_ok = p_tot = p_ok = t_tot = t_ok = 0
    tp = fp = fn = 0
    for name, gold in labels.items():
        r = app.analyze(f"{name}.txt", (ROOT / "data" / "samples" / f"{name}.txt").read_bytes())
        ids[name] = r["id"]
        got_clauses = [c["label"] for c in r["clauses"] if c["num"] not in ("0", "end")]
        clause_err = [(i + 1, g, p) for i, (g, p) in enumerate(zip(gold["clauses"], got_clauses)) if g != p]
        clause_err += [("count", len(gold["clauses"]), len(got_clauses))] if len(got_clauses) != len(gold["clauses"]) else []
        c_tot += len(gold["clauses"])
        c_ok += len(gold["clauses"]) - len([e for e in clause_err if e[0] != "count"])
        field_err = []
        for k, v in gold["fields"].items():
            got = r["fields"].get(k, {}).get("value")
            f_tot += 1
            if _eq(v, got):
                f_ok += 1
            else:
                field_err.append((k, v, got))
        party_err = []
        if "parties" in gold:
            got_p = {clean_name(p["name"]).lower() for p in r["parties"]}
            for p in gold["parties"]:
                p_tot += 1
                if clean_name(p).lower() in got_p:
                    p_ok += 1
                else:
                    party_err.append(p)
        got_r, exp_r = {f["id"] for f in r["risks"]}, set(gold["risks"])
        tp += len(got_r & exp_r)
        fp += len(got_r - exp_r)
        fn += len(exp_r - got_r)
        tl = {e["label"]: e["date"] for e in r["timeline"]}
        tl_err = []
        for k, v in gold["timeline"].items():
            t_tot += 1
            if tl.get(k) == v:
                t_ok += 1
            else:
                tl_err.append((k, v, tl.get(k)))
        rows.append({"doc": name, "doc_type_ok": r["doc_type"] == gold["doc_type"], "doc_type": r["doc_type"],
                     "clause_errors": clause_err, "field_errors": field_err, "party_errors": party_err,
                     "extra_risks": sorted(got_r - exp_r), "missed_risks": sorted(exp_r - got_r),
                     "timeline_errors": tl_err, "latency_ms": r["latency_ms"]})

    qa_rows, qa_ok = [], 0
    for q in json.loads((ROOT / "eval" / "qa.json").read_text()):
        a = app.ask(ids[q["doc"]], q["question"])
        if q["clause"] is None:
            ok = not a["answerable"]
        else:
            cited = [c["clause"] for c in a["citations"]]
            ok = a["answerable"] and q["clause"] in cited and all(_norm(f) in _norm(a["answer"] + " " + " ".join(
                c["quote"] for c in a["citations"])) for f in q["facts"])
        qa_ok += ok
        qa_rows.append({"doc": q["doc"], "question": q["question"], "ok": ok, "answer": a["answer"],
                        "cited": [c["clause"] for c in a["citations"]]})

    s = {"provider": app.llm.provider, "documents": len(rows),
         "doc_type_accuracy": round(sum(r["doc_type_ok"] for r in rows) / len(rows), 3),
         "clause_accuracy": round(c_ok / c_tot, 3), "clauses": c_tot,
         "field_accuracy": round(f_ok / f_tot, 3), "fields": f_tot,
         "party_accuracy": round(p_ok / p_tot, 3) if p_tot else 1.0,
         "risk_precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
         "risk_recall": round(tp / (tp + fn), 3) if tp + fn else 1.0, "risk_flags_expected": tp + fn,
         "timeline_accuracy": round(t_ok / t_tot, 3) if t_tot else 1.0,
         "qa_accuracy": round(qa_ok / len(qa_rows), 3), "qa_questions": len(qa_rows),
         "avg_latency_ms": int(sum(r["latency_ms"] for r in rows) / len(rows)), "seconds": round(time.time() - t0, 1)}
    return {"summary": s, "docs": rows, "qa": qa_rows}


def report(res: dict) -> str:
    s = res["summary"]
    out = [f"# NyayaLens evaluation (provider: `{s['provider']}`)", "", "| Metric | Value |", "|---|---|"]
    for k in ("doc_type_accuracy", "clause_accuracy", "field_accuracy", "party_accuracy", "risk_precision",
              "risk_recall", "timeline_accuracy", "qa_accuracy", "avg_latency_ms"):
        out.append(f"| {k} | {s[k]} |")
    out += ["", f"{s['documents']} documents, {s['clauses']} clauses, {s['fields']} fields, "
            f"{s['risk_flags_expected']} expected risk flags, {s['qa_questions']} questions.", "",
            "| Document | Type | Clause errors | Field errors | Risk differences |", "|---|---|---|---|---|"]
    for r in res["docs"]:
        risk = ", ".join([f"+{x}" for x in r["extra_risks"]] + [f"-{x}" for x in r["missed_risks"]])
        out.append(f"| {r['doc']} | {r['doc_type']} | {'; '.join(f'§{n}: {g} -> {p}' for n, g, p in r['clause_errors']) or '-'} | "
                   f"{'; '.join(f'{k}: {g} -> {p}' for k, g, p in r['field_errors']) or '-'} | {risk or '-'} |")
    out += ["", "| Q&A | Question | Correct | Cited |", "|---|---|---|---|"]
    for q in res["qa"]:
        out.append(f"| {q['doc']} | {q['question']} | {'yes' if q['ok'] else 'NO'} | {', '.join(q['cited']) or '-'} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--out")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()
    res = run(args.provider)
    print(json.dumps(res["summary"], indent=2))
    if args.show_errors:
        for r in res["docs"]:
            errs = {k: r[k] for k in ("clause_errors", "field_errors", "party_errors", "extra_risks", "missed_risks",
                                      "timeline_errors") if r[k]}
            if errs or not r["doc_type_ok"]:
                print(r["doc"], r["doc_type"], errs)
        for q in res["qa"]:
            if not q["ok"]:
                print("QA", q["doc"], q["question"], "->", q["answer"][:150], q["cited"])
    if args.out:
        Path(args.out).write_text(report(res))
    if args.gate:
        failed = [k for k, v in GATES.items() if res["summary"][k] < v]
        if failed:
            print("QUALITY GATE FAILED:", failed)
            sys.exit(1)
        print("quality gates passed")


if __name__ == "__main__":
    main()

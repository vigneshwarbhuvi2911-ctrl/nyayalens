"""Runs the review playbook (data/playbook.yaml) against an analysed document.

Each flag carries its evidence: the clause reference and the quote that triggered it (or, for a missing
clause, a statement of what is missing). Checks are small named functions; which ones run, and their
thresholds, come from the YAML file.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from app.extract import Field
from app.parse import Document

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


class Playbook:
    def __init__(self, path: Path):
        self.rules = yaml.safe_load(path.read_text())

    def run(self, doc: Document, doc_type: str, fields: dict[str, Field], today: date) -> list[dict]:
        flags = []
        for rule in self.rules.get(doc_type, []):
            hit = CHECKS[rule["check"]](doc, fields, today, **rule.get("params", {}))
            if not hit:
                continue
            severity = rule["severity"]
            hi = rule.get("params", {}).get("high_limit")
            if hi is not None and isinstance(hit.get("value"), (int, float)) and hit["value"] > hi:
                severity = "high"
            values = {k: f.value for k, f in fields.items()} | rule.get("params", {}) | hit.get("vars", {})
            flags.append({"id": rule["id"], "severity": severity, "title": rule["title"],
                          "why": rule["why"].format_map(_Safe(values)), "clause": hit.get("clause", ""),
                          "quote": hit.get("quote", "")})
        return sorted(flags, key=lambda f: (SEVERITY_ORDER[f["severity"]], f["id"]))


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"

    def __getitem__(self, key):
        v = dict.__getitem__(self, key) if key in self else self.__missing__(key)
        return f"{v:g}" if isinstance(v, float) else v


def _ev(f: Field) -> dict:
    return {"clause": f.clause, "quote": f.quote, "value": f.value}


def field_above(doc, fields, today, field, limit, high_limit=None):
    f = fields.get(field)
    if f and isinstance(f.value, (int, float)) and not isinstance(f.value, bool) and f.value > limit:
        return _ev(f)


def field_below(doc, fields, today, field, limit):
    f = fields.get(field)
    if f and isinstance(f.value, (int, float)) and f.value < limit:
        return _ev(f)


def field_equals(doc, fields, today, field, value):
    f = fields.get(field)
    if f and f.value == value:
        return _ev(f)


def clause_matches(doc, fields, today, labels, pattern):
    for c in doc.of(*labels):
        m = re.search(pattern, c.text, re.I)
        if m:
            from app.extract import _sentence
            return {"clause": c.ref, "quote": _sentence(c.text, m.start(), m.end())}


def text_matches(doc, fields, today, pattern):
    for c in doc.numbered():
        m = re.search(pattern, c.text, re.I)
        if m:
            from app.extract import _sentence
            return {"clause": c.ref, "quote": _sentence(c.text, m.start(), m.end())}


def clause_missing(doc, fields, today, label):
    if not doc.of(label):
        return {"clause": "", "quote": f"No clause about {label.replace('_', ' ')} was found."}


def field_missing_if_clause(doc, fields, today, field, label):
    cs = doc.of(label)
    if cs and field not in fields:
        return {"clause": cs[0].ref, "quote": " ".join(cs[0].text.split())}


def one_sided_termination(doc, fields, today):
    for c in doc.of("termination", "term"):
        m = re.search(r"(?:Lessor|Landlord|Owner|Licensor)\s+may\s+terminate[^.]*\bat\s+any\s+time", c.text, re.I)
        if m:
            from app.extract import _sentence
            return {"clause": c.ref, "quote": _sentence(c.text, m.start(), m.end())}


def uncapped_indemnity(doc, fields, today):
    ind = doc.of("indemnity")
    if ind and not doc.of("liability_cap"):
        return {"clause": ind[0].ref, "quote": " ".join(ind[0].text.split())}


def deadline_passed(doc, fields, today):
    nd, days = fields.get("notice_date"), fields.get("compliance_days")
    if nd and days and isinstance(nd.value, date):
        from datetime import timedelta
        deadline = nd.value + timedelta(days=days.value)
        if deadline < today:
            return {"clause": days.clause, "quote": days.quote, "vars": {"deadline": deadline.strftime("%d %b %Y")}}


CHECKS = {f.__name__: f for f in (field_above, field_below, field_equals, clause_matches, text_matches,
                                   clause_missing, field_missing_if_clause, one_sided_termination,
                                   uncapped_indemnity, deadline_passed)}

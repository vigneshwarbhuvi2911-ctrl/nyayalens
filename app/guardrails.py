"""Masks personal identifiers before any text is sent to an LLM.

Names of parties are kept, because they are the substance of a legal document; phone numbers, email
addresses, Aadhaar, PAN, card and account numbers are replaced with placeholders.
"""
from __future__ import annotations

import re

_PII = [
    ("CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("PHONE", re.compile(r"(?:\+91[ -]?)?\b[6-9]\d{9}\b")),
    ("ACCOUNT_NO", re.compile(r"\b\d{11,18}\b")),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, rx in _PII:
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[{label}]", text)
    return text, found

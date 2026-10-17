"""Two scikit-learn classifiers, trained at start-up on synthetic examples (data/training/, about 1 s):

* document type: loan_agreement | rent_agreement | nda | legal_notice  (below 0.5 confidence -> "unknown")
* clause type:   30 labels such as interest, deposit, non_compete, demand

Both use TF-IDF on word 1-2 grams plus character 3-5 grams (robust to OCR noise and spelling variants),
with logistic regression, so each prediction comes with a probability that the UI shows as confidence.
Training from text at start-up, rather than shipping a pickled model, keeps the repository reviewable and
avoids loading untrusted binary files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from app.parse import Document

DOC_TYPES = ("loan_agreement", "rent_agreement", "nda", "legal_notice")
DOC_TYPE_NAMES = {"loan_agreement": "Loan agreement", "rent_agreement": "Rent / lease agreement",
                  "nda": "Non-disclosure agreement", "legal_notice": "Legal notice", "unknown": "Unknown document"}


def _pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("clf", LogisticRegression(C=c, max_iter=2000)),
    ])


def _load(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


class Classifiers:
    def __init__(self, training_dir: Path):
        x, y = _load(training_dir / "clauses.jsonl")
        self.clause_model = _pipeline(8.0).fit(x, y)
        x, y = _load(training_dir / "docs.jsonl")
        self.doc_model = _pipeline(4.0).fit(x, y)
        self.clause_labels = [str(c) for c in self.clause_model.classes_]

    def doc_type(self, doc: Document) -> tuple[str, float, dict]:
        probs = self.doc_model.predict_proba([doc.text[:6000]])[0]
        dist = {c: round(float(p), 3) for c, p in zip(self.doc_model.classes_, probs)}
        best = int(np.argmax(probs))
        label = str(self.doc_model.classes_[best])
        return (label if probs[best] >= 0.5 else "unknown"), float(probs[best]), dist

    def label_clauses(self, doc: Document, doc_type: str) -> None:
        """Labels each numbered clause. Labels that cannot occur in this document type are excluded,
        e.g. 'deposit' in a loan agreement."""
        allowed = ALLOWED.get(doc_type)
        targets = doc.numbered()
        if not targets:
            return
        probs = self.clause_model.predict_proba([c.body for c in targets])
        classes = self.clause_model.classes_
        mask = np.array([allowed is None or c in allowed for c in classes], dtype=float)
        for c, p in zip(targets, probs):
            p = p * mask
            p = p / p.sum() if p.sum() else p
            k = int(np.argmax(p))
            c.label, c.confidence = str(classes[k]), round(float(p[k]), 3)
        doc.preamble.label = "parties"
        for c in doc.clauses:
            if c.num == "end":
                c.label = "execution"


COMMON = {"definitions", "disputes", "notices", "indemnity", "term", "termination", "remedies"}
ALLOWED = {
    "loan_agreement": COMMON | {"principal", "interest", "repayment", "prepayment", "penal_charges", "default",
                                "security", "set_off"},
    "rent_agreement": COMMON | {"property", "rent", "deposit", "escalation", "maintenance", "use"},
    "nda": COMMON | {"purpose", "confidentiality", "exclusions", "non_compete", "liability_cap",
                     "return_of_info"},
    "legal_notice": {"facts", "demand", "consequences"},
}

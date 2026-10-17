"""Reading documents (TXT, MD, PDF, scanned PDF, DOCX, images) and splitting them into clauses.

A clause starts at a line beginning with a number ("5." or "5)"). The text before the first clause is the
preamble (title, date, parties); text after the last clause's paragraph, such as the signature block, is the
closing. Each clause keeps its number so answers and risk flags can cite "clause 5".
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SUPPORTED = (".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg")


class ParseError(ValueError):
    pass


@dataclass
class Clause:
    num: str                 # "0" = preamble, "5" = clause 5, "end" = closing block
    heading: str
    text: str
    start: int               # character offsets in the full document text
    end: int
    label: str = ""
    confidence: float = 0.0

    @property
    def ref(self) -> str:
        return {"0": "preamble", "end": "closing"}.get(self.num, f"clause {self.num}")

    @property
    def body(self) -> str:
        return f"{self.heading}\n{self.text}".strip()

    def public(self) -> dict:
        return asdict(self) | {"ref": self.ref}


@dataclass
class Document:
    text: str
    clauses: list[Clause] = field(default_factory=list)
    source: str = ""
    ocr: bool = False

    @property
    def preamble(self) -> Clause:
        return self.clauses[0]

    def numbered(self) -> list[Clause]:
        return [c for c in self.clauses if c.num not in ("0", "end")]

    def of(self, *labels: str) -> list[Clause]:
        return [c for c in self.clauses if c.label in labels]

    def clause(self, num: str) -> Clause | None:
        return next((c for c in self.clauses if c.num == num), None)


# ------------------------------------------------------------------------------------------- reading
def read_bytes(name: str, data: bytes) -> tuple[str, bool]:
    """Returns (text, used_ocr)."""
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ParseError(f"unsupported file type {suffix or '(none)'}; supported: {', '.join(SUPPORTED)}")
    if suffix in (".txt", ".md"):
        try:
            return data.decode("utf-8"), False
        except UnicodeDecodeError:
            return data.decode("latin-1"), False
    if suffix == ".docx":
        from docx import Document as Docx
        return "\n".join(p.text for p in Docx(io.BytesIO(data)).paragraphs), False
    if suffix == ".pdf":
        import pdfplumber
        pages, used_ocr = [], False
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if len(t.strip()) < 20:          # scanned page: no text layer
                    t, used_ocr = ocr_image(page.to_image(resolution=300).original), True
                pages.append(t)
        return "\n".join(pages), used_ocr
    from PIL import Image
    return ocr_image(Image.open(io.BytesIO(data))), True


def ocr_image(img) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img, config="--psm 4")
    except Exception as exc:  # tesseract binary missing or unreadable image
        raise ParseError(f"OCR failed ({type(exc).__name__}); is Tesseract installed?") from exc


# ---------------------------------------------------------------------------------------- segmenting
_CLAUSE_START = re.compile(r"^\s*(\d{1,2})[.)]\s+(.*)$")
_CLOSING = re.compile(r"^\s*(IN WITNESS|In witness|Signed|Agreed and signed|Executed|A copy of this notice|"
                      r"The Borrower confirms|For and on behalf|Yours)", re.I)


def _split_heading(first_line: str) -> tuple[str, str]:
    """'INTEREST' on its own line is a heading; 'The Loan shall...' is body text."""
    s = first_line.strip()
    if s and len(s) <= 60 and not s.endswith(".") and (s.isupper() or (s.istitle() and len(s.split()) <= 6)):
        return s, ""
    return "", s


def segment(text: str) -> Document:
    text = text.replace("\r\n", "\n")
    if not text.strip():
        raise ParseError("no text found in the document")
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    starts = [i for i, ln in enumerate(lines) if _CLAUSE_START.match(ln)]
    clauses: list[Clause] = []
    first = starts[0] if starts else len(lines)
    pre = "\n".join(lines[:first]).strip()
    clauses.append(Clause("0", "", pre, 0, offsets[first] if first < len(lines) else len(text)))
    for k, i in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else len(lines)
        block = lines[i:nxt]
        m = _CLAUSE_START.match(block[0])
        num, rest = m.group(1), m.group(2)
        heading, first_body = _split_heading(rest)
        body_lines = ([first_body] if first_body else []) + block[1:]
        # the last clause stops at the closing block (signatures, witness lines, advocate's sign-off)
        end_idx = len(body_lines)
        if k + 1 == len(starts):
            seen_text = False
            for j, ln in enumerate(body_lines):
                if not ln.strip() and seen_text:
                    end_idx = j
                    break
                seen_text = seen_text or bool(ln.strip())
        body = "\n".join(body_lines[:end_idx]).strip()
        end_line = i + 1 + end_idx - (1 if first_body else 0)
        end_off = offsets[end_line] if end_line < len(lines) else len(text)
        clauses.append(Clause(num, heading, body, offsets[i], end_off))
        if k + 1 == len(starts):
            closing = "\n".join(body_lines[end_idx:]).strip()
            if closing:
                clauses.append(Clause("end", "", closing, end_off, len(text)))
    if not starts and pre:
        clauses[0].end = len(text)
    return Document(text, clauses)


def load(name: str, data: bytes) -> Document:
    text, used_ocr = read_bytes(name, data)
    doc = segment(text)
    doc.source, doc.ocr = name, used_ocr
    return doc

"""Parsing of the values that matter in Indian legal documents: rupee amounts (including lakh/crore and
Indian digit grouping), percentages with their period, dates in the formats lawyers use, and durations.

Every parser returns the exact matched text alongside the normalised value, so an extracted value can
always be shown with the words it came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty-four": 24,
            "thirty": 30, "sixty": 60, "ninety": 90}


@dataclass
class Match:
    value: object
    text: str
    start: int
    end: int


# ------------------------------------------------------------------------------------------- amounts
_AMOUNT = re.compile(
    r"(?:Rs\.?|INR|₹)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:/-)?\s*(lakhs?|lacs?|crores?)?(?:\s*/-)?", re.I)


def amounts(text: str) -> list[Match]:
    out = []
    for m in _AMOUNT.finditer(text):
        v = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit.startswith(("lakh", "lac")):
            v *= 100_000
        elif unit.startswith("crore"):
            v *= 10_000_000
        out.append(Match(int(round(v)), m.group(0).strip(), m.start(), m.end()))
    return out


def format_inr(v: int | float) -> str:
    """12345678 -> 'Rs 1,23,45,678' (Indian digit grouping)."""
    s = str(int(round(v)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"Rs {s}"


# --------------------------------------------------------------------------------------- percentages
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)(\s*(?:per\s+annum|p\.\s?a\.?|per\s+year|"
                  r"per\s+month|p\.\s?m\.?|a\s+month|per\s+mensem))?", re.I)


def percents(text: str) -> list[Match]:
    """Value is (rate, period) with period 'year', 'month' or ''."""
    out = []
    for m in _PCT.finditer(text):
        per = (m.group(2) or "").lower()
        period = "month" if ("month" in per or "p.m" in per or "mensem" in per) else ("year" if per else "")
        out.append(Match((float(m.group(1)), period), m.group(0).strip(), m.start(), m.end()))
    return out


# --------------------------------------------------------------------------------------------- dates
_D1 = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_D2 = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_D3 = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")


def dates(text: str) -> list[Match]:
    """Indian convention: 05/08/2026 is 5 August 2026."""
    found = []
    for rx, order in ((_D1, "dmy_word"), (_D2, "dmy_num"), (_D3, "mdy_word")):
        for m in rx.finditer(text):
            try:
                if order == "dmy_word":
                    mon = MONTHS.get(m.group(2).lower())
                    d = date(int(m.group(3)), mon, int(m.group(1))) if mon else None
                elif order == "dmy_num":
                    d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                else:
                    mon = MONTHS.get(m.group(1).lower())
                    d = date(int(m.group(3)), mon, int(m.group(2))) if mon else None
            except ValueError:
                d = None
            if d:
                found.append(Match(d, m.group(0), m.start(), m.end()))
    found.sort(key=lambda x: x.start)
    out: list[Match] = []
    for f in found:      # drop overlaps
        if not out or f.start >= out[-1].end:
            out.append(f)
    return out


# ----------------------------------------------------------------------------------------- durations
_DUR = re.compile(r"\b(\d{1,3}|" + "|".join(sorted(WORD_NUM, key=len, reverse=True)) +
                  r")\s*(?:\(\w+\)\s*)?(days?|months?|years?)(?:'s|')?", re.I)


def durations(text: str) -> list[Match]:
    """Value is (number, unit) with unit 'day', 'month' or 'year'."""
    out = []
    for m in _DUR.finditer(text):
        raw = m.group(1).lower()
        num = int(raw) if raw.isdigit() else WORD_NUM[raw]
        unit = m.group(2).lower().rstrip("s")
        out.append(Match((num, unit), m.group(0), m.start(), m.end()))
    return out


def to_days(num: int, unit: str) -> int:
    return num * {"day": 1, "month": 30, "year": 365}[unit]


def to_months(num: int, unit: str) -> int:
    return {"day": max(1, round(num / 30)), "month": num, "year": num * 12}[unit] if unit != "day" else round(num / 30)


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    import calendar
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


# ------------------------------------------------------------------------------------------- tokens
STOP = set("""a an the and or of to in on for at by with from as is are was were be been it its this that these those
shall will may any all such under said such whose which who whom what when where how do does i you my your
your our we he she his her there their them they not no if than then into upon herein thereof hereby""".split())
_TOK = re.compile(r"[a-z0-9]+")


def stem(w: str) -> str:
    if len(w) <= 4 or w.isdigit():
        return w
    for suf in ("ation", "ment", "ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOK.findall(text.lower()) if t not in STOP]


# --------------------------------------------------------------------------------------- sentences
_ABBR = r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bMrs\.)(?<!\bM/s\.)(?<!\bSt\.)(?<!\s[A-Z]\.)"
SENT_END = re.compile(_ABBR + r"(?<=[.;])\s+(?=[A-Z(\"“])")


def split_sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in SENT_END.split(flat) if s.strip()]


def sentence_at(text: str, start: int, end: int) -> str:
    """The sentence containing text[start:end], ignoring full stops in 'Rs.', 'No.', 'Mr.' and similar."""
    bounds = [0] + [m.end() for m in SENT_END.finditer(text)] + [len(text)]
    bounds += [m.end() for m in re.finditer(r"\n", text)]
    bounds = sorted(set(bounds))
    a = max(b for b in bounds if b <= start)
    b = min((x for x in bounds if x > end), default=len(text))
    return " ".join(text[a:b].split())

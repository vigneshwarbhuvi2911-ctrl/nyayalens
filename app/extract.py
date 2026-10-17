"""Field extraction. Each field is looked for only in the clauses where it belongs (the interest rate in the
clause classified as `interest`, the deposit in the `deposit` clause, and so on), which is what makes
simple, auditable rules accurate. Every value carries the exact quote and the clause it came from.

When an LLM is configured, it may fill fields the rules could not find, and its answer is accepted only
if its quote appears in the document and the value can be re-read from that quote (see `llm_fill`).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field as dc_field
from datetime import date

from app.parse import Clause, Document
from app.textutil import Match, add_months, amounts, dates, durations, percents, to_days


@dataclass
class Field:
    name: str
    value: object
    display: str
    quote: str
    clause: str          # "clause 3", "preamble"
    source: str = "rule"  # rule | llm
    extra: dict = dc_field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        if isinstance(self.value, date):
            d["value"] = self.value.isoformat()
        return d


LABELS = {
    "parties": "Parties", "agreement_date": "Date of agreement", "loan_amount": "Loan amount",
    "interest_rate": "Interest rate (p.a.)", "tenure_months": "Tenure (months)", "emi": "EMI",
    "first_emi_date": "First EMI date", "penal_rate_pa": "Penal interest (p.a.)",
    "prepayment_charge": "Prepayment charge", "dispute_city": "Courts / arbitration at",
    "start_date": "Start date", "term_months": "Term (months)", "lock_in_months": "Lock-in (months)",
    "monthly_rent": "Monthly rent", "deposit": "Security deposit", "deposit_months": "Deposit (months of rent)",
    "escalation_pct": "Rent escalation", "notice_days": "Termination notice (days)",
    "deposit_refund_days": "Deposit refund within (days)", "confidentiality": "Confidentiality period",
    "non_compete_months": "Non-compete / non-solicit (months)", "liability_capped": "Liability capped",
    "advocate": "Sent by (advocate)", "client": "On behalf of", "recipient": "Addressed to",
    "notice_date": "Date of notice", "amount_demanded": "Amount demanded", "interest_claimed": "Interest claimed (p.a.)",
    "compliance_days": "Time given to comply (days)",
}

FIELDS = {
    "loan_agreement": ["agreement_date", "loan_amount", "interest_rate", "tenure_months", "emi", "first_emi_date",
                       "penal_rate_pa", "prepayment_charge", "dispute_city"],
    "rent_agreement": ["agreement_date", "start_date", "term_months", "lock_in_months", "monthly_rent", "deposit",
                       "deposit_months", "escalation_pct", "notice_days", "deposit_refund_days", "dispute_city"],
    "nda": ["agreement_date", "confidentiality", "non_compete_months", "liability_capped", "dispute_city"],
    "legal_notice": ["advocate", "client", "recipient", "notice_date", "amount_demanded", "interest_claimed",
                     "compliance_days"],
}


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence (or line) containing text[start:end]."""
    from app.textutil import sentence_at
    return sentence_at(text, start, end)


def _near(matches: list[Match], text: str, keyword: str) -> Match | None:
    """The match closest to a keyword (e.g. the duration nearest to 'lock-in')."""
    kws = [m.start() for m in re.finditer(keyword, text, re.I)]
    if not matches or not kws:
        return None
    return min(matches, key=lambda m: min(abs(m.start - k) for k in kws))


def _f(name, value, display, clause: Clause, m: Match | None = None, quote: str | None = None) -> Field:
    q = quote if quote is not None else (_sentence(clause.text, m.start, m.end) if m else "")
    return Field(name, value, display, q, clause.ref)


def _first(doc: Document, labels: tuple[str, ...], finder, name: str, fmt, keyword: str | None = None):
    for c in doc.of(*labels):
        ms = finder(c.text)
        if keyword:
            m = _near(ms, c.text, keyword)
        else:
            m = ms[0] if ms else None
        if m:
            return _f(name, m.value, fmt(m.value), c, m)
    return None


def _rupees(v) -> str:
    from app.textutil import format_inr
    return format_inr(v)


# ------------------------------------------------------------------------------------------- parties
_ROLE = re.compile(r"\((?:hereinafter[^\"“()]{0,40}?)?(?:the\s+)?[\"“]([A-Z][A-Za-z ]{1,30})[\"”]\)")
_BOUNDARY = re.compile(r"\b(?:between|BETWEEN|AND)\b|(?:^|,|\))\s*and\s+(?=[A-Z])|\n\s*and\s+(?=[A-Z])")
NOT_PARTIES = {"effective date", "premises", "property", "purpose", "agreement", "loan", "facility", "schedule",
               "confidential information", "party", "parties", "emi", "due date", "demised premises"}
HONORIFICS = re.compile(r"^(?:Mr|Mrs|Ms|Dr|Shri|Smt|M/s)\.?\s+", re.I)


def clean_name(s: str) -> str:
    return HONORIFICS.sub("", " ".join(s.split()).strip(" ,.;:")).strip()


def agreement_parties(doc: Document) -> list[dict]:
    pre = doc.preamble.text
    out, last_end = [], 0
    for m in _ROLE.finditer(pre):
        seg = pre[last_end:m.start()]
        b = list(_BOUNDARY.finditer(seg))
        if b:
            seg = seg[b[-1].end():]
        name = re.split(r",|\s\(|\n", seg.strip(), maxsplit=1)[0]
        if name.strip() and m.group(1).strip().lower() not in NOT_PARTIES:
            out.append({"role": m.group(1).strip(), "name": clean_name(name), "quote": " ".join(seg.split())[:160]})
        last_end = m.end()
    return out


def notice_parties(doc: Document) -> dict[str, Field]:
    pre, out = doc.preamble, {}
    adv = re.search(r"^\s*(?:From:?\s*\n\s*)?([A-Z][A-Za-z. ]{2,40}?),?\s*(?:B\.A\.,?\s*B\.L\.?,?)?\s*\n?\s*Advocate",
                    pre.text, re.M)
    if adv:
        nm = adv.group(1).strip()
        nm = nm.title() if nm.isupper() else nm
        out["advocate"] = _f("advocate", nm, nm, pre, quote=" ".join(adv.group(0).split()))
    to = re.search(r"^\s*To:?\s*\n\s*(.+)", pre.text, re.M)
    if to:
        nm = clean_name(to.group(1).rstrip(","))
        out["recipient"] = _f("recipient", nm, nm, pre, quote=to.group(1).strip())
    cl = re.search(r"my client,?\s+(?:M/s\.?\s+)?(.+?)(?:,|\s+I\s|\n)", pre.text)
    if cl:
        nm = clean_name(cl.group(1))
        out["client"] = _f("client", nm, nm, pre, quote=" ".join(cl.group(0).split()))
    return out


# --------------------------------------------------------------------------------------------- extract
def extract(doc: Document, doc_type: str) -> dict[str, Field]:
    f: dict[str, Field] = {}

    def put(field: Field | None):
        if field is not None:
            f[field.name] = field

    pre = doc.preamble
    pre_dates = dates(pre.text)
    if doc_type == "legal_notice":
        f.update(notice_parties(doc))
        if pre_dates:
            put(_f("notice_date", pre_dates[0].value, pre_dates[0].value.strftime("%d %b %Y"), pre, pre_dates[0]))
        put(_first(doc, ("demand",), amounts, "amount_demanded", _rupees))
        put(_first(doc, ("demand",), lambda t: [d for d in durations(t) if d.value[1] == "day"], "compliance_days",
                   lambda v: f"{v[0]} days", keyword="within"))
        if "compliance_days" in f:
            f["compliance_days"].value = f["compliance_days"].value[0]
        put(_first(doc, ("facts", "demand"), percents, "interest_claimed", lambda v: f"{v[0]:g}% p.a."))
        if "interest_claimed" in f:
            r, per = f["interest_claimed"].value
            f["interest_claimed"].value = r * 12 if per == "month" else r
        return f

    if pre_dates:
        put(_f("agreement_date", pre_dates[0].value, pre_dates[0].value.strftime("%d %b %Y"), pre, pre_dates[0]))
    put(_first(doc, ("disputes",), _cities, "dispute_city", str))

    if doc_type == "loan_agreement":
        put(_first(doc, ("principal",), amounts, "loan_amount", _rupees))
        put(_first(doc, ("interest",), percents, "interest_rate", lambda v: f"{v[0]:g}% p.a."))
        put(_first(doc, ("repayment",), amounts, "emi", _rupees))
        put(_first(doc, ("repayment",), dates, "first_emi_date", lambda d: d.strftime("%d %b %Y")))
        put(_first(doc, ("repayment",), _tenure, "tenure_months", lambda v: f"{v} months"))
        put(_first(doc, ("penal_charges",), percents, "penal_rate_pa",
                   lambda v: f"{v[0]:g}% per {'month' if v[1] == 'month' else 'year'}"))
        put(_first(doc, ("prepayment",), _prepay, "prepayment_charge", lambda v: f"{v:g}%"))
        for k in ("interest_rate", "penal_rate_pa"):
            if k in f and isinstance(f[k].value, tuple):
                r, per = f[k].value
                f[k].value = round(r * 12, 2) if per == "month" else r
    elif doc_type == "rent_agreement":
        put(_first(doc, ("term",), dates, "start_date", lambda d: d.strftime("%d %b %Y")))
        put(_first(doc, ("term",), _months, "term_months", lambda v: f"{v} months", keyword=r"\bfor\b|period|term"))
        put(_first(doc, ("term",), _months, "lock_in_months", lambda v: f"{v} months", keyword="lock-in"))
        put(_first(doc, ("rent",), amounts, "monthly_rent", _rupees))
        put(_first(doc, ("deposit",), amounts, "deposit", _rupees))
        put(_first(doc, ("escalation",), percents, "escalation_pct", lambda v: f"{v[0]:g}%"))
        if "escalation_pct" in f:
            f["escalation_pct"].value = f["escalation_pct"].value[0]
        put(_notice_days(doc))
        put(_first(doc, ("deposit",), lambda t: [d for d in durations(t) if d.value[1] == "day"],
                   "deposit_refund_days", lambda v: f"{v[0]} days", keyword="within"))
        if "deposit_refund_days" in f:
            f["deposit_refund_days"].value = f["deposit_refund_days"].value[0]
        if "deposit" in f and "monthly_rent" in f and f["monthly_rent"].value:
            months = round(f["deposit"].value / f["monthly_rent"].value, 1)
            d = f["deposit"]
            f["deposit_months"] = Field("deposit_months", months, f"{months:g} months' rent", d.quote, d.clause)
    elif doc_type == "nda":
        put(_confidentiality(doc))
        put(_first(doc, ("non_compete",), _months, "non_compete_months", lambda v: f"{v} months"))
        caps = doc.of("liability_cap")
        f["liability_capped"] = (_f("liability_capped", True, "Yes", caps[0], quote=_first_sentence(caps[0].text))
                                 if caps else Field("liability_capped", False, "No (no limitation clause)", "", ""))
    return f


def _first_sentence(t: str) -> str:
    return " ".join(t.split()).split(". ")[0].rstrip(".") + "."


def _cities(text: str) -> list[Match]:
    rx = re.compile(r"(?i:courts?\s+(?:at|in)|seat\s+of\s+arbitration\s+shall\s+be|venue\s+of\s+arbitration\s+shall\s+be|"
                    r"arbitration\s+at|seat\s+at)\s+([A-Z][a-z]+)")
    return [Match(m.group(1), m.group(0), m.start(), m.end()) for m in rx.finditer(text)]


def _tenure(text: str) -> list[Match]:
    out = []
    for m in re.finditer(r"(\d{1,3})\s+(?:equated\s+)?monthly\s+instalments", text, re.I):
        out.append(Match(int(m.group(1)), m.group(0), m.start(), m.end()))
    for m in re.finditer(r"tenure\s+of\s+(?:the\s+loan\s+is\s+)?(\d{1,2})\s+years", text, re.I):
        out.append(Match(int(m.group(1)) * 12, m.group(0), m.start(), m.end()))
    return sorted(out, key=lambda m: m.start)


def _months(text: str) -> list[Match]:
    return [Match(d.value[0] * (12 if d.value[1] == "year" else 1), d.text, d.start, d.end)
            for d in durations(text) if d.value[1] in ("month", "year")]


def _prepay(text: str) -> list[Match]:
    m = re.search(r"without\s+any\s+(?:prepayment|foreclosure)\s+charge|no\s+(?:prepayment\s+)?charges?\s+shall|"
                  r"\bnil\b", text, re.I)
    if m:
        return [Match(0.0, m.group(0), m.start(), m.end())]
    return [Match(p.value[0], p.text, p.start, p.end) for p in percents(text)]


def _notice_days(doc: Document) -> Field | None:
    best = None
    for c in doc.of("termination"):
        for d in durations(c.text):
            after = c.text[d.end:d.end + 30].lower()
            if "notice" in after:
                days = to_days(*d.value)
                if best is None or days > best[0]:
                    best = (days, c, d)
    return _f("notice_days", best[0], f"{best[0]} days", best[1], best[2]) if best else None


def _confidentiality(doc: Document) -> Field | None:
    for c in doc.of("term", "confidentiality"):
        m = re.search(r"perpetu\w*|indefinite\w*", c.text, re.I)
        if m:
            return _f("confidentiality", "perpetual", "Perpetual (no end date)", c, Match(None, m.group(0), m.start(), m.end()))
    for c in doc.of("term", "confidentiality"):
        ys = [d for d in durations(c.text) if d.value[1] == "year"]
        m = _near(ys, c.text, r"surviv|confiden") or (ys[0] if ys else None)
        if m:
            after = bool(re.search(r"after\s+(?:its\s+)?(?:expiry|termination)", _sentence(c.text, m.start, m.end), re.I))
            fld = Field("confidentiality", m.value[0],
                        f"{m.value[0]} years" + (" after the agreement ends" if after else ""),
                        _sentence(c.text, m.start, m.end), c.ref)
            if after:
                fld.extra["agreement_years"] = next((d.value[0] for d in ys if d is not m), 0)
            return fld
    return None


# ------------------------------------------------------------------------------------------- timeline
def timeline(doc_type: str, f: dict[str, Field]) -> list[dict]:
    ev = []

    def add(label, d: date | None, basis: str, kind: str = "date"):
        if d:
            ev.append({"label": label, "date": d.isoformat(), "basis": basis, "kind": kind})

    v = {k: x.value for k, x in f.items()}
    if doc_type == "loan_agreement" and isinstance(v.get("first_emi_date"), date):
        add("First EMI due", v["first_emi_date"], f["first_emi_date"].quote)
        if v.get("tenure_months"):
            add("Last EMI due", add_months(v["first_emi_date"], v["tenure_months"] - 1),
                f"First EMI + {v['tenure_months'] - 1} months ({v['tenure_months']} instalments)")
    if doc_type == "rent_agreement" and isinstance(v.get("start_date"), date):
        s = v["start_date"]
        add("Tenancy starts", s, f["start_date"].quote)
        if v.get("lock_in_months"):
            add("Lock-in period ends", add_months(s, v["lock_in_months"]), f"Start + {v['lock_in_months']} months")
        if v.get("term_months"):
            end = add_months(s, v["term_months"])
            add("Agreement expires", end, f"Start + {v['term_months']} months")
            if v.get("notice_days"):
                from datetime import timedelta
                add("Last day to give notice before expiry", end - timedelta(days=v["notice_days"]),
                    f"Expiry - {v['notice_days']} days notice", "deadline")
    if doc_type == "nda" and isinstance(v.get("agreement_date"), date) and isinstance(v.get("confidentiality"), int):
        years = v["confidentiality"] + f["confidentiality"].extra.get("agreement_years", 0)
        add("Confidentiality obligations end", add_months(v["agreement_date"], 12 * years), f["confidentiality"].quote)
    if doc_type == "legal_notice" and isinstance(v.get("notice_date"), date) and v.get("compliance_days"):
        from datetime import timedelta
        add("Deadline to comply / reply", v["notice_date"] + timedelta(days=v["compliance_days"]),
            f"Notice date + {v['compliance_days']} days. The period runs from receipt of the notice, so check the "
            f"date it was received.", "deadline")
    return sorted(ev, key=lambda e: e["date"])


def to_ics(title: str, events: list[dict]) -> str:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//NyayaLens//EN", "CALSCALE:GREGORIAN"]
    for i, e in enumerate(events):
        d = e["date"].replace("-", "")
        summary = f"{e['label']} - {title}".replace(",", "\\,").replace(";", "\\;")
        desc = e["basis"].replace("\n", " ").replace(",", "\\,").replace(";", "\\;")
        lines += ["BEGIN:VEVENT", f"UID:{d}-{i}@nyayalens", f"DTSTAMP:{stamp}", f"DTSTART;VALUE=DATE:{d}",
                  f"SUMMARY:{summary}", f"DESCRIPTION:{desc}"]
        if e["kind"] == "deadline":
            lines += ["BEGIN:VALARM", "TRIGGER:-P3D", "ACTION:DISPLAY", f"DESCRIPTION:{summary}", "END:VALARM"]
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"

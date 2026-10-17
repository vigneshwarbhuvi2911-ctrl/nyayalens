"""Summary, question answering, version comparison, reply drafting and LLM gap-filling.

Every feature works offline with deterministic logic, and uses the LLM when one is configured. LLM output
is checked before it is shown: quotes must exist in the document, and amounts, percentages and dates
must appear in the document (or in the user's instructions). Anything that fails the check is dropped,
and the offline result is used instead.
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from datetime import date

from app.extract import FIELDS, LABELS, Field, _sentence
from app.guardrails import mask_pii
from app.llm import FILL_SYSTEM, QA_SYSTEM, REPLY_SYSTEM, SUMMARY_SYSTEM, LLMClient, LLMError, Usage
from app.parse import Document
from app.textutil import amounts, dates, durations, format_inr, percents, tokens

DISCLAIMER = ("NyayaLens highlights points for review. It is not legal advice; consult an advocate before acting.")


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _facts(text: str) -> set[str]:
    """Amounts, percentages and dates in a text, normalised, for checking LLM output against the document."""
    out = {f"amt:{m.value}" for m in amounts(text)}
    out |= {f"pct:{m.value[0]:g}" for m in percents(text)}
    out |= {f"date:{m.value.isoformat()}" for m in dates(text)}
    return out


def unsupported_facts(output: str, *sources: str) -> set[str]:
    allowed = set().union(*(_facts(s) for s in sources))
    return _facts(output) - allowed


# ---------------------------------------------------------------------------------------- LLM gap fill
KIND = {"agreement_date": "date", "start_date": "date", "first_emi_date": "date", "notice_date": "date",
        "loan_amount": "amount", "emi": "amount", "monthly_rent": "amount", "deposit": "amount",
        "amount_demanded": "amount", "interest_rate": "pct", "penal_rate_pa": "pct", "prepayment_charge": "pct",
        "escalation_pct": "pct", "interest_claimed": "pct", "tenure_months": "months", "term_months": "months",
        "lock_in_months": "months", "non_compete_months": "months", "notice_days": "days",
        "deposit_refund_days": "days", "compliance_days": "days"}


def _value_from_quote(kind: str, quote: str):
    if kind == "date":
        m = dates(quote)
        return (m[0].value, m[0].value.strftime("%d %b %Y")) if m else None
    if kind == "amount":
        m = amounts(quote)
        return (m[0].value, format_inr(m[0].value)) if m else None
    if kind == "pct":
        m = percents(quote)
        if not m:
            return None
        r, per = m[0].value
        return (r * 12 if per == "month" else r), m[0].text
    if kind in ("months", "days"):
        m = durations(quote)
        if not m:
            return None
        n, unit = m[0].value
        val = {"months": {"day": round(n / 30), "month": n, "year": n * 12},
               "days": {"day": n, "month": n * 30, "year": n * 365}}[kind][unit]
        return val, m[0].text
    return None


def llm_fill(llm: LLMClient, doc: Document, doc_type: str, fields: dict[str, Field], usage: Usage) -> list[str]:
    """Asks the model only for fields the rules did not find. A value is accepted only if its quote is in the
    document and the value can be read back from that quote. Returns the names of fields that were filled."""
    missing = [k for k in FIELDS.get(doc_type, []) if k not in fields and k in KIND]
    if llm.is_mock or not missing:
        return []
    masked, _ = mask_pii(doc.text)
    prompt = f"Fields wanted: {', '.join(missing)}\n\nDocument:\n<<<\n{masked[:12000]}\n>>>"
    try:
        raw = llm.complete_json(FILL_SYSTEM, prompt, tier="small", step="fill_fields", usage=usage)
    except LLMError:
        return []
    filled, norm_doc = [], _norm(doc.text)
    for name, item in (raw.get("fields") or {}).items():
        if name not in missing or not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        if not quote or _norm(quote) not in norm_doc:
            continue
        got = _value_from_quote(KIND[name], quote)
        if got is None:
            continue
        clause = next((c.ref for c in doc.clauses if _norm(quote) in _norm(c.text)), "")
        fields[name] = Field(name, got[0], str(got[1]), quote, clause, source="llm")
        filled.append(name)
    return filled


# ---------------------------------------------------------------------------------------------- summary
def offline_summary(doc_type: str, parties: list[dict], f: dict[str, Field]) -> list[str]:
    d = {k: v.display for k, v in f.items()}
    names = " and ".join(f"{p['name']} ({p['role']})" for p in parties)
    out = []
    if doc_type == "loan_agreement":
        out.append(f"Loan agreement between {names}" + (f", dated {d['agreement_date']}." if "agreement_date" in d else "."))
        if "loan_amount" in d:
            out.append(f"Loan of {d['loan_amount']}" + (f" at {d['interest_rate']}" if "interest_rate" in d else "") +
                       (f", repayable in {d['tenure_months']}" if "tenure_months" in d else "") +
                       (f" by EMIs of {d['emi']}" if "emi" in d else "") +
                       (f" starting {d['first_emi_date']}." if "first_emi_date" in d else "."))
        extras = [f"penal interest {d['penal_rate_pa']}" if "penal_rate_pa" in d else "",
                  f"prepayment charge {d['prepayment_charge']}" if "prepayment_charge" in d else "no prepayment clause"]
        out.append("Charges: " + ", ".join(x for x in extras if x) + ".")
    elif doc_type == "rent_agreement":
        out.append(f"Rent agreement between {names}" + (f", dated {d['agreement_date']}." if "agreement_date" in d else "."))
        if "monthly_rent" in d:
            out.append(f"Rent {d['monthly_rent']} a month" + (f", deposit {d['deposit']} ({d['deposit_months']})" if "deposit" in d and "deposit_months" in d else "") + ".")
        if "term_months" in d:
            out.append(f"Term {d['term_months']}" + (f" from {d['start_date']}" if "start_date" in d else "") +
                       (f", lock-in {d['lock_in_months']}" if "lock_in_months" in d else "") +
                       (f", notice {d['notice_days']}" if "notice_days" in d else "") + ".")
    elif doc_type == "nda":
        out.append(f"Non-disclosure agreement between {names}" + (f", dated {d['agreement_date']}." if "agreement_date" in d else "."))
        if "confidentiality" in d:
            out.append(f"Confidentiality period: {d['confidentiality']}.")
        if "non_compete_months" in d:
            out.append(f"Restriction on competing or soliciting: {d['non_compete_months']}.")
        out.append(f"Liability capped: {d.get('liability_capped', 'not stated')}.")
    elif doc_type == "legal_notice":
        out.append(f"Legal notice from {d.get('advocate', 'an advocate')}, for {d.get('client', 'the client')}, "
                   f"to {d.get('recipient', 'the recipient')}" + (f", dated {d['notice_date']}." if "notice_date" in d else "."))
        if "amount_demanded" in d:
            out.append(f"Demands {d['amount_demanded']}" + (f" with interest at {d['interest_claimed']}" if "interest_claimed" in d else "") +
                       (f" within {d['compliance_days']}." if "compliance_days" in d else "."))
    return out


def llm_summary(llm: LLMClient, doc: Document, usage: Usage) -> list[str] | None:
    if llm.is_mock:
        return None
    masked, _ = mask_pii(doc.text)
    try:
        raw = llm.complete_json(SUMMARY_SYSTEM, f"Document:\n<<<\n{masked[:12000]}\n>>>", tier="small",
                                step="summary", usage=usage)
    except LLMError:
        return None
    bullets = [str(b) for b in raw.get("bullets") or [] if str(b).strip()]
    if not bullets or unsupported_facts(" ".join(bullets), doc.text):
        return None
    return bullets


# ------------------------------------------------------------------------------------------------- Q&A
INTENTS = [
    (r"what happens if|consequence|if i (?:do not|don't|miss|fail)|\bmiss(?:ed)?\b|default", ["default", "penal_charges", "consequences"], None),
    (r"prepay|foreclos|repay\b.*\bearly|close\b.*\bearly|\bearly\b|part[- ]pay", ["prepayment"], "prepayment_charge"),
    (r"emi amount|instal?ment amount|how much is (?:the |each |my )?(?:emi|instal)", ["repayment"], "emi"),
    (r"tenure|how many (?:emi|instal|months)", ["repayment"], "tenure_months"),
    (r"first emi|emi start|when (?:does|do|will) .*(?:emi|instal)", ["repayment"], "first_emi_date"),
    (r"set[- ]?off|\blien\b|other accounts|take (?:my )?money", ["set_off"], None),
    (r"guarant|security(?! deposit)|collateral|mortgage", ["security"], None),
    (r"interest rate|rate of interest|\binterest\b(?! claimed)", ["interest"], "interest_rate"),
    (r"penal|late (?:payment|fee)|overdue|delay", ["penal_charges"], "penal_rate_pa"),
    (r"\bemi\b|instal?ment|repay", ["repayment"], None),
    (r"loan amount|how much.*(?:loan|borrow)|sanction", ["principal"], "loan_amount"),
    (r"deposit|advance", ["deposit"], "deposit"),
    (r"lock[- ]?in", ["term"], "lock_in_months"),
    (r"increase|escalat|hike|go(?:es)? up|raise", ["escalation"], "escalation_pct"),
    (r"\brent\b", ["rent"], "monthly_rent"),
    (r"notice period|terminate|termination|vacate|end the (?:lease|agreement)", ["termination"], "notice_days"),
    (r"repair|maintenance|electricity|water|property tax", ["maintenance"], None),
    (r"court|jurisdiction|arbitrat|dispute", ["disputes"], "dispute_city"),
    (r"confidential", ["term", "confidentiality"], "confidentiality"),
    (r"compet|solicit|\bjoin\b", ["non_compete"], "non_compete_months"),
    (r"liabilit|indemn", ["liability_cap", "indemnity"], None),
    (r"how much.*(?:pay|owe|demand|claim)|amount (?:demanded|claimed)", ["demand"], "amount_demanded"),
    (r"deadline|how many days|by when|last date|time to (?:pay|reply)", ["demand"], "compliance_days"),
    (r"\bwant|demand|asking|call(?:s|ed)? upon", ["demand"], None),
]


def answer(question: str, doc: Document, fields: dict[str, Field], llm: LLMClient, usage: Usage,
           min_score: float = 0.25) -> dict:
    ranked = _rank(question, doc)
    if not llm.is_mock and ranked:
        res = _llm_answer(question, ranked[:4], doc, llm, usage)
        if res:
            return res
    q = question.lower()
    for pattern, labels, field in INTENTS:
        if re.search(pattern, q):
            if field and field in fields and fields[field].quote:
                f = fields[field]
                text = f"{LABELS.get(field, field)}: {f.display}".rstrip(".") + "."
                if field == "compliance_days" and isinstance(getattr(fields.get("notice_date"), "value", None), date):
                    from datetime import timedelta
                    due = fields["notice_date"].value + timedelta(days=f.value)
                    text = (f"Within {f.value} days of receiving the notice. Counted from the notice date, that is "
                            f"{due.strftime('%d %b %Y')}; count from the actual date of receipt.")
                return _ans(text, [(f.clause, f.quote)], "field")
            cands = doc.of(*labels)[:2]
            if cands:
                sents = [(c.ref, _short_answer(question, c.text)) for c in cands]
                return _ans(" ".join(s for _, s in sents), sents, "clause")
            break
    if ranked and ranked[0][1] >= min_score:
        c = ranked[0][0]
        s = _best_sentence(question, c.text)
        return _ans(s, [(c.ref, s)], "clause")
    return {"answerable": False, "answer": "I could not find this in the document.", "citations": [], "source": "none"}


def _ans(text: str, cites: list[tuple[str, str]], source: str) -> dict:
    return {"answerable": True, "answer": text, "citations": [{"clause": c, "quote": q} for c, q in cites],
            "source": source}


def _rank(question: str, doc: Document) -> list[tuple]:
    clauses = doc.numbered()
    toks = [Counter(tokens(c.body)) for c in clauses]
    df = Counter(t for tc in toks for t in tc)
    n = len(clauses) or 1
    q = set(tokens(question))
    if not q:
        return []
    out = []
    for c, tc in zip(clauses, toks):
        w = sum(math.log(1 + n / (1 + df[t])) for t in q)
        hit = sum(math.log(1 + n / (1 + df[t])) for t in q if t in tc)
        out.append((c, hit / w if w else 0.0))
    return sorted(out, key=lambda x: -x[1])


def _short_answer(question: str, text: str) -> str:
    """A short clause (up to 3 sentences) is quoted whole; otherwise the sentence that best matches."""
    from app.textutil import split_sentences
    sents = split_sentences(text)
    return " ".join(sents) if len(sents) <= 3 else _best_sentence(question, text)


def _best_sentence(question: str, text: str) -> str:
    q = set(tokens(question))
    from app.textutil import split_sentences
    sents = split_sentences(text)
    return max(sents, key=lambda s: (len(q & set(tokens(s))), -len(s))) if sents else text


def _llm_answer(question: str, ranked: list[tuple], doc: Document, llm: LLMClient, usage: Usage) -> dict | None:
    ctx = "\n\n".join(f"[{c.ref}] {c.body}" for c, _ in ranked)
    masked, _ = mask_pii(ctx)
    try:
        raw = llm.complete_json(QA_SYSTEM, f"Clauses:\n<<<\n{masked}\n>>>\n\nQuestion: {question}", tier="small",
                                step="answer", usage=usage)
    except LLMError:
        return None
    if raw.get("answerable") is False:
        return {"answerable": False, "answer": "I could not find this in the document.", "citations": [],
                "source": "llm"}
    refs = {c.ref: c for c, _ in ranked}
    kept, cites = [], []
    for s in raw.get("sentences") or []:
        text = str(s.get("text", "")).strip()
        cited = [refs[r] for r in s.get("clauses") or [] if r in refs]
        if text and cited and not unsupported_facts(text, *(c.text for c in cited)):
            kept.append(text)
            cites += [(c.ref, " ".join(c.text.split())) for c in cited]
    if not kept:
        return None
    return _ans(" ".join(kept), list(dict.fromkeys(cites)), "llm")


# --------------------------------------------------------------------------------------------- compare
def compare(a: dict, b: dict) -> dict:
    """Differences between two analyses (for example, draft v1 and v2 of the same agreement)."""
    fa, fb = a["fields"], b["fields"]
    field_changes = []
    for k in list(dict.fromkeys(list(fa) + list(fb))):
        va, vb = fa.get(k, {}).get("display"), fb.get(k, {}).get("display")
        if va != vb:
            field_changes.append({"field": k, "label": LABELS.get(k, k), "before": va, "after": vb})
    ca = {c["label"]: c for c in a["clauses"] if c["num"] not in ("0", "end")}
    cb = {c["label"]: c for c in b["clauses"] if c["num"] not in ("0", "end")}
    clause_changes = []
    for label in list(dict.fromkeys(list(ca) + list(cb))):
        x, y = ca.get(label), cb.get(label)
        if x and y and _norm(x["text"]) == _norm(y["text"]):
            continue
        clause_changes.append({
            "label": label, "before_ref": x["ref"] if x else None, "after_ref": y["ref"] if y else None,
            "status": "added" if not x else "removed" if not y else "changed",
            "diff": _word_diff(x["text"] if x else "", y["text"] if y else "")})
    ia, ib = {f["id"]: f for f in a["risks"]}, {f["id"]: f for f in b["risks"]}
    return {"field_changes": field_changes, "clause_changes": clause_changes,
            "new_risks": [ib[i] for i in ib if i not in ia], "resolved_risks": [ia[i] for i in ia if i not in ib],
            "changed_risks": [ib[i] for i in ib if i in ia and ib[i]["severity"] != ia[i]["severity"]]}


def _word_diff(a: str, b: str) -> list[dict]:
    wa, wb = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append({"op": "same", "text": " ".join(wa[i1:i2])})
        if op in ("delete", "replace"):
            out.append({"op": "del", "text": " ".join(wa[i1:i2])})
        if op in ("insert", "replace"):
            out.append({"op": "add", "text": " ".join(wb[j1:j2])})
    return out


# ------------------------------------------------------------------------------------------ reply draft
def reply_draft(doc: Document, fields: dict[str, Field], instructions: str, llm: LLMClient, usage: Usage,
                today: date) -> dict:
    v = {k: f.display for k, f in fields.items()}
    template = "\n".join([
        "DRAFT - for review and completion by an advocate. Not legal advice.",
        "",
        f"Date: {today.strftime('%d %b %Y')}",
        "",
        "To",
        f"{v.get('advocate', '[name of advocate]')}, Advocate" + (f", for {v['client']}" if "client" in v else ""),
        "",
        f"Sub: Reply to your notice dated {v.get('notice_date', '[date of notice]')}",
        "",
        "Sir/Madam,",
        "",
        f"Under instructions from my client, {v.get('recipient', '[name of client]')}, I reply to your notice as follows:",
        "",
        f"1. My client received your notice on [date of receipt].",
        "2. My client denies all allegations in the notice except those specifically admitted in this reply.",
        f"3. Regarding the demand for {v.get('amount_demanded', '[amount]')}: [state my client's position, with supporting facts and documents].",
        ("4. The claim for interest at " + v["interest_claimed"] + " is not admitted, as no such rate was agreed. [confirm with client]")
        if "interest_claimed" in v else "4. [Any further points in reply].",
        "5. " + (f"My client's instructions: {instructions.strip()}" if instructions.strip() else "[Proposed way forward, for example a settlement or a request for documents]."),
        "",
        "This reply is sent without prejudice to my client's rights and remedies, all of which are reserved.",
        "",
        "[Advocate's name and signature]",
    ])
    result = {"draft": template, "source": "template", "warnings": []}
    if llm.is_mock:
        return result
    masked, _ = mask_pii(doc.text)
    try:
        raw = llm.complete_json(REPLY_SYSTEM, f"Notice:\n<<<\n{masked[:10000]}\n>>>\n\nClient's instructions: "
                                f"{instructions or '(none given)'}\n\nToday's date: {today.isoformat()}",
                                tier="large", step="reply", usage=usage, max_tokens=1200)
        draft = str(raw.get("draft", "")).strip()
    except LLMError:
        return result | {"warnings": ["The model was unavailable, so the template draft is shown."]}
    bad = unsupported_facts(draft, doc.text, instructions, today.isoformat(), today.strftime("%d %B %Y"))
    if not draft or bad:
        return result | {"warnings": [f"The model's draft contained figures not found in the notice "
                                      f"({', '.join(sorted(bad))}), so the template draft is shown."]}
    return {"draft": "DRAFT - for review and completion by an advocate. Not legal advice.\n\n" + draft,
            "source": "llm", "warnings": []}

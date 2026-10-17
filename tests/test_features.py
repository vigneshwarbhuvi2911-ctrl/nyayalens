"""Q&A, comparison, reply drafts, and every LLM path with its checks."""
from app.llm import LLMError
from tests.conftest import sample


def test_questions_answered_from_fields_and_clauses(engine, analysed):
    rid = analysed["LOAN-1"]["id"]
    a = engine.ask(rid, "What is the interest rate?")
    assert a["answer"] == "Interest rate (p.a.): 21% p.a." and a["citations"][0]["clause"] == "clause 3"
    a = engine.ask(rid, "What happens if I miss an EMI?")
    assert "3% per month" in a["answer"] and {c["clause"] for c in a["citations"]} == {"clause 6", "clause 7"}
    a = engine.ask(analysed["NOTICE-1"]["id"], "By when do I have to pay?")
    assert "17 Oct 2026" in a["answer"] and "actual date of receipt" in a["answer"]


def test_questions_outside_the_document_are_declined(engine, analysed):
    a = engine.ask(analysed["RENT-1"]["id"], "Can I keep a pet dog?")
    assert not a["answerable"] and a["citations"] == []


def test_compare_versions(engine, analysed):
    c = engine.compare(analysed["RENT-1"]["id"], analysed["RENT-1-v2"]["id"])
    changes = {x["field"]: (x["before"], x["after"]) for x in c["field_changes"]}
    assert changes["deposit"] == ("Rs 2,50,000", "Rs 3,00,000") and changes["lock_in_months"] == ("6 months", "11 months")
    assert {x["id"] for x in c["new_risks"]} == {"R4", "R6"} and [x["id"] for x in c["changed_risks"]] == ["R1"]
    term = next(x for x in c["clause_changes"] if x["label"] == "termination")
    assert {"op": "add", "text": "at any time"} in term["diff"] and {"op": "del", "text": "one month's"} in term["diff"]


def test_reply_template_only_uses_document_facts(engine, analysed):
    d = engine.reply(analysed["NOTICE-1"]["id"], "Client disputes the interest rate")
    assert d["source"] == "template" and d["draft"].startswith("DRAFT")
    assert "K. Venkatesh, Advocate, for Thamarai Textiles" in d["draft"] and "[date of receipt]" in d["draft"]
    assert "Client disputes the interest rate" in d["draft"]
    import pytest
    with pytest.raises(ValueError):
        engine.reply(analysed["LOAN-1"]["id"])


# ------------------------------------------------------------------------------------------- LLM paths
def test_llm_fills_only_grounded_missing_fields(llm_engine):
    e, llm = llm_engine({"fill_fields": {"fields": {
        "prepayment_charge": {"value": "2%", "quote": "prepayment is allowed with 2% charge"},     # not in document
        "loan_amount": {"value": "Rs 99", "quote": "Rs. 99/-"}}},                                 # not requested
        "summary": {"bullets": []}})
    r = e.analyze("LOAN-3.txt", sample("LOAN-3"))
    assert r["llm_filled"] == [] and "prepayment_charge" not in r["fields"]
    assert r["fields"]["loan_amount"]["value"] == 1200000
    asked = next(c for c in llm.calls if c["step"] == "fill_fields")["user"]
    assert asked.startswith("Fields wanted: prepayment_charge")


def test_llm_fill_reads_the_value_from_the_quote(llm_engine):
    text = sample("RENT-2").decode().replace("15% at the end of every 12 months", "fifteen percent yearly")
    e, _ = llm_engine({"fill_fields": {"fields": {"escalation_pct": {
        "value": "20%", "quote": "The monthly rent shall stand increased by fifteen percent yearly."}}}})
    r = e.analyze_text("rent.txt", text)
    assert "escalation_pct" not in r["fields"]           # quote has no figure the parser can verify -> rejected
    e2, _ = llm_engine({"fill_fields": {"fields": {"escalation_pct": {
        "value": "20%", "quote": "Rs. 40,000/- in advance"}}}})
    r2 = e2.analyze_text("rent.txt", text)
    assert "escalation_pct" not in r2["fields"]          # quote exists but is not a percentage


def test_llm_summary_rejected_if_it_invents_numbers(llm_engine):
    e, _ = llm_engine({"summary": {"bullets": ["Loan of Rs 6,00,000 at 21% p.a."]}})
    r = e.analyze("LOAN-1.txt", sample("LOAN-1"))
    assert r["summary_source"] == "template"
    e2, _ = llm_engine({"summary": {"bullets": ["A Rs 5,00,000 personal loan at 21% p.a. over 36 EMIs."]}})
    r2 = e2.analyze("LOAN-1.txt", sample("LOAN-1"))
    assert r2["summary_source"] == "llm" and r2["usage"]["calls"] >= 1


def test_no_contact_details_reach_the_model(llm_engine):
    e, llm = llm_engine({"summary": {"bullets": []}})
    e.analyze("NDA-2.txt", sample("NDA-2"))
    e.analyze("LOAN-1.txt", sample("LOAN-1"))
    sent = " ".join(c["user"] for c in llm.calls)
    assert "farah.sheikh@example.com" not in sent and "9840012345" not in sent and "[EMAIL]" in sent


def test_llm_answer_checked_against_cited_clause(llm_engine, analysed):
    good = {"answerable": True, "sentences": [{"text": "Penal interest is 3% per month.", "clauses": ["clause 6"]}]}
    bad = {"answerable": True, "sentences": [{"text": "Penal interest is 5% per month.", "clauses": ["clause 6"]}]}
    e, _ = llm_engine({"answer": good})
    assert e.ask(analysed["LOAN-1"]["id"], "What is the penalty for late payment?")["source"] == "llm"
    e2, _ = llm_engine({"answer": bad})
    a = e2.ask(analysed["LOAN-1"]["id"], "What is the penalty for late payment?")
    assert a["source"] != "llm" and "3% per month" in a["answer"]       # falls back to the checked answer
    e3, _ = llm_engine({"answer": LLMError("timeout")})
    assert e3.ask(analysed["LOAN-1"]["id"], "What is the interest rate?")["answerable"]


def test_llm_reply_rejected_if_it_invents_amounts(llm_engine, analysed):
    e, llm = llm_engine({"reply": {"draft": "My client offers Rs. 1,00,000/- in full settlement."}})
    d = e.reply(analysed["NOTICE-1"]["id"], "Offer a settlement")
    assert d["source"] == "template" and "amt:100000" in d["warnings"][0]
    assert llm.calls[-1]["tier"] == "large"
    e2, _ = llm_engine({"reply": {"draft": "My client disputes the claim of Rs. 3,75,000/- and the interest at 24% per annum."}})
    d2 = e2.reply(analysed["NOTICE-1"]["id"], "")
    assert d2["source"] == "llm" and d2["draft"].startswith("DRAFT")

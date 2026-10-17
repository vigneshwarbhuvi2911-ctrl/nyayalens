"""Classification, extraction, risk checks, timeline and calendar on the sample documents."""
from datetime import date
from pathlib import Path

import yaml

from app.parse import segment
from app.risks import Playbook
from tests.conftest import ROOT

EXPECTED_TYPES = {"LOAN": "loan_agreement", "RENT": "rent_agreement", "NDA": "nda", "NOTICE": "legal_notice"}


def test_document_types(analysed):
    for name, r in analysed.items():
        assert r["doc_type"] == EXPECTED_TYPES[name.split("-")[0]], name
        assert r["doc_type_confidence"] >= 0.5


def test_unrelated_text_is_unknown(engine):
    r = engine.analyze_text("recipe.txt", "Tomato rasam. 1. Boil tamarind water with salt. 2. Add tomatoes, "
                                          "pepper and cumin. 3. Temper with mustard seeds and curry leaves.")
    assert r["doc_type"] == "unknown" and r["risks"] == [] and r["fields"] == {}


def test_clause_labels_respect_document_type(analysed):
    labels = [c["label"] for c in analysed["LOAN-1"]["clauses"] if c["num"] not in ("0", "end")]
    assert labels == ["definitions", "principal", "interest", "repayment", "prepayment", "penal_charges", "default",
                      "disputes", "notices"]
    assert all(c["label"] != "deposit" for c in analysed["LOAN-2"]["clauses"])


def test_fields_have_values_quotes_and_clauses(analysed):
    f = analysed["LOAN-1"]["fields"]
    assert f["loan_amount"]["value"] == 500000 and f["loan_amount"]["display"] == "Rs 5,00,000"
    assert f["interest_rate"]["value"] == 21.0 and f["interest_rate"]["clause"] == "clause 3"
    assert f["penal_rate_pa"]["value"] == 36.0 and f["penal_rate_pa"]["display"] == "3% per month"
    assert f["interest_rate"]["quote"] == "The Loan shall carry interest at the rate of 21% per annum, calculated on the reducing balance."
    assert analysed["LOAN-2"]["fields"]["loan_amount"]["value"] == 4500000           # "Rs. 45 lakh"
    assert analysed["LOAN-2"]["fields"]["prepayment_charge"]["value"] == 0.0         # "without any prepayment charge"
    assert "prepayment_charge" in analysed["LOAN-3"]["missing_fields"]


def test_parties(analysed):
    assert [(p["role"], p["name"]) for p in analysed["LOAN-2"]["parties"]] == [
        ("Lender", "Nilgiri Housing Finance Limited"), ("Borrower", "Meena Krishnan")]
    assert [p["name"] for p in analysed["NDA-1"]["parties"]] == ["Zentrix Software Private Limited", "Orbis Analytics LLP"]
    n = analysed["NOTICE-2"]["fields"]
    assert (n["advocate"]["value"], n["client"]["value"], n["recipient"]["value"]) == ("R. Priya", "Vel Agencies", "P. Murugan")


def test_rent_terms_and_derived_values(analysed):
    f = analysed["RENT-2"]["fields"]
    assert (f["lock_in_months"]["value"], f["term_months"]["value"], f["notice_days"]["value"]) == (24, 36, 90)
    assert f["deposit_months"]["value"] == 10.0 and "deposit_refund_days" not in f


def test_risk_flags_with_evidence(analysed):
    ids = {k: [x["id"] for x in r["risks"]] for k, r in analysed.items()}
    assert ids["LOAN-1"] == ["L1", "L2", "L4", "L3", "L7"]           # sorted by severity
    assert ids["LOAN-2"] == [] and ids["RENT-3"] == [] and ids["NDA-1"] == []
    assert set(ids["RENT-2"]) == {"R1", "R2", "R3", "R4", "R6", "R7"}
    assert set(ids["NDA-2"]) == {"N1", "N2", "N4", "N5"}
    l4 = next(x for x in analysed["LOAN-1"]["risks"] if x["id"] == "L4")
    assert l4["clause"] == "clause 3" and "sole discretion" in l4["quote"]
    r1 = next(x for x in analysed["RENT-1"]["risks"] if x["id"] == "R1")
    assert r1["severity"] == "medium" and "10 months of rent" in r1["why"]


def test_deadline_passed_depends_on_today(analysed):
    assert "D2" in [x["id"] for x in analysed["NOTICE-2"]["risks"]]      # due 17 Sep 2026, today 17 Oct 2026
    assert "D2" not in [x["id"] for x in analysed["NOTICE-1"]["risks"]]  # due today


def test_playbook_thresholds_are_configuration(engine, tmp_path):
    rules = yaml.safe_load((ROOT / "data" / "playbook.yaml").read_text())
    next(r for r in rules["loan_agreement"] if r["id"] == "L1")["params"]["limit"] = 8
    p = tmp_path / "pb.yaml"
    p.write_text(yaml.safe_dump(rules))
    doc = segment((ROOT / "data/samples/LOAN-2.txt").read_text())
    engine.models.label_clauses(doc, "loan_agreement")
    from app.extract import extract
    flags = Playbook(Path(p)).run(doc, "loan_agreement", extract(doc, "loan_agreement"), date(2026, 10, 17))
    assert [f["id"] for f in flags] == ["L1"]


def test_timeline_and_calendar(engine, analysed):
    tl = {e["label"]: e["date"] for e in analysed["RENT-1"]["timeline"]}
    assert tl == {"Tenancy starts": "2026-06-01", "Lock-in period ends": "2026-12-01",
                  "Last day to give notice before expiry": "2027-04-01", "Agreement expires": "2027-05-01"}
    n = analysed["NOTICE-1"]
    assert n["timeline"][0]["date"] == "2026-10-17" and n["timeline"][0]["kind"] == "deadline"
    ics = engine.calendar(n["id"])
    assert ics.startswith("BEGIN:VCALENDAR") and "DTSTART;VALUE=DATE:20261017" in ics and "BEGIN:VALARM" in ics
    assert ics.count("BEGIN:VEVENT") == 1 and ics.endswith("END:VCALENDAR\r\n")


def test_summary_offline(analysed):
    s = analysed["LOAN-1"]["summary"]
    assert s[0] == "Loan agreement between Kaveri Finance Limited (Lender) and Arun Prakash (Borrower), dated 05 Mar 2026."
    assert analysed["LOAN-1"]["pii_detected"] == ["PHONE"]

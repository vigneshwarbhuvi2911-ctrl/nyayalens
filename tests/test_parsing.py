"""Value parsers, reading files (TXT, DOCX, PDF, scanned PDF) and clause segmentation."""
import io
from datetime import date

import pytest

from app.parse import ParseError, load, segment
from app.textutil import amounts, dates, durations, format_inr, percents, sentence_at, split_sentences
from tests.conftest import ROOT, sample, tesseract


def test_amounts_understand_indian_formats():
    got = [m.value for m in amounts("Rs. 5,00,000/- and Rs. 45 lakh and INR 2,50,000 and ₹1.5 crore and Rs 300")]
    assert got == [500000, 4500000, 250000, 15000000, 300]
    assert format_inr(12345678) == "Rs 1,23,45,678" and format_inr(500) == "Rs 500"


def test_percent_periods_and_dates_in_indian_order():
    assert [m.value for m in percents("21% per annum, 3% per month, 2.5 percent p.a., 15%")] == [
        (21.0, "year"), (3.0, "month"), (2.5, "year"), (15.0, "")]
    got = [m.value for m in dates("5th day of March, 2026; 12/01/2026; Date: 10.10.2026; August 20, 2025")]
    assert got == [date(2026, 3, 5), date(2026, 1, 12), date(2026, 10, 10), date(2025, 8, 20)]
    assert dates("31/02/2026") == []                                  # invalid dates are ignored


def test_durations_including_words():
    assert [m.value for m in durations("one month's notice, 30 days' notice, 3 months, 2 years")] == [
        (1, "month"), (30, "day"), (3, "month"), (2, "year")]


def test_sentences_ignore_abbreviations_and_initials():
    t = "The loan is guaranteed by Mrs. G. Lakshmi under a deed. Liability shall not exceed Rs. 25,00,000/-. Done."
    assert split_sentences(t)[0] == "The loan is guaranteed by Mrs. G. Lakshmi under a deed."
    i = t.find("25,00,000")
    assert sentence_at(t, i, i + 9) == "Liability shall not exceed Rs. 25,00,000/-."


def test_segmentation_preamble_clauses_closing():
    d = load("LOAN-1.txt", sample("LOAN-1"))
    assert [c.num for c in d.clauses] == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "end"]
    assert d.clause("3").heading == "INTEREST" and d.clause("3").text.startswith("The Loan shall carry interest")
    assert "Kaveri Finance Limited" in d.preamble.text and d.clauses[-1].text.startswith("IN WITNESS")
    n = load("NOTICE-1.txt", sample("NOTICE-1"))
    assert n.clause("4").heading == "" and n.clause("5").text.endswith("costs and consequences.")
    assert n.clauses[-1].text.startswith("A copy of this notice")


def test_docx_and_text_pdf_are_read():
    from docx import Document
    d = Document()
    for line in sample("RENT-3").decode().splitlines():
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    doc = load("lease.docx", buf.getvalue())
    assert doc.clause("3").text.startswith("The monthly rent shall be Rs. 60,000/-")

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in sample("NOTICE-3").decode().splitlines():
        c.drawString(40, y, line[:110])
        y -= 14
    c.save()
    pdf = load("notice.pdf", buf.getvalue())
    assert not pdf.ocr and pdf.clause("3") is not None


@tesseract
def test_scanned_pdf_goes_through_ocr():
    doc = load("scan.pdf", (ROOT / "data" / "scanned" / "NOTICE-2-scanned.pdf").read_bytes())
    assert doc.ocr and "Rs. 1,50,000" in doc.text and len(doc.numbered()) == 4


def test_bad_input():
    with pytest.raises(ParseError, match="unsupported"):
        load("x.exe", b"MZ")
    with pytest.raises(ParseError, match="no text"):
        segment("   \n  ")

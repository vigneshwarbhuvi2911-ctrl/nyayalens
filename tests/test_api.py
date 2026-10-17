import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import sample


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine))


def test_health_and_samples(client):
    assert client.get("/health").json()["status"] == "ok"
    names = client.get("/samples").json()
    assert "LOAN-1.txt" in names and len(names) == 13
    assert client.post("/samples/..%2Fplaybook.yaml/analyze").status_code == 404


def test_upload_analyse_ask_calendar(client):
    r = client.post("/analyze", files={"file": ("notice.txt", sample("NOTICE-1"), "text/plain")}).json()
    assert r["doc_type"] == "legal_notice" and r["fields"]["amount_demanded"]["value"] == 375000
    assert client.get(f"/analyses/{r['id']}").json()["id"] == r["id"]
    a = client.post(f"/analyses/{r['id']}/ask", json={"question": "How much am I asked to pay?"}).json()
    assert "3,75,000" in a["answer"]
    ics = client.get(f"/analyses/{r['id']}/calendar.ics")
    assert ics.headers["content-type"].startswith("text/calendar") and "BEGIN:VEVENT" in ics.text
    d = client.post(f"/analyses/{r['id']}/reply", json={"instructions": "Ask for 60 days"}).json()
    assert d["draft"].startswith("DRAFT")
    assert client.get("/analyses").json()[0]["id"] == r["id"]


def test_errors(client):
    assert client.post("/analyze", files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 422
    assert client.get("/analyses/nope").status_code == 404
    assert client.post("/analyze/text", json={"text": "short"}).status_code == 422
    loan = client.post("/samples/LOAN-1.txt/analyze").json()
    assert client.post(f"/analyses/{loan['id']}/reply", json={}).status_code == 422


def test_compare_endpoint(client):
    a = client.post("/samples/RENT-1.txt/analyze").json()["id"]
    b = client.post("/samples/RENT-1-v2.txt/analyze").json()["id"]
    c = client.get(f"/compare?a={a}&b={b}").json()
    assert c["same_type"] and {x["id"] for x in c["new_risks"]} == {"R4", "R6"}


def test_web_page(client):
    assert "NyayaLens" in client.get("/").text


@pytest.mark.parametrize("ok", [True])
def test_evaluation_gates(ok):
    from eval.run import GATES, run
    os.environ.pop("OPENAI_API_KEY", None)
    s = run("mock")["summary"]
    for k, v in GATES.items():
        assert s[k] >= v, (k, s[k])

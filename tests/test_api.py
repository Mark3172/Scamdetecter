import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


BANK_SCAM = (
    "URGENT: Your bank account has been locked. "
    "Verify immediately at http://secure-bank-login.com"
)
DINNER = "Hey, are we still on for dinner at 7 tonight?"


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_web_ui_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Scam Message Detector" in response.text


def test_detect_scam_includes_explanation(client):
    response = client.post("/api/v1/detect", json={"text": BANK_SCAM})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Scam"
    assert body["original_text"] == BANK_SCAM
    assert 0 <= body["confidence_score"] <= 1
    assert 0 <= body["scam_probability"] <= 1
    assert body["advice"]
    assert isinstance(body["signals"], list)
    assert "cues" in body
    assert any(cue["id"] == "url" for cue in body["cues"])
    assert body["highlights"]
    assert any(span["term"] == "link / URL" for span in body["highlights"])


def test_detect_legitimate(client):
    response = client.post("/api/v1/detect", json={"text": DINNER})
    assert response.status_code == 200
    assert response.json()["prediction"] == "Legitimate"


def test_detect_rejects_whitespace(client):
    response = client.post("/api/v1/detect", json={"text": "   "})
    assert response.status_code == 422


def test_batch_detect(client):
    response = client.post(
        "/api/v1/detect/batch",
        json={"messages": [{"text": BANK_SCAM}, {"text": DINNER}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scam_count"] == 1
    assert body["legitimate_count"] == 1
    assert len(body["results"]) == 2


def test_examples_endpoint(client):
    response = client.get("/api/v1/examples")
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 4
    assert {"id", "title", "text"} <= set(body[0].keys())


def test_batch_csv_download(client):
    response = client.post(
        "/api/v1/detect/batch.csv",
        json={"messages": [{"text": BANK_SCAM}, {"text": DINNER}]},
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "Scam" in response.text
    assert "Legitimate" in response.text


def test_feedback_and_stats(client):
    stats = client.get("/api/v1/model/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["corpus_size"] >= 20
    assert body["accuracy"] >= 0.9

    marked = client.post(
        "/api/v1/feedback",
        json={"text": DINNER, "predicted": "Legitimate", "actual": "Legitimate"},
    )
    assert marked.status_code == 200
    assert marked.json()["record"]["correct"] is True
    summary = client.get("/api/v1/feedback")
    assert summary.status_code == 200
    assert summary.json()["count"] >= 1

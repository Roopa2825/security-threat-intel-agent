"""API tests for endpoints that need no LLM or database.
Skipped automatically if FastAPI is not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_check_url_flags_typosquatting():
    response = client.post("/check-url", json={"url": "http://paypa1.com/login/verify"})
    assert response.status_code == 200
    assert response.json()["verdict"] == "high risk"


def test_ask_rejects_too_short_question():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_invalid_cve_id_returns_400():
    assert client.get("/cve/log4shell").status_code == 400

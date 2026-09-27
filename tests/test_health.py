from fastapi.testclient import TestClient

from conduix.app import app


def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["openai_codex"] == "0.157.1"

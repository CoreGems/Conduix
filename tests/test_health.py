from fastapi.testclient import TestClient

from conduix import app as app_module
from conduix.app import app


class FakeBackend:
    def __init__(self, status):
        self._status = status

    async def status(self):
        return self._status


def test_health_ok(monkeypatch):
    monkeypatch.setattr(app_module, "backend", FakeBackend({
        "codex": "ok", "codex_version": "0.157.1", "logged_in": True,
        "account_type": "chatgpt", "plan_type": "plus",
    }))
    # No `with`: the lifespan (which starts real Codex) doesn't run.
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["plan_type"] == "plus"


def test_health_degraded_when_codex_down(monkeypatch):
    monkeypatch.setattr(app_module, "backend", FakeBackend({"codex": "error", "detail": "boom"}))
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["detail"] == "boom"

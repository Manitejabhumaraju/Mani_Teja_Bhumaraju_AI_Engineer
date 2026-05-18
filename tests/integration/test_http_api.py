"""
FastAPI HTTP-surface tests.

We don't spin up the lifespan (which boots MCP + LLM) - we test only the
REST handlers in isolation. The lifespan is needed for the WS handler
which calls into runner; we skip live WS tests and rely on the graph
flow tests for graph correctness.

Pattern: use TestClient with a no-op lifespan override so we don't need
NVIDIA_API_KEY to test auth/session endpoints.
"""

import pytest
from contextlib import asynccontextmanager
from fastapi.testclient import TestClient

from app.auth import auth_store
from app.sessions import session_manager


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


@pytest.fixture
def client():
    # Import here so we can monkey-patch the lifespan first
    from app.main import app
    # Replace lifespan with a no-op so TestClient doesn't try to boot
    # MCP/LLM/Redis. The REST endpoints don't need any of that.
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.router.lifespan_context = original


@pytest.fixture(autouse=True)
def reset_state():
    # Clear cross-test state
    auth_store._tokens.clear()  # type: ignore[attr-defined]
    session_manager._by_user.clear()  # type: ignore[attr-defined]
    yield


class TestHealthAndAuth:
    def test_health_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True

    def test_login_good_creds(self, client):
        r = client.post("/api/login", json={"username": "demo", "password": "demo"})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == "demo"
        assert body["token"]

    def test_login_bad_creds(self, client):
        r = client.post("/api/login", json={"username": "demo", "password": "wrong"})
        assert r.status_code == 401

    def test_login_unknown_user(self, client):
        r = client.post("/api/login", json={"username": "nobody", "password": "x"})
        assert r.status_code == 401

    def test_logout_revokes_token(self, client):
        r = client.post("/api/login", json={"username": "demo", "password": "demo"})
        token = r.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Authenticated call works
        assert client.get("/api/sessions", headers=headers).status_code == 200

        # Logout
        assert client.post("/api/logout", headers=headers).status_code == 200

        # Same token now rejected
        assert client.get("/api/sessions", headers=headers).status_code == 401


class TestSessionsCRUD:
    def _login(self, client, user="demo"):
        r = client.post("/api/login", json={"username": user, "password": user})
        return r.json()["token"]

    def test_initial_session_list_empty(self, client):
        token = self._login(client)
        r = client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json() == {"sessions": []}

    def test_create_then_list(self, client):
        token = self._login(client)
        headers = {"Authorization": f"Bearer {token}"}
        c = client.post("/api/sessions", json={}, headers=headers)
        assert c.status_code == 200
        sid = c.json()["session"]["session_id"]

        r = client.get("/api/sessions", headers=headers)
        assert r.status_code == 200
        sessions = r.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sid

    def test_named_session(self, client):
        token = self._login(client)
        headers = {"Authorization": f"Bearer {token}"}
        r = client.post("/api/sessions", json={"name": "Q4 review"}, headers=headers)
        assert r.json()["session"]["name"] == "Q4 review"

    def test_rename(self, client):
        token = self._login(client)
        headers = {"Authorization": f"Bearer {token}"}
        sid = client.post("/api/sessions", json={}, headers=headers).json()["session"]["session_id"]
        r = client.post(
            f"/api/sessions/{sid}/rename",
            json={"name": "Renamed"},
            headers=headers,
        )
        assert r.status_code == 200
        assert r.json()["session"]["name"] == "Renamed"

    def test_delete(self, client):
        token = self._login(client)
        headers = {"Authorization": f"Bearer {token}"}
        sid = client.post("/api/sessions", json={}, headers=headers).json()["session"]["session_id"]
        r = client.delete(f"/api/sessions/{sid}", headers=headers)
        assert r.status_code == 200
        assert client.get("/api/sessions", headers=headers).json()["sessions"] == []

    def test_user_isolation(self, client):
        """demo's sessions invisible to analyst, and vice versa."""
        demo_token = self._login(client, "demo")
        analyst_token = self._login(client, "analyst")
        client.post(
            "/api/sessions", json={"name": "Demo's session"},
            headers={"Authorization": f"Bearer {demo_token}"},
        )
        client.post(
            "/api/sessions", json={"name": "Analyst's session"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        demo_list = client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {demo_token}"},
        ).json()["sessions"]
        analyst_list = client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {analyst_token}"},
        ).json()["sessions"]

        assert len(demo_list) == 1
        assert demo_list[0]["name"] == "Demo's session"
        assert len(analyst_list) == 1
        assert analyst_list[0]["name"] == "Analyst's session"


class TestAuthRequired:
    def test_sessions_without_auth_401(self, client):
        assert client.get("/api/sessions").status_code == 401

    def test_sessions_with_bad_token_401(self, client):
        r = client.get(
            "/api/sessions",
            headers={"Authorization": "Bearer junk"},
        )
        assert r.status_code == 401

    def test_sessions_without_bearer_prefix_401(self, client):
        r = client.get(
            "/api/sessions",
            headers={"Authorization": "Basic foo"},
        )
        assert r.status_code == 401

"""
Locust load profile for the Loadshare RCA Agent.

Run:
  locust -f load/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to set users / spawn rate.

The chat endpoint is WebSocket and Locust isn't great at WS. Two
approaches we use:

  1. ChatUser exercises the REST surface (login + session list/create).
     Easy, fast, useful as a baseline for the HTTP path.

  2. ChatWSUser uses gevent + websocket-client to do a one-shot WS turn
     per task. Less throughput-realistic than a long-lived WS but
     better than nothing and demonstrates the streaming endpoint under
     concurrent load.

Metrics we care about:
  - p50/p95/p99 latency for /api/login and /api/sessions
  - p95 wall clock for a full WS turn (open + send + drain to 'final')
  - failure rate
"""

from __future__ import annotations

import json
import random
import time

import gevent
import websocket
from locust import HttpUser, between, events, task


SAMPLE_QUERIES = [
    "How did Bangalore do on 2026-04-22?",
    "What about STORE_003?",
    "Walk me through the morning hours.",
    "Tell me about Mumbai.",
    "How did Chennai do?",
    "Why was STORE_004 slow?",
    "Evening hours please.",
]

CREDENTIALS = [("demo", "demo"), ("analyst", "analyst")]


class ChatUser(HttpUser):
    """Exercises the REST surface - login, list sessions, create session."""
    wait_time = between(1, 3)

    def on_start(self):
        username, password = random.choice(CREDENTIALS)
        r = self.client.post("/api/login", json={"username": username, "password": password})
        if r.status_code != 200:
            self.environment.runner.quit()
            return
        self.token = r.json()["token"]
        self.user_id = r.json()["user_id"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

        # Create one session up front
        sr = self.client.post("/api/sessions", json={}, headers=self.headers)
        self.session_id = sr.json()["session"]["session_id"] if sr.status_code == 200 else None

    @task(3)
    def list_sessions(self):
        self.client.get("/api/sessions", headers=self.headers)

    @task(1)
    def create_then_delete_session(self):
        r = self.client.post("/api/sessions", json={}, headers=self.headers, name="/api/sessions [POST]")
        if r.status_code == 200:
            sid = r.json()["session"]["session_id"]
            self.client.delete(f"/api/sessions/{sid}", headers=self.headers, name="/api/sessions/{id} [DELETE]")

    @task(2)
    def health_check(self):
        self.client.get("/api/health")


class ChatWSUser(HttpUser):
    """One-shot WebSocket turn per task. Slower than HTTP, more realistic."""
    wait_time = between(2, 5)

    def on_start(self):
        username, password = random.choice(CREDENTIALS)
        r = self.client.post("/api/login", json={"username": username, "password": password})
        if r.status_code != 200:
            self.environment.runner.quit()
            return
        self.token = r.json()["token"]
        self.user_id = r.json()["user_id"]
        sr = self.client.post(
            "/api/sessions", json={},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.session_id = sr.json()["session"]["session_id"] if sr.status_code == 200 else None

        # Build WS URL from the HTTP host
        host = self.host.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{host}/ws/chat?token={self.token}&session_id={self.session_id}"

    @task
    def one_chat_turn(self):
        if not self.session_id:
            return
        message = random.choice(SAMPLE_QUERIES)
        start = time.time()
        exc = None
        try:
            ws = websocket.create_connection(self.ws_url, timeout=30)
            ws.send(json.dumps({"message": message}))
            # Drain until 'final'
            while True:
                raw = ws.recv()
                if not raw:
                    break
                ev = json.loads(raw)
                if ev.get("type") == "final":
                    break
            ws.close()
        except Exception as e:
            exc = e

        elapsed_ms = (time.time() - start) * 1000
        events.request.fire(
            request_type="WS",
            name="/ws/chat one-turn",
            response_time=elapsed_ms,
            response_length=len(message),
            exception=exc,
        )

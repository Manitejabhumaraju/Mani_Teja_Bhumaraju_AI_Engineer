"""
Session manager.

Per-user collection of sessions. Each session has:
  - session_id: stable UUID-ish string
  - name: user-friendly label ("Session 1", or whatever they rename it to)
  - created_at, last_active_at

LangGraph's MemorySaver checkpointer keys conversation state on
thread_id = "{user_id}:{session_id}", so it stays naturally isolated.

This module just tracks the metadata (which sessions exist for whom).
The actual conversation history lives in the graph checkpointer.

In-memory dict, lost on restart. Brief says no persistence required.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional


log = logging.getLogger(__name__)


@dataclass
class SessionMeta:
    session_id: str
    user_id: str
    name: str
    created_at: float
    last_active_at: float

    def to_dict(self) -> dict:
        return asdict(self)


class SessionManager:
    """Per-user session registry. Thread-safe."""

    def __init__(self):
        self._by_user: dict[str, dict[str, SessionMeta]] = {}
        self._lock = threading.Lock()

    def list_for_user(self, user_id: str) -> list[SessionMeta]:
        with self._lock:
            sessions = list(self._by_user.get(user_id, {}).values())
        # Most recently active first
        sessions.sort(key=lambda s: -s.last_active_at)
        return sessions

    def create(self, user_id: str, name: Optional[str] = None) -> SessionMeta:
        session_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            user_sessions = self._by_user.setdefault(user_id, {})
            default_name = name or f"Session {len(user_sessions) + 1}"
            meta = SessionMeta(
                session_id=session_id,
                user_id=user_id,
                name=default_name,
                created_at=now,
                last_active_at=now,
            )
            user_sessions[session_id] = meta
        log.info("created session %s for user=%s", session_id, user_id)
        return meta

    def touch(self, user_id: str, session_id: str) -> Optional[SessionMeta]:
        """Update last_active_at. Returns the meta if found, None otherwise."""
        with self._lock:
            user_sessions = self._by_user.get(user_id, {})
            meta = user_sessions.get(session_id)
            if meta is None:
                return None
            meta.last_active_at = time.time()
            return meta

    def get(self, user_id: str, session_id: str) -> Optional[SessionMeta]:
        with self._lock:
            return self._by_user.get(user_id, {}).get(session_id)

    def delete(self, user_id: str, session_id: str) -> bool:
        with self._lock:
            user_sessions = self._by_user.get(user_id, {})
            return user_sessions.pop(session_id, None) is not None

    def rename(self, user_id: str, session_id: str, new_name: str) -> bool:
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        with self._lock:
            user_sessions = self._by_user.get(user_id, {})
            meta = user_sessions.get(session_id)
            if meta is None:
                return False
            meta.name = new_name[:80]   # cap to keep UI sane
            return True


session_manager = SessionManager()

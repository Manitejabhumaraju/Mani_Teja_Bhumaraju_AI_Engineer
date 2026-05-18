"""
Mock authentication.

Hardcoded credentials, in-memory token store. The brief says auth is out
of scope; this is just enough to demonstrate multi-user isolation in the
walkthrough.

Tokens are random URL-safe strings, stored {token -> user_id} server-side.
On server restart, all tokens invalidate (intentionally - state is in
memory by design).

If we ever needed real auth, the replacement points are well-marked:
  - verify_credentials() -> swap for bcrypt + DB lookup
  - issue_token()        -> swap for JWT signing
  - resolve_token()      -> swap for JWT verification

DO NOT use this for anything that touches real users.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.config import settings


log = logging.getLogger(__name__)


TOKEN_TTL_SECONDS = 8 * 3600   # 8 hours - generous for a demo session


@dataclass
class TokenRecord:
    user_id: str
    issued_at: float
    last_seen: float


class AuthStore:
    """
    Thread-safe in-memory token store. FastAPI workers share a process so
    a plain dict + lock is fine. If we ever ran multi-worker we'd need
    Redis-backed storage; flagged in README.
    """

    def __init__(self):
        self._tokens: dict[str, TokenRecord] = {}
        self._lock = threading.Lock()

    def verify_credentials(self, username: str, password: str) -> bool:
        expected = settings.demo_users.get(username)
        if expected is None:
            return False
        # Constant-time compare - paranoid for a mock but the right habit.
        return secrets.compare_digest(expected, password)

    def issue_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            self._tokens[token] = TokenRecord(
                user_id=user_id, issued_at=now, last_seen=now,
            )
        log.info("issued token for user=%s", user_id)
        return token

    def resolve_token(self, token: Optional[str]) -> Optional[str]:
        """Returns user_id if token is valid + not expired, else None."""
        if not token:
            return None
        now = time.time()
        with self._lock:
            rec = self._tokens.get(token)
            if rec is None:
                return None
            if now - rec.issued_at > TOKEN_TTL_SECONDS:
                # Lazy expiry - clean up on access.
                self._tokens.pop(token, None)
                log.info("token expired for user=%s", rec.user_id)
                return None
            rec.last_seen = now
            return rec.user_id

    def revoke_token(self, token: str) -> bool:
        with self._lock:
            return self._tokens.pop(token, None) is not None


# Module-level singleton. Imported by FastAPI handlers.
auth_store = AuthStore()

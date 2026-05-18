"""
Transaction log — durable record of every chat turn.

Two reasons this exists:

1. Session continuity. The LangGraph MemorySaver is in-process; if the
   backend restarts, every prior turn is gone. Saving turns to SQLite means
   that switching back to Session 1 after talking in Session 2 can re-load
   the full message history.

2. Audit + analytics. "What was my last question?" is a legitimate user
   query that the RCA path can't answer (it's not in the orders_gold data).
   The transaction log can.

Schema is intentionally narrow: one row per (user, session, turn). Columns
follow the user's spec — user_id, question, response, plus session_id and
a timestamp formatted YYYY-MM-DD/HH:MM for readability.

Stored separately from data/loadshare.db so the RCA dataset stays read-only
and the transaction log can grow without touching it.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_PATH = "data/transactions.db"


class TransactionLog:
    """Append-only log of chat turns, keyed by user and session."""

    def __init__(self, db_path: str = DEFAULT_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # New connection per call — SQLite + threads + long-lived connections
        # is a known footgun. WAL mode lets concurrent reads happen while
        # one writer is active, which is plenty for this scale.
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT    NOT NULL,
                    session_id  TEXT    NOT NULL,
                    question    TEXT    NOT NULL,
                    response    TEXT    NOT NULL,
                    intent      TEXT,
                    ts_unix     REAL    NOT NULL,
                    ts_pretty   TEXT    NOT NULL
                )
            """)
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_user_session
                ON turns(user_id, session_id, ts_unix)
            """)
            c.commit()

    def record(
        self,
        *,
        user_id: str,
        session_id: str,
        question: str,
        response: str,
        intent: Optional[str] = None,
    ) -> int:
        """Insert one turn. Returns the inserted row id."""
        now = time.time()
        pretty = datetime.fromtimestamp(now).strftime("%Y-%m-%d/%H:%M")
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO turns "
                "(user_id, session_id, question, response, intent, ts_unix, ts_pretty) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, question, response, intent or "", now, pretty),
            )
            c.commit()
            row_id = cur.lastrowid
        log.debug("logged turn %d for user=%s session=%s", row_id, user_id, session_id)
        return row_id

    def history_for_session(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """All turns for a session, oldest first. Powers session resume."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT question, response, intent, ts_pretty "
                "FROM turns "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY ts_unix ASC "
                "LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        return [
            {
                "question": q,
                "response": r,
                "intent": intent,
                "ts": ts,
            }
            for q, r, intent, ts in rows
        ]

    def last_question(self, user_id: str, session_id: str) -> Optional[dict]:
        """Most recent turn for a session. Powers 'what did I just ask' queries."""
        with self._conn() as c:
            row = c.execute(
                "SELECT question, response, intent, ts_pretty "
                "FROM turns "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY ts_unix DESC "
                "LIMIT 1",
                (user_id, session_id),
            ).fetchone()
        if row is None:
            return None
        q, r, intent, ts = row
        return {"question": q, "response": r, "intent": intent, "ts": ts}

    def recent_questions(
        self,
        user_id: str,
        session_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """Last N questions for 'what have I been asking' queries."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT question, intent, ts_pretty "
                "FROM turns "
                "WHERE user_id = ? AND session_id = ? "
                "ORDER BY ts_unix DESC "
                "LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        return [
            {"question": q, "intent": intent, "ts": ts}
            for q, intent, ts in rows
        ]


# Singleton — same pattern as session_manager and auth_store
transaction_log = TransactionLog()

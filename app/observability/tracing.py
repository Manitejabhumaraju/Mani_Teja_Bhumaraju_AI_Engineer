"""
Observability.

Two things this module does:

1. setup_tracing(): one-call init for LangSmith. If LANGCHAIN_TRACING_V2 is
   set in env, LangGraph and the LangChain core libs auto-trace - we just
   verify the env vars are wired correctly and log the project name.

2. log_turn_metrics(state, latency_ms): emits a structured log line per
   turn so we can grep / parse / pipe to a log aggregator. Even without
   LangSmith, this gives us the per-turn data the eval harness needs:
   nodes_executed, intent, tokens, cache hit, latency.

We deliberately don't wrap nodes with @traceable - LangGraph already does
that under the hood when LANGCHAIN_TRACING_V2=true. Re-decorating would
just nest spans.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from app.config import settings


log = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGCHAIN_TRACING_V2", "").strip().lower() in ("1", "true", "yes"))


def setup_tracing() -> None:
    """
    Validate LangSmith env, log status. Call once at app startup.

    LangChain auto-traces when these env vars are set:
      LANGCHAIN_TRACING_V2=true
      LANGCHAIN_API_KEY=lsv2_...
      LANGCHAIN_PROJECT=loadshare-rca-agent   (optional, sensible default)
    """
    if not tracing_enabled():
        log.info("LangSmith tracing OFF (set LANGCHAIN_TRACING_V2=true to enable)")
        return

    key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not key:
        log.warning(
            "LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is empty - "
            "traces will not be uploaded. Get a key at https://smith.langchain.com"
        )
        return

    project = os.environ.get("LANGCHAIN_PROJECT") or settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = project
    log.info("LangSmith tracing ON, project=%s", project)


def log_turn_metrics(
    state: dict[str, Any],
    *,
    latency_ms: float,
    extra: dict | None = None,
) -> None:
    """
    Emit a structured per-turn log line.

    Used by the FastAPI handler after run_turn completes. Format is JSON
    on one line so log shippers can parse it.
    """
    rec = {
        "ts": time.time(),
        "user_id": state.get("user_id"),
        "session_id": state.get("session_id"),
        "intent": state.get("intent"),
        "nodes_executed": state.get("nodes_executed") or [],
        "input_guard_decision": state.get("input_guard_decision"),
        "output_guard_decision": state.get("output_guard_decision"),
        "cache_hit": bool(state.get("cache_hit")),
        "latency_ms": round(latency_ms, 1),
        "response_chars": len(state.get("final_response") or ""),
        "report_chars": len(state.get("report") or ""),
        "error": state.get("error"),
    }
    if extra:
        rec.update(extra)

    # Single-line JSON for grep / jq friendliness.
    log.info("turn_metric %s", json.dumps(rec, default=str))

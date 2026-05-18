"""
Evaluation and observability metrics.

CAST framework:
  C — Correctness:  does RCA identify causes correctly vs playbook ground truth
  A — Accuracy:     are numbers grounded in the deterministic report
  S — Speed:        latency P50/P95 across turns
  T — Trust:        DAST block rate, output guard catch rate

All metrics are emitted as structured log lines (JSON-serialisable dicts)
so they can be ingested by any log aggregator or LangSmith dashboard.
"""

from __future__ import annotations

import time
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger("app.metrics")


# ── CAST metrics accumulator ──────────────────────────────────────────────────

@dataclass
class TurnMetrics:
    session_id: str
    turn_number: int
    intent: str
    latency_ms: float
    cache_hit: bool
    input_blocked: bool
    output_blocked: bool
    nodes_executed: list[str]
    city: Optional[str] = None
    store: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class SessionMetricsCollector:
    """
    Collects per-turn metrics for one session.
    Computes CAST summary at end of session.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[TurnMetrics] = []

    def record(self, turn: TurnMetrics) -> None:
        self.turns.append(turn)
        log.info("TURN_METRIC %s", turn.to_dict())

    def cast_summary(self) -> dict:
        if not self.turns:
            return {}

        latencies = [t.latency_ms for t in self.turns]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        # Speed (S)
        p50_latency = latencies_sorted[n // 2]
        p95_latency = latencies_sorted[max(0, int(n * 0.95) - 1)]

        # Trust (T)
        input_block_rate = sum(1 for t in self.turns if t.input_blocked) / n
        output_block_rate = sum(1 for t in self.turns if t.output_blocked) / n
        cache_hit_rate = sum(1 for t in self.turns if t.cache_hit) / n

        # Accuracy (A) — proxy: output not blocked = grounded response
        accuracy_proxy = 1.0 - output_block_rate

        return {
            "session_id": self.session_id,
            "turn_count": n,
            "cast": {
                "C_correctness": "requires_ground_truth",   # needs human eval
                "A_accuracy_proxy": round(accuracy_proxy, 3),
                "S_latency_p50_ms": round(p50_latency, 1),
                "S_latency_p95_ms": round(p95_latency, 1),
                "T_input_block_rate": round(input_block_rate, 3),
                "T_output_block_rate": round(output_block_rate, 3),
                "T_cache_hit_rate": round(cache_hit_rate, 3),
            },
        }


# ── Global log-based metrics helper ───────────────────────────────────────────

def log_turn_metrics(state: dict, latency_ms: float) -> None:
    """
    Emit a structured log line from a completed graph state.
    Called from the WebSocket handler after each turn completes.
    """
    ctx = state.get("context") or {}
    log.info(
        "TURN_COMPLETE | intent=%s | city=%s | store=%s | "
        "latency_ms=%.1f | cache_hit=%s | "
        "input_guard=%s | output_guard=%s | nodes=%s",
        state.get("intent", "?"),
        ctx.get("last_city") or ctx.get("city"),
        ctx.get("last_store") or ctx.get("store"),
        latency_ms,
        state.get("cache_hit", False),
        state.get("input_guard_decision", "?"),
        state.get("output_guard_decision", "?"),
        "->".join(state.get("nodes_executed") or []),
    )

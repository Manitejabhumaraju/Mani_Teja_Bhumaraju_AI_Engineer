"""
Eval runner.

Two suites:
  - eval/conversation_suite.jsonl: multi-turn correctness. Needs a live LLM.
  - eval/adversarial.jsonl: input-guard coverage. Pure deterministic, fast.

Usage:
    python eval/run_eval.py                 # both
    python eval/run_eval.py --only dast     # only adversarial
    python eval/run_eval.py --only convo    # only conversation

The conversation suite calls the real NIM endpoint - it costs LLM tokens
per turn. The DAST suite hits only the regex guard so it costs nothing
and runs in milliseconds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.guardrails import check_input, Decision   # noqa: E402
from app.config import settings                     # noqa: E402


logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("eval")


# === DAST suite =======================================================

def _build_dast_input(case: dict) -> str:
    """Some cases have input_repeat/count to test length limits."""
    if "input_repeat" in case:
        return case["input_repeat"] * case.get("input_repeat_count", 1)
    return case.get("input", "")


def run_dast(path: Path) -> tuple[int, int, list[dict]]:
    """Returns (passed, total, failures)."""
    total = 0
    passed = 0
    failures = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        total += 1

        msg = _build_dast_input(case)
        verdict = check_input(msg)
        expected_blocker = case.get("expect_blocked_by")

        if expected_blocker is None:
            # Control case: must be allowed
            ok = verdict.allowed
        else:
            # Attack: must be blocked
            ok = not verdict.allowed

        if ok:
            passed += 1
        else:
            failures.append({
                "id": case["id"],
                "category": case.get("category"),
                "verdict": verdict.decision.value,
                "reason": verdict.reason,
                "input_sample": msg[:120],
            })

    return passed, total, failures


# === Conversation suite ===============================================

async def run_conversation(path: Path) -> tuple[int, int, list[dict]]:
    """Build a real GraphRunner against the live LLM and walk each scenario."""
    # Lazy imports - heavy
    from app.llm.nim_client import NimClient
    from app.rca.store_resolver import StoreResolver
    from app.graph import build_graph, GraphRunner
    from app.graph.nodes import NodeDeps

    try:
        llm = NimClient()
    except RuntimeError as e:
        print(f"SKIP conversation suite: {e}")
        return 0, 0, []

    store_resolver = StoreResolver.from_sqlite(str(settings.db_path))
    deps = NodeDeps(
        llm_client=llm,
        db_path=str(settings.db_path),
        store_resolver=store_resolver,
        cache=None,
        mcp_manager=None,
    )
    compiled = build_graph(deps)
    runner = GraphRunner(compiled, deps, cache=None)

    total = 0
    passed = 0
    failures = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        case = json.loads(line)
        case_id = case["id"]
        # Fresh session per case so coreference doesn't bleed across cases
        session_id = f"eval-{case_id}-{int(time.time())}"

        case_pass = True
        case_failures = []

        for i, turn in enumerate(case.get("turns", [])):
            user_msg = turn["user"]
            try:
                result = await runner.run_turn(
                    user_id="eval_user",
                    session_id=session_id,
                    user_message=user_msg,
                )
            except Exception as e:
                case_pass = False
                case_failures.append(f"turn {i}: exception {e}")
                break

            exp = turn.get("expect", {})
            if "intent" in exp and result.get("intent") != exp["intent"]:
                case_pass = False
                case_failures.append(
                    f"turn {i}: intent {result.get('intent')!r} != expected {exp['intent']!r}"
                )

            for needle in exp.get("report_contains", []):
                if needle.lower() not in (result.get("report") or "").lower():
                    case_pass = False
                    case_failures.append(
                        f"turn {i}: report missing substring {needle!r}"
                    )

            for k, v in exp.get("context", {}).items():
                ctx = result.get("context") or {}
                if ctx.get(k) != v:
                    case_pass = False
                    case_failures.append(
                        f"turn {i}: context.{k}={ctx.get(k)!r} != expected {v!r}"
                    )

        total += 1
        if case_pass:
            passed += 1
        else:
            failures.append({"id": case_id, "failures": case_failures})

    return passed, total, failures


# === Main =============================================================

def print_summary(name: str, passed: int, total: int, failures: list[dict]):
    pct = (passed / total * 100) if total else 0
    print(f"\n=== {name}: {passed}/{total} passed ({pct:.1f}%) ===")
    for f in failures:
        print(f"  FAIL {f['id']}: {f.get('failures') or f.get('reason')}")


async def amain(args):
    repo_root = Path(__file__).resolve().parent.parent

    if args.only in (None, "dast"):
        dast_path = repo_root / "eval" / "adversarial.jsonl"
        p, t, fails = run_dast(dast_path)
        print_summary("DAST", p, t, fails)
        if fails and args.fail_on_error:
            return 1

    if args.only in (None, "convo"):
        convo_path = repo_root / "eval" / "conversation_suite.jsonl"
        p, t, fails = await run_conversation(convo_path)
        print_summary("Conversation", p, t, fails)
        if fails and args.fail_on_error:
            return 1

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["dast", "convo"], default=None)
    parser.add_argument("--fail-on-error", action="store_true",
                        help="exit non-zero if any case fails")
    args = parser.parse_args()
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()

"""
Graph nodes — fixed for production quality.

Key fixes vs original:
  1. Context resolver: city stored as both `city` AND `last_city` so downstream
     nodes always find it. No merge_context — each turn replaces city/store
     completely unless user is explicitly referencing "same" entity.
  2. City/store lookup: uses match_city() which normalises spaces so
     "mumbai" → "Mumbai  north" (two spaces) works correctly.
  3. mem0ai: injected into context resolver (reads prior memories for
     coreference hints) and persist_turn (writes new memories after each turn).
  4. P50/P95: surfaced in city and store reports from aggregations.
  5. Free-form SQLite MCP node: unchanged (used when WSL/Linux, skipped otherwise).
"""

from __future__ import annotations

import json
import re
import logging
import time
from typing import Any, Optional

from app.graph.state import GraphState, ConversationContext
from app.guardrails import check_input, check_output, Decision
from app.llm.schemas import ContextOutput, IntentOutput, SQLOutput
from app.llm.prompts import (
    CONTEXT_RESOLVER_SYSTEM,
    INTENT_CLASSIFIER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    REFUSAL_TEMPLATE,
)
from app.rca import aggregations, engine, time_ranges
from app.rca.store_resolver import StoreResolver


log = logging.getLogger(__name__)

DEFAULT_DATE = "2026-04-22"


# ── Dependencies ──────────────────────────────────────────────────────────────

class NodeDeps:
    """Carries shared dependencies into each node. Built once at app boot."""

    def __init__(
        self,
        *,
        llm_client,
        db_path: str,
        store_resolver: StoreResolver,
        cache=None,
        mcp_manager=None,
        memory=None,       # mem0ai Memory instance or None
    ):
        self.llm = llm_client
        self.db_path = db_path
        self.stores = store_resolver
        self.cache = cache
        self.mcp = mcp_manager
        self.memory = memory   # may be None if mem0ai unavailable


def _append_node(state: GraphState, name: str) -> list[str]:
    existing = list(state.get("nodes_executed") or [])
    existing.append(name)
    return existing


# ── Node 1: input guard ───────────────────────────────────────────────────────

def make_input_guard_node(_deps: NodeDeps):

    async def input_guard_node(state: GraphState) -> dict:
        msg = state.get("user_message", "")
        verdict = check_input(msg)
        out: dict = {
            "input_guard_decision": verdict.decision.value,
            "input_guard_reason": verdict.reason,
            "nodes_executed": _append_node(state, "input_guard"),
        }
        if not verdict.allowed:
            out["final_response"] = REFUSAL_TEMPLATE
            log.info("input blocked: %s", verdict.reason)
        return out

    return input_guard_node


# ── Node 2: context resolver (with mem0ai) ────────────────────────────────────

def make_context_resolver_node(deps: NodeDeps):

    async def context_resolver_node(state: GraphState) -> dict:
        prior: ConversationContext = state.get("context") or {}  # type: ignore[assignment]
        user_msg = state.get("user_message", "")
        user_id = state.get("user_id", "default")
        session_id = state.get("session_id", "default")

        cities = deps.stores.known_cities()
        ranges = time_ranges.known_ranges()

        # ── mem0ai: pull relevant memories ──────────────────────────────────
        memory_hint = ""
        if deps.memory is not None:
            try:
                results = deps.memory.search(user_msg, user_id=user_id, limit=3)
                mems = results.get("results", results) if isinstance(results, dict) else results
                if mems:
                    memory_hint = "Session memory:\n" + "\n".join(
                        f"  - {m.get('memory', m) if isinstance(m, dict) else m}"
                        for m in mems[:3]
                    ) + "\n\n"
            except Exception as e:
                log.debug("mem0ai search failed (non-fatal): %s", e)

        # ── Build system prompt ──────────────────────────────────────────────
        cities_list = "\n".join(f"  - {c}" for c in cities)
        system = CONTEXT_RESOLVER_SYSTEM.format(
            cities=cities_list,
            time_ranges=", ".join(ranges),
        )

        # ── Build user prompt with CRITICAL entity-replacement rule ──────────
        prior_city = prior.get("last_city") or prior.get("city")
        prior_store = prior.get("last_store") or prior.get("store")
        prior_date = prior.get("last_date") or prior.get("date")

        prior_note = ""
        if prior_city or prior_store:
            parts = []
            if prior_city:
                parts.append(f"city={prior_city!r}")
            if prior_store:
                parts.append(f"store={prior_store!r}")
            if prior_date:
                parts.append(f"date={prior_date!r}")
            prior_note = f"[Previous turn: {', '.join(parts)}]\n\n"

        user_prompt = (
            f"{memory_hint}"
            f"{prior_note}"
            f'Message: "{user_msg}"\n\n'
            f"PARSE THIS. If user mentions a city name, match it to the known cities "
            f"list and output the EXACT name from the list (including spaces). "
            f"Do NOT keep prior city/store unless user says 'same', 'that', 'it'. "
            f"'what about X' / 'tell me about X' / 'same way for X' → X is NEW entity."
        )

        try:
            result = await deps.llm.complete_json(
                system=system, user=user_prompt,
                temperature=0.0, max_tokens=256,
            )
        except Exception as e:
            log.exception("context resolver LLM call failed: %s", e)
            return {
                "error": "context resolver unavailable",
                "nodes_executed": _append_node(state, "context_resolver"),
            }

        # ── Validate LLM output via Pydantic ──────────────────────────────────
        try:
            validated_ctx = ContextOutput.model_validate(result)
            result = validated_ctx.model_dump()
        except Exception as e:
            log.warning("context_resolver: LLM output failed validation (%s); using safe defaults", e)
            result = {"scope": "clarification", "city": None, "store": None,
                      "date": None, "time_range": None, "is_followup": False,
                      "clarification_question": None}

        # ── Normalise city to exact DB name ───────────────────────────────────
        raw_city: Optional[str] = result.get("city")
        log.info("context_resolver: LLM returned city=%r store=%r scope=%r (prior city=%r)",
                 raw_city, result.get("store"), result.get("scope"), prior_city)
        
        if raw_city:
            matched = deps.stores.match_city(raw_city)
            if matched:
                raw_city = matched
                log.info("context_resolver: matched city=%r", raw_city)
            # else keep what LLM returned — store_rca / city_rca will handle unknown

        raw_store: Optional[str] = result.get("store")

        # Explicit-entity override (deterministic, runs BEFORE sticky fallback)
        # 
        # The LLM sometimes echoes the prior turn's sticky store when the user
        # mentions a different (often unknown) store ID like "TBC8". Detect a
        # store-ID pattern in the raw user message and use that literal value.
        # If the regex finds one, drop sticky fallback for `store`.
        _store_re = re.compile(r"\b(?:TBC|TBF|TBL|TBM|TBS|STORE_)[A-Z0-9]+\b", re.IGNORECASE)
        _explicit_store_match = _store_re.search(user_msg or "")
        explicit_store_in_msg = bool(_explicit_store_match)
        if explicit_store_in_msg:
            raw_store = _explicit_store_match.group(0).upper().replace("STORE_", "STORE_")
            log.info("context_resolver: explicit store %r in message — overriding LLM output", raw_store)

        # Explicit-city override using the resolver's known city list
        known_cities_lower = {c.lower(): c for c in deps.stores.known_cities()}
        explicit_city_in_msg = False
        msg_lower = (user_msg or "").lower()
        for city_lower, city_proper in known_cities_lower.items():
            # Match city name as a whole word, case-insensitive, with at least
            # one word boundary on each side
            if re.search(r"\b" + re.escape(city_lower.split()[0]) + r"\b", msg_lower):
                raw_city = city_proper
                explicit_city_in_msg = True
                log.info("context_resolver: explicit city %r in message — overriding LLM output", raw_city)
                break

        resolved_date: str = result.get("date") or prior_date or DEFAULT_DATE

        # ── Resolve time range ────────────────────────────────────────────────
        time_range: Optional[str] = result.get("time_range")
        hours: Optional[list[int]] = None
        if time_range:
            try:
                hours = time_ranges.resolve(time_range)
            except ValueError:
                hours = None

        # ── Build new context (NO merge — full replacement) ──────────────────
        # Both `city` and `last_city` are set so downstream nodes find it
        # regardless of which key they read (ctx.get("city") or ctx.get("last_city")).
        scope = result.get("scope") or "clarification"
        new_context: ConversationContext = {
            "scope": scope,
            "city": raw_city,
            "store": raw_store,
            "time_range": time_range,
            "hours": hours,
            "date": resolved_date,
            "is_followup": bool(result.get("is_followup")),
            "clarification_question": result.get("clarification_question"),
            # sticky slots — same values as current-turn for next-turn coreference
            # Sticky slots for next-turn coreference. Rule: if the user explicitly
            # named a new city, drop the prior store from sticky (scope switch).
            "last_city": raw_city,
            "last_store": None if explicit_city_in_msg else raw_store,
            "last_hours": hours,
            "last_date": resolved_date,
        }

        return {
            "context": new_context,
            "nodes_executed": _append_node(state, "context_resolver"),
        }

    return context_resolver_node


# ── Node 3: intent classifier ─────────────────────────────────────────────────

# Regex hints for meta-queries about conversation history. Cheap pre-check
# that runs before the LLM so "what did I just ask" doesn't fall into the
# RCA path and get blocked by the output guard.
_META_QUERY_PATTERNS = (
    "what was my last question",
    "what did i just ask",
    "what did i ask",
    "what have i been asking",
    "show me my history",
    "show my history",
    "what did we discuss",
    "what have we talked about",
    "what did i say",
    "previous questions",
    "my recent questions",
)


def _looks_like_meta_query(message: str) -> bool:
    """Pre-classifier hint. Catches the obvious phrasings cheaply."""
    lower = message.lower().strip()
    return any(pat in lower for pat in _META_QUERY_PATTERNS)


def make_intent_classifier_node(deps: NodeDeps):

    async def intent_classifier_node(state: GraphState) -> dict:
        ctx = state.get("context") or {}
        user_msg = state.get("user_message", "")

        # Pre-check: meta-queries about conversation history bypass the LLM.
        # These don't touch orders_gold and shouldn't go through RCA.
        if _looks_like_meta_query(user_msg):
            return {
                "intent": "meta_query",
                "intent_reason": "user asking about conversation history",
                "nodes_executed": _append_node(state, "intent_classifier"),
            }

        if ctx.get("scope") == "clarification":
            return {
                "intent": "refuse",
                "intent_reason": "clarification needed",
                "nodes_executed": _append_node(state, "intent_classifier"),
            }

        try:
            result = await deps.llm.complete_json(
                system=INTENT_CLASSIFIER_SYSTEM,
                user=(
                    f"Resolved context: {json.dumps(ctx, default=str)}\n"
                    f"User message: {user_msg}"
                ),
                temperature=0.0, max_tokens=80,
            )
        except Exception as e:
            log.exception("intent classifier failed: %s", e)
            return {
                "error": "intent classifier unavailable",
                "nodes_executed": _append_node(state, "intent_classifier"),
            }

        # Validate via Pydantic — bad shape → safe refuse, not a crash
        try:
            validated = IntentOutput.model_validate(result)
            intent = validated.intent
            intent_reason = validated.intent_reason
        except Exception as e:
            log.warning("intent_classifier: LLM output failed validation (%s); refusing", e)
            intent = "refuse"
            intent_reason = "intent validation failed"

        return {
            "intent": intent,
            "intent_reason": intent_reason,
            "nodes_executed": _append_node(state, "intent_classifier"),
        }

    return intent_classifier_node


# ── Node 4a: city RCA ─────────────────────────────────────────────────────────

def make_city_rca_node(deps: NodeDeps):

    async def city_rca_node(state: GraphState) -> dict:
        ctx = state.get("context") or {}
        # Both keys are set by context_resolver now; fallback is just safety
        city_input = ctx.get("city") or ctx.get("last_city")
        date = ctx.get("date") or ctx.get("last_date") or DEFAULT_DATE

        if not city_input:
            return _err(state, "city_rca", "no city resolved")

        # ── Normalise city name (handles space differences) ───────────────────
        city = deps.stores.match_city(city_input)
        if not city:
            return _err(
                state, "city_rca",
                f"unknown city {city_input!r}. Known cities: "
                f"{', '.join(deps.stores.known_cities())}",
            )

        rollup = aggregations.city_day_rollup(deps.db_path, city, date)
        if rollup:
            rollup["_db_path"] = deps.db_path  # pass-through for v2 report sub-queries
        if not rollup:
            return _err(state, "city_rca", f"no data for {city!r} on {date}")

        worst = aggregations.worst_stores_in_city(deps.db_path, city, date, limit=5)
        report = _format_city_report(city, date, rollup, worst)

        return {
            "report": report,
            "raw_data": {"city_rollup": rollup, "worst_stores": worst},
            "nodes_executed": _append_node(state, "city_rca"),
        }

    return city_rca_node


# ── Node 4b: store RCA ────────────────────────────────────────────────────────

def make_store_rca_node(deps: NodeDeps):

    async def store_rca_node(state: GraphState) -> dict:
        ctx = state.get("context") or {}
        store_input = ctx.get("store") or ctx.get("last_store")
        date = ctx.get("date") or ctx.get("last_date") or DEFAULT_DATE

        if not store_input:
            return _err(state, "store_rca", "no store resolved")

        resolution = deps.stores.resolve(store_input)
        if not resolution.found:
            suggestions = (
                ", ".join(f"{c} ({city})" for c, city in resolution.suggestions[:3])
                if resolution.suggestions else "no close matches"
            )
            report = (
                f"Store '{store_input}' is not in the dataset.\n"
                f"Closest matches: {suggestions}.\n"
                f"Known cities: {', '.join(deps.stores.known_cities())}."
            )
            return {
                "report": report,
                "raw_data": {"store_lookup_failed": store_input,
                             "suggestions": resolution.suggestions},
                "nodes_executed": _append_node(state, "store_rca"),
            }

        store = resolution.match
        city = resolution.city
        rollup = aggregations.store_day_rollup(deps.db_path, store, date)
        if rollup:
            rollup["_db_path"] = deps.db_path  # pass-through for v2 report sub-queries
        if not rollup:
            return _err(state, "store_rca", f"no data for {store!r} on {date}")

        problem = aggregations.problem_hours_for_store(deps.db_path, store, date)
        all_hours = aggregations.all_hours_for_store(deps.db_path, store, date)
        findings = engine.analyze_hours(problem, all_hours)

        report = _format_store_report(store, date, rollup, findings)
        return {
            "report": report,
            "raw_data": {"store_rollup": rollup, "findings_count": len(findings)},
            "nodes_executed": _append_node(state, "store_rca"),
        }

    return store_rca_node


# ── Node 4c: hour drill ───────────────────────────────────────────────────────

def make_hour_drill_node(deps: NodeDeps):

    async def hour_drill_node(state: GraphState) -> dict:
        ctx = state.get("context") or {}
        store_input = ctx.get("store") or ctx.get("last_store")
        hours = ctx.get("hours") or ctx.get("last_hours")
        date = ctx.get("date") or ctx.get("last_date") or DEFAULT_DATE

        if not store_input:
            return _err(state, "hour_drill", "no store for hour drill")
        if not hours:
            return _err(state, "hour_drill", "no hour range specified")

        resolution = deps.stores.resolve(store_input)
        if not resolution.found:
            return _err(state, "hour_drill", f"unknown store {store_input!r}")
        store = resolution.match

        problem = aggregations.problem_hours_for_store(
            deps.db_path, store, date, hours=hours,
        )
        all_h = aggregations.all_hours_for_store(
            deps.db_path, store, date, hours=hours,
        )
        findings = engine.analyze_hours(problem, all_h)

        label = ctx.get("time_range") or f"hours {hours[0]}-{hours[-1]}"
        report = _format_hour_drill_report(store, date, label, hours, findings, len(all_h))

        return {
            "report": report,
            "raw_data": {"findings_count": len(findings), "hours_range": hours},
            "nodes_executed": _append_node(state, "hour_drill"),
        }

    return hour_drill_node


# ── Node 4d: free-form (SQLite MCP) ──────────────────────────────────────────

def make_free_form_node(deps: NodeDeps):
    """
    Falls back to SQLite MCP for ad-hoc queries not covered by the playbook.
    Works only on WSL/Linux where MCP stdio is available.
    On Windows (mcp_manager=None) this node returns a friendly error.
    """

    async def free_form_node(state: GraphState) -> dict:
        if not deps.mcp or not deps.mcp.sqlite:
            return {
                "report": (
                    "Ad-hoc SQL queries require MCP integration (WSL/Linux). "
                    "For this query, please ask about a specific city, store, or time range."
                ),
                "nodes_executed": _append_node(state, "free_form"),
            }

        user_msg = state.get("user_message", "")
        cols = await deps.mcp.sqlite.describe_table("orders_gold")
        schema = "\n".join(
            f"  - {c['name']}: {c.get('type', 'unknown')}"
            for c in cols if isinstance(c, dict) and "name" in c
        )

        system = (
            "You write a single read-only SELECT against orders_gold. "
            'Return strict JSON: {"sql": "<SELECT>", "rationale": "<one line>"}. '
            "No DDL, no DML, no semicolons.\n\n"
            f"Schema:\n{schema}"
        )

        try:
            result = await deps.llm.complete_json(
                system=system, user=user_msg, temperature=0.0, max_tokens=300,
            )
        except Exception as e:
            return _err(state, "free_form", f"LLM unavailable: {e}")

        try:
            validated_sql = SQLOutput.model_validate(result)
            sql = validated_sql.sql
        except Exception as e:
            return _err(state, "free_form", f"SQL output failed validation: {e}")

        try:
            rows = await deps.mcp.sqlite.read_query(sql)
        except Exception as e:
            return _err(state, "free_form", f"SQL refused or failed: {e}")

        rows_shown = rows[:25]
        truncated = len(rows) > 25
        report = _format_freeform_report(sql, rows_shown, len(rows), truncated)

        return {
            "report": report,
            "raw_data": {"sql": sql, "row_count": len(rows)},
            "nodes_executed": _append_node(state, "free_form"),
        }

    return free_form_node


# ── Node 4e: meta query (conversation-history questions) ─────────────────────

def make_meta_query_node(deps: NodeDeps):
    """
    Answers questions about prior turns ('what did I just ask').

    Pulls from the transaction log keyed by (user_id, session_id) and
    produces a deterministic report so the synthesizer has clean ground
    truth — no LLM hallucination of past conversation.
    """

    async def meta_query_node(state: GraphState) -> dict:
        from app.transaction_log import transaction_log

        user_id = state.get("user_id", "unknown")
        session_id = state.get("session_id", "unknown")
        user_msg = state.get("user_message", "").lower()

        # "last question" / "what did I just ask"
        if "last" in user_msg or "just ask" in user_msg or "previous" in user_msg:
            last = transaction_log.last_question(user_id, session_id)
            if last is None:
                report = (
                    "No prior turns recorded in this session yet. "
                    "Ask a question about a city or store and I'll start tracking."
                )
            else:
                report = (
                    f"# Last question\n\n"
                    f"At {last['ts']}, you asked:\n\n"
                    f"> {last['question']}\n\n"
                    f"I classified it as {last.get('intent') or 'unclassified'} and replied with a "
                    f"{len(last['response'])}-character response."
                )
        else:
            # "show me my history" / "what have I been asking"
            recent = transaction_log.recent_questions(user_id, session_id, limit=10)
            if not recent:
                report = "No prior turns in this session."
            else:
                lines = [f"# Recent questions in this session ({len(recent)} most recent)\n"]
                for i, r in enumerate(recent, 1):
                    lines.append(
                        f"{i}. [{r['ts']}] ({r.get('intent') or 'unclassified'}): {r['question']}"
                    )
                report = "\n".join(lines)

        return {
            "report": report,
            "raw_data": {"meta_query": True},
            "nodes_executed": _append_node(state, "meta_query"),
        }

    return meta_query_node




# ── Node 5: synthesizer ───────────────────────────────────────────────────────

def make_synthesizer_node(deps: NodeDeps):

    async def synthesizer_node(state: GraphState) -> dict:
        report = state.get("report") or ""
        if not report:
            return {
                "final_response": REFUSAL_TEMPLATE,
                "nodes_executed": _append_node(state, "synthesizer"),
            }

        user_msg = state.get("user_message", "")
        # NOTE: history is intentionally NOT passed to synthesizer.
        # Past turns can leak entities (e.g. "Bangalore" data into a
        # "Chennai" answer). The current report is the only source of truth.

        try:
            chunks: list[str] = []
            async for piece in deps.llm.complete_stream(
                system=SYNTHESIZER_SYSTEM,
                history=[],  # explicit: no history to avoid cross-turn data leakage
                user=(
                    f"User asked: {user_msg}\n\n"
                    f"=== RCA REPORT (the ONLY source of truth) ===\n"
                    f"{report}\n"
                    f"=== END REPORT ===\n\n"
                    f"Now phrase this report naturally. Use ONLY numbers and "
                    f"findings from the report above. Do not reference any "
                    f"prior conversation. If the report is about a different "
                    f"entity than what the user asked, that is the correct "
                    f"answer — the engine resolved the entity."
                ),
                temperature=0.2, max_tokens=600,
            ):
                chunks.append(piece)
            text = "".join(chunks).strip()
        except Exception as e:
            log.exception("synthesizer failed: %s", e)
            return _err(state, "synthesizer", f"synthesizer error: {e}")

        return {
            "final_response": text,
            "nodes_executed": _append_node(state, "synthesizer"),
        }

    return synthesizer_node


# ── Node 6: output guard ──────────────────────────────────────────────────────

def make_output_guard_node(_deps: NodeDeps):

    async def output_guard_node(state: GraphState) -> dict:
        response = state.get("final_response", "")
        report = state.get("report", "")
        verdict = check_output(response, report)
        out: dict = {
            "output_guard_decision": verdict.decision.value,
            "output_guard_reason": verdict.reason,
            "nodes_executed": _append_node(state, "output_guard"),
        }
        if not verdict.allowed:
            log.warning("output blocked: %s", verdict.reason)
            out["final_response"] = (
                "[Output guard: response contained ungrounded claims. "
                "Falling back to deterministic report.]\n\n" + (report or REFUSAL_TEMPLATE)
            )
        return out

    return output_guard_node


# ── Node 7: persist turn (with mem0ai) ────────────────────────────────────────

def make_persist_turn_node(deps: NodeDeps):
    """
    Append (user, assistant) pair to messages.
    Also saves memories via mem0ai if available.
    """

    async def persist_turn_node(state: GraphState) -> dict:
        user_msg = state.get("user_message", "")
        final_resp = state.get("final_response", "")
        user_id = state.get("user_id", "default")
        ctx = state.get("context") or {}

        # ── mem0ai: save what happened this turn ─────────────────────────────
        if deps.memory is not None:
            try:
                memory_text = f"User asked: {user_msg}"
                if ctx.get("city"):
                    memory_text += f" | Resolved city: {ctx['city']}"
                if ctx.get("store"):
                    memory_text += f" | Resolved store: {ctx['store']}"
                deps.memory.add(
                    [{"role": "user", "content": memory_text}],
                    user_id=user_id,
                )
            except Exception as e:
                log.debug("mem0ai add failed (non-fatal): %s", e)

        # ── transaction log: durable record for session resume + meta-queries
        try:
            from app.transaction_log import transaction_log
            transaction_log.record(
                user_id=user_id,
                session_id=state.get("session_id", "unknown"),
                question=user_msg,
                response=final_resp,
                intent=state.get("intent"),
            )
        except Exception as e:
            log.debug("transaction_log write failed (non-fatal): %s", e)

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": final_resp},
            ],
            "nodes_executed": _append_node(state, "persist_turn"),
        }

    return persist_turn_node


# ── Formatting helpers (deterministic) ───────────────────────────────────────

def _format_city_report(city: str, date: str, rollup: dict, worst: list[dict]) -> str:
    """
    Structured city RCA report.
    
    Layout (in order):
      1. Headline    — city, date, weighted OR2A, breach rate
      2. Delivery    — total/completed/cancelled/RTO
      3. Performers  — top 3 vs worst 3 (two-column)
      4. Breach      — hour-wise peaks + store concentration
      5. Pileup      — by slot
      6. Forecast    — actual vs projected + model note
      7. Causes      — top combos across problem hours (best-effort)
      8. Drill-down  — one specific next-question suggestion
      9. P50/P95     — tail metrics for completeness
    """
    from app.rca import aggregations as A
    db_path = rollup.get("_db_path") or "data/loadshare.db"

    # Pull the v2 aggregations
    ds   = A.delivery_summary(db_path, "city", city, date)
    hbp  = A.hourly_breach_profile(db_path, "city", city, date, top_n=3)
    pbs  = A.pileup_by_slot(db_path, "city", city, date)
    tops = A.top_performers_in_city(db_path, city, date, limit=3)
    fa   = A.forecast_accuracy(db_path, "city", city, date)

    p50 = rollup.get("p50_or2a", 0.0)
    p95 = rollup.get("p95_or2a", 0.0)
    breach_pct = rollup.get("weighted_breached_rate", 0.0) * 100

    lines = [
        f"# City RCA — {city} — {date}",
        "",
        # 1. Headline
        f"**Stores observed:** {rollup.get('store_count', 0)}  ·  "
        f"**Hours:** {rollup.get('hours_observed', 0)}  ·  "
        f"**Weighted OR2A:** {rollup.get('weighted_avg_or2a', 0.0):.1f} min  ·  "
        f"**Weighted breach rate:** {breach_pct:.1f}%",
        "",
        # 2. Delivery summary
        "## Delivery summary",
        f"  Total orders:   {ds['total_orders']:>8,}",
        f"  Completed:      {ds['completed']:>8,}  ({ds['completion_rate']:.1%})",
        f"  Cancelled:      {ds['cancelled']:>8,}  ({ds['cancellation_rate']:.1%})",
        f"  RTO:            {ds['rto']:>8,}  ({ds['rto_rate']:.1%})",
        "",
        # 3. Top + worst performers (two-column visual)
        "## Performers",
    ]

    if tops or worst:
        lines.append(f"{'Top performers (by completion)':<46}{'Worst performers (by breach)':<46}")
        max_rows = max(len(tops), len(worst))
        for i in range(max_rows):
            left = ""
            right = ""
            if i < len(tops):
                t = tops[i]
                left = f"  {t['store']}  {t['completion_rate']:.1%}  ({t['total_orders']:,} orders)"
            if i < len(worst):
                w = worst[i]
                right = f"  {w['store']}  breach {w['weighted_breached_rate']:.1%}  OR2A {w['weighted_avg_or2a']:.1f}m"
            lines.append(f"{left:<46}{right:<46}")
        lines.append("")

    # 4. Breach pattern
    lines.append("## Breach pattern")
    lines.append(f"  Hours with breach rate > 20%: {hbp['hours_over_20pct']} of {hbp['total_hours_observed']}")
    if hbp["top_hours"]:
        lines.append("  Worst hours:")
        for h in hbp["top_hours"]:
            lines.append(
                f"    hour {h['hour']:>2}: {h['breach_rate']:.1%} breach "
                f"on {h['total_orders']:,} orders"
            )
    lines.append("")

    # 5. Pileup by slot
    lines.append("## Pileup concentration")
    if not pbs:
        lines.append("  No pileup recorded.")
    else:
        total_pileup = sum(s["pileup_orders"] for s in pbs)
        for s in pbs[:3]:
            share = (s["pileup_orders"] / total_pileup) if total_pileup else 0
            lines.append(
                f"  {s['slot_name']:<20}  {s['pileup_orders']:>6,} orders "
                f"across {s['pileup_hours']:>3} hours  ({share:.0%} of pileup)"
            )
    lines.append("")

    # 6. Forecast accuracy + ML model line
    lines.append("## Demand vs forecast")
    lines.append(f"  Actual:    {fa['actual']:>8,}")
    lines.append(f"  Projected: {fa['projected']:>8,}")
    lines.append(f"  Ratio:     {fa['ratio']:.2f}")
    lines.append(f"  → {fa['model_note']}")
    lines.append("")

    # 7. Cause attribution headline (city level needs problem-hour findings;
    #    we don't compute them per-city by default, so we use the worst-stores
    #    rollup as the strongest signal available)
    if worst:
        lines.append("## Headline pattern")
        worst_total_breach = sum(w["weighted_breached_rate"] for w in worst) / max(len(worst), 1)
        lines.append(
            f"  The {len(worst)} worst stores averaged {worst_total_breach:.1%} breach rate "
            f"— compared with the city's {breach_pct:.1f}% overall."
        )
        lines.append("")

    # 8. Drill-down suggestion
    if worst:
        target = worst[0]["store"]
        lines.append(f"## Suggested next step")
        lines.append(f"  Drill into {target}: \"Why did {target} underperform on {date}?\"")
        lines.append("")

    # 9. P50/P95 details
    lines.append("## OR2A distribution")
    lines.append(f"  P50 (median):       {p50:>6.1f} min")
    lines.append(f"  P95 (worst tail):   {p95:>6.1f} min")

    return "\n".join(lines)


def _format_store_report(store: str, date: str, rollup: dict, findings: list) -> str:
    """
    Structured store RCA report.
    
    Layout (in order):
      1. Headline       — store, city, date, weighted OR2A
      2. Breach pattern — hour-wise (the diagnosis)
      3. Pileup timing  — within the store's hours
      4. Forecast       — actual vs projected for this store
      5. Delivery       — total/completed/cancelled/RTO
      6. Hour-by-hour   — playbook format for each problem hour
      7. Drill-down     — one specific next-question suggestion
    """
    from app.rca import aggregations as A
    from app.rca import engine
    db_path = rollup.get("_db_path") or "data/loadshare.db"

    ds  = A.delivery_summary(db_path, "store", store, date)
    hbp = A.hourly_breach_profile(db_path, "store", store, date, top_n=3)
    pbs = A.pileup_by_slot(db_path, "store", store, date)
    fa  = A.forecast_accuracy(db_path, "store", store, date)
    ca  = A.cause_attribution(findings)

    p50 = rollup.get("p50_or2a", 0.0)
    p95 = rollup.get("p95_or2a", 0.0)
    breach_pct = rollup.get("breach_rate", 0.0) * 100

    lines = [
        f"# Store RCA — {store} ({rollup.get('city', '')}) — {date}",
        "",
        # 1. Headline
        f"**Hours observed:** {rollup.get('hours_observed', 0)}  ·  "
        f"**Weighted OR2A:** {rollup.get('weighted_avg_or2a', 0.0):.1f} min  ·  "
        f"**Breach rate:** {breach_pct:.1f}%  ·  "
        f"**Problem hours:** {rollup.get('problem_hour_count', 0)}",
        "",
        # 2. Breach pattern — hour-wise (the diagnosis)
        "## Breach pattern (hour-wise)",
    ]
    lines.append(f"  Hours with breach rate > 20%: {hbp['hours_over_20pct']} of {hbp['total_hours_observed']}")
    if hbp["top_hours"]:
        lines.append("  Worst hours:")
        for h in hbp["top_hours"]:
            lines.append(
                f"    hour {h['hour']:>2}: {h['breach_rate']:.1%} breach "
                f"on {h['total_orders']:,} orders"
            )
    lines.append("")

    # 3. Pileup timing
    lines.append("## Pileup timing")
    if not pbs:
        lines.append("  No pileup recorded.")
    else:
        for s in pbs:
            lines.append(
                f"  {s['slot_name']:<20}  {s['pileup_orders']:>6,} orders "
                f"across {s['pileup_hours']:>3} hours"
            )
    lines.append("")

    # 4. Forecast vs actual for this store
    lines.append("## Demand vs forecast")
    lines.append(f"  Actual:    {fa['actual']:>6,}")
    lines.append(f"  Projected: {fa['projected']:>6,}")
    lines.append(f"  Ratio:     {fa['ratio']:.2f}  → {fa['model_note']}")
    lines.append("")

    # 5. Delivery summary (secondary for store)
    lines.append("## Delivery summary")
    lines.append(f"  Total orders:   {ds['total_orders']:,}")
    lines.append(f"  Completed:      {ds['completed']:,}  ({ds['completion_rate']:.1%})")
    lines.append(f"  Cancelled:      {ds['cancelled']:,}  ({ds['cancellation_rate']:.1%})")
    lines.append(f"  RTO:            {ds['rto']:,}  ({ds['rto_rate']:.1%})")
    lines.append("")

    # 5b. Cause attribution if findings exist
    if ca["total_problem_hours"] > 0 and ca["top_combos"]:
        lines.append("## Cause attribution")
        for entry in ca["top_combos"]:
            lines.append(
                f"  {entry['pct']:.0%} of problem hours: {entry['combo']} "
                f"({entry['count']} hours)"
            )
        lines.append("")

    # 6. Hour-by-hour findings (existing playbook format)
    if findings:
        lines.append("## Hour-by-hour findings (playbook format)")
        lines.append("")
        for f in findings:
            lines.append(engine.format_hour_finding(f))
    else:
        lines.append("## Hour-by-hour findings")
        lines.append("  No problem hours flagged for this store.")
        lines.append("")

    # 7. Drill-down suggestion
    lines.append("## Suggested next step")
    if hbp["top_hours"]:
        worst_h = hbp["top_hours"][0]["hour"]
        lines.append(f"  Walk through the {worst_h:02d}:00 window: \"Why did {store} breach at hour {worst_h}?\"")
    else:
        lines.append(f"  Compare to peers: \"How did {rollup.get('city', 'the city')} do on {date}?\"")

    # P50/P95 tail (kept for completeness)
    lines.append("")
    lines.append(f"OR2A P50 (median):     {p50:.1f} min")
    lines.append(f"OR2A P95 (worst-tail): {p95:.1f} min")

    return "\n".join(lines)


def _format_hour_drill_report(
    store: str, date: str, label: str, hours: list[int],
    findings: list, hours_in_range: int,
) -> str:
    lines = [
        f"# Hour Drill — {store} — {date} — {label}",
        "",
        f"Hours analyzed: {hours[0] if hours else 0}-{hours[-1] if hours else 0}",
        f"Problem hours in range: {len(findings)} of {hours_in_range}",
        "",
    ]
    if not findings:
        lines.append("No problem hours in this range. Performance was within SLA.")
        return "\n".join(lines)
    for f in findings:
        lines.append(engine.format_hour_finding(f))
    return "\n".join(lines)


def _format_freeform_report(sql: str, rows: list[dict], total: int, truncated: bool) -> str:
    lines = [
        "# Free-form SQL result",
        "",
        f"Query: `{sql}`",
        f"Rows returned: {total}" + (" (showing first 25)" if truncated else ""),
        "",
    ]
    if not rows:
        lines.append("No rows.")
        return "\n".join(lines)
    cols = list(rows[0].keys())
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, ""))[:30] for c in cols) + " |")
    return "\n".join(lines)


def _err(state: GraphState, node_name: str, msg: str) -> dict:
    log.warning("%s: %s", node_name, msg)
    return {
        "report": f"[{node_name}] {msg}",
        "nodes_executed": _append_node(state, node_name),
    }

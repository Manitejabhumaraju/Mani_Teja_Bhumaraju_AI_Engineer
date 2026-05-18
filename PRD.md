# PRD — Delivery Operations RCA Agent

**Author:** AI Engineering candidate submission  
**Status:** Implemented, walkthrough-ready  
**Last updated:** May 2026

---

## 1. Background

Loadshare operates last-mile delivery for Amazon's quick-commerce business across nine cities and ~233 dark stores. The primary SLA metric is **OR2A** — the minutes between an order becoming ready at a store and a rider being assigned to it. When OR2A breaches SLA at a store-hour, the ops team needs to figure out why.

Today this RCA is run manually: an analyst opens SQL, applies a written playbook, and produces a written report. The work is repetitive, the playbook is mechanical, and the SLA on producing an answer is "before the next ops review meeting." This is a strong fit for a conversational agent.

## 2. Problem

A new analyst sits down at 9 AM, sees yesterday's breach numbers were elevated for Bangalore, and needs to answer three questions before standup:

1. **Which stores drove it?** (city-level rollup, ranked)
2. **Why those stores?** (store-level RCA across all problem hours)
3. **Is this a one-off or a pattern?** (drill into specific hour ranges)

Manual SQL takes 30–45 minutes if the analyst knows the schema. A new analyst takes longer. An agent that can answer all three in conversational form, with the same playbook logic, should compress this to under 5 minutes.

## 3. Users

| User | Daily activity | Pain |
|---|---|---|
| **City ops analyst** | Reviews previous day's performance, drills into outliers, writes a one-page summary for the morning standup | SQL + playbook in two windows; numbers don't always reconcile |
| **Regional ops manager** | Reviews patterns across multiple cities and weeks | No drill-down tooling above SQL; questions phrased in business language don't map to schema |
| **New analyst (week 1–4)** | Learning the playbook + schema simultaneously | Steep ramp; playbook lives in a doc, schema lives elsewhere |

The agent prioritizes the city ops analyst use case (the brief's five sample queries reflect this). Other users benefit from the same surface.

## 4. Goals

**Primary**
- Answer the brief's five sample queries correctly using real data.
- Support 5+ multi-turn conversations within a session: drill-downs, follow-ups, context switches.
- Encode the playbook's thresholds (1.10 demand spike, 0.90 booking gap, 0.85 utilization gap, 3+ sustained pileup) verbatim — no LLM drift.
- Match the playbook's required output format (numbered checks, summary line, all three checks reported per hour even when negative).

**Secondary**
- Survive a backend restart (transaction log → session-resume endpoint).
- Answer meta queries ("what did I just ask") from the transaction log without going through RCA.
- Provide a Streamlit frontend wired to the agent over WebSocket so the multi-turn behavior demos cleanly.

**Non-goals (per the brief)**
- Recommendations engine (the agent diagnoses, it doesn't prescribe).
- Cross-day or cross-week trend analysis (one day of data provided).
- Production auth (demo login `demo`/`demo` is intentional).
- Fancy UI.

## 5. Functional requirements

### FR1 — City RCA
Input: a city name + optional date.  
Output: weighted breach rate, weighted avg OR2A, P50/P95 OR2A, top 3 best and worst stores, hour-wise breach peaks, pileup concentration by slot, demand-vs-forecast ratio with model note, headline cause attribution, drill-down suggestion.

### FR2 — Store RCA
Input: a store ID + optional date.  
Output: hour-wise breach pattern (the diagnosis), pileup timing, demand-vs-forecast for the store, delivery summary, hour-by-hour playbook findings (numbered checks per the brief), drill-down suggestion.

### FR3 — Hour drill
Input: a store + a time range (e.g. "morning hours").  
Output: per-hour playbook findings for hours in the range, scoped to the named time window.

### FR4 — Unknown entity handling
When the user names a store that doesn't exist in the dataset (e.g. "TBC8"), the agent must say so explicitly and not return data for any other store. Regex pre-check in the context resolver enforces this even when the LLM tries to echo a prior turn's sticky store.

### FR5 — Free-form analytics
Questions that don't map to FR1-FR3 (e.g. "top 5 stores by total orders") route to an LLM-generated SELECT executed via the MCP SQLite server. SQL output is validated against `SQLOutput` Pydantic schema — DDL/DML rejected before the MCP server is called.

### FR6 — Meta queries
"What was my last question," "show my history," and similar phrases are intercepted before intent classification and answered from the transaction log. No RCA call, no LLM fabrication.

### FR7 — Multi-turn coreference
"What about Chennai?" after a Bangalore turn must switch context to Chennai cleanly. "Same way for the morning hours?" after a store turn must keep the store and apply the time range. Sticky context survives across turns; explicit entity mentions override sticky.

### FR8 — Streaming responses
The synthesizer streams tokens over WebSocket so the UI feels responsive. Final state ships in a separate `{type: "final"}` event with the full report and metadata.

## 6. Non-functional requirements

| Requirement | Target | How met |
|---|---|---|
| Latency (cached) | < 1 s | Redis semantic cache hits skip the graph entirely |
| Latency (cold, RCA path) | < 30 s | 3 NIM calls per turn (resolver, classifier, synthesizer); P50 around 20 s in observed runs |
| Correctness (numerical) | 100% | Numbers come from SQL, not LLM; cross-checked against the CSV |
| Playbook compliance | 100% | Engine constants match the brief; output format matches the brief |
| Observability | Per-turn structured log | Structured turn_metric log line; LangSmith trace if configured |
| Graceful degradation | All optional integrations | Redis/LangSmith/mem0ai all degrade to a no-op fallback if not configured |

## 7. Architecture

### Stack
- **Backend:** Python 3.10+, FastAPI, LangGraph, Pydantic 2
- **LLM:** NVIDIA NIM (`meta/llama-3.3-70b-instruct`) via OpenAI-compatible API
- **MCP:** Node Express HTTP server for SQLite (one server, four endpoints)
- **DB:** SQLite (`data/loadshare.db`, built from CSV at install time) + SQLite (`data/transactions.db`, written per turn)
- **Frontend:** Streamlit + WebSocket
- **Optional:** Redis (semantic cache), LangSmith (tracing), mem0ai (semantic memory)

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Orchestration | LangGraph StateGraph with deterministic edges | Known intent → handler paths; LLM-as-router would just add latency |
| Playbook encoding | Python constants + analyze loop | Brief's thresholds are explicit; LLM must not "improve" them |
| Number generation | SQL aggregations | Reproducible; reviewable; no hallucination surface |
| Phrasing | LLM (synthesizer) | The one thing LLMs are better at than templates |
| MCP scope | One server (SQLite over HTTP) | The brief requires "at least one"; a filesystem MCP would duplicate the engine |
| MCP transport | HTTP, not stdio | Windows-native demo; container-friendly; survives backend restart |
| Memory model | Two layers (LangGraph MemorySaver + transaction log) | One ephemeral (per-thread), one durable (cross-session, cross-restart) |
| Auth | Bearer token, in-memory | Brief says auth is out of scope; built minimal version for multi-user demo |

### Graph topology

8 functional nodes:

```
input_guard
  → context_resolver        (LLM call #1: extract entities, normalize)
    → intent_classifier     (LLM call #2: pick a path)
      → [city_rca | store_rca | hour_drill | free_form | meta_query]
        → synthesizer       (LLM call #3: phrase the report, streaming)
          → output_guard    (grounding check)
            → persist_turn  (mem0 write + transaction log)
              → END
```

Plus conditional edges that skip RCA paths for refused or meta-queried intents.

### Pydantic-validated boundaries

Every LLM JSON output passes through a Pydantic model:

| Boundary | Schema |
|---|---|
| Context resolver | `ContextOutput` (scope, city, store, date, time_range, is_followup) |
| Intent classifier | `IntentOutput` (literal of allowed intents) |
| Free-form SQL agent | `SQLOutput` (validated SELECT-only, rejects DML/DDL) |
| Input guard LLM judge | `InputGuardVerdict` |
| WebSocket events | `StageEvent`, `TokenEvent`, `FinalEvent`, `ErrorEvent` |

Validation failure → safe default → continue. Never a crash.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM rewrites unknown store IDs to a sticky prior store | Regex pre-check in context resolver — explicit entity in user message overrides LLM output |
| Output guard refuses a valid response | Falls back to the deterministic report text directly; the guard is advisory, not load-bearing |
| MCP server unreachable | Backend startup logs a warning and continues with MCP disabled; free-form path returns a friendly error |
| Backend restart loses MemorySaver state | Transaction log + session-resume endpoint reload prior turns on session pick |
| Cross-turn entity leak (Bangalore data in Chennai answer) | Synthesizer is called with `history=[]`; report is the only source of truth |
| Cost (NIM API) | Semantic cache via Redis; repeat questions are free |

## 9. Test plan

| Layer | Where | What |
|---|---|---|
| Unit | `tests/unit/` | Aggregations math, store resolver fuzzy matching, engine threshold logic |
| Integration | `tests/integration/` | Graph runner end-to-end with mocked NIM + real SQLite |
| Conversation | `tests/conversations/` | Multi-turn sequences from the brief + sticky-context regressions |
| Security | `tests/security/` | SQL injection attempts, prompt injection attempts, guardrail bypass |
| Eval | `eval/run_eval.py` | Scripted runs against `conversation_suite.jsonl` and `adversarial.jsonl` |

## 10. What this submission does NOT do

Spelled out so the walkthrough doesn't get sidetracked:

- **No cross-day analytics.** One day of data; no trend lines.
- **No write operations.** SQL allowlist rejects DML/DDL; agent is read-only by design.
- **No fine-tuned models.** Out-of-the-box NIM, prompted but not adapted.
- **No real auth.** `demo/demo` login is for the multi-turn demo to have a session.
- **No production deployment story.** Dockerfile and compose are for review-time reproducibility, not scale.

## 11. References

- Brief: `candidate_brief (AI Engineer).md`
- Playbook: `docs/quick_commerce_rca_logic.md`
- Schema: `docs/quick_commerce_orders_gold.md`
- OR2A definition: `docs/order_ready_to_assignment.md`
- Test queries: `TEST_QUERIES.md`

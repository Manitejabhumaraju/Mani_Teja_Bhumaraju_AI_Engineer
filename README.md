# Loadshare RCA Agent

A conversational root-cause-analysis agent for Amazon's quick-commerce delivery operations, built on **LangGraph** with **MCP** for tool access and **NVIDIA NIM** for inference.

Built for the Loadshare AI Engineering take-home. Walkthrough-ready.

---

## What it does

Ops analysts ask plain-English questions about delivery performance and the agent walks the [RCA playbook](docs/quick_commerce_rca_logic.md) for them. Multi-turn, context-aware, deterministic where it matters and LLM-driven where it helps.

```
> How did Bangalore do on 2026-04-22?
> Why did STORE_091 underperform that day?
> Walk me through the morning hours.
> What about STORE_003?
```

Each answer is built from the same playbook a human analyst would follow: weighted breach rate at the city/store/day level, then drill into problem hours, then check the three causes (demand, pileup, supply) on each one. Numbers come from SQL, phrasing comes from an LLM, the playbook is enforced in Python.

---

## How to run it (one command)

**Prerequisites:** Python 3.10+, Node.js 18+, an NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/) (free tier is fine).

```bash
# 1. Configure your NVIDIA key
cp .env.example .env
# Open .env, set NVIDIA_API_KEY=nvapi-...

# 2. Run everything
run.bat                # Windows
./run.sh               # macOS / Linux
```

`run.bat` installs Python deps, installs Node packages for the MCP server, builds the SQLite DB from the CSV, and starts three processes:

| Service | Port | What it is |
|---|---|---|
| MCP SQLite server | 3002 | Tool layer — read-only `/query`, `/tables`, `/schema/:t`, `/health` |
| FastAPI backend | 8000 | LangGraph orchestrator + WebSocket chat |
| Streamlit frontend | 8501 | Login `demo` / `demo`, open at http://localhost:8501 |

Five sample queries to confirm it works: see [TEST_QUERIES.md](TEST_QUERIES.md).

### Docker (alternative)

```bash
docker compose up --build
# open http://localhost:8501
```

---

## Architecture

### Graph topology

```
┌───────────────────────────────────────────────────────────────┐
│                      LangGraph StateGraph                     │
│                                                               │
│   START → input_guard → context_resolver → intent_classifier │
│                                  │                            │
│            ┌─────────────────────┼─────────────────────┐     │
│            ▼                     ▼                     ▼     │
│        city_rca              store_rca           hour_drill  │
│            │                     │                     │     │
│            │                  free_form          meta_query  │
│            │                     │                     │     │
│            └────────────►  synthesizer  ◄──────────────┘     │
│                                  ▼                            │
│                            output_guard                       │
│                                  ▼                            │
│                            persist_turn → END                 │
└───────────────────────────────────────────────────────────────┘
```

Each node is a callable `Agent` class (`app/agents/*.py`) so the graph reads as supervisor + specialized handlers, but routing is the deterministic LangGraph edge layer — fast, traceable, no LLM-as-judge on the hot path.

### Where decisions live

| Concern | Where | Why |
|---|---|---|
| Coreference, intent | LLM (context_resolver, intent_classifier) | Natural language; LLMs are good at this |
| Numeric aggregation | SQL (`app/rca/aggregations.py`) | Reproducible, fast, cheap |
| Playbook thresholds (1.10, 0.90, 0.85, 3+ pileup) | Python constants (`app/rca/engine.py`) | The brief calls them out specifically; can't be left to LLM drift |
| Report layout & ordering | Python (`_format_city_report`, `_format_store_report`) | Deterministic — analyst sees the same sections every time |
| Natural-language phrasing | LLM (synthesizer node) | The one thing that benefits from LLM creativity |

### MCP integration

The brief asks for at least one open-source MCP server consumed by the agent. We use **SQLite-over-HTTP**: a small Node Express server (`mcp-servers/sqlite-server.js`) exposes `/tables`, `/schema/:t`, `/query` as REST endpoints. The Python client (`app/mcp_clients/sqlite_client.py`) hits these over httpx. Used by the `free_form` agent when a question doesn't fit the playbook.

Why HTTP and not stdio: stdio MCP requires WSL or subprocess pipes on Windows; HTTP works natively, deploys as its own container, and survives backend restarts. Trade-off is one more port to expose.

### Pydantic everywhere LLMs hand back JSON

`app/llm/schemas.py` defines `ContextOutput`, `IntentOutput`, `SQLOutput`, etc. Every LLM JSON call is validated by `.model_validate()`. Invalid output drops to safe defaults rather than crashing — the synthesizer never sees garbage.

### Memory & session continuity

Two layers:

1. **LangGraph `MemorySaver` checkpointer.** Keys conversation state by `thread_id = user_id:session_id`. Lets a single turn's nodes share state.
2. **`app/transaction_log.py` SQLite log.** Durable record of every (user, session, question, response, timestamp). Survives backend restart, powers the session-resume endpoint (`GET /api/sessions/{id}/history`), and answers meta queries like "what was my last question."

If `OPENAI_API_KEY` is set, mem0ai layers on top for semantic memory; otherwise the in-process `SimpleMemory` (`app/memory_store.py`) handles coreference hints with simple keyword matching.

### Optional integrations

| Service | Activated by | What it adds |
|---|---|---|
| Redis | `REDIS_URL=redis://localhost:6379/0` | Semantic cache — repeat questions skip resolver + RCA + LLM, served from cache |
| LangSmith | `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY=...` | Per-node trace with input/output/latency |
| mem0ai | `OPENAI_API_KEY` | Vector-backed memory instead of dict |

All three degrade gracefully when not configured. Health check shows what's on.

---

## Project layout

```
loadshare-rca-agent/
├── app/
│   ├── main.py                    # FastAPI app, WebSocket /ws/chat, REST /api/*
│   ├── graph/
│   │   ├── builder.py             # StateGraph wiring, GraphRunner with streaming
│   │   ├── nodes.py               # Node implementations (each agent's run logic)
│   │   ├── router.py              # Conditional-edge functions
│   │   └── state.py               # GraphState TypedDict with add_messages reducer
│   ├── agents/                    # OO agent wrappers (BaseAgent + 10 concrete agents)
│   ├── rca/
│   │   ├── engine.py              # Playbook thresholds + hour-finding analysis
│   │   ├── aggregations.py        # SQL aggregations (rollups, percentiles, profiles)
│   │   └── store_resolver.py      # Fuzzy city/store matching
│   ├── llm/
│   │   ├── nim_client.py          # NVIDIA NIM via OpenAI-compatible API
│   │   ├── prompts.py             # System prompts (versioned)
│   │   └── schemas.py             # Pydantic models for LLM outputs
│   ├── guardrails/                # SQL allowlist, input guard, output grounding check
│   ├── mcp_clients/               # SQLite MCP HTTP client
│   ├── cache/                     # Semantic cache (Redis + bge-small) with NoOp fallback
│   ├── observability/             # LangSmith hooks + structured turn-metric logs
│   ├── auth.py                    # Bearer token issue/resolve/revoke
│   ├── sessions.py                # Per-user session registry
│   ├── transaction_log.py         # Durable turn log (SQLite)
│   └── memory_store.py            # SimpleMemory — mem0ai-compatible fallback
├── mcp-servers/
│   ├── sqlite-server.js           # SQLite-over-HTTP MCP server
│   └── package.json
├── frontend/
│   └── app.py                     # Streamlit chat UI with session sidebar
├── docs/
│   ├── quick_commerce_orders_gold.md
│   ├── order_ready_to_assignment.md
│   └── quick_commerce_rca_logic.md   # ← The playbook the engine encodes
├── data/
│   └── quick_commerce_orders_gold_20260422.csv
├── scripts/
│   └── load_csv_to_sqlite.py      # CSV → SQLite, run once at install
├── eval/                          # Conversation + adversarial test suites
├── tests/                         # Unit, integration, conversation, security
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.bat                        # One-command setup + run (Windows)
└── README.md                      # ← You are here
```

---

## How the brief's questions get answered

| Question | Path through the graph | Source of numbers |
|---|---|---|
| "How did Bangalore do on 2026-04-22?" | input_guard → context (city=Bangalore, date=…) → intent (city_rca) → city_rca → synthesizer | `aggregations.city_day_rollup` + `worst_stores_in_city` + `hourly_breach_profile` + `pileup_by_slot` + `forecast_accuracy` |
| "Why did TBC8 underperform?" | input_guard → context (store=TBC8 via regex override) → intent (store_rca) → store_rca | `store_day_rollup` returns no rows → reports "not in dataset" cleanly |
| "Walk me through morning at STORE_091" | input_guard → context (store=STORE_091, time=morning, is_followup) → intent (hour_drill) → hour_drill | `problem_hours_for_store` + `engine.analyze_hours` |
| "What about STORE_003?" | input_guard → context (store=STORE_003 via regex; sticky cleared) → intent (store_rca) → store_rca | Same as above for STORE_003 |
| "what was my last question" | input_guard → context → intent (meta_query, regex pre-classifier) → meta_query | `transaction_log.last_question` |

---

## What I used AI tools for

This repo was built with Claude as the primary pair-programming assistant.

**Claude wrote:** the initial graph topology, LangGraph node skeletons, agent class refactor, Pydantic schemas, MCP HTTP server, frontend Streamlit shell, Dockerfile, and most test fixtures.

**I wrote:** the RCA playbook encoding in `engine.py` (the thresholds and the analysis loop), the report layout decisions (section order, what's in city vs store reports), the sticky-context fix design after debugging a real failure case, the MCP transport choice (HTTP over stdio for Windows compatibility), and architectural calls on what goes in the LLM vs SQL vs Python.

**I discarded:** an early DeepAgents supervisor variant (worked but added orchestration complexity without solving the brief's known-path queries any better than LangGraph edges), an MCP filesystem server (the playbook is hardcoded in Python; a filesystem MCP would be re-reading the same docs every turn), and an initial Postgres setup (SQLite handles 3,813 rows fine and saves a docker service).

---

## License & attribution

Take-home submission. Not licensed for production use without permission.

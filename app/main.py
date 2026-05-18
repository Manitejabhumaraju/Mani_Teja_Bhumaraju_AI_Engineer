"""
FastAPI app - HTTP SQLite MCP

Surface:
  POST /api/login          {username, password}      -> {token, user_id}
  POST /api/logout         (Bearer)                  -> {ok: true}
  GET  /api/sessions       (Bearer)                  -> {sessions: [...]}
  POST /api/sessions       (Bearer) {name?}          -> {session}
  POST /api/sessions/{sid}/rename (Bearer) {name}    -> {session}
  DELETE /api/sessions/{sid} (Bearer)                -> {ok: true}
  GET  /api/health                                   -> {ok: true, ...}
  WS   /ws/chat?token=...&session_id=...
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import auth_store
from app.cache.embeddings import EmbeddingModel
from app.cache.semantic_cache import SemanticCache, NoOpCache
from app.config import settings
from app.graph import build_graph, GraphRunner
from app.graph.nodes import NodeDeps
from app.guardrails import validate_sql
from app.llm.nim_client import NimClient
from app.mcp_clients.manager import MCPManager
from app.observability import setup_tracing, log_turn_metrics
from app.rca.store_resolver import StoreResolver
from app.sessions import session_manager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("app.main")


def _init_memory():
    """
    Initialize memory store.
    
    Tries mem0ai first (needs OpenAI key), falls back to SimpleMemory
    which works out-of-the-box with no external dependencies.
    """
    # Try mem0ai if OpenAI key is available
    import os
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from mem0 import Memory
            mem = Memory()
            log.info("mem0ai initialized (with OpenAI backend)")
            return mem
        except Exception as e:
            log.warning("mem0ai init failed (%s), using SimpleMemory", e)
    
    # Fallback: SimpleMemory (in-process, no external deps)
    from app.memory_store import SimpleMemory
    mem = SimpleMemory()
    log.info("SimpleMemory initialized (in-process, no API keys needed)")
    return mem


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()

    # SQLite MCP server (HTTP) - must be started separately
    mcp_mgr = MCPManager(
        docs_dir=settings.docs_dir,
        db_path=settings.db_path,
        sql_validator=validate_sql,
    )
    try:
        await mcp_mgr.start()
        log.info("✅ SQLite MCP connected")
    except ConnectionRefusedError:
        log.error(
            "❌ SQLite MCP server not running!\n"
            "   Start it first: cd mcp-servers && npm start"
        )
        mcp_mgr = None
    except Exception as e:
        log.exception("MCP startup failed: %s", e)
        mcp_mgr = None

    # LLM client
    try:
        llm = NimClient()
    except RuntimeError as e:
        log.error("LLM client init failed: %s. Did you set NVIDIA_API_KEY?", e)
        raise

    # mem0ai memory
    memory = _init_memory()

    # Cache (Redis optional)
    cache = NoOpCache()
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(settings.redis_url)
        await redis_client.ping()
        embedder = EmbeddingModel(settings.embedding_model)
        embedder.warmup()
        cache = SemanticCache(
            redis_client=redis_client,
            embedder=embedder,
            similarity_threshold=settings.semantic_cache_threshold,
        )
        log.info("semantic cache initialized")
    except Exception as e:
        log.warning("Redis unavailable; using NoOpCache: %s", e)

    # Store resolver
    store_resolver = StoreResolver.from_sqlite(str(settings.db_path))

    # Graph - with memory support
    deps = NodeDeps(
        llm_client=llm,
        db_path=str(settings.db_path),
        store_resolver=store_resolver,
        cache=cache,
        mcp_manager=mcp_mgr,
        memory=memory,
    )
    compiled = build_graph(deps)
    runner = GraphRunner(compiled, deps, cache=cache)

    app.state.runner = runner
    app.state.llm = llm
    app.state.mcp = mcp_mgr
    app.state.cache = cache
    app.state.memory = memory

    log.info(
        "🚀 loadshare-rca-agent ready | MCP=%s | mem0ai=%s",
        "on" if mcp_mgr else "off",
        "on" if memory else "off",
    )

    try:
        yield
    finally:
        if mcp_mgr is not None:
            await mcp_mgr.stop()
        log.info("shutdown complete")


app = FastAPI(title="Loadshare RCA Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Auth helpers ====================================================

def _user_from_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="expected Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user_id = auth_store.resolve_token(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user_id


# === REST endpoints ==================================================

class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/api/health")
async def health(request: Request):
    state = request.app.state
    return {
        "ok": True,
        "mcp_ready": getattr(state, "mcp", None) is not None,
        "cache_kind": type(getattr(state, "cache", None)).__name__,
        "model": settings.nim_model,
    }


@app.post("/api/login")
async def login(payload: LoginRequest):
    if not auth_store.verify_credentials(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = auth_store.issue_token(payload.username)
    return {"token": token, "user_id": payload.username}


@app.post("/api/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    auth_store.revoke_token(token)
    return {"ok": True}


@app.get("/api/sessions")
async def list_sessions(authorization: Optional[str] = Header(None)):
    user_id = _user_from_bearer(authorization)
    sessions = [s.to_dict() for s in session_manager.list_for_user(user_id)]
    return {"sessions": sessions}


class CreateSessionRequest(BaseModel):
    name: Optional[str] = None


@app.post("/api/sessions")
async def create_session(
    payload: CreateSessionRequest,
    authorization: Optional[str] = Header(None),
):
    user_id = _user_from_bearer(authorization)
    meta = session_manager.create(user_id, payload.name)
    return {"session": meta.to_dict()}


class RenameRequest(BaseModel):
    name: str


@app.post("/api/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    payload: RenameRequest,
    authorization: Optional[str] = Header(None),
):
    user_id = _user_from_bearer(authorization)
    if not session_manager.rename(user_id, session_id, payload.name):
        raise HTTPException(status_code=404, detail="session not found")
    return {"session": session_manager.get(user_id, session_id).to_dict()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    authorization: Optional[str] = Header(None),
):
    user_id = _user_from_bearer(authorization)
    if not session_manager.delete(user_id, session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


@app.get("/api/sessions/{session_id}/history")
async def session_history(
    session_id: str,
    authorization: Optional[str] = Header(None),
    limit: int = 200,
):
    """
    Return prior turns for a session — used by the frontend when the
    user switches back to an older session and we want to re-render the
    conversation. Reads from the durable transaction_log, not the
    in-memory checkpointer.
    """
    user_id = _user_from_bearer(authorization)
    if session_manager.get(user_id, session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    from app.transaction_log import transaction_log
    turns = transaction_log.history_for_session(user_id, session_id, limit=limit)
    return {"session_id": session_id, "turns": turns, "count": len(turns)}


# === WebSocket endpoint ==============================================

@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    token = websocket.query_params.get("token")
    session_id = websocket.query_params.get("session_id")

    user_id = auth_store.resolve_token(token)
    if user_id is None:
        await websocket.close(code=4401, reason="invalid token")
        return

    if not session_id:
        await websocket.close(code=4400, reason="session_id required")
        return

    if session_manager.get(user_id, session_id) is None:
        await websocket.close(code=4404, reason="session not found")
        return

    await websocket.accept()
    log.info("WS opened user=%s session=%s", user_id, session_id)

    runner: GraphRunner = websocket.app.state.runner

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                user_message = msg.get("message", "")
            except json.JSONDecodeError:
                user_message = raw

            if not user_message.strip():
                await websocket.send_json({
                    "type": "error", "message": "empty message",
                })
                continue

            session_manager.touch(user_id, session_id)
            t0 = time.perf_counter()
            final_state: dict = {}

            try:
                async for event in runner.astream_tokens(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                ):
                    if event.get("type") == "final":
                        state = event.get("state", {})
                        if "messages" in state:
                            state["messages"] = [
                                {"role": m.type if hasattr(m, "type") else "unknown",
                                "content": m.content if hasattr(m, "content") else str(m)}
                                for m in state.get("messages", [])
                            ]
                        event["state"] = state

                    await websocket.send_json(event)
                    if event.get("type") == "final":
                        final_state = event.get("state") or {}
            except WebSocketDisconnect:
                log.info("client disconnected mid-turn user=%s", user_id)
                return
            except Exception as e:
                log.exception("turn failed: %s", e)
                await websocket.send_json({
                    "type": "error", "message": f"agent error: {e}",
                })
                continue

            latency_ms = (time.perf_counter() - t0) * 1000
            log_turn_metrics(final_state, latency_ms=latency_ms)

    except WebSocketDisconnect:
        log.info("WS closed user=%s session=%s", user_id, session_id)
    except Exception as e:
        log.exception("WS handler crashed: %s", e)
        try:
            await websocket.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass


@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.exception("unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error"},
    )

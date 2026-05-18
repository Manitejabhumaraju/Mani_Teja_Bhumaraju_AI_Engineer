"""
Streamlit chat UI for the Loadshare RCA Agent.

Runs separately from the FastAPI backend. Default backend URL is
http://localhost:8000 (override with API_URL env).

Three screens:
  1. Login - username/password form, hits POST /api/login
  2. Session sidebar - list / new / rename / delete (no message history
     across reloads - state is in the graph's checkpointer server-side,
     not here)
  3. Chat - sends messages over WS, streams stage + token events to a
     placeholder

Notes:
  - We use the synchronous `websockets` client inside the Streamlit
    callback. Streamlit reruns on every interaction, so opening a
    persistent WS connection across reruns is fragile. Per-turn open-
    send-recv-close is simpler and works correctly with reruns.
  - We DO NOT persist chat history in st.session_state. The graph's
    MemorySaver is server-side authority. Each turn we just append the
    user message and the streamed assistant response to a local rolling
    transcript for display - that's lost on browser refresh, which is
    fine (server still has it; reload the session to resume).
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import requests
import streamlit as st
import websockets


API_URL = os.environ.get("API_URL", "http://localhost:8000")
WS_URL = os.environ.get("WS_URL", "ws://localhost:8000")


# ============================ Helpers =================================

def api_post(path: str, json_body: dict, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(f"{API_URL}{path}", json=json_body, headers=headers, timeout=10)
    if r.status_code >= 400:
        return None, r.json().get("detail", f"HTTP {r.status_code}")
    return r.json(), None


def api_get(path: str, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"{API_URL}{path}", headers=headers, timeout=10)
    if r.status_code >= 400:
        return None, r.json().get("detail", f"HTTP {r.status_code}")
    return r.json(), None


def api_delete(path: str, token: Optional[str] = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.delete(f"{API_URL}{path}", headers=headers, timeout=10)
    if r.status_code >= 400:
        return None, r.json().get("detail", f"HTTP {r.status_code}")
    return r.json(), None


# ============================ State ===================================

if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "transcript" not in st.session_state:
    # local display only - server holds the real history
    st.session_state.transcript = []   # list of {role, content}


# ============================ Login ===================================

def render_login():
    st.title("Loadshare RCA Agent")
    st.caption("Sign in with the demo credentials (see README).")

    with st.form("login"):
        username = st.text_input("Username", value="demo")
        password = st.text_input("Password", value="demo", type="password")
        submit = st.form_submit_button("Sign in")

    if submit:
        data, err = api_post("/api/login", {"username": username, "password": password})
        if err:
            st.error(err)
            return
        st.session_state.token = data["token"]
        st.session_state.user_id = data["user_id"]
        st.rerun()


# ============================ Session sidebar =========================

def render_session_sidebar():
    with st.sidebar:
        st.markdown(f"**Signed in:** `{st.session_state.user_id}`")
        if st.button("Sign out", use_container_width=True):
            api_post("/api/logout", {}, token=st.session_state.token)
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.subheader("Sessions")

        data, err = api_get("/api/sessions", token=st.session_state.token)
        if err:
            st.error(f"Could not list sessions: {err}")
            return
        sessions = data.get("sessions", [])

        if st.button("➕ New session", use_container_width=True):
            new_data, _ = api_post("/api/sessions", {}, token=st.session_state.token)
            if new_data:
                st.session_state.session_id = new_data["session"]["session_id"]
                st.session_state.transcript = []
                st.rerun()

        if not sessions:
            st.caption("No sessions yet. Create one above.")
            return

        for s in sessions:
            sid = s["session_id"]
            label = s["name"]
            is_current = (sid == st.session_state.session_id)
            prefix = "▶ " if is_current else "   "
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(prefix + label, key=f"s_{sid}", use_container_width=True):
                    st.session_state.session_id = sid
                    st.session_state.transcript = []   # clear local display only
                    st.rerun()
            with col2:
                if st.button("✕", key=f"d_{sid}", help="Delete session"):
                    api_delete(f"/api/sessions/{sid}", token=st.session_state.token)
                    if st.session_state.session_id == sid:
                        st.session_state.session_id = None
                        st.session_state.transcript = []
                    st.rerun()


# ============================ Chat ====================================

async def stream_turn(token: str, session_id: str, user_message: str):
    """
    Open a WS, send one user message, yield events until the 'final' frame.

    Returns the final state so the caller can stash metadata.
    """
    url = f"{WS_URL}/ws/chat?token={token}&session_id={session_id}"
    final_state = None
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"message": user_message}))
        while True:
            raw = await ws.recv()
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "final":
                final_state = event.get("state") or {}
                break
            yield event
    yield {"type": "final", "state": final_state}


def run_async_gen_collect(coro_gen):
    """
    Helper to run an async generator from sync Streamlit code, accumulating
    yielded items. We don't actually stream into the UI live with this
    pattern - we collect all events, then render. Trade-off documented:
    Streamlit reruns + async generators don't compose cleanly enough to
    show per-token live updates without a custom component.

    For true per-token live streaming you'd swap this for st.empty() +
    asyncio.run with a callback. The agent backend supports it
    (astream_tokens yields piece-by-piece). The UI demo currently shows
    the response as one chunk after streaming completes server-side.
    """
    events: list[dict] = []

    async def _runner():
        async for ev in coro_gen:
            events.append(ev)

    asyncio.run(_runner())
    return events


def render_chat():
    sid = st.session_state.session_id
    if not sid:
        st.info("Select or create a session from the sidebar to start chatting.")
        return

    st.title("RCA Chat")
    st.caption(f"Session: `{sid}`")

    # Display rolling transcript
    for turn in st.session_state.transcript:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_message = st.chat_input(
        "Ask about a city, a store, or a time window (e.g. 'How did Bangalore do on 2026-04-22?')"
    )

    if user_message:
        # Show the user message immediately
        st.session_state.transcript.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        # Stream the response WITH LIVE UPDATES
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            response_text = ""
            final_state = {}
            current_stage = None
            
            try:
                # This runs the async generator and yields events live
                async def stream_and_display():
                    nonlocal response_text, final_state, current_stage
                    
                    async for event in stream_turn(
                        st.session_state.token, sid, user_message
                    ):
                        kind = event.get("type")
                        
                        if kind == "stage":
                            stage_name = event.get("name", "")
                            # Map technical names to user-friendly labels
                            stage_labels = {
                                "input_guard": "🛡️ Checking input...",
                                "context_resolver": "🔍 Understanding context...",
                                "intent_classifier": "🎯 Routing query...",
                                "city_rca": "📊 Analyzing city data...",
                                "store_rca": "🏪 Analyzing store data...",
                                "hour_drill": "⏰ Drilling into hours...",
                                "free_form": "💾 Running SQL query...",
                                "synthesizer": "✍️ Writing response...",
                                "output_guard": "✅ Validating output...",
                                "cache_hit": "⚡ Cache hit!",
                            }
                            current_stage = stage_labels.get(stage_name, f"Processing {stage_name}...")
                            status_placeholder.info(current_stage)
                        
                        elif kind == "token":
                            delta = event.get("delta", "")
                            response_text += delta
                            # Update the placeholder with accumulated text
                            response_placeholder.markdown(response_text + "▌")
                        
                        elif kind == "final":
                            final_state = event.get("state", {})
                            # Clear the cursor
                            response_placeholder.markdown(response_text)
                            status_placeholder.empty()
                        
                        elif kind == "error":
                            error_msg = event.get("message", "Unknown error")
                            status_placeholder.error(f"Error: {error_msg}")
                            return

                # Run the async function
                asyncio.run(stream_and_display())

            except Exception as e:
                response_placeholder.error(f"Stream error: {e}")
                return

            # Use final_state response if token stream was empty
            if not response_text and final_state.get("final_response"):
                response_text = final_state["final_response"]
                response_placeholder.markdown(response_text)

            # Show trace in expander
            if final_state:
                with st.expander("🔍 Trace"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Intent:**", final_state.get("intent"))
                        st.write("**Cache hit:**", final_state.get("cache_hit", False))
                    with col2:
                        st.write("**Input guard:**", final_state.get("input_guard_decision"))
                        st.write("**Output guard:**", final_state.get("output_guard_decision"))
                    st.write("**Nodes executed:**")
                    st.write(", ".join(final_state.get("nodes_executed", [])))

        st.session_state.transcript.append({"role": "assistant", "content": response_text})
        st.rerun()  # Refresh to clear the input box

# ============================ Main ====================================

def main():
    st.set_page_config(page_title="Loadshare RCA Agent", page_icon="📦", layout="wide")

    if not st.session_state.token:
        render_login()
        return

    render_session_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

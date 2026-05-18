"""
Graph builder + runner.

The builder wires up nodes, edges, and the checkpointer once at app boot.
The runner is the public surface the FastAPI/WebSocket layer calls. It
handles the semantic-cache layer on top of the graph (cache lookup before
running, cache write after) and exposes async streaming.

Why the cache wraps the graph instead of being a node:
  The cache stores deterministic RCA reports, not LLM-synthesized
  responses. On a hit, we still want the synthesizer to phrase the
  cached report in the current conversation's voice. So the cache sits
  between the input guard and the RCA path - if we hit, we skip the
  resolver/classifier/RCA nodes and feed the cached report straight to
  the synthesizer.

  Implementing this as a graph node would mean a conditional branch from
  cache_check to synthesizer-with-cached-report. Cleaner to keep cache
  logic in the runner and the graph focused on the analytical flow.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional

from flask import config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState, empty_context
from app.graph import nodes, router
from app.graph.nodes import NodeDeps


log = logging.getLogger(__name__)


def build_graph(deps: NodeDeps) -> Any:
    """
    Construct and compile the StateGraph.

    Topology:

        START
          ↓
        input_guard ──blocked──→ output_guard ──→ persist_turn ──→ END
          ↓ allowed
        context_resolver ──error──→ output_guard ─── ...
          ↓
        intent_classifier ──error/refuse──→ output_guard ─── ...
          ↓
        {city_rca | store_rca | hour_drill | free_form}
          ↓
        synthesizer
          ↓
        output_guard
          ↓
        persist_turn
          ↓
        END
    """
    g = StateGraph(GraphState)

    # Register agents as graph nodes. Each agent is a callable class
    # (state -> partial state); LangGraph treats it the same as a function.
    from app.agents import (
        InputGuardAgent, ContextAgent, IntentAgent,
        CityRCAAgent, StoreRCAAgent, HourDrillAgent, FreeFormAgent,
        SynthesizerAgent, OutputGuardAgent, PersistTurnAgent,
    )
    g.add_node("input_guard",       InputGuardAgent(deps))
    g.add_node("context_resolver",  ContextAgent(deps))
    g.add_node("intent_classifier", IntentAgent(deps))
    g.add_node("city_rca",          CityRCAAgent(deps))
    g.add_node("store_rca",         StoreRCAAgent(deps))
    g.add_node("hour_drill",        HourDrillAgent(deps))
    g.add_node("free_form",         FreeFormAgent(deps))
    # Meta-query: answers questions about prior turns from the transaction log
    from app.graph.nodes import make_meta_query_node
    g.add_node("meta_query",        make_meta_query_node(deps))
    g.add_node("synthesizer",       SynthesizerAgent(deps))
    g.add_node("output_guard",      OutputGuardAgent(deps))
    g.add_node("persist_turn",      PersistTurnAgent(deps))

    # Edges
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", router.after_input_guard)
    g.add_conditional_edges("context_resolver", router.after_context_resolver)
    g.add_conditional_edges("intent_classifier", router.after_intent_classifier)

    # All RCA paths converge at synthesizer
    for path in ("city_rca", "store_rca", "hour_drill", "free_form", "meta_query"):
        g.add_edge(path, "synthesizer")

    g.add_edge("synthesizer", "output_guard")
    g.add_edge("output_guard", "persist_turn")
    g.add_edge("persist_turn", END)

    # MemorySaver gives us thread-keyed checkpoints. thread_id is the
    # (user_id, session_id) tuple stringified at the call site.
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


# === Runner ===========================================================

class GraphRunner:
    """
    Public entry point. Wraps the compiled graph with cache logic +
    streaming-friendly interfaces.

    The runner is the only thing the WebSocket layer needs to know about.
    """

    def __init__(
        self,
        compiled_graph,
        deps: NodeDeps,
        cache=None,
    ):
        self._graph = compiled_graph
        self._deps = deps
        self._cache = cache

    @staticmethod
    def thread_id(user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    async def run_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> dict:
        """
        Run one full turn. Returns the final state dict.

        Use this for unit/eval testing where you want the whole result at
        once. For UI streaming use astream() below.
        """
        config = {"configurable": {"thread_id": self.thread_id(user_id, session_id)}}
        prior_snapshot = self._graph.get_state(config)
        prior_state = prior_snapshot.values if prior_snapshot else {}
        # Build the inbound state. The checkpointer merges this with prior
        # state for the same thread_id - so 'messages' accumulates across
        # turns and so does 'context'.
        initial_state: GraphState = {
            "user_id": user_id,
            "session_id": session_id,
            "user_message": user_message,
            "nodes_executed": [],
            "context": prior_state.get("context", empty_context()),
        }

        # Pre-check the semantic cache. If we hit, we still run the graph
        # but seed it with the cached report - synthesizer phrases it in
        # the current conversational voice.
        cache_hit_report = None
        if self._cache is not None:
            try:
                entry = await self._cache.get(user_id, user_message)
                if entry is not None:
                    cache_hit_report = entry.report
                    initial_state["cache_hit"] = True
            except Exception as e:
                log.warning("cache lookup failed: %s", e)

        if cache_hit_report is not None:
            # Bypass the analytical nodes - feed the cached report
            # directly to synthesizer. We do this by populating the
            # state ourselves and invoking the graph from a different
            # entry point... actually simpler: just call the synthesizer
            # node function directly, then output_guard.
            return await self._run_with_cached_report(
                initial_state, cache_hit_report,
            )

        final_state = await self._graph.ainvoke(initial_state, config=config)

        # Cache the deterministic report (if we have one and the response
        # was allowed). Not the synthesized text - that depends on tone /
        # context and shouldn't be frozen.
        if (
            self._cache is not None
            and final_state.get("report")
            and final_state.get("output_guard_decision") == "allow"
            and final_state.get("input_guard_decision") == "allow"
        ):
            try:
                await self._cache.put(
                    user_id, user_message, final_state["report"],
                )
            except Exception as e:
                log.warning("cache write failed: %s", e)

        return final_state

    async def _run_with_cached_report(
        self, initial_state: GraphState, report: str,
    ) -> dict:
        """
        Cache-hit path. We re-run synthesizer + output guard + persist with
        a manually-set report. Doesn't go through context resolution since
        the cached report already captures what the user asked.
        """
        state: GraphState = dict(initial_state)  # type: ignore[assignment]
        state["report"] = report
        state["intent"] = "cached"
        state["intent_reason"] = "served from semantic cache"
        state["context"] = empty_context()

        # Manually walk the remaining nodes.
        synth = nodes.make_synthesizer_node(self._deps)
        out_guard = nodes.make_output_guard_node(self._deps)
        persist = nodes.make_persist_turn_node(self._deps)

        for node_fn in (synth, out_guard, persist):
            updates = await node_fn(state)
            state.update(updates)  # type: ignore[arg-type]

        return state

    async def astream_tokens(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
    ) -> AsyncIterator[dict]:
        """
        Streaming variant for the WebSocket.

        Yields dicts of shape:
          {"type": "stage", "name": "context_resolver"}
          {"type": "token", "delta": "Bangalore"}
          {"type": "final", "state": {...full final state...}}

        The stage events are derived from graph.astream_events. Token
        events come from the synthesizer's LLM stream. The final event
        carries the whole settled state for the UI to inspect (intent,
        nodes_executed, etc).

        Note: LangGraph's astream_events doesn't natively forward
        sub-token streams from inside a node, so for streaming we run
        the graph up to the synthesizer normally, then call the
        synthesizer's stream manually. Trade-off documented below.
        """
        # Run the graph normally (gets us through input guard, context,
        # classifier, RCA path) - emit a stage event after each node.
        config = {"configurable": {"thread_id": self.thread_id(user_id, session_id)}}
        prior_snapshot = self._graph.get_state(config)
        prior_state = prior_snapshot.values if prior_snapshot else {}
        initial_state: GraphState = {
            "user_id": user_id,
            "session_id": session_id,
            "user_message": user_message,
            "nodes_executed": [],
            "context": prior_state.get("context", empty_context()),
        }

        # Cache check
        cache_hit_report = None
        if self._cache is not None:
            try:
                entry = await self._cache.get(user_id, user_message)
                if entry is not None:
                    cache_hit_report = entry.report
            except Exception as e:
                log.warning("cache lookup failed: %s", e)

        if cache_hit_report is not None:
            yield {"type": "stage", "name": "cache_hit"}
            # Stream synthesis of the cached report
            async for tok in self._stream_synthesis_only(
                user_id, session_id, user_message, cache_hit_report,
            ):
                yield tok
            return

        # Otherwise stream node-by-node updates via astream
        async for event in self._graph.astream(
            initial_state, config=config, stream_mode="updates",
        ):
            # event is {"node_name": state_updates}
            for node_name, updates in event.items():
                yield {"type": "stage", "name": node_name}

                # When we reach synthesizer's updates, the final_response is
                # already there as one piece because the node ran to completion.
                # For per-token streaming we'd need to refactor synthesizer
                # into the runner. This is the documented trade-off:
                #   - true per-token streaming requires bypassing the graph
                #     for that node, which we do in cache-hit path
                #   - graph-driven streaming gives stage granularity,
                #     useful for "thinking..." UX
                if node_name == "synthesizer":
                    response = updates.get("final_response", "")
                    # Emit the response as one chunk for now. UI can
                    # paint it with a typewriter effect if desired.
                    yield {"type": "token", "delta": response}

        # Pull the settled state from the checkpointer
        snapshot = self._graph.get_state(config)
        final_state = snapshot.values if snapshot else {}

        # Cache write
        if (
            self._cache is not None
            and final_state.get("report")
            and final_state.get("output_guard_decision") == "allow"
            and final_state.get("input_guard_decision") == "allow"
        ):
            try:
                await self._cache.put(
                    user_id, user_message, final_state["report"],
                )
            except Exception as e:
                log.warning("cache write failed: %s", e)

        yield {"type": "final", "state": final_state}

    async def _stream_synthesis_only(
        self, user_id: str, session_id: str,
        user_message: str, report: str,
    ) -> AsyncIterator[dict]:
        """Streams the synthesizer directly when we have a cached report."""
        from app.llm.prompts import SYNTHESIZER_SYSTEM

        # NOTE: NOT passing history to avoid cross-turn data leakage
        # (e.g. Bangalore data bleeding into a Chennai answer)
        
        yield {"type": "stage", "name": "synthesizer"}
        chunks: list[str] = []
        try:
            async for piece in self._deps.llm.complete_stream(
                system=SYNTHESIZER_SYSTEM,
                history=[],  # explicit: report is the only source of truth
                user=(
                    f"User asked: {user_message}\n\n"
                    f"=== RCA REPORT (the ONLY source of truth) ===\n"
                    f"{report}\n"
                    f"=== END REPORT ===\n\n"
                    f"Use ONLY numbers from the report. Do not reference prior turns."
                ),
                temperature=0.2,
                max_tokens=600,
            ):
                chunks.append(piece)
                yield {"type": "token", "delta": piece}
        except Exception as e:
            log.exception("cached-path synthesis failed: %s", e)
            yield {"type": "token", "delta": report}  # fall back to raw report

        yield {"type": "final", "state": {
            "user_id": user_id, "session_id": session_id,
            "cache_hit": True,
            "final_response": "".join(chunks),
            "report": report,
        }}

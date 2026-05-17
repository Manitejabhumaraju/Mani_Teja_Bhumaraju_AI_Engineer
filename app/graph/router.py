"""
Conditional edge routers.

LangGraph's conditional_edges hook expects callables of shape
(state) -> str_node_name. We keep them dead simple: no side effects,
no async, no state mutation. Just look at the state, decide where to go.
"""

from __future__ import annotations

from app.graph.state import GraphState


def after_input_guard(state: GraphState) -> str:
    """If the input was blocked, jump straight to output (which serves the refusal)."""
    if state.get("input_guard_decision") == "block":
        return "output_guard"
    return "context_resolver"


def after_context_resolver(state: GraphState) -> str:
    """If the LLM call failed, refuse. Otherwise classify intent."""
    if state.get("error"):
        return "output_guard"
    return "intent_classifier"


def after_intent_classifier(state: GraphState) -> str:
    """Route by intent. The five paths."""
    if state.get("error"):
        return "output_guard"

    intent = state.get("intent") or "refuse"
    mapping = {
        "city_rca":     "city_rca",
        "store_rca":    "store_rca",
        "hour_drill":   "hour_drill",
        "free_form":    "free_form",
        "schema_query": "free_form",   # collapse - schema is a free-form question
        "meta_query":   "meta_query",
        "refuse":       "output_guard",
    }
    return mapping.get(intent, "output_guard")


def after_rca_path(state: GraphState) -> str:
    """All RCA paths converge at the synthesizer (unless one of them errored)."""
    return "synthesizer"


def after_synthesizer(_state: GraphState) -> str:
    return "output_guard"


def after_output_guard(_state: GraphState) -> str:
    return "persist_turn"

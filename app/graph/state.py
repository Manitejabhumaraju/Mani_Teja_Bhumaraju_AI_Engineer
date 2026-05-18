"""
Shared graph state.

Anything that needs to be visible across nodes goes here. Anything that's
local to a node stays in node-local variables.

Key design choice: we keep a 'context' subdict explicitly. The LLM resolves
references like "what about TBC8?" by reading prior context + the current
message and emitting an updated context. That resolved context becomes the
input to the deterministic RCA routing. Doing it this way means coreference
is one focused LLM call, not a hidden side-effect of message history.

The 'messages' list is the LangChain-style chat history that flows into
the synthesizer. It's append-only across turns within a session.

We keep 'guard_verdict' and 'report' separate from 'final_response' so the
graph has clear hand-off points - guardrails see the synth response before
the user does, retries can re-enter the synthesizer with the same report,
and observability tools see each stage independently.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class ConversationContext(TypedDict, total=False):
    """
    The resolved view of what the user is asking about.

    Populated by the context_resolver node, consumed by intent_classifier
    and the RCA path nodes. Persists across turns - "morning hours?" on
    turn 3 reads last_store from turn 2.
    """
    scope: str                 # "city" | "store" | "hour_range" | "clarification"
    city: Optional[str]
    store: Optional[str]
    time_range: Optional[str]  # named range from time_ranges.py
    hours: Optional[list[int]] # resolved hour ints
    date: Optional[str]        # ISO YYYY-MM-DD
    is_followup: bool
    clarification_question: Optional[str]

    # Sticky values from the last turn - used when current turn omits them.
    last_city: Optional[str]
    last_store: Optional[str]
    last_hours: Optional[list[int]]


class GraphState(TypedDict, total=False):
    """
    Flows through every node. LangGraph merges per-node returns into this.

    `messages` uses LangGraph's add_messages reducer so node returns can
    append rather than overwrite. Everything else is overwrite-on-return,
    which is what we want.
    """
    # Identity
    user_id: str
    session_id: str

    # This turn
    user_message: str
    context: ConversationContext

    # Routing decision
    intent: str                # "city_rca" | "store_rca" | "hour_drill" | "schema_query" | "free_form" | "refuse"
    intent_reason: str

    # Tool / engine outputs
    report: str                # deterministic RCA output, fed to synthesizer
    raw_data: dict             # structured payload (rollups, findings) - for logs / debug

    # Guard verdicts (keep history for observability)
    input_guard_decision: str  # "allow" | "block"
    input_guard_reason: str
    output_guard_decision: str
    output_guard_reason: str

    # Final
    final_response: str
    cache_hit: bool

    # Conversation history (for the synthesizer's context window)
    messages: Annotated[list[dict], add_messages]

    # Metrics
    nodes_executed: list[str]  # order of nodes traversed, for tracing
    error: Optional[str]       # set on unrecoverable error - stops the graph


def empty_context() -> ConversationContext:
    """Initial blank context. Used when a session has no prior turns."""
    return ConversationContext(
        scope="clarification",
        city=None,
        store=None,
        time_range=None,
        hours=None,
        date=None,
        is_followup=False,
        clarification_question=None,
        last_city=None,
        last_store=None,
        last_hours=None,
    )


def merge_context(prior: ConversationContext, resolved: ConversationContext) -> ConversationContext:
    """
    Carry forward last_* slots from prior turn so coreference works.

    The resolver fills the current-turn slots. We separately propagate
    last_city/last_store/last_hours from the previous turn so 'morning
    hours?' can find STORE_003 even though the user didn't say it again.
    """
    merged: ConversationContext = dict(resolved)  # type: ignore[assignment]

    # The current turn's resolved values become "last" for the next turn.
    # We update last_* with whichever is more specific.
    merged["last_city"] = resolved.get("city") or prior.get("last_city")
    merged["last_store"] = resolved.get("store") or prior.get("last_store")
    merged["last_hours"] = resolved.get("hours") or prior.get("last_hours")
    return merged

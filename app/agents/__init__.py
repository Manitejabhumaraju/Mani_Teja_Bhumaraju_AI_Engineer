"""
First-class Agent classes.

Each agent is a callable that takes GraphState and returns a partial state
dict. LangGraph happily accepts callable objects as nodes, so the agents
plug into the existing StateGraph without an adapter layer.

Why classes instead of node-factory closures (the previous design):
  - Each agent is one file with one responsibility — easier to reason about
    in isolation, easier to unit test (inject a mock deps object).
  - The agent name and its prompt template live in the same file.
  - Sub-agents can be composed: a SupervisorAgent can hold instances of
    several sub-agents and dispatch to them, which is what the DeepAgents
    variant builds on.
"""

from app.agents.base import BaseAgent, AgentDeps
from app.agents.input_guard_agent import InputGuardAgent
from app.agents.context_agent import ContextAgent
from app.agents.intent_agent import IntentAgent
from app.agents.city_rca_agent import CityRCAAgent
from app.agents.store_rca_agent import StoreRCAAgent
from app.agents.hour_drill_agent import HourDrillAgent
from app.agents.free_form_agent import FreeFormAgent
from app.agents.synthesizer_agent import SynthesizerAgent
from app.agents.output_guard_agent import OutputGuardAgent
from app.agents.persist_turn_agent import PersistTurnAgent

__all__ = [
    "BaseAgent", "AgentDeps",
    "InputGuardAgent", "ContextAgent", "IntentAgent",
    "CityRCAAgent", "StoreRCAAgent", "HourDrillAgent", "FreeFormAgent",
    "SynthesizerAgent", "OutputGuardAgent", "PersistTurnAgent",
]

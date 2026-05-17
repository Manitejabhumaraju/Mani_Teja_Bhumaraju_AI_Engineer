"""
Base agent class.

An agent is a callable: state in, partial-state-update dict out. That's the
contract LangGraph expects from a node, so agents are literally graph nodes
with a class shape.

AgentDeps is a thin re-export of NodeDeps. We keep two names because the
old code calls it NodeDeps and downstream tests still import that; aliasing
lets the refactor land without churn.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.graph.nodes import NodeDeps as AgentDeps  # alias
from app.graph.state import GraphState


log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract callable agent.

    Subclasses implement `run(state) -> dict`. The base class wraps `run`
    with consistent error handling, structured logging, and the
    `nodes_executed` trail update so the graph trace shows which agents
    actually ran.
    """

    # Set by subclasses for log/trace breadcrumbs.
    name: str = "base"

    def __init__(self, deps: AgentDeps):
        self.deps = deps

    @abstractmethod
    async def run(self, state: GraphState) -> dict:
        """Return a partial state update. Raise to abort the turn."""
        ...

    async def __call__(self, state: GraphState) -> dict:
        """LangGraph entrypoint. Catches exceptions so one agent error
        doesn't take down the whole turn — the graph will route to
        output_guard which serves a refusal."""
        try:
            patch = await self.run(state)
            return self._append_trace(patch)
        except Exception as e:
            log.exception("%s agent raised: %s", self.name, e)
            return {
                "error": f"{self.name} failed: {e}",
                "nodes_executed": self._trail(state),
            }

    def _append_trace(self, patch: dict) -> dict:
        """Ensure every successful run appends this agent's name to the
        execution trail, without overwriting anything else."""
        trail = patch.get("nodes_executed")
        if trail is None:
            patch["nodes_executed"] = self._trail({})
        return patch

    def _trail(self, state: GraphState) -> list[str]:
        existing = list(state.get("nodes_executed") or [])
        existing.append(self.name)
        return existing

"""
OutputGuardAgent: agent wrapper around the output_guard node factory.

The factory function is the existing, tested implementation. The agent
class is the OO surface that the graph builder now consumes. Keeping the
internals as a closure means the agent's logic stays in one place and the
class file stays small — easy to reason about, easy to test.

To replace the implementation, override `run()` here. The base class still
handles error capture and the execution trail.
"""

from __future__ import annotations

from app.agents.base import BaseAgent, AgentDeps
from app.graph.nodes import make_output_guard_node
from app.graph.state import GraphState


class OutputGuardAgent(BaseAgent):
    name = "output_guard"

    def __init__(self, deps: AgentDeps):
        super().__init__(deps)
        # Build the underlying node closure once; reuse on every call.
        self._fn = make_output_guard_node(deps)

    async def run(self, state: GraphState) -> dict:
        return await self._fn(state)

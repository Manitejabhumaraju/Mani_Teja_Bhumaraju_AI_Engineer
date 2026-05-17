"""
Simple in-memory store - drop-in replacement for mem0ai.

Mimics the mem0 interface so the rest of the code doesn't need to change.
No external dependencies, no API keys needed.

For production, swap for mem0ai with NVIDIA NIM or Qdrant backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from collections import defaultdict


log = logging.getLogger(__name__)


class SimpleMemory:
    """
    In-memory key-value store with mem0-like interface.
    
    Stores memories per user_id, useful for tracking entities discussed
    across multi-turn conversations.
    """
    
    def __init__(self):
        # user_id -> list of memory dicts
        self._memories: dict[str, list[dict]] = defaultdict(list)
        log.info("SimpleMemory initialized (in-process)")
    
    def add(self, messages: str | list, user_id: str = "default", **kwargs) -> dict:
        """
        Add a memory. Accepts either a string or list of messages.
        
        Mimics mem0.Memory.add() interface.
        """
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            # Handle [{"role": "user", "content": "..."}] format
            text = " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m)
                for m in messages
            )
        else:
            text = str(messages)
        
        memory = {
            "id": f"mem_{len(self._memories[user_id])}_{int(time.time())}",
            "memory": text,
            "user_id": user_id,
            "created_at": time.time(),
            "metadata": kwargs.get("metadata", {}),
        }
        
        # Keep last 20 memories per user
        self._memories[user_id].append(memory)
        if len(self._memories[user_id]) > 20:
            self._memories[user_id] = self._memories[user_id][-20:]
        
        return {"results": [memory]}
    
    def search(self, query: str, user_id: str = "default", limit: int = 5, **kwargs) -> dict:
        """
        Search memories with simple keyword matching.
        
        Mimics mem0.Memory.search() interface.
        Returns most recent memories matching query keywords.
        """
        if user_id not in self._memories:
            return {"results": []}
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score memories by word overlap
        scored = []
        for mem in self._memories[user_id]:
            mem_lower = mem["memory"].lower()
            mem_words = set(mem_lower.split())
            
            # Simple overlap score
            overlap = len(query_words & mem_words)
            if overlap > 0 or any(word in mem_lower for word in query_words):
                scored.append((overlap, mem))
        
        # Sort by score desc, then recency desc
        scored.sort(key=lambda x: (-x[0], -x[1]["created_at"]))
        results = [mem for _, mem in scored[:limit]]
        
        return {"results": results}
    
    def get_all(self, user_id: str = "default", **kwargs) -> dict:
        """Get all memories for a user."""
        return {"results": list(self._memories.get(user_id, []))}
    
    def delete_all(self, user_id: str = "default", **kwargs) -> dict:
        """Clear all memories for a user."""
        if user_id in self._memories:
            self._memories[user_id] = []
        return {"results": "deleted"}
    
    def reset(self) -> None:
        """Clear all memories for all users."""
        self._memories.clear()

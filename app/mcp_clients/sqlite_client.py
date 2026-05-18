"""
SQLite HTTP client - simple REST API.
"""

import httpx
import logging
from typing import Any

log = logging.getLogger(__name__)


class SqliteMCPClient:
    """HTTP client for SQLite server."""
    
    def __init__(self, base_url: str = "http://localhost:3002"):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None
    
    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
    
    async def list_tables(self) -> list[str]:
        """List all tables."""
        await self._ensure_client()
        resp = await self._client.get(f"{self.base_url}/tables")
        resp.raise_for_status()
        return resp.json()["tables"]
    
    async def describe_table(self, table: str) -> list[dict]:
        """Get table schema."""
        await self._ensure_client()
        resp = await self._client.get(f"{self.base_url}/schema/{table}")
        resp.raise_for_status()
        return resp.json()["columns"]
    
    async def read_query(self, sql: str) -> list[dict]:
        """Execute SELECT query."""
        await self._ensure_client()
        resp = await self._client.post(
            f"{self.base_url}/query",
            json={"sql": sql}
        )
        resp.raise_for_status()
        return resp.json()["rows"]
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

"""
MCP Manager - HTTP SQLite only.
"""

import logging
from pathlib import Path
from typing import Callable, Optional
import httpx

from app.mcp_clients.sqlite_client import SqliteMCPClient

log = logging.getLogger(__name__)


class MCPManager:
    """Manages HTTP SQLite connection."""
    
    def __init__(
        self,
        docs_dir: Path,
        db_path: Path,
        sql_validator: Callable[[str], None],
        sqlite_url: str = "http://localhost:3002",
    ):
        self._sqlite_url = sqlite_url
        self._sql_validator = sql_validator
        self.fs = None
        self.sqlite: Optional[SqliteMCPClient] = None
    
    async def start(self) -> None:
        """Connect to SQLite HTTP server."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._sqlite_url}/health")
                resp.raise_for_status()
                log.info("✅ SQLite MCP server reachable at %s", self._sqlite_url)
        except httpx.ConnectError:
            log.error(
                "❌ Cannot connect to SQLite MCP server at %s\n"
                "   Start it first: cd mcp-servers && npm start",
                self._sqlite_url
            )
            raise ConnectionRefusedError(f"SQLite MCP not running at {self._sqlite_url}")
        except Exception as e:
            log.error("SQLite MCP health check failed: %s", e)
            raise
        
        self.sqlite = SqliteMCPClient(self._sqlite_url)
        log.info("✅ SQLite MCP connected via HTTP")
    
    async def stop(self) -> None:
        if self.sqlite:
            await self.sqlite.close()
            self.sqlite = None
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

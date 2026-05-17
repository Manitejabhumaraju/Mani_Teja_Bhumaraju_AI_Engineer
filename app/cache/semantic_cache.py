"""
Semantic cache for the RCA agent.

What we cache:
  The deterministic RCA report (the engine output, not the LLM's wording).
  We re-synthesize the conversational wrapping each time so cached entries
  don't freeze the agent's tone or break coreference across turns.

What we DON'T cache:
  Raw LLM completions. Those depend on conversation history, which means
  "How did Bangalore do?" returns different wording on turn 1 vs turn 5,
  and freezing turn 1's wording into the cache would be wrong.

Lookup approach:
  At demo scale (likely <100 entries per session), a numpy dot-product scan
  is sub-millisecond. We do not pretend to need ANN. At 100k+ entries we
  would either move to Redis Vector Search (built into Redis 8) or stand up
  a proper vector store. That migration is one file change.

Scoping:
  Cache keys include user_id so demo's questions don't leak answers to
  analyst. Same query from two different users is two different cache
  entries. The wasted-work cost is acceptable for the privacy benefit.

Persistence:
  Redis. If Redis is unreachable we degrade to a no-op cache rather than
  crashing - the agent works fine without caching, just slower on repeats.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from app.cache.embeddings import EmbeddingModel


log = logging.getLogger(__name__)


# === Stored entry shape ==============================================

@dataclass
class CacheEntry:
    """One cached (query -> report) pair plus metadata."""
    query: str
    report: str                       # the deterministic RCA output
    embedding_b64: str                # base64-encoded float32 vector
    user_id: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            query=d["query"],
            report=d["report"],
            embedding_b64=d["embedding_b64"],
            user_id=d["user_id"],
            created_at=float(d.get("created_at", time.time())),
            hit_count=int(d.get("hit_count", 0)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# === Helpers =========================================================

def _b64_encode(vec: np.ndarray) -> str:
    import base64
    return base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")


def _b64_decode(s: str, dim: int) -> np.ndarray:
    import base64
    raw = base64.b64decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape((dim,))


def _query_key(user_id: str, query: str) -> str:
    """Used for exact-match short-circuit before the vector scan."""
    h = hashlib.sha1(f"{user_id}|{query.strip().lower()}".encode("utf-8")).hexdigest()
    return f"rcacache:exact:{h}"


def _user_index_key(user_id: str) -> str:
    """A Redis SET holding all entry keys belonging to a user. Used for the scan."""
    return f"rcacache:idx:{user_id}"


def _entry_key(user_id: str, query: str) -> str:
    """Storage key for a full CacheEntry."""
    h = hashlib.sha1(f"{user_id}|{query.strip().lower()}".encode("utf-8")).hexdigest()
    return f"rcacache:entry:{h}"


# === Cache ===========================================================

class SemanticCache:
    """
    Per-user semantic cache backed by Redis.

    Methods are async because Redis I/O is async and we don't want to
    block the event loop on cache calls. The embedder's encode is CPU-
    bound but small (<10ms on bge-small with 384 dims).
    """

    def __init__(
        self,
        redis_client,
        embedder: EmbeddingModel,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_entries_per_user: int = 100,
    ):
        self._redis = redis_client
        self._embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries_per_user = max_entries_per_user

        # Lazy state - we discover the embedding dim from the first encode.
        self._dim: Optional[int] = None

    async def _dim_ready(self) -> int:
        if self._dim is None:
            self._dim = self._embedder.dim
        return self._dim

    async def get(self, user_id: str, query: str) -> Optional[CacheEntry]:
        """
        Return cached entry if either:
          1. exact match on normalized query (cheap), or
          2. semantic match above threshold (vector scan).

        Returns None on miss or any Redis error.
        """
        if not user_id or not query or not query.strip():
            return None

        try:
            # 1. Exact-match short circuit.
            exact_key = _entry_key(user_id, query)
            raw = await self._redis.get(exact_key)
            if raw:
                entry = CacheEntry.from_dict(json.loads(raw))
                await self._record_hit(exact_key, entry)
                log.info("cache HIT (exact) user=%s query=%r", user_id, query[:60])
                return entry

            # 2. Semantic scan.
            return await self._semantic_lookup(user_id, query)

        except Exception as e:
            log.warning("cache.get failed (degrading to miss): %s", e)
            return None

    async def _semantic_lookup(
        self, user_id: str, query: str
    ) -> Optional[CacheEntry]:
        idx_key = _user_index_key(user_id)
        keys = await self._redis.smembers(idx_key)
        if not keys:
            return None

        # Normalize set members - python redis client may return bytes
        key_list = [k.decode("utf-8") if isinstance(k, bytes) else k for k in keys]

        # Load all entries in one MGET. Capped by max_entries_per_user.
        raws = await self._redis.mget(key_list)
        entries: list[tuple[str, CacheEntry]] = []
        for k, raw in zip(key_list, raws):
            if raw is None:
                # Entry was evicted; clean up index. Best-effort, ignore failure.
                try:
                    await self._redis.srem(idx_key, k)
                except Exception:
                    pass
                continue
            try:
                entries.append((k, CacheEntry.from_dict(json.loads(raw))))
            except Exception:
                continue

        if not entries:
            return None

        # Encode the query and score against every stored vector. Dot product
        # works because embeddings are L2-normalized in EmbeddingModel.encode.
        dim = await self._dim_ready()
        q_vec = await self._embedder.encode_async(query)
        matrix = np.vstack([_b64_decode(e.embedding_b64, dim) for _, e in entries])
        scores = matrix @ q_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.similarity_threshold:
            best_key, best_entry = entries[best_idx]
            await self._record_hit(best_key, best_entry)
            log.info(
                "cache HIT (semantic) user=%s score=%.3f query=%r matched=%r",
                user_id, best_score, query[:60], best_entry.query[:60],
            )
            return best_entry

        log.debug(
            "cache MISS (best semantic score %.3f below threshold %.3f) query=%r",
            best_score, self.similarity_threshold, query[:60],
        )
        return None

    async def put(self, user_id: str, query: str, report: str) -> None:
        """
        Store a (query, report) pair for this user. Caps per-user entries
        so a single user can't fill the cache.
        """
        if not user_id or not query.strip() or not report.strip():
            return

        try:
            dim = await self._dim_ready()
            vec = await self._embedder.encode_async(query)
            assert vec.shape == (dim,), f"unexpected vec shape {vec.shape}"

            entry = CacheEntry(
                query=query,
                report=report,
                embedding_b64=_b64_encode(vec),
                user_id=user_id,
            )

            entry_key = _entry_key(user_id, query)
            idx_key = _user_index_key(user_id)

            # Best-effort eviction: if user already at cap, drop the oldest.
            await self._maybe_evict(user_id, idx_key)

            pipe = self._redis.pipeline()
            pipe.set(entry_key, json.dumps(entry.to_dict()), ex=self.ttl_seconds)
            pipe.sadd(idx_key, entry_key)
            pipe.expire(idx_key, self.ttl_seconds * 2)   # index outlives entries a bit
            await pipe.execute()

            log.debug("cache PUT user=%s query=%r", user_id, query[:60])

        except Exception as e:
            log.warning("cache.put failed (continuing without cache): %s", e)

    async def _maybe_evict(self, user_id: str, idx_key: str) -> None:
        """If user is at cap, drop the oldest entry. Best-effort."""
        try:
            count = await self._redis.scard(idx_key)
            if count < self.max_entries_per_user:
                return

            keys = await self._redis.smembers(idx_key)
            key_list = [k.decode("utf-8") if isinstance(k, bytes) else k for k in keys]
            raws = await self._redis.mget(key_list)

            oldest_key = None
            oldest_ts = float("inf")
            for k, raw in zip(key_list, raws):
                if raw is None:
                    await self._redis.srem(idx_key, k)
                    continue
                try:
                    ts = float(json.loads(raw).get("created_at", 0))
                except Exception:
                    continue
                if ts < oldest_ts:
                    oldest_ts = ts
                    oldest_key = k

            if oldest_key:
                pipe = self._redis.pipeline()
                pipe.delete(oldest_key)
                pipe.srem(idx_key, oldest_key)
                await pipe.execute()
                log.debug("cache EVICT (LRU-ish) user=%s key=%s", user_id, oldest_key)
        except Exception as e:
            log.debug("eviction skipped: %s", e)

    async def _record_hit(self, entry_key: str, entry: CacheEntry) -> None:
        """Bump hit count. Best-effort - failures don't break the read path."""
        try:
            entry.hit_count += 1
            await self._redis.set(
                entry_key, json.dumps(entry.to_dict()), ex=self.ttl_seconds, xx=True
            )
        except Exception:
            pass

    async def clear_user(self, user_id: str) -> int:
        """Clear all cache for one user. Returns number of entries removed."""
        try:
            idx_key = _user_index_key(user_id)
            keys = await self._redis.smembers(idx_key)
            if not keys:
                return 0
            key_list = [k.decode("utf-8") if isinstance(k, bytes) else k for k in keys]
            pipe = self._redis.pipeline()
            for k in key_list:
                pipe.delete(k)
            pipe.delete(idx_key)
            await pipe.execute()
            return len(key_list)
        except Exception as e:
            log.warning("cache.clear_user failed: %s", e)
            return 0


# === No-op cache (used when Redis isn't available) ====================

class NoOpCache:
    """
    Stand-in when Redis isn't configured or is unreachable. Same interface
    as SemanticCache so callers don't need to know which they have.
    """

    async def get(self, user_id: str, query: str) -> Optional[CacheEntry]:
        return None

    async def put(self, user_id: str, query: str, report: str) -> None:
        return None

    async def clear_user(self, user_id: str) -> int:
        return 0

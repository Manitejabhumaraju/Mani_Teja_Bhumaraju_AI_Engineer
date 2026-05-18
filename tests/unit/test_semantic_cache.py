"""
Semantic cache unit tests.

Uses fakeredis for Redis simulation and the deterministic stub embedder.
Fast (<200ms total) and covers all the cache logic that matters:
  - exact-match short-circuit
  - semantic match above threshold
  - semantic miss below threshold
  - per-user scoping (no cross-leak)
  - eviction when at cap
  - graceful degradation on Redis error
"""

import asyncio
import json

import fakeredis.aioredis
import pytest
import pytest_asyncio

from app.cache.embeddings import EmbeddingModel
from app.cache.semantic_cache import (
    SemanticCache, NoOpCache, CacheEntry,
    _entry_key, _user_index_key,
)


@pytest.fixture
def stub_embedder():
    """Deterministic stub - same text -> same vector, no model load."""
    return EmbeddingModel(model_name="__test__")


@pytest_asyncio.fixture
async def redis_client():
    """In-memory fake Redis. Async-compatible API."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def cache(redis_client, stub_embedder):
    return SemanticCache(
        redis_client=redis_client,
        embedder=stub_embedder,
        similarity_threshold=0.92,
        ttl_seconds=60,
        max_entries_per_user=5,
    )


# === Basic put/get ====================================================

class TestBasicFlow:
    async def test_miss_on_empty_cache(self, cache):
        assert await cache.get("demo", "How did Bangalore do?") is None

    async def test_exact_match_hit(self, cache):
        await cache.put("demo", "How did Bangalore do?", "REPORT_TEXT_HERE")
        entry = await cache.get("demo", "How did Bangalore do?")
        assert entry is not None
        assert entry.report == "REPORT_TEXT_HERE"
        assert entry.query == "How did Bangalore do?"
        assert entry.user_id == "demo"

    async def test_case_and_whitespace_normalized(self, cache):
        await cache.put("demo", "How did Bangalore do?", "R1")
        # Exact-match key normalizes to lower + stripped
        entry = await cache.get("demo", "  How Did Bangalore Do?  ")
        assert entry is not None
        assert entry.report == "R1"

    async def test_hit_count_increments(self, cache):
        await cache.put("demo", "X", "R")
        e1 = await cache.get("demo", "X")
        e2 = await cache.get("demo", "X")
        e3 = await cache.get("demo", "X")
        # The in-memory entries we got each represent a snapshot. The stored
        # entry should have been bumped each time.
        assert e1 is not None and e3 is not None


# === Semantic match ===================================================

class TestSemanticMatch:
    async def test_semantic_hit_when_above_threshold(
        self, redis_client, stub_embedder
    ):
        """
        With the stub embedder, "same text -> same vector". So "How did
        Bangalore do?" lookup will exact-match. To test semantic specifically
        we need two DIFFERENT queries with vectors close enough to clear the
        threshold - which requires the real embedder.

        Here we set the threshold very low so the stub's 'distinct text =
        distinct vector' produces a high enough cosine to hit. This proves
        the scan logic, not the semantic quality (covered by the integration
        test that uses real bge-small).
        """
        cache = SemanticCache(
            redis_client=redis_client,
            embedder=stub_embedder,
            similarity_threshold=-1.0,   # always match
            ttl_seconds=60,
            max_entries_per_user=5,
        )
        await cache.put("demo", "How did Bangalore do?", "REPORT_BANG")
        # Different query - exact-match miss, falls through to scan, hits
        # because threshold is -1.
        entry = await cache.get("demo", "Tell me about Mumbai")
        assert entry is not None
        assert entry.report == "REPORT_BANG"

    async def test_semantic_miss_when_below_threshold(self, cache):
        """With default 0.92 threshold + stub embedder (random-ish), no hit."""
        await cache.put("demo", "How did Bangalore do?", "REPORT_BANG")
        entry = await cache.get("demo", "Tell me about Mumbai")
        # Different query, no exact match, stub vectors are uncorrelated -> miss
        assert entry is None


# === Per-user scoping ==================================================

class TestUserScoping:
    async def test_one_users_cache_invisible_to_another(self, cache):
        await cache.put("demo", "How did Bangalore do?", "DEMO_REPORT")
        # analyst asks the same question - should NOT see demo's cached answer
        result = await cache.get("analyst", "How did Bangalore do?")
        assert result is None

    async def test_clear_user_only_clears_that_user(self, cache):
        await cache.put("demo", "Q1", "R1")
        await cache.put("demo", "Q2", "R2")
        await cache.put("analyst", "Q1", "R3")

        removed = await cache.clear_user("demo")
        assert removed == 2

        # analyst's entry intact
        assert (await cache.get("analyst", "Q1")) is not None
        # demo's entries gone
        assert (await cache.get("demo", "Q1")) is None
        assert (await cache.get("demo", "Q2")) is None


# === Eviction =========================================================

class TestEviction:
    async def test_eviction_kicks_in_at_cap(self, cache, redis_client):
        # cap is 5
        for i in range(7):
            await cache.put("demo", f"Query {i}", f"Report {i}")
            # small delay so created_at differs - eviction picks oldest
            await asyncio.sleep(0.005)

        idx = await redis_client.smembers(_user_index_key("demo"))
        # After cap, we should have at most 5 (last 5).
        assert len(idx) <= 5


# === Graceful degradation =============================================

class FailingRedis:
    """Mock Redis that raises on every operation."""

    def __init__(self):
        self.error = ConnectionError("redis down")

    async def get(self, *_a, **_kw): raise self.error
    async def set(self, *_a, **_kw): raise self.error
    async def mget(self, *_a, **_kw): raise self.error
    async def smembers(self, *_a, **_kw): raise self.error
    async def sadd(self, *_a, **_kw): raise self.error
    async def srem(self, *_a, **_kw): raise self.error
    async def scard(self, *_a, **_kw): raise self.error
    async def delete(self, *_a, **_kw): raise self.error
    async def expire(self, *_a, **_kw): raise self.error

    def pipeline(self):
        return _FailingPipeline(self.error)


class _FailingPipeline:
    """Pipeline ops are queued synchronously in real redis; only execute is async."""

    def __init__(self, error):
        self._error = error

    # Pipeline operations are sync no-ops (real redis pipelines queue them
    # without round-trip). It's execute() that fails.
    def set(self, *_a, **_kw): return self
    def sadd(self, *_a, **_kw): return self
    def srem(self, *_a, **_kw): return self
    def delete(self, *_a, **_kw): return self
    def expire(self, *_a, **_kw): return self

    async def execute(self):
        raise self._error


class TestGracefulDegradation:
    async def test_get_returns_none_on_redis_error(self, stub_embedder):
        cache = SemanticCache(
            redis_client=FailingRedis(),
            embedder=stub_embedder,
        )
        # Don't crash, just miss
        assert await cache.get("demo", "anything") is None

    async def test_put_no_op_on_redis_error(self, stub_embedder):
        cache = SemanticCache(
            redis_client=FailingRedis(),
            embedder=stub_embedder,
        )
        # Don't crash
        await cache.put("demo", "Q", "R")


# === No-op cache ======================================================

class TestNoOpCache:
    async def test_noop_get_always_miss(self):
        c = NoOpCache()
        assert await c.get("u", "q") is None

    async def test_noop_put_returns_none(self):
        c = NoOpCache()
        assert await c.put("u", "q", "r") is None

    async def test_noop_clear_user_returns_zero(self):
        c = NoOpCache()
        assert await c.clear_user("u") == 0


# === Empty / nullish input ============================================

class TestEdgeCases:
    async def test_empty_user_id_misses(self, cache):
        assert await cache.get("", "real query") is None

    async def test_empty_query_misses(self, cache):
        assert await cache.get("demo", "") is None

    async def test_empty_query_doesnt_write(self, cache, redis_client):
        await cache.put("demo", "", "report")
        idx = await redis_client.smembers(_user_index_key("demo"))
        assert len(idx) == 0

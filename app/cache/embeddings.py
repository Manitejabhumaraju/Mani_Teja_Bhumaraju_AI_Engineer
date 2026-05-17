"""
Embedding model wrapper.

Uses sentence-transformers locally with bge-small-en-v1.5 by default.
384-dim vectors, ~80MB model, CPU-fine for our throughput (one embedding
per user message, batched to one per query).

Why local and not NVIDIA NIM's embedding endpoint:
  - No API rate limits during a demo
  - No network latency for a hot-path call (embedding the incoming query
    happens before we know whether the cache will hit)
  - bge-small is consistently near the top of MTEB for its size, free,
    and ships under MIT-equivalent terms

Threading:
  - sentence-transformers under the hood uses torch; encoding is
    GIL-released and safe to call from threads. We don't need an async
    wrapper - the model call is CPU-bound, not I/O-bound, so wrapping in
    asyncio.to_thread is the right pattern when called from async code.

Lifecycle:
  - First .encode() call loads the model (~2-3s cold start on CPU).
  - We force-load at app boot via .warmup() to keep the first user request
    fast.
"""

from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Singleton-style local embedder. Construct once, share everywhere.

    Test note: passing model_name="__test__" returns a stub model that
    generates deterministic-but-distinct vectors from a hash. Lets cache
    tests run in <1s without loading 80MB of weights.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._dim: Optional[int] = None
        self._load_lock = Lock()

    def warmup(self) -> None:
        """Force model load at app boot rather than on first user message."""
        self._load()
        log.info("embedding model %s warmed up (dim=%d)", self.model_name, self._dim)

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:   # double-check after lock
                return

            if self.model_name == "__test__":
                self._model = _DeterministicStubModel(dim=64)
                self._dim = 64
                return

            # Defer import so test stub path doesn't pay the cost.
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Probe dim with a tiny encode call. Cheaper than hardcoding.
            probe = self._model.encode("probe", convert_to_numpy=True)
            self._dim = int(probe.shape[-1])

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim  # type: ignore[return-value]

    def encode(self, text: str) -> np.ndarray:
        """Synchronous encode. L2-normalized so cosine = dot product."""
        self._load()
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    async def encode_async(self, text: str) -> np.ndarray:
        """Async-friendly wrapper. Offloads to a thread - the encode itself is CPU-bound."""
        return await asyncio.to_thread(self.encode, text)


class _DeterministicStubModel:
    """
    Test-only stub. Produces stable, distinct vectors from text hash.
    NOT semantically meaningful - just lets cache plumbing be tested.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str, convert_to_numpy: bool = True,
               normalize_embeddings: bool = True) -> np.ndarray:
        import hashlib
        # 64-byte hash gives us up to 512 bits of entropy; truncate to dim*4 bytes
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch hash to required length
        while len(h) < self.dim * 4:
            h = h + hashlib.sha256(h).digest()
        vec = np.frombuffer(h[: self.dim * 4], dtype=np.uint32).astype(np.float32)
        # Center around 0 and normalize so cosine works
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        if norm > 0 and normalize_embeddings:
            vec = vec / norm
        return vec

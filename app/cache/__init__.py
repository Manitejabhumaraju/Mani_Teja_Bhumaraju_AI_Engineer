"""Semantic cache: local embeddings + Redis backend."""

from app.cache.embeddings import EmbeddingModel
from app.cache.semantic_cache import SemanticCache, CacheEntry

__all__ = ["EmbeddingModel", "SemanticCache", "CacheEntry"]

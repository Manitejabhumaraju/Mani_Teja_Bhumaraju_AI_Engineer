"""
Store name resolution with space-normalised city matching.

Two jobs:
  1. Confirm a store exists (case/space insensitive).
  2. When it doesn't, suggest closest known stores by string similarity.

The brief's sample questions use store codes (TBC8, TBF1) that aren't in the
dataset - we refuse to hallucinate a match. Instead we return suggestions and
let the agent surface them to the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import process, fuzz


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or '').strip()).lower()


@dataclass(frozen=True)
class StoreResolution:
    input: str
    match: Optional[str]
    city: Optional[str]
    suggestions: list[tuple[str, str]]

    @property
    def found(self) -> bool:
        return self.match is not None


class StoreResolver:
    MIN_SUGGEST_SCORE = 60

    def __init__(self, store_to_city: dict[str, str]):
        if not store_to_city:
            raise ValueError("StoreResolver requires at least one store")
        self._stores = store_to_city
        self._codes = list(store_to_city.keys())

    @classmethod
    def from_sqlite(cls, db_path: str) -> "StoreResolver":
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT store, city FROM orders_gold"
            ).fetchall()
        finally:
            conn.close()
        return cls({store: city for store, city in rows})

    def resolve(self, query: str, top_k: int = 3) -> StoreResolution:
        q = (query or "").strip()
        if not q:
            return StoreResolution(input=query, match=None, city=None, suggestions=[])

        # Exact case-insensitive match
        for code in self._codes:
            if code.lower() == q.lower():
                return StoreResolution(input=query, match=code, city=self._stores[code], suggestions=[])

        # Fuzzy fallback
        results = process.extract(q, self._codes, scorer=fuzz.token_sort_ratio, limit=top_k)
        suggestions = [
            (code, self._stores[code]) for code, score, _ in results
            if score >= self.MIN_SUGGEST_SCORE
        ]
        return StoreResolution(input=query, match=None, city=None, suggestions=suggestions)

    def stores_in_city(self, city: str) -> list[str]:
        city_n = _norm(city)
        return sorted(code for code, c in self._stores.items() if _norm(c) == city_n)

    def known_cities(self) -> list[str]:
        """Unique cities from DB, sorted."""
        return sorted(set(self._stores.values()))

    def match_city(self, city: str) -> Optional[str]:
        """
        Return exact DB city name for a user-supplied city.
        Handles partial names: "mumbai" -> "Mumbai  north", "pune" -> "Pune city east"
        """
        q = _norm(city)
        cities = list(set(self._stores.values()))

        # Pass 1: exact normalised match
        for c in cities:
            if _norm(c) == q:
                return c

        # Pass 2: db city starts with query ("mumbai north" startswith "mumbai")
        for c in cities:
            if _norm(c).startswith(q):
                return c

        # Pass 3: query starts with city prefix
        for c in cities:
            if q.startswith(_norm(c)):
                return c

        # Pass 4: word overlap ("delhi" -> "New delhi")
        q_words = set(q.split())
        for c in cities:
            c_words = set(_norm(c).split())
            if q_words & c_words:
                return c

        return None

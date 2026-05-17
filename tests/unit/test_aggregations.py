"""
Aggregation tests run against the real DB. The point isn't to test SQL
syntax - it's to confirm the weighted formulas match what the playbook
says they should.

Hand-computed expected values: where the smoke test gave us numbers,
we use those as truth and let pytest catch any regression.
"""

from app.rca import aggregations
import pytest


DATE = "2026-04-22"


class TestCityRollup:
    def test_bangalore_rollup_shape(self, db_path):
        result = aggregations.city_day_rollup(db_path, "Bangalore", DATE)
        assert result is not None
        assert result["city"] == "Bangalore"
        assert result["store_count"] == 97       # known from CSV
        assert result["total_orders"] > 0
        assert 0 <= result["weighted_breached_rate"] <= 1.0

    def test_unknown_city_returns_none(self, db_path):
        assert aggregations.city_day_rollup(db_path, "Atlantis", DATE) is None

    def test_unknown_date_returns_none(self, db_path):
        assert aggregations.city_day_rollup(db_path, "Bangalore", "2020-01-01") is None

    def test_weighted_breach_rate_not_simple_mean(self, db_path, conn):
        """The weighted rate should NOT match a simple AVG of hourly rates.
        If it does, our weighting is broken."""
        rollup = aggregations.city_day_rollup(db_path, "Bangalore", DATE)
        simple_avg = conn.execute(
            "SELECT AVG(breached_rate) FROM orders_gold "
            "WHERE city='Bangalore' AND charge_date=?",
            (DATE,),
        ).fetchone()[0]

        # These should differ - simple AVG is unweighted, rollup is volume-weighted
        assert abs(rollup["weighted_breached_rate"] - simple_avg) > 0.001


class TestStoreRollup:
    def test_known_store(self, db_path):
        result = aggregations.store_day_rollup(db_path, "STORE_003", DATE)
        assert result is not None
        assert result["store"] == "STORE_003"
        assert result["city"] == "Bangalore"

    def test_unknown_store_returns_none(self, db_path):
        assert aggregations.store_day_rollup(db_path, "DOES_NOT_EXIST", DATE) is None


class TestWorstStores:
    def test_returns_sorted_desc(self, db_path):
        rows = aggregations.worst_stores_in_city(db_path, "Bangalore", DATE, limit=10)
        rates = [r["weighted_breached_rate"] for r in rows]
        assert rates == sorted(rates, reverse=True)

    def test_limit_applied(self, db_path):
        assert len(aggregations.worst_stores_in_city(db_path, "Bangalore", DATE, limit=3)) <= 3

    def test_low_volume_stores_excluded(self, db_path, conn):
        """The HAVING clause filters out stores with <20 orders/day to avoid
        surfacing noise (e.g. a store that had 1 breach out of 2 orders)."""
        rows = aggregations.worst_stores_in_city(db_path, "Bangalore", DATE, limit=100)
        for r in rows:
            assert r["total_orders"] >= 20


class TestProblemHoursFilter:
    def test_only_returns_problem_hours(self, db_path):
        rows = aggregations.problem_hours_for_store(db_path, "STORE_003", DATE)
        for r in rows:
            assert r["is_problem_hour"] == 1

    def test_hour_filter_works(self, db_path):
        morning = [7, 8, 9, 10, 11]
        rows = aggregations.problem_hours_for_store(db_path, "STORE_003", DATE, hours=morning)
        for r in rows:
            assert int(r["hour"]) in morning

    def test_all_hours_includes_non_problem(self, db_path):
        all_rows = aggregations.all_hours_for_store(db_path, "STORE_003", DATE)
        problem_rows = aggregations.problem_hours_for_store(db_path, "STORE_003", DATE)
        assert len(all_rows) >= len(problem_rows)

"""
Unit tests for the RCA engine.

These don't touch the database. They feed synthetic hour-rows into the
engine and assert on the structured output. The point is to lock the
threshold behavior so a future refactor can't silently shift the numbers.
"""

import pytest

from app.rca import engine


# === Factory for fake hour rows ======================================

def _row(**overrides):
    """Default hour row that triggers no flags. Override fields to trigger checks."""
    base = {
        "store": "STORE_X",
        "city": "Bangalore",
        "charge_date": "2026-04-22",
        "hour": 10.0,
        "total_orders": 100,
        "breached_count": 20,
        "breached_rate": 0.20,
        "is_problem_hour": 1,
        "pileup_count": 0,
        "pileup_flag": 0,
        "avg_or2a": 15.0,
        "order_projection": 100.0,
        "current_size": 20,
        "booked_size": 20,                 # 100% booked => no booking gap
        "rider_hours_per_hour": 1200,
        "booked_hours_per_hour": 1200,
        "man_hour": 1.0,                   # full utilization
        "noshow_count": 0,
        "completed_count": 20,
    }
    base.update(overrides)
    return base


# === Threshold tests =================================================

class TestDemandCheck:
    def test_at_threshold_no_spike(self):
        # exactly 1.10x is NOT a spike (strict >)
        r = _row(total_orders=110, order_projection=100)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].demand_spike is False

    def test_just_over_threshold_spike(self):
        r = _row(total_orders=111, order_projection=100)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].demand_spike is True
        assert findings[0].demand_pct_over == pytest.approx(11.0)

    def test_under_projection_no_spike(self):
        r = _row(total_orders=80, order_projection=100)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].demand_spike is False
        assert findings[0].demand_pct_over == pytest.approx(-20.0)

    def test_zero_projection_handled(self):
        r = _row(total_orders=50, order_projection=0)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].demand_spike is False
        assert findings[0].demand_pct_over == 0.0


class TestBookingCheck:
    def test_at_threshold_no_gap(self):
        # exactly 0.90 is NOT a gap (strict <)
        r = _row(booked_size=9, current_size=10)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].booking_gap is False

    def test_below_threshold_gap(self):
        r = _row(booked_size=8, current_size=10)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].booking_gap is True
        assert findings[0].booking_pct == pytest.approx(80.0)

    def test_zero_current_size(self):
        r = _row(booked_size=0, current_size=0)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].booking_gap is False


class TestUtilizationCheck:
    def test_at_threshold_no_gap(self):
        r = _row(man_hour=0.85)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].utilization_gap is False

    def test_below_threshold_gap(self):
        r = _row(man_hour=0.80)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].utilization_gap is True

    def test_null_man_hour_not_a_gap(self):
        # Per the schema, NULL means booked_hours was 0. There's no supply
        # to underutilize, so this shouldn't flag.
        r = _row(man_hour=None)
        findings = engine.analyze_hours([r], [r])
        assert findings[0].utilization_gap is False


class TestPileupSustainedDetection:
    def test_two_consecutive_not_sustained(self):
        rows = [
            _row(hour=10, pileup_flag=1),
            _row(hour=11, pileup_flag=1),
        ]
        findings = engine.analyze_hours(rows, rows)
        for f in findings:
            assert f.pileup is True
            assert f.pileup_sustained is False

    def test_three_consecutive_sustained(self):
        rows = [
            _row(hour=10, pileup_flag=1),
            _row(hour=11, pileup_flag=1),
            _row(hour=12, pileup_flag=1),
        ]
        findings = engine.analyze_hours(rows, rows)
        for f in findings:
            assert f.pileup_sustained is True

    def test_gap_breaks_streak(self):
        # 1, 1, 0, 1, 1  - no run of 3 because hour 12 isn't a pileup
        rows = [
            _row(hour=10, pileup_flag=1),
            _row(hour=11, pileup_flag=1),
            _row(hour=12, pileup_flag=0),
            _row(hour=13, pileup_flag=1),
            _row(hour=14, pileup_flag=1),
        ]
        findings = engine.analyze_hours(rows, rows)
        for f in findings:
            assert f.pileup_sustained is False

    def test_sustained_when_problem_hour_only_overlaps_partly(self):
        """
        Real-world case: we filter problem_hours but sustained detection
        should still see hour 11 in a run of [10, 11, 12] all pileup_flag=1.
        """
        all_rows = [
            _row(hour=10, pileup_flag=1, is_problem_hour=0),
            _row(hour=11, pileup_flag=1, is_problem_hour=1),
            _row(hour=12, pileup_flag=1, is_problem_hour=0),
        ]
        problem_only = [all_rows[1]]
        findings = engine.analyze_hours(problem_only, all_rows)
        assert findings[0].pileup_sustained is True


# === Output format tests =============================================

class TestOutputFormat:
    def test_all_flags_triggered(self):
        r = _row(
            total_orders=150, order_projection=100,
            pileup_flag=1, pileup_count=20,
            booked_size=10, current_size=20,
            man_hour=0.5,
            avg_or2a=45.0, breached_count=80,
        )
        findings = engine.analyze_hours([r], [r])
        out = engine.format_hour_finding(findings[0])

        assert "DEMAND SPIKE" in out or "Demand Spike: YES" in out
        assert "Pileup: YES" in out
        assert "BOOKING GAP" in out
        assert "UTILIZATION GAP" in out
        assert "45.0 min" in out

    def test_no_flags_triggered_still_renders(self):
        r = _row()
        findings = engine.analyze_hours([r], [r])
        out = engine.format_hour_finding(findings[0])

        assert "Demand Spike: NO" in out
        assert "Pileup: NO" in out
        assert "no playbook flags triggered" in out

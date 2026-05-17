from app.rca import time_ranges
import pytest


def test_morning_is_seven_to_eleven():
    assert time_ranges.resolve("morning") == [7, 8, 9, 10, 11]


def test_full_day():
    assert time_ranges.resolve("full_day") == list(range(24))


def test_alias_resolves():
    assert time_ranges.resolve("AM") == time_ranges.resolve("morning")
    assert time_ranges.resolve("peak") == time_ranges.resolve("dinner_peak")


def test_case_and_whitespace_tolerant():
    assert time_ranges.resolve("  Morning ") == time_ranges.resolve("morning")
    assert time_ranges.resolve("EARLY-MORNING") == time_ranges.resolve("early_morning")


def test_unknown_raises():
    with pytest.raises(ValueError, match="Unknown time range"):
        time_ranges.resolve("midmorning_ish")


def test_known_ranges_returns_canonical_only():
    # Aliases shouldn't appear in this list - it's what we surface to the LLM.
    known = time_ranges.known_ranges()
    assert "morning" in known
    assert "AM" not in known
    assert "peak" not in known


def test_describe_roundtrip():
    hours = time_ranges.resolve("evening")
    assert time_ranges.describe(hours) == "evening"


def test_describe_custom():
    assert time_ranges.describe([5, 6, 7]) == "custom"

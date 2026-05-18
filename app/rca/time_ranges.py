"""
Natural-language hour ranges.

Kept as a flat dict so the LLM can be told "pick one of these names" and
the deterministic code converts the name to a list of hour ints. We never
let the LLM emit hour numbers directly - it picks a slot name, we apply it.

Ranges are closed-open: morning = hours where 6 <= h < 12, i.e. hours [6..11].
"""

from typing import Iterable


# Hours are 0-23. The data uses hour as the start of the window.
HOUR_RANGES: dict[str, tuple[int, int]] = {
    "early_morning": (4, 7),     # 4am - 6am
    "morning":       (7, 12),    # 7am - 11am
    "afternoon":     (12, 17),   # 12pm - 4pm
    "evening":       (17, 21),   # 5pm - 8pm
    "night":         (21, 24),   # 9pm - 11pm
    "late_night":    (0, 4),     # midnight - 3am
    "lunch_peak":    (12, 15),   # 12pm - 2pm
    "dinner_peak":   (19, 22),   # 7pm - 9pm
    "full_day":      (0, 24),
}

# Synonyms the LLM might emit. Map them onto canonical names above.
ALIASES: dict[str, str] = {
    "am":            "morning",
    "pm":            "afternoon",
    "lunch":         "lunch_peak",
    "dinner":        "dinner_peak",
    "peak":          "dinner_peak",      # quick-commerce peak skews evening
    "rush":          "dinner_peak",
    "day":           "full_day",
    "all_day":       "full_day",
    "whole_day":     "full_day",
}


def resolve(name: str) -> list[int]:
    """Return the hour list for a named range. Raises on unknown name."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)

    if key not in HOUR_RANGES:
        valid = ", ".join(sorted(HOUR_RANGES))
        raise ValueError(f"Unknown time range '{name}'. Valid: {valid}")

    start, end = HOUR_RANGES[key]
    return list(range(start, end))


def describe(hours: Iterable[int]) -> str:
    """Reverse: given a list of hours, name the closest range if any."""
    hours_set = set(hours)
    for name, (start, end) in HOUR_RANGES.items():
        if hours_set == set(range(start, end)):
            return name
    return "custom"


def known_ranges() -> list[str]:
    """For surfacing in prompts so the LLM picks from a closed set."""
    return list(HOUR_RANGES.keys())

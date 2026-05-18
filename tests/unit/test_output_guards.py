"""
Output guardrail tests.

The grounding logic is the interesting one - it has to be tight enough to
catch hallucinations but not so tight that legitimate paraphrasing fails.
"""

import pytest

from app.guardrails.output_guards import (
    check_grounding, detect_refusal, check_output,
    NUMERIC_TOLERANCE_ABS, NUMERIC_TOLERANCE_REL,
)


# === A realistic report we'll reference ===============================

REPORT = """\
### STORE_003 — Hour 7 — avg OR2A: 390.0 min (threshold: 8 min)

1. Demand Spike: NO — 62 orders vs 86 projected (-27.9%)
2. Pileup: YES — 10 orders carried from previous hour (SUSTAINED — 3+ consecutive hours)
3. Supply:
   a. Booking: 17 of 41 slots booked (41%) — BOOKING GAP
   b. Utilization: man_hour ratio 0.72 (0 no-shows) — UTILIZATION GAP

**Summary**: OR2A was 390.0 min, driven by SUSTAINED PILEUP, BOOKING GAP, UTILIZATION GAP.
"""


# === Grounding: PASS cases ============================================

class TestGroundingPass:
    def test_response_using_only_report_numbers_passes(self):
        response = (
            "STORE_003 had OR2A of 390 min at hour 7, driven by sustained "
            "pileup and a booking gap (17 of 41 slots booked)."
        )
        result = check_grounding(response, REPORT)
        assert result.verdict.allowed, result.verdict.reason

    def test_rounded_numbers_within_tolerance_pass(self):
        # Report says 390.0; response says 390 - exact match after rounding
        response = "OR2A was around 390 min."
        result = check_grounding(response, REPORT)
        assert result.verdict.allowed

    def test_qualitative_response_with_no_numbers_passes(self):
        response = "Sustained pileup was the dominant cause across the morning."
        result = check_grounding(response, REPORT)
        assert result.verdict.allowed

    def test_safe_numbers_like_zero_allowed(self):
        # 0 is in _SAFE_NUMBERS - "0 no-shows" / "first store" don't need grounding
        response = "There were 0 no-shows. This is the 1st priority store."
        result = check_grounding(response, REPORT)
        assert result.verdict.allowed


# === Grounding: BLOCK cases ===========================================

class TestGroundingBlock:
    def test_fabricated_number_blocked(self):
        # Report says 390 min OR2A; response says 450 min - made up.
        response = "OR2A was 450 minutes."
        result = check_grounding(response, REPORT)
        assert not result.verdict.allowed
        assert 450.0 in result.unverified_numbers

    def test_fabricated_breach_rate_blocked(self):
        # 76% breach rate isn't in this report at all.
        response = "Breach rate hit 76% during the morning."
        result = check_grounding(response, REPORT)
        assert not result.verdict.allowed

    def test_empty_response_blocked(self):
        result = check_grounding("", REPORT)
        assert not result.verdict.allowed

    def test_multiple_fabricated_numbers_blocked(self):
        response = "We saw 200 orders at hour 9 with OR2A of 555 min and a 99% breach rate."
        result = check_grounding(response, REPORT)
        assert not result.verdict.allowed
        # all three numbers are unverified
        assert len(result.unverified_numbers) >= 3


# === Grounding: tolerance =============================================

class TestGroundingTolerance:
    def test_within_absolute_tolerance_passes(self):
        # Report has 41 (slots), response says 40 - within 1.0 absolute tolerance
        response = "About 17 of 40 slots were booked."
        result = check_grounding(response, REPORT)
        # 17 matches exactly; 40 is within 1.0 of 41 in report
        assert result.verdict.allowed

    def test_within_relative_tolerance_passes(self):
        # Report has 390.0; response says 400 - within 5% (380-410 range)
        response = "OR2A was approximately 400 min."
        result = check_grounding(response, REPORT)
        assert result.verdict.allowed

    def test_outside_relative_tolerance_blocked(self):
        # Report has 390.0; response says 480 - too far off (~23% off)
        response = "OR2A was approximately 480 min."
        result = check_grounding(response, REPORT)
        assert not result.verdict.allowed


# === Refusal detection ================================================

class TestRefusalDetection:
    @pytest.mark.parametrize("text", [
        "I'm sorry, but I can't help with that.",
        "As an AI language model, I don't have access to that data.",
        "I cannot provide an answer to this question.",
        "I don't have access to the underlying data.",
        "I can't help you with delivery operations.",
    ])
    def test_refusal_phrases_detected(self, text):
        assert detect_refusal(text) is not None

    @pytest.mark.parametrize("text", [
        "STORE_003 had a tough morning with sustained pileup.",
        "OR2A was 390 min.",
        "No issues found for that store.",
        "",
    ])
    def test_normal_responses_not_flagged(self, text):
        assert detect_refusal(text) is None


# === Composite check_output ===========================================

class TestCheckOutputComposite:
    def test_grounded_non_refusal_passes(self):
        response = "STORE_003 OR2A was 390 min at hour 7 — sustained pileup."
        assert check_output(response, REPORT).allowed

    def test_refusal_blocked_even_if_grounded(self):
        response = "I'm sorry, but I can't help with that. (OR2A was 390 min.)"
        verdict = check_output(response, REPORT)
        assert not verdict.allowed
        assert verdict.matched_pattern == "unexpected_refusal"

    def test_ungrounded_blocked(self):
        response = "OR2A peaked at 999 min last Tuesday."
        verdict = check_output(response, REPORT)
        assert not verdict.allowed
        assert verdict.matched_pattern == "grounding_fail"

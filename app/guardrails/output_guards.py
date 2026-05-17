"""
Output guardrails.

Run AFTER the synthesizer LLM but BEFORE we send the response to the user.

The hard problem the output guard solves: LLMs sometimes invent numbers.
The deterministic engine says "12% breach rate, 33 min OR2A" and the LLM,
free-styling its way through the response, writes "around 15% breach rate
with OR2A near 40 min." Both are plausible but the second one is fabricated.

Approach:
  1. Pull every numeric token from the LLM response.
  2. Pull every numeric token from the report we gave the LLM.
  3. For each number in the response, require either:
       a. it appears in the report (exact or near-match within tolerance), or
       b. it's a "safe" common number (0, 100, percentages of obvious facts)
  4. If unverified numbers exist, flag the response.

We DO NOT silently rewrite. We surface the violation to the graph, which
either retries with a stricter prompt or sends a refusal. This is the
classic "evidence-backed responses" guardrail.

There's a second check that's simpler but useful: detect refusals from
the LLM. Sometimes Llama 3.3 will refuse a perfectly fine question because
it pattern-matched on a keyword. We catch the refusal and turn it into a
retry signal.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from app.guardrails.input_guards import GuardVerdict, Decision


log = logging.getLogger(__name__)


# === Number extraction =================================================

# Catches integers and decimals, with optional trailing % or units.
# Examples matched: 12, 12.5, 12%, 33 min, 1.10x, 245
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# Numbers we don't bother grounding. These are scaffolding, not claims.
_SAFE_NUMBERS = {
    0, 1, 2, 3,                 # ordinals, "the 1st step"
    10, 100,                    # round figures
    24,                         # hours in a day
    60,                         # minutes in an hour, etc
}

# Acceptable absolute error when comparing two numbers from different sources.
# A report might say "33.35 min" and the LLM might write "33 min" - both fine.
NUMERIC_TOLERANCE_ABS = 1.0      # absolute tolerance for small numbers
NUMERIC_TOLERANCE_REL = 0.05     # 5% relative tolerance for larger numbers


def _extract_numbers(text: str) -> list[float]:
    if not text:
        return []
    return [float(m.group()) for m in _NUM_RE.finditer(text)]


def _matches_any(value: float, allowed: list[float]) -> bool:
    """Is `value` close to any allowed number within tolerance?"""
    if value in _SAFE_NUMBERS:
        return True
    for ref in allowed:
        diff = abs(value - ref)
        if diff <= NUMERIC_TOLERANCE_ABS:
            return True
        if ref != 0 and diff / abs(ref) <= NUMERIC_TOLERANCE_REL:
            return True
    return False


# === Grounding check ==================================================

@dataclass
class GroundingResult:
    verdict: GuardVerdict
    unverified_numbers: list[float] = field(default_factory=list)
    response_numbers: list[float] = field(default_factory=list)
    report_numbers: list[float] = field(default_factory=list)


def check_grounding(response: str, report: str) -> GroundingResult:
    """
    Verify the numeric claims in `response` are supported by `report`.

    `report` is the full text of the deterministic RCA output that was fed
    to the synthesizer. Anything the synthesizer cites should be derivable
    from it.

    Returns GroundingResult. If `verdict.allowed` is False, the response
    contains unverified numbers and should be retried or refused.
    """
    if not response or not response.strip():
        return GroundingResult(
            verdict=GuardVerdict(Decision.BLOCK, "empty response", severity="warn"),
        )

    resp_nums = _extract_numbers(response)
    rep_nums = _extract_numbers(report)

    # Edge case: report has no numbers (e.g. "no problem hours found").
    # Then any number the response invents is suspect.
    unverified = [n for n in resp_nums if not _matches_any(n, rep_nums)]

    if unverified:
        log.warning(
            "grounding violation: %d unverified number(s) in response: %s",
            len(unverified),
            unverified[:5],
        )
        return GroundingResult(
            verdict=GuardVerdict(
                decision=Decision.BLOCK,
                reason=f"{len(unverified)} unverified number(s) in response",
                matched_pattern="grounding_fail",
                severity="warn",
            ),
            unverified_numbers=unverified,
            response_numbers=resp_nums,
            report_numbers=rep_nums,
        )

    return GroundingResult(
        verdict=GuardVerdict(Decision.ALLOW, "grounded"),
        response_numbers=resp_nums,
        report_numbers=rep_nums,
    )


# === Refusal detection ================================================
# Sometimes the LLM refuses a question it shouldn't. We catch this so the
# graph can retry with a clearer prompt rather than dumping the LLM's
# refusal text on the user.

_REFUSAL_MARKERS = [
    r"\bI\s+(can'?t|cannot|am\s+not\s+able\s+to)\s+(help|assist|answer|provide)\b",
    r"\bAs\s+an?\s+AI\s+(language\s+)?model\b",
    r"\bI\s+(don'?t|do\s+not)\s+have\s+access\s+to\b",
    r"\bI'?m\s+sorry,?\s+but\s+I\s+(can'?t|cannot)\b",
]

_REFUSAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_MARKERS]


def detect_refusal(response: str) -> Optional[str]:
    """
    Return the first refusal marker found in the response, or None if clean.

    Refusals are a problem when they come from the synthesizer node, which
    was given concrete data and asked to phrase it. If we see one there,
    something went wrong in the prompt - retry.
    """
    if not response:
        return None
    for rx in _REFUSAL_COMPILED:
        m = rx.search(response)
        if m:
            return m.group(0)
    return None


# === Composite output check ===========================================

def check_output(response: str, report: str) -> GuardVerdict:
    """
    Convenience wrapper: refusal + grounding in one call.

    Used by the graph's output node. If this returns BLOCK, the graph
    either retries the synthesizer (cheap) or sends a refusal to the user
    after N retries.
    """
    refusal = detect_refusal(response)
    if refusal:
        log.warning("synthesizer refusal detected: %r", refusal)
        return GuardVerdict(
            decision=Decision.BLOCK,
            reason=f"synthesizer refused unexpectedly: {refusal!r}",
            matched_pattern="unexpected_refusal",
            severity="warn",
        )

    return check_grounding(response, report).verdict

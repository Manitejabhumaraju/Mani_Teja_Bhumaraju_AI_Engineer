"""
RCA engine. Pure logic. No LLM calls, no I/O of its own.

Encodes the playbook from docs/quick_commerce_rca_logic.md verbatim:

    Check 1 (Demand):  total_orders > order_projection * 1.10
    Check 2 (Pileup):  pileup_flag == 1; sustained if 3+ consecutive
    Check 3a (Booking):  current_capacity_booked < 0.90
    Check 3b (Util):     man_hour < 0.85

Thresholds are constants at the top of the file. The interviewer will look
for them - if they're scattered or hardcoded inline, that's a smell.

The engine takes already-fetched hour rows (from aggregations.py) and
returns structured findings. The graph nodes call this, format the output
via the playbook's template, and the LLM only does the final phrasing pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# === Thresholds (from docs/quick_commerce_rca_logic.md) ===========================

DEMAND_SPIKE_MULTIPLIER   = 1.10   # actual > projection * this
PILEUP_SUSTAINED_HOURS    = 3      # consecutive hours flagged
BOOKING_GAP_RATIO         = 0.90   # current_capacity_booked floor
UTILIZATION_GAP_RATIO     = 0.85   # man_hour floor


# === Result dataclasses ===============================================

@dataclass
class HourFinding:
    """RCA output for a single problem hour."""
    store: str
    hour: int
    total_orders: int
    order_projection: float
    avg_or2a: float
    breached_count: int

    demand_spike: bool
    demand_pct_over: float        # (actual / projection - 1) * 100

    pileup: bool
    pileup_count: int
    pileup_sustained: bool        # part of a 3+ consecutive run

    booking_gap: bool
    booked_size: int
    current_size: int
    booking_pct: float

    utilization_gap: bool
    man_hour: float | None
    noshow_count: int

    def triggered_flags(self) -> list[str]:
        flags = []
        if self.demand_spike:
            flags.append("DEMAND SPIKE")
        if self.pileup_sustained:
            flags.append("SUSTAINED PILEUP")
        elif self.pileup:
            flags.append("PILEUP")
        if self.booking_gap:
            flags.append("BOOKING GAP")
        if self.utilization_gap:
            flags.append("UTILIZATION GAP")
        return flags


@dataclass
class StoreDayFindings:
    """Findings rolled up at the store-day level. List of hour-level findings."""
    store: str
    city: str
    date: str
    weighted_breached_rate: float
    weighted_avg_or2a: float
    total_orders: int
    problem_hour_count: int
    hours: list[HourFinding] = field(default_factory=list)

    def headline_causes(self) -> dict[str, int]:
        """Frequency of each cause across problem hours. For summary lines."""
        tally: dict[str, int] = {}
        for h in self.hours:
            for flag in h.triggered_flags():
                tally[flag] = tally.get(flag, 0) + 1
        return tally


# === Engine ===========================================================

def _check_demand(total_orders: int, order_projection: float) -> tuple[bool, float]:
    """Returns (is_spike, pct_over_projection)."""
    if not order_projection or order_projection <= 0:
        return False, 0.0
    pct = (total_orders / order_projection - 1) * 100
    return total_orders > order_projection * DEMAND_SPIKE_MULTIPLIER, pct


def _check_booking(booked_size: int, current_size: int) -> tuple[bool, float]:
    """Returns (is_gap, pct_filled)."""
    if not current_size or current_size <= 0:
        return False, 0.0
    ratio = booked_size / current_size
    return ratio < BOOKING_GAP_RATIO, ratio * 100


def _check_utilization(man_hour: float | None) -> bool:
    """man_hour can be NULL when booked_hours=0; treat as not-a-gap (no booked supply to underutilize)."""
    if man_hour is None:
        return False
    return man_hour < UTILIZATION_GAP_RATIO


def _mark_sustained_pileups(hours_sorted: list[dict]) -> set[int]:
    """
    Given hour rows sorted by hour ascending, return the set of hour values
    that are part of a 3+ consecutive run of pileup_flag=1.

    Consecutive here means hour+1, not "next problem hour". A gap of even
    one non-pileup hour breaks the run.
    """
    if not hours_sorted:
        return set()

    sustained: set[int] = set()
    run: list[int] = []

    # Walk all hours we have. We need to know about non-pileup hours in
    # between to detect actual breaks in continuity.
    for row in hours_sorted:
        h = int(row["hour"])
        if row.get("pileup_flag") == 1:
            if run and h == run[-1] + 1:
                run.append(h)
            else:
                run = [h]
            if len(run) >= PILEUP_SUSTAINED_HOURS:
                sustained.update(run)
        else:
            run = []

    return sustained


def analyze_hours(problem_hour_rows: Iterable[dict],
                  all_hour_rows: Iterable[dict]) -> list[HourFinding]:
    """
    Run the three checks on each problem hour.

    `problem_hour_rows` is the filtered set to analyze.
    `all_hour_rows` is the full store-day used for sustained-pileup detection
    (we need to see the non-problem hours to know if pileup is broken).
    """
    all_sorted = sorted(all_hour_rows, key=lambda r: int(r["hour"]))
    sustained_hours = _mark_sustained_pileups(all_sorted)

    findings: list[HourFinding] = []
    for row in problem_hour_rows:
        hour = int(row["hour"])

        demand_spike, demand_pct = _check_demand(
            int(row["total_orders"]),
            float(row["order_projection"]),
        )

        booking_gap, booking_pct = _check_booking(
            int(row["booked_size"]),
            int(row["current_size"]),
        )

        util_gap = _check_utilization(row.get("man_hour"))

        pileup = bool(row.get("pileup_flag") == 1)

        findings.append(HourFinding(
            store=row["store"],
            hour=hour,
            total_orders=int(row["total_orders"]),
            order_projection=float(row["order_projection"]),
            avg_or2a=float(row["avg_or2a"]) if row.get("avg_or2a") is not None else 0.0,
            breached_count=int(row["breached_count"]),
            demand_spike=demand_spike,
            demand_pct_over=demand_pct,
            pileup=pileup,
            pileup_count=int(row.get("pileup_count") or 0),
            pileup_sustained=hour in sustained_hours,
            booking_gap=booking_gap,
            booked_size=int(row["booked_size"]),
            current_size=int(row["current_size"]),
            booking_pct=booking_pct,
            utilization_gap=util_gap,
            man_hour=row.get("man_hour"),
            noshow_count=int(row.get("noshow_count") or 0),
        ))

    return findings


# === Output formatter =================================================

def format_hour_finding(f: HourFinding, threshold_min: float = 8.0) -> str:
    """
    Renders the playbook's required output format for a single hour.

    Kept deterministic so the agent's output is the same regardless of LLM
    temperature. The LLM only adds the conversational wrapper around this.
    """
    flags = f.triggered_flags()
    summary_causes = ", ".join(flags) if flags else "no playbook flags triggered (review manually)"

    pileup_note = ""
    if f.pileup_sustained:
        pileup_note = " (SUSTAINED — 3+ consecutive hours)"

    man_hour_str = f"{f.man_hour:.2f}" if f.man_hour is not None else "n/a"

    return (
        f"### {f.store} — Hour {f.hour} — avg OR2A: {f.avg_or2a:.1f} min "
        f"(threshold: {threshold_min:.0f} min)\n"
        f"\n"
        f"1. Demand Spike: {'YES' if f.demand_spike else 'NO'} — "
        f"{f.total_orders} orders vs {f.order_projection:.0f} projected "
        f"({f.demand_pct_over:+.1f}%)\n"
        f"2. Pileup: {'YES' if f.pileup else 'NO'} — "
        f"{f.pileup_count} orders carried from previous hour{pileup_note}\n"
        f"3. Supply:\n"
        f"   a. Booking: {f.booked_size} of {f.current_size} slots booked "
        f"({f.booking_pct:.0f}%)"
        f"{' — BOOKING GAP' if f.booking_gap else ''}\n"
        f"   b. Utilization: man_hour ratio {man_hour_str} "
        f"({f.noshow_count} no-shows)"
        f"{' — UTILIZATION GAP' if f.utilization_gap else ''}\n"
        f"\n"
        f"**Summary**: OR2A was {f.avg_or2a:.1f} min, driven by {summary_causes}.\n"
    )

"""
Rollups from hour-grain to store-day and city-day.

Weighted metrics per playbook:
  - Weighted breached_rate = SUM(breached_count) / SUM(total_orders)
  - Weighted avg_or2a     = SUM(avg_or2a * total_orders) / SUM(total_orders)

P50/P95 are added for operational insight: P50 = median OR2A per store-day,
P95 = worst-tail experience. Calculated from per-hour avg_or2a weighted by
total_orders (approximate percentile from grouped data).

No ORM, no SQLAlchemy - sqlite3 + dict. Data is small (3,813 rows).
"""

import sqlite3
import re
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Collapse multi-spaces, strip, lowercase. For fuzzy city/store matching."""
    return re.sub(r'\s+', ' ', (s or '').strip()).lower()


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _match_city(cur: sqlite3.Cursor, city: str) -> Optional[str]:
    """
    Return exact DB city name matching the input.
    Strategy:
      1. Exact match after space-normalisation (handles double-space issue)
      2. Starts-with match: "mumbai" -> "Mumbai  north" because
         norm("mumbai north") startswith norm("mumbai")
      3. Contains match as last resort.
    """
    cur.execute("SELECT DISTINCT city FROM orders_gold")
    db_cities = [row[0] for row in cur.fetchall()]
    q = _norm(city)

    # Pass 1: exact normalised match
    for c in db_cities:
        if _norm(c) == q:
            return c

    # Pass 2: starts-with (handles "mumbai" -> "mumbai north", "hyderabad" -> "hyderabad city")
    for c in db_cities:
        if _norm(c).startswith(q):
            return c

    # Pass 3: query starts with city prefix
    for c in db_cities:
        if q.startswith(_norm(c)):
            return c

    # Pass 4: any word in query matches any word in city name
    # "delhi" -> "New delhi" because "delhi" is a word in "New delhi"
    q_words = set(q.split())
    for c in db_cities:
        c_words = set(_norm(c).split())
        if q_words & c_words:  # intersection
            return c

    return None


def _match_store(cur: sqlite3.Cursor, store: str) -> Optional[str]:
    """Return exact DB store name matching the input (case-insensitive)."""
    cur.execute("SELECT DISTINCT store FROM orders_gold")
    for (db_store,) in cur.fetchall():
        if _norm(db_store) == _norm(store):
            return db_store
    return None


def _percentile_from_hours(rows: list[dict], col: str, p: float) -> float:
    """
    Approximate percentile of `col` weighted by total_orders.

    We can't get true order-level distribution from a pre-aggregated table,
    so we treat each store-hour avg as one data point, sort by value, and
    find the p-th percentile by count (unweighted) — good enough for ops.
    """
    vals = sorted(
        float(r[col]) for r in rows
        if r.get(col) is not None and float(r[col]) > 0
    )
    if not vals:
        return 0.0
    idx = max(0, int(len(vals) * p / 100) - 1)
    return vals[min(idx, len(vals) - 1)]


# ── city-level ────────────────────────────────────────────────────────────────

def city_day_rollup(db_path: str, city: str, date: str) -> Optional[dict]:
    """City-level aggregation for one day. Returns None if no data."""
    with _conn(db_path) as conn:
        cur = conn.cursor()
        matched = _match_city(cur, city)
        if not matched:
            return None

        cur.execute("""
            SELECT
                city,
                charge_date,
                COUNT(DISTINCT store)                                    AS store_count,
                COUNT(DISTINCT CAST(hour AS INTEGER))                    AS hours_observed,
                SUM(total_orders)                                        AS total_orders,
                SUM(breached_count)                                      AS total_breached,
                SUM(CASE WHEN is_problem_hour = 1 THEN 1 ELSE 0 END)    AS problem_hour_count,
                SUM(CASE WHEN pileup_flag = 1 THEN 1 ELSE 0 END)        AS pileup_hour_count,
                CAST(SUM(breached_count) AS REAL)
                    / NULLIF(SUM(total_orders), 0)                       AS weighted_breached_rate,
                SUM(avg_or2a * total_orders)
                    / NULLIF(SUM(total_orders), 0)                       AS weighted_avg_or2a
            FROM orders_gold
            WHERE city = ? AND charge_date = ?
            GROUP BY city, charge_date
        """, (matched, date))

        row = cur.fetchone()
        if not row:
            return None
        result = dict(row)

        # Add P50/P95 from hour-level rows
        cur.execute("""
            SELECT avg_or2a, total_orders FROM orders_gold
            WHERE city = ? AND charge_date = ? AND avg_or2a > 0
        """, (matched, date))
        hour_rows = [{"avg_or2a": r[0], "total_orders": r[1]} for r in cur.fetchall()]
        result["p50_or2a"] = _percentile_from_hours(hour_rows, "avg_or2a", 50)
        result["p95_or2a"] = _percentile_from_hours(hour_rows, "avg_or2a", 95)
        return result


def worst_stores_in_city(db_path: str, city: str, date: str, limit: int = 5) -> list[dict]:
    """Top N worst stores in a city on a given day, by weighted breach rate."""
    with _conn(db_path) as conn:
        cur = conn.cursor()
        matched = _match_city(cur, city)
        if not matched:
            return []

        cur.execute("""
            SELECT
                store,
                city,
                SUM(total_orders)                                        AS total_orders,
                SUM(breached_count)                                      AS total_breached,
                CAST(SUM(breached_count) AS REAL)
                    / NULLIF(SUM(total_orders), 0)                       AS weighted_breached_rate,
                SUM(avg_or2a * total_orders)
                    / NULLIF(SUM(total_orders), 0)                       AS weighted_avg_or2a,
                SUM(CASE WHEN is_problem_hour = 1 THEN 1 ELSE 0 END)    AS problem_hour_count
            FROM orders_gold
            WHERE city = ? AND charge_date = ?
            GROUP BY store, city
            HAVING SUM(total_orders) > 0
            ORDER BY weighted_breached_rate DESC, weighted_avg_or2a DESC
            LIMIT ?
        """, (matched, date, limit))

        return [dict(r) for r in cur.fetchall()]


# ── store-level ───────────────────────────────────────────────────────────────

def store_day_rollup(db_path: str, store: str, date: str) -> Optional[dict]:
    """Store-level aggregation for one day."""
    with _conn(db_path) as conn:
        cur = conn.cursor()
        matched = _match_store(cur, store)
        if not matched:
            return None

        cur.execute("""
            SELECT
                store, city, charge_date,
                COUNT(DISTINCT CAST(hour AS INTEGER))                    AS hours_observed,
                SUM(total_orders)                                        AS total_orders,
                SUM(breached_count)                                      AS total_breached,
                SUM(CASE WHEN is_problem_hour = 1 THEN 1 ELSE 0 END)    AS problem_hour_count,
                SUM(CASE WHEN pileup_flag = 1 THEN 1 ELSE 0 END)        AS pileup_hour_count,
                CAST(SUM(breached_count) AS REAL)
                    / NULLIF(SUM(total_orders), 0)                       AS breach_rate,
                SUM(avg_or2a * total_orders)
                    / NULLIF(SUM(total_orders), 0)                       AS weighted_avg_or2a
            FROM orders_gold
            WHERE store = ? AND charge_date = ?
            GROUP BY store, city, charge_date
        """, (matched, date))

        row = cur.fetchone()
        if not row:
            return None
        result = dict(row)

        # P50/P95 for this store
        cur.execute("""
            SELECT avg_or2a, total_orders FROM orders_gold
            WHERE store = ? AND charge_date = ? AND avg_or2a > 0
        """, (matched, date))
        hour_rows = [{"avg_or2a": r[0], "total_orders": r[1]} for r in cur.fetchall()]
        result["p50_or2a"] = _percentile_from_hours(hour_rows, "avg_or2a", 50)
        result["p95_or2a"] = _percentile_from_hours(hour_rows, "avg_or2a", 95)
        return result


def problem_hours_for_store(
    db_path: str, store: str, date: str, hours: Optional[list[int]] = None
) -> list[dict]:
    """All problem hours for a store-day (is_problem_hour=1), optionally filtered."""
    with _conn(db_path) as conn:
        cur = conn.cursor()
        matched = _match_store(cur, store)
        if not matched:
            return []

    sql = """
        SELECT
            store, city, charge_date, hour,
            total_orders, breached_count, breached_rate, is_problem_hour,
            pileup_count, pileup_flag, avg_or2a, order_projection,
            current_size, booked_size, current_capacity_booked,
            rider_hours_per_hour, booked_hours_per_hour, man_hour,
            noshow_count, completed_count
        FROM orders_gold
        WHERE store = ? AND charge_date = ? AND is_problem_hour = 1
    """
    params: list = [matched, date]

    if hours:
        placeholders = ",".join("?" * len(hours))
        sql += f" AND CAST(hour AS INTEGER) IN ({placeholders})"
        params.extend(hours)

    sql += " ORDER BY hour"

    with _conn(db_path) as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def all_hours_for_store(
    db_path: str, store: str, date: str, hours: Optional[list[int]] = None
) -> list[dict]:
    """All hours for a store-day (no is_problem_hour filter), optionally filtered."""
    with _conn(db_path) as conn:
        cur = conn.cursor()
        matched = _match_store(cur, store)
        if not matched:
            return []

    sql = """
        SELECT
            store, city, charge_date, hour,
            total_orders, breached_count, breached_rate, is_problem_hour,
            pileup_count, pileup_flag, avg_or2a, order_projection,
            current_size, booked_size, current_capacity_booked,
            rider_hours_per_hour, booked_hours_per_hour, man_hour,
            noshow_count, completed_count
        FROM orders_gold
        WHERE store = ? AND charge_date = ?
    """
    params: list = [matched, date]

    if hours:
        placeholders = ",".join("?" * len(hours))
        sql += f" AND CAST(hour AS INTEGER) IN ({placeholders})"
        params.extend(hours)

    sql += " ORDER BY hour"

    with _conn(db_path) as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


# === New v2 aggregations for richer reports =====================================
#
# These power the structured city/store report layout. Pure SQL, no LLM, no
# dependence on already-fetched data: each function takes (db, scope_value, date)
# and returns a small dict/list ready for the formatter.
#
# Kept separate from the legacy functions above so the existing tests don't
# need to change. Both old and new live side by side.


def delivery_summary(
    db_path: str,
    scope: str,
    scope_value: str,
    date: str,
) -> dict:
    """
    Total / completed / cancelled / RTO across the scope.
    
    scope in ("city", "store"). Single SUM query, fast.
    Returns: total_orders, completed, cancelled, rto, completion_rate,
             cancellation_rate, rto_rate.
    """
    col = "city" if scope == "city" else "store"
    with _conn(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM(total_orders), 0)            AS total,
                COALESCE(SUM(completed_order_count), 0)   AS completed,
                COALESCE(SUM(cancelled_order_count), 0)   AS cancelled,
                COALESCE(SUM(rto_order_count), 0)         AS rto
            FROM orders_gold
            WHERE {col} = ? AND charge_date = ?
            """,
            (scope_value, date),
        ).fetchone()

    total = int(row[0]) if row[0] else 0
    completed = int(row[1]) if row[1] else 0
    cancelled = int(row[2]) if row[2] else 0
    rto = int(row[3]) if row[3] else 0

    return {
        "total_orders": total,
        "completed": completed,
        "cancelled": cancelled,
        "rto": rto,
        "completion_rate": (completed / total) if total else 0.0,
        "cancellation_rate": (cancelled / total) if total else 0.0,
        "rto_rate": (rto / total) if total else 0.0,
    }


def hourly_breach_profile(
    db_path: str,
    scope: str,
    scope_value: str,
    date: str,
    top_n: int = 3,
) -> dict:
    """
    Hour-by-hour breach profile within the scope.
    
    Returns: top_hours (N worst), total_hours_observed, hours_over_20pct.
    Each top_hour: hour, total_orders, breach_rate.
    """
    col = "city" if scope == "city" else "store"
    with _conn(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                CAST(hour AS INT)                       AS hr,
                SUM(total_orders)                       AS orders,
                CASE WHEN SUM(total_orders) > 0
                     THEN SUM(breached_count) * 1.0 / SUM(total_orders)
                     ELSE 0.0
                END                                     AS br_rate
            FROM orders_gold
            WHERE {col} = ? AND charge_date = ?
            GROUP BY hr
            ORDER BY br_rate DESC
            """,
            (scope_value, date),
        ).fetchall()

    all_hours = [
        {"hour": int(r[0]), "total_orders": int(r[1] or 0), "breach_rate": float(r[2] or 0.0)}
        for r in rows
    ]
    over_20 = sum(1 for r in all_hours if r["breach_rate"] > 0.20)

    return {
        "top_hours": all_hours[:top_n],
        "total_hours_observed": len(all_hours),
        "hours_over_20pct": over_20,
    }


def pileup_by_slot(
    db_path: str,
    scope: str,
    scope_value: str,
    date: str,
) -> list[dict]:
    """
    Pileup concentration by slot. One row per slot_name with non-zero pileup.
    
    Returns list of {slot_name, slot_type, pileup_hours, pileup_orders}
    sorted by pileup_orders desc.
    """
    col = "city" if scope == "city" else "store"
    with _conn(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                COALESCE(slot_name, 'unknown') AS sn,
                COALESCE(slot_type, '')        AS st,
                SUM(pileup_flag)               AS pileup_hours,
                SUM(pileup_count)              AS pileup_orders
            FROM orders_gold
            WHERE {col} = ? AND charge_date = ?
              AND pileup_flag = 1
            GROUP BY sn, st
            ORDER BY pileup_orders DESC
            """,
            (scope_value, date),
        ).fetchall()

    return [
        {
            "slot_name": r[0],
            "slot_type": r[1],
            "pileup_hours": int(r[2] or 0),
            "pileup_orders": int(r[3] or 0),
        }
        for r in rows
    ]


def top_performers_in_city(
    db_path: str,
    city: str,
    date: str,
    limit: int = 3,
) -> list[dict]:
    """
    Best stores in the city by completion rate. Min volume floor of 50 orders
    so we don't surface tiny stores with 1-of-1 completions.
    """
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                store,
                SUM(total_orders)                       AS total,
                SUM(completed_order_count)              AS completed,
                CASE WHEN SUM(total_orders) > 0
                     THEN SUM(breached_count) * 1.0 / SUM(total_orders)
                     ELSE 0.0
                END                                     AS br_rate,
                AVG(avg_or2a)                           AS avg_or2a
            FROM orders_gold
            WHERE city = ? AND charge_date = ?
            GROUP BY store
            HAVING SUM(total_orders) >= 50
            ORDER BY (SUM(completed_order_count) * 1.0 / SUM(total_orders)) DESC
            LIMIT ?
            """,
            (city, date, limit),
        ).fetchall()

    return [
        {
            "store": r[0],
            "total_orders": int(r[1] or 0),
            "completed": int(r[2] or 0),
            "completion_rate": (int(r[2] or 0) / int(r[1] or 1)) if r[1] else 0.0,
            "breach_rate": float(r[3] or 0.0),
            "avg_or2a": float(r[4] or 0.0),
        }
        for r in rows
    ]


def forecast_accuracy(
    db_path: str,
    scope: str,
    scope_value: str,
    date: str,
) -> dict:
    """
    Demand vs forecast.
    Returns: total_orders, total_projection, ratio, model_note.
    
    The model_note is a one-line analyst-style assessment of forecast quality —
    keeps the output deterministic so the synthesizer doesn't have to invent.
    """
    col = "city" if scope == "city" else "store"
    with _conn(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT
                SUM(total_orders)        AS actual,
                SUM(order_projection)    AS projected
            FROM orders_gold
            WHERE {col} = ? AND charge_date = ?
            """,
            (scope_value, date),
        ).fetchone()

    actual = int(row[0] or 0)
    projected = float(row[1] or 0.0)
    ratio = (actual / projected) if projected > 0 else 0.0

    # Analyst-style one-liner based on the ratio. Keep the buckets tight.
    if projected <= 0:
        note = "No forecast available to compare against."
    elif ratio >= 1.30:
        pct = (ratio - 1) * 100
        note = f"Forecast under-predicted by {pct:.0f}%. Model recalibration warranted."
    elif ratio >= 1.10:
        pct = (ratio - 1) * 100
        note = f"Forecast was low by {pct:.0f}% — actual demand exceeded plan."
    elif ratio >= 0.90:
        pct = (ratio - 1) * 100
        sign = "high" if pct < 0 else "low"
        note = f"Forecast tracked actuals within {abs(pct):.0f}% ({sign})."
    elif ratio >= 0.70:
        pct = (1 - ratio) * 100
        note = f"Forecast over-predicted by {pct:.0f}% — actuals fell short."
    else:
        pct = (1 - ratio) * 100
        note = f"Forecast over-predicted by {pct:.0f}%. Model recalibration warranted."

    return {
        "actual": actual,
        "projected": int(projected),
        "ratio": ratio,
        "model_note": note,
    }


def cause_attribution(findings: list) -> dict:
    """
    Headline cause attribution across the findings list.
    
    Operates on already-computed HourFindings (no DB hit). Returns:
      total_problem_hours, top_combos (list of {combo, count, pct}).
    
    Combo = sorted comma-joined flag tuple. e.g. "DEMAND SPIKE+PILEUP".
    """
    if not findings:
        return {"total_problem_hours": 0, "top_combos": []}

    combo_counts: dict[str, int] = {}
    for f in findings:
        flags = f.triggered_flags()
        combo = " + ".join(flags) if flags else "no flags (review manually)"
        combo_counts[combo] = combo_counts.get(combo, 0) + 1

    total = len(findings)
    sorted_combos = sorted(combo_counts.items(), key=lambda x: -x[1])

    return {
        "total_problem_hours": total,
        "top_combos": [
            {"combo": c, "count": n, "pct": n / total}
            for c, n in sorted_combos[:3]
        ],
    }

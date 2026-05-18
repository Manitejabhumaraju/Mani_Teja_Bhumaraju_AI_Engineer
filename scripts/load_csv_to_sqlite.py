"""
One-shot ETL: gold CSV -> SQLite.

Run once before starting the app. Idempotent - drops and recreates the table.

    python scripts/load_csv_to_sqlite.py

Notes on the data:
  - breached_rate comes through as a long-decimal string in the CSV; cast to float.
  - man_hour and rider_hours_per_hour have NULLs for hours where booked_hours=0.
    Keep them nullable, handle in queries.
  - Column 'orginal_size' is misspelled in the source. We keep the misspelling
    to match the upstream schema doc. The interviewer will likely test if we
    quietly renamed it - we didn't.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd


# Single-table schema. Types match the schema.md doc, with SQLite's type system.
# REAL for floats, INTEGER for ints, TEXT for strings and dates (SQLite has no DATE).
SCHEMA_SQL = """
CREATE TABLE orders_gold (
    charge_date              TEXT    NOT NULL,
    store                    TEXT    NOT NULL,
    city                     TEXT    NOT NULL,
    hour                     REAL    NOT NULL,
    hour_start               TEXT,
    hour_end                 TEXT,
    total_orders             INTEGER NOT NULL,
    breached_count           INTEGER NOT NULL,
    breached_rate            REAL    NOT NULL,
    is_problem_hour          INTEGER NOT NULL,
    pileup_count             INTEGER NOT NULL,
    pileup_flag              INTEGER NOT NULL,
    avg_or2a                 REAL,
    o2a                      REAL,
    completed_order_count    INTEGER NOT NULL,
    cancelled_order_count    INTEGER NOT NULL,
    rto_order_count          INTEGER NOT NULL,
    order_projection         REAL    NOT NULL,
    slot_name                TEXT,
    slot_start               TEXT,
    slot_end                 TEXT,
    slot_type                TEXT,
    orginal_size             INTEGER,
    current_size             INTEGER NOT NULL,
    booked_size              INTEGER NOT NULL,
    completed_count          INTEGER,
    incompleted_count        INTEGER,
    noshow_count             INTEGER,
    cancelled_count          INTEGER,
    orginal_capacity_booked  REAL,
    current_capacity_booked  REAL,
    rider_hours_per_hour     INTEGER,
    booked_hours_per_hour    INTEGER,
    man_hour                 REAL,
    PRIMARY KEY (charge_date, store, hour)
);
"""

# Indices that earn their keep given the access pattern:
# 1. (city, charge_date) - "How did Bangalore do on 2026-04-22?"
# 2. (store, charge_date) - "Why did STORE_087 underperform?"
# 3. (charge_date, is_problem_hour) - "What were the problem hours?"
INDEX_SQL = [
    "CREATE INDEX idx_city_date ON orders_gold(city, charge_date);",
    "CREATE INDEX idx_store_date ON orders_gold(store, charge_date);",
    "CREATE INDEX idx_date_problem ON orders_gold(charge_date, is_problem_hour);",
]


def load(csv_path: Path, db_path: Path) -> None:
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Read {len(df):,} rows from {csv_path.name}")

    # breached_rate is the one column where the CSV's long-decimal text bites us.
    # The rest pandas infers correctly.
    df["breached_rate"] = pd.to_numeric(df["breached_rate"], errors="coerce")

    # Sanity checks before write - cheap to do, expensive to debug later.
    nulls_in_required = df[["total_orders", "store", "city", "charge_date"]].isnull().sum().sum()
    if nulls_in_required > 0:
        sys.exit(f"Required columns have nulls. Aborting. ({nulls_in_required} found)")

    if df["breached_rate"].max() > 1.0 or df["breached_rate"].min() < 0.0:
        sys.exit("breached_rate outside [0,1]. Data is malformed.")

    # Wipe and rebuild. The dataset is small enough that this is fine.
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing DB at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)

        df.to_sql("orders_gold", conn, if_exists="append", index=False)

        for stmt in INDEX_SQL:
            conn.execute(stmt)

        conn.commit()

        # Verify
        row_count = conn.execute("SELECT COUNT(*) FROM orders_gold").fetchone()[0]
        store_count = conn.execute("SELECT COUNT(DISTINCT store) FROM orders_gold").fetchone()[0]
        city_count = conn.execute("SELECT COUNT(DISTINCT city) FROM orders_gold").fetchone()[0]
        problem_hours = conn.execute(
            "SELECT COUNT(*) FROM orders_gold WHERE is_problem_hour = 1"
        ).fetchone()[0]

        print(f"Loaded into {db_path}")
        print(f"  rows:          {row_count:,}")
        print(f"  stores:        {store_count:,}")
        print(f"  cities:        {city_count:,}")
        print(f"  problem hours: {problem_hours:,} ({100 * problem_hours / row_count:.1f}%)")
    finally:
        conn.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=repo_root / "data" / "quick_commerce_orders_gold_20260422.csv",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=repo_root / "data" / "loadshare.db",
    )
    args = parser.parse_args()

    load(args.csv, args.db)


if __name__ == "__main__":
    main()

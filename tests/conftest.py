"""Test fixtures shared across the suite."""

import sqlite3
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "loadshare.db"


@pytest.fixture(scope="session")
def db_path() -> str:
    """Path to the loaded SQLite DB. Requires load_csv_to_sqlite.py to have run."""
    if not DB_PATH.exists():
        pytest.skip(f"DB not found at {DB_PATH}. Run scripts/load_csv_to_sqlite.py first.")
    return str(DB_PATH)


@pytest.fixture(scope="session")
def conn(db_path):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    yield c
    c.close()

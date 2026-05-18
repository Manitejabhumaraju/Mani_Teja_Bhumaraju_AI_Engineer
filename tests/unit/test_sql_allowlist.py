"""
SQL allowlist unit tests.

These don't touch MCP or the DB - the allowlist is pure validation logic.
The point is to exhaustively cover the attack surface so a future change
can't silently relax it.
"""

import pytest

from app.guardrails.sql_allowlist import validate_sql, SQLPolicyError


# === Pass cases =======================================================

@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders_gold",
    "select count(*) from orders_gold where city='Bangalore'",
    "SELECT store, AVG(avg_or2a) FROM orders_gold GROUP BY store",
    "  SELECT 1 FROM orders_gold  ",                           # whitespace ok
    "SELECT * FROM orders_gold;",                              # trailing ; ok
    "WITH x AS (SELECT * FROM orders_gold) SELECT * FROM x",   # WITH/CTE ok
    "SELECT a.* FROM orders_gold a JOIN orders_gold b ON a.store = b.store",
])
def test_allowed_queries(sql):
    validate_sql(sql)  # no exception = pass


# === Reject: DML/DDL ==================================================

@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE orders_gold",                "DDL"),
    ("DELETE FROM orders_gold WHERE 1=1",     "DML"),
    ("UPDATE orders_gold SET total_orders=0", "DML"),
    ("INSERT INTO orders_gold VALUES (...)",  "DML"),
    ("CREATE TABLE foo (id INT)",             "DDL"),
    ("ALTER TABLE orders_gold ADD COLUMN x",  "DDL"),
    ("TRUNCATE orders_gold",                  "DDL"),
    ("REPLACE INTO orders_gold VALUES (1)",   "DML"),
])
def test_rejects_dml_ddl(sql, reason):
    with pytest.raises(SQLPolicyError):
        validate_sql(sql)


# === Reject: SQLite-specific dangerous statements =====================

@pytest.mark.parametrize("sql", [
    "PRAGMA table_info(orders_gold)",
    "ATTACH DATABASE '/etc/passwd' AS leak",
    "DETACH DATABASE main",
    "VACUUM",
    "REINDEX orders_gold",
])
def test_rejects_sqlite_specials(sql):
    with pytest.raises(SQLPolicyError):
        validate_sql(sql)


# === Reject: statement stacking =======================================

def test_rejects_statement_stacking():
    with pytest.raises(SQLPolicyError, match="multiple statements"):
        validate_sql("SELECT * FROM orders_gold; DROP TABLE orders_gold")


def test_rejects_stacking_via_union_then_pragma():
    # The keyword check catches this even before the ; check
    with pytest.raises(SQLPolicyError):
        validate_sql("SELECT * FROM orders_gold; PRAGMA table_info(x)")


# === Reject: table allowlist ==========================================

@pytest.mark.parametrize("sql", [
    "SELECT * FROM sqlite_master",
    "SELECT name FROM sqlite_schema",
    "SELECT * FROM users",
    "SELECT a.* FROM orders_gold a JOIN secret_table s ON a.x = s.x",
])
def test_rejects_unknown_tables(sql):
    with pytest.raises(SQLPolicyError, match="allowlist"):
        validate_sql(sql)


def test_rejects_query_with_no_table_ref():
    # `SELECT 1` would let an attacker probe whether the agent has a working
    # SQL channel even without authorised data access.
    with pytest.raises(SQLPolicyError, match="must reference orders_gold"):
        validate_sql("SELECT 1")


# === Edge: empty input ================================================

@pytest.mark.parametrize("sql", ["", "   ", "\n\t", None])
def test_rejects_empty(sql):
    with pytest.raises((SQLPolicyError, AttributeError, TypeError)):
        validate_sql(sql)


# === Edge: keyword as identifier doesn't false-positive ===============

def test_column_name_containing_update_substring_is_fine():
    # 'updated_at' should NOT trigger the UPDATE rule because of word boundaries.
    validate_sql("SELECT updated_at_col FROM orders_gold")
    # But the actual UPDATE keyword should still trigger.
    with pytest.raises(SQLPolicyError):
        validate_sql("UPDATE orders_gold SET x=1")


# === The "must start with SELECT/WITH" rule ===========================

def test_select_can_start_after_whitespace():
    validate_sql("   SELECT * FROM orders_gold")


def test_comments_at_start_are_rejected():
    # A leading comment could hide an injected statement. Conservative: reject.
    # We don't try to be clever about stripping comments.
    with pytest.raises(SQLPolicyError):
        validate_sql("-- friendly comment\nDROP TABLE orders_gold")

"""
SQL allowlist validator.

Sits in front of the SQLite MCP server's read_query tool. Rejects any SQL
that isn't a pure SELECT against the orders_gold table.

This is intentionally syntactic and conservative. We're not parsing SQL
properly - we're saying "if it doesn't look like a SELECT against our one
table, refuse it." That's the right tradeoff: false positives (rejecting
legitimate SQL) are recoverable, false negatives (letting through
something destructive) are not.

The reason we don't use sqlparse or sqlglot here:
  - More deps for marginal value
  - The official sqlite server is in front of a real SQLite engine; even
    if our regex misses something, the engine refuses writes when launched
    in read-only mode... except the official server doesn't actually have
    a read-only mode, so we're the only line of defense. Keep it strict.

We get more sophisticated guardrails in app.guardrails.input_guards for
the natural-language layer.
"""

from __future__ import annotations

import re


class SQLPolicyError(ValueError):
    """Raised when a query violates the allowlist."""


# Anything that looks like a write or DDL statement. Word-boundary anchored
# so 'updated_at' in a column name doesn't trigger the UPDATE rule.
_FORBIDDEN_KEYWORDS = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b",
    r"\bDROP\b", r"\bCREATE\b", r"\bALTER\b", r"\bTRUNCATE\b",
    r"\bATTACH\b", r"\bDETACH\b",
    r"\bPRAGMA\b",
    r"\bREPLACE\b",
    r"\bVACUUM\b",
    r"\bREINDEX\b",
]

# Tables the agent may reference. Anything else is rejected so the LLM
# can't fish around in sqlite_master or attach external dbs.
_ALLOWED_TABLES = frozenset({"orders_gold"})

# Detect a FROM/JOIN clause and capture the table identifier so we can
# allowlist-check it. Very lightweight - won't catch every form, but the
# forbidden-keyword check above is the primary defense.
_TABLE_REF_RE = re.compile(
    r"(?:\bFROM\b|\bJOIN\b)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    re.IGNORECASE,
)

# Capture CTE names: WITH foo AS (...), bar AS (...) -> {foo, bar}
# Standard SQL only - no recursive CTEs needed for this scope.
_CTE_NAME_RE = re.compile(
    r"(?:\bWITH\b|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> None:
    """
    Raise SQLPolicyError if the query is not allowed.

    Returns None on success. Caller passes through to MCP.
    """
    if not sql or not sql.strip():
        raise SQLPolicyError("empty query")

    stripped = sql.strip().rstrip(";")

    # Must start with SELECT or WITH (common in analytical SQL).
    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise SQLPolicyError(
            f"only SELECT/WITH queries allowed; got: {stripped[:40]!r}..."
        )

    # No statement separators - one statement only.
    # (Strip trailing ; first, then any inner ; is a stack of statements.)
    if ";" in stripped:
        raise SQLPolicyError("multiple statements not allowed")

    # Forbidden keywords (DML/DDL).
    for pattern in _FORBIDDEN_KEYWORDS:
        if re.search(pattern, sql, re.IGNORECASE):
            raise SQLPolicyError(
                f"forbidden keyword in query: {pattern.strip(chr(92) + 'b')}"
            )

    # Table allowlist. CTE names defined in this query are allowed too -
    # they're aliases for subqueries that themselves reference orders_gold,
    # which the FROM check still enforces.
    referenced = {m.group(1).lower() for m in _TABLE_REF_RE.finditer(sql)}
    cte_names = {m.group(1).lower() for m in _CTE_NAME_RE.finditer(sql)}
    allowed_here = _ALLOWED_TABLES | cte_names

    disallowed = referenced - allowed_here
    if disallowed:
        raise SQLPolicyError(
            f"table(s) not in allowlist: {', '.join(sorted(disallowed))}"
        )

    # Must reference orders_gold somewhere (either directly or transitively
    # through a CTE that itself references it). If it's a CTE-only query,
    # check the inner FROMs include orders_gold.
    if "orders_gold" not in referenced:
        raise SQLPolicyError("query must reference orders_gold")

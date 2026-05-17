"""
Pydantic schemas for LLM structured outputs.

Every LLM call that returns JSON validates its output against one of these
schemas. If validation fails, the calling node returns a refusal rather than
passing garbage downstream. This is the contract layer between the LLM and
the deterministic engine.

Why Pydantic and not jsonschema:
- runtime validation with one line: ContextOutput.model_validate(d)
- coercion on common shapes (e.g. "true"/"false" → bool)
- model_dump() round-trips cleanly for logging
- ValidationError gives field-by-field reasons that go straight into a
  structured log entry
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ── Context resolver output ──────────────────────────────────────────────

class ContextOutput(BaseModel):
    """
    What the context_resolver / context agent extracts from a user message.
    
    The LLM is instructed to emit exactly these fields. If it omits one,
    we treat it as None (model_validate is lenient because total=False
    behavior is what we want).
    """
    model_config = ConfigDict(extra="ignore")
    
    scope: Literal["city", "store", "hour_range", "clarification"] = "clarification"
    city: Optional[str] = None
    store: Optional[str] = None
    date: Optional[str] = None
    time_range: Optional[str] = None
    is_followup: bool = False
    clarification_question: Optional[str] = None
    
    @field_validator("date")
    @classmethod
    def _date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        # cheap ISO check — full parse would be heavier and the resolver
        # downstream already tolerates malformed dates with a default
        if len(v) != 10 or v[4] != "-" or v[7] != "-":
            return None
        return v


# ── Intent classifier output ─────────────────────────────────────────────

INTENT_VALUES = (
    "city_rca",
    "store_rca",
    "hour_drill",
    "free_form",
    "schema_query",
    "meta_query",
    "refuse",
)


class IntentOutput(BaseModel):
    """What the intent_classifier emits."""
    model_config = ConfigDict(extra="ignore")
    
    intent: Literal[
        "city_rca", "store_rca", "hour_drill",
        "free_form", "schema_query", "meta_query", "refuse",
    ] = "refuse"
    intent_reason: str = Field(default="", max_length=200)


# ── Free-form SQL output ─────────────────────────────────────────────────

class SQLOutput(BaseModel):
    """What the free_form node's LLM produces — a SELECT plus rationale."""
    model_config = ConfigDict(extra="ignore")
    
    sql: str = Field(..., min_length=6)
    rationale: str = Field(default="", max_length=300)
    
    @field_validator("sql")
    @classmethod
    def _must_be_select(cls, v: str) -> str:
        stripped = v.strip().upper()
        if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
            raise ValueError("only SELECT or WITH allowed")
        forbidden = ("DROP ", "DELETE ", "INSERT ", "UPDATE ", "ALTER ", "CREATE ")
        if any(tok in stripped for tok in forbidden):
            raise ValueError("DML/DDL keywords not allowed")
        return v.strip()


# ── Input guard verdict ──────────────────────────────────────────────────

class InputGuardVerdict(BaseModel):
    """
    Output of the LLM-judge layer of the input guard.
    The cheap allowlist/denylist runs before this and may short-circuit.
    """
    model_config = ConfigDict(extra="ignore")
    
    decision: Literal["allow", "block"] = "allow"
    reason: str = Field(default="", max_length=200)


# ── Structured RCA report payloads ───────────────────────────────────────
# These are not LLM outputs — they're the shape of `raw_data` that the
# deterministic engine produces and that the synthesizer reads back. We
# keep them here so the same schemas are used by tests and by the
# observability metrics layer.

class CityRollup(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    city: str
    charge_date: str
    store_count: int
    hours_observed: int
    total_orders: int
    total_breached: int
    problem_hour_count: int
    pileup_hour_count: int
    weighted_breached_rate: float = Field(..., ge=0.0, le=1.0)
    weighted_avg_or2a: float = Field(..., ge=0.0)
    p50_or2a: float = Field(..., ge=0.0)
    p95_or2a: float = Field(..., ge=0.0)


class WorstStoreEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    store: str
    total_orders: int
    breached_count: int
    breach_rate: float = Field(..., ge=0.0, le=1.0)
    avg_or2a: float = Field(..., ge=0.0)


class CityRCAPayload(BaseModel):
    """raw_data shape produced by city_rca node."""
    model_config = ConfigDict(extra="ignore")
    
    city_rollup: CityRollup
    worst_stores: list[WorstStoreEntry]


# ── Final API response wrappers ──────────────────────────────────────────
# These are what the WebSocket emits to the client. Having them as Pydantic
# models means the frontend contract is documented in code.

class StageEvent(BaseModel):
    type: Literal["stage"] = "stage"
    name: str


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    delta: str


class FinalEvent(BaseModel):
    type: Literal["final"] = "final"
    state: dict


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str

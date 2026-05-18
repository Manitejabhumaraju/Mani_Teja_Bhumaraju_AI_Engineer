from app.rca.store_resolver import StoreResolver
import pytest


@pytest.fixture
def resolver():
    return StoreResolver({
        "STORE_001": "Bangalore",
        "STORE_002": "Bangalore",
        "STORE_003": "Bangalore",
        "STORE_101": "Chennai",
        "STORE_102": "Chennai",
        "STORE_201": "Mumbai north",
    })


def test_exact_match(resolver):
    result = resolver.resolve("STORE_001")
    assert result.found is True
    assert result.match == "STORE_001"
    assert result.city == "Bangalore"
    assert result.suggestions == []


def test_case_insensitive_match(resolver):
    result = resolver.resolve("store_002")
    assert result.found is True
    assert result.match == "STORE_002"


def test_unknown_store_returns_suggestions(resolver):
    result = resolver.resolve("STORE_005")
    assert result.found is False
    assert result.match is None
    # STORE_001, STORE_002, STORE_003 are similar - should suggest those
    suggested_codes = [s[0] for s in result.suggestions]
    assert any(c.startswith("STORE_00") for c in suggested_codes)


def test_garbage_input_returns_no_suggestions(resolver):
    result = resolver.resolve("zxcvbnm")
    assert result.found is False
    assert result.suggestions == []


def test_empty_input_handled(resolver):
    result = resolver.resolve("")
    assert result.found is False
    assert result.suggestions == []


def test_tbc8_case_from_brief(resolver):
    """Specifically: the brief uses TBC8 which isn't in any plausible dataset.
    Agent must not hallucinate. Suggestions are allowed if any score well enough."""
    result = resolver.resolve("TBC8")
    assert result.found is False
    # Whatever happens with suggestions is fine - the important thing is no false match.
    assert result.match is None


def test_stores_in_city(resolver):
    assert resolver.stores_in_city("Bangalore") == ["STORE_001", "STORE_002", "STORE_003"]


def test_city_case_insensitive(resolver):
    assert resolver.stores_in_city("bangalore") == resolver.stores_in_city("Bangalore")


def test_empty_resolver_rejected():
    with pytest.raises(ValueError):
        StoreResolver({})

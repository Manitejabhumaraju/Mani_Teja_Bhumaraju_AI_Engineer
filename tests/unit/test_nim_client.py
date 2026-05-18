"""
Tests for the LLM client's defensive JSON parser.

We deliberately don't test the network round-trip here - that needs an
NVIDIA API key and is what the eval harness is for. Unit tests cover the
deterministic parsing logic that catches LLM output sins.
"""

import pytest

from app.llm.nim_client import _parse_json_loose


def test_clean_json():
    assert _parse_json_loose('{"k": 1}') == {"k": 1}


def test_pretty_printed_json():
    text = """{
        "scope": "store",
        "store": "STORE_003"
    }"""
    assert _parse_json_loose(text) == {"scope": "store", "store": "STORE_003"}


def test_markdown_fenced_json():
    text = '```json\n{"scope": "city", "city": "Bangalore"}\n```'
    assert _parse_json_loose(text) == {"scope": "city", "city": "Bangalore"}


def test_unlabeled_fenced_json():
    text = '```\n{"k": "v"}\n```'
    assert _parse_json_loose(text) == {"k": "v"}


def test_preamble_then_json():
    """LLMs often say 'Here is the JSON:' first. Strip it."""
    text = "Here is the result:\n{\"scope\": \"store\", \"store\": \"STORE_001\"}"
    assert _parse_json_loose(text) == {"scope": "store", "store": "STORE_001"}


def test_trailing_commentary_then_json():
    text = "{\"k\": 1}\nLet me know if you need clarification."
    # Our parser finds the first { and the last } so commentary after the
    # closing brace ends up in the search range. The find/rfind strategy
    # only works if the trailing text doesn't contain another }.
    assert _parse_json_loose(text) == {"k": 1}


def test_empty_returns_empty_dict():
    assert _parse_json_loose("") == {}
    assert _parse_json_loose("   ") == {}


def test_unparseable_returns_error_marker():
    text = "I cannot do that."
    result = _parse_json_loose(text)
    assert result.get("_parse_error") is True
    assert "_raw" in result


def test_nested_json_works():
    text = '{"a": {"b": [1, 2, 3]}, "c": "ok"}'
    assert _parse_json_loose(text) == {"a": {"b": [1, 2, 3]}, "c": "ok"}

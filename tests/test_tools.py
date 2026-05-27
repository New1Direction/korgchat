"""Tests for the v0.4.1 tool registry."""

from __future__ import annotations

import pytest

from korgchat.tools import Tool, ToolRegistry, default_tools


def test_default_registry_has_three_builtins():
    reg = default_tools()
    assert sorted(reg.names()) == ["add", "echo", "get_time"]
    assert len(reg) == 3


def test_echo_round_trips_input():
    reg = default_tools()
    out = reg.get("echo").call({"input": "hello korg"})
    assert out == {"echoed": "hello korg"}


def test_add_returns_sum():
    reg = default_tools()
    out = reg.get("add").call({"a": 2, "b": 3})
    assert out == {"sum": 5}


def test_add_rejects_non_numeric():
    reg = default_tools()
    with pytest.raises(TypeError):
        reg.get("add").call({"a": "two", "b": 3})


def test_add_rejects_missing_input():
    reg = default_tools()
    with pytest.raises(ValueError):
        reg.get("add").call({"a": 2})


def test_get_time_frozen_is_deterministic():
    reg = default_tools(frozen_time=1234567890.0)
    a = reg.get("get_time").call({})
    b = reg.get("get_time").call({})
    assert a == b == {"unix_seconds": 1234567890.0}


def test_get_time_unfrozen_returns_recent_value():
    import time as _time

    reg = default_tools()
    out = reg.get("get_time").call({})
    # Sanity: within a couple seconds of "now" (test is not running on a TARDIS).
    assert "unix_seconds" in out
    assert abs(out["unix_seconds"] - _time.time()) < 5.0


def test_register_duplicate_rejected():
    reg = ToolRegistry()
    t = Tool(name="t", description="", input_schema={}, handler=lambda _: None)
    reg.register(t)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(t)


def test_get_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="unknown tool"):
        reg.get("nope")


def test_anthropic_schema_shape():
    reg = default_tools()
    schemas = reg.to_anthropic_tools()
    assert len(schemas) == 3
    for s in schemas:
        assert set(s.keys()) == {"name", "description", "input_schema"}
        assert isinstance(s["input_schema"], dict)

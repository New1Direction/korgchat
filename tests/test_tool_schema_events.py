"""Tests for tool-schema snapshot + conformance events (task #7).

A replayed conversation goes stale the moment a tool's schema changes:
the journal records the *call* (args, result) but not the *contract* it
was made against. So you can't tell, six months later, whether the model
called a tool correctly — the `input_schema` it was shown is gone.

This pins the fix. For every tool execution KorgChat now records:

  * `tool_schema_snapshot` — BEFORE the call: the tool's `input_schema`,
    `description`, and a deterministic `schema_hash`. Freezes the contract
    the model was operating against at that point in time.
  * `tool_validation`     — AFTER the call: did the input the model sent
    conform to the declared `input_schema`? Did the call succeed? A
    `valid` flag + any violations.

Causal chain for one tool call:

    llm_inference (L)
      └─ tool_schema_snapshot   triggered_by = L
      └─ <tool_call>            triggered_by = L   (sibling)
      └─ tool_validation        triggered_by = <tool_call>

The schema_hash is canonical (sorted keys, compact separators, sha256)
so it matches across re-snapshots of an unchanged schema and DIVERGES
the instant the schema changes — that's the whole point.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from korgchat import ChatSession, MockResponder
from korgchat.schema import schema_hash, validate_input
from korgchat.tools import Tool, ToolRegistry, default_tools


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


def _by_name(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"]["tool_name"] == name]


# ── schema_hash: deterministic + canonical ─────────────────────────────


def test_schema_hash_is_deterministic():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    assert schema_hash(schema) == schema_hash(schema)


def test_schema_hash_is_key_order_invariant():
    """Canonicalization sorts keys: logically-identical schemas written in
    different key orders must hash identically."""
    s1 = {"type": "object", "required": ["a"], "properties": {"a": {"type": "number"}}}
    s2 = {"properties": {"a": {"type": "number"}}, "required": ["a"], "type": "object"}
    assert schema_hash(s1) == schema_hash(s2)


def test_schema_hash_diverges_on_change():
    """The instant the schema changes, the hash changes — the property that
    makes a replay able to detect 'this tool's contract drifted.'"""
    s1 = {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}
    s2 = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert schema_hash(s1) != schema_hash(s2)


def test_schema_hash_is_sha256_hex():
    h = schema_hash({"type": "object"})
    assert len(h) == 64
    int(h, 16)  # valid hex


# ── validate_input: minimal JSON-Schema conformance ────────────────────


def test_validate_input_accepts_conforming_call():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    res = validate_input(schema, {"a": 1, "b": 2})
    assert res.valid is True
    assert res.violations == []


def test_validate_input_flags_missing_required():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }
    res = validate_input(schema, {"a": 1})
    assert res.valid is False
    assert any("b" in v for v in res.violations)


def test_validate_input_flags_wrong_type():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}},
        "required": ["a"],
    }
    res = validate_input(schema, {"a": "not a number"})
    assert res.valid is False
    assert any("a" in v for v in res.violations)


def test_validate_input_string_type():
    schema = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }
    assert validate_input(schema, {"input": "hi"}).valid is True
    assert validate_input(schema, {"input": 5}).valid is False


# ── ChatSession emits snapshot + validation events ─────────────────────


def test_tool_call_emits_schema_snapshot_and_validation(tmp_journal):
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        tools=default_tools(),
    )
    # MockResponder free mode parses the [tool:...] marker and calls it.
    s.send("please [tool:add(a=2, b=3)]")

    events = _events(tmp_journal)
    snaps = _by_name(events, "tool_schema_snapshot")
    vals = _by_name(events, "tool_validation")
    assert len(snaps) == 1
    assert len(vals) == 1

    snap = snaps[0]["event"]
    # Snapshot captures the declared contract.
    assert snap["args"]["tool_name"] == "add"
    assert snap["result"]["input_schema"]["required"] == ["a", "b"]
    assert "Add two numbers" in snap["result"]["description"]
    assert len(snap["result"]["schema_hash"]) == 64

    val = vals[0]["event"]
    assert val["args"]["tool_name"] == "add"
    # A correct call validates clean.
    assert val["result"]["valid"] is True
    assert val["result"]["violations"] == []
    assert val["result"]["call_succeeded"] is True
    # The validation references the snapshot's schema_hash so a replay can
    # tie the verdict to the exact contract it was made against.
    assert val["result"]["schema_hash"] == snap["result"]["schema_hash"]


def test_schema_snapshot_precedes_tool_call_which_precedes_validation(tmp_journal):
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        tools=default_tools(),
    )
    s.send("[tool:add(a=1, b=1)]")

    events = _events(tmp_journal)
    order = [e["event"]["tool_name"] for e in events]
    i_snap = order.index("tool_schema_snapshot")
    i_call = order.index("add")
    i_val = order.index("tool_validation")
    assert i_snap < i_call < i_val


def test_schema_events_causal_links(tmp_journal):
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        tools=default_tools(),
    )
    s.send("[tool:add(a=1, b=1)]")

    events = _events(tmp_journal)
    by_seq = {e["seq_id"]: e for e in events}

    snap = _by_name(events, "tool_schema_snapshot")[0]
    call = _by_name(events, "add")[0]
    val = _by_name(events, "tool_validation")[0]

    # The producing llm_inference triggers both the snapshot and the call.
    llm_seq = call["metadata"]["triggered_by"]
    assert by_seq[llm_seq]["event"]["tool_name"] == "llm_inference"
    assert snap["metadata"]["triggered_by"] == llm_seq

    # The validation chains from the tool_call it judges.
    assert val["metadata"]["triggered_by"] == call["seq_id"]


def test_validation_flags_bad_call(tmp_journal):
    """A call that violates the schema is recorded as invalid — and the
    journal still tells the full story (snapshot + call + validation)."""
    # add requires numbers; feed it a string via a custom tool whose
    # handler will raise, exercising the call_succeeded=False path too.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    def _add(args):
        return {"sum": args["a"] + args["b"]}

    reg = ToolRegistry(
        [Tool(name="add", description="Add.", input_schema=schema, handler=_add)]
    )
    s = ChatSession(
        journal_path=tmp_journal, responder=MockResponder(), tools=reg
    )
    # 'a' is a string here → schema violation (and the handler will raise
    # on str + int, so call_succeeded should be False).
    s.send('[tool:add(a="x", b=3)]')

    events = _events(tmp_journal)
    val = _by_name(events, "tool_validation")[0]["event"]
    assert val["result"]["valid"] is False
    assert any("a" in v for v in val["result"]["violations"])
    assert val["result"]["call_succeeded"] is False


def test_no_schema_events_without_tools(tmp_journal):
    """A plain text turn (no tool call) emits neither snapshot nor
    validation — these events are strictly per-tool-execution."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("just a haiku please")

    events = _events(tmp_journal)
    assert _by_name(events, "tool_schema_snapshot") == []
    assert _by_name(events, "tool_validation") == []


def test_unknown_tool_still_validates(tmp_journal):
    """If the model calls a tool that doesn't exist, there's no declared
    schema to snapshot — but we still record a validation event marking the
    call invalid so the replay isn't silent about it."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        tools=default_tools(),
    )
    s.send("[tool:nonexistent(x=1)]")

    events = _events(tmp_journal)
    vals = _by_name(events, "tool_validation")
    assert len(vals) == 1
    val = vals[0]["event"]
    assert val["args"]["tool_name"] == "nonexistent"
    assert val["result"]["valid"] is False
    assert val["result"]["call_succeeded"] is False
    # No declared schema → empty/None schema_hash, flagged as unknown.
    assert val["result"].get("schema_known") is False

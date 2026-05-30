"""Tests for context_injection ledger events (task #5).

The auto-context preamble that the model actually sees was a *ghost*: the
journal recorded only the user's ORIGINAL prompt, never the injected
recall context. A replay couldn't reconstruct what the model was shown.

This pins the fix: whenever auto-context injects a preamble, a
`context_injection` event lands in the journal BEFORE the `llm_inference`
it feeds, capturing:

  * the preamble text,
  * the recall query (the user's prompt),
  * the matched seq_ids + scores,

and is causally linked: user_prompt → context_injection → llm_inference.

These tests run without `fastembed` by relying on substring-mode
auto-context (overlapping query terms), so they're deterministic in CI.
The original-prompt-pristine invariant from test_auto_context.py still
holds — the user_prompt event is untouched; the injected context is a
*separate, first-class* event.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from korgchat import ChatSession, MockResponder, Reply
from korgchat.context import AutoContextEngine


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


def _by_name(events: list[dict], name: str) -> list[dict]:
    return [e for e in events if e["event"]["tool_name"] == name]


# ── AutoContextEngine surfaces structured matches ──────────────────────


def test_engine_build_context_returns_preamble_and_matches(tmp_journal):
    """build_context() must expose the matched events (seq_id + score),
    not just the rendered preamble string. The ledger event needs them."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust borrow checker explainer")  # seq 1 user, 2 llm

    engine = AutoContextEngine(s, min_score=0.30)
    result = engine.build_context("rust borrow checker", exclude_seqs={3})
    assert result is not None
    preamble, matches = result.preamble, result.matches
    assert preamble is not None
    assert matches, "expected at least one matched prior event"
    # Each match carries the structured fields the ledger event records.
    for m in matches:
        assert isinstance(m.seq_id, int)
        assert isinstance(m.score, float)
        assert isinstance(m.kind, str)
    # The matched seq should be the prior user_prompt (seq=1), not the
    # excluded seq=3.
    seqs = {m.seq_id for m in matches}
    assert 1 in seqs
    assert 3 not in seqs


def test_engine_build_context_none_when_no_match(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust borrow checker")
    engine = AutoContextEngine(s, min_score=0.40)
    # Disjoint terms — AND-substring recall finds nothing.
    assert engine.build_context("quantum chromodynamics gauge") is None


# ── ChatSession emits a context_injection event ────────────────────────


def test_send_emits_context_injection_event(tmp_journal):
    """When a preamble is injected, a context_injection event is recorded."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
    )
    # Seed a prior turn whose prompt is a term-superset of the next one,
    # so AND-substring recall (no fastembed in CI) fires deterministically.
    s.send("rust borrow checker explainer please")
    # Every term of this prompt appears in the seeded one → recall fires.
    s.send("rust borrow checker please")

    events = _events(tmp_journal)
    injections = _by_name(events, "context_injection")
    assert injections, "expected a context_injection event in the journal"

    ev = injections[0]["event"]
    args = ev["args"]
    result = ev["result"]
    # Recall query = the user's (original) prompt that triggered injection.
    assert args["query"] == "rust borrow checker please"
    # The preamble text the model actually saw is captured.
    assert "Relevant prior conversation" in result["preamble"]
    # Matched seq_ids + scores are recorded.
    assert result["match_count"] >= 1
    assert isinstance(result["matches"], list)
    first = result["matches"][0]
    assert "seq_id" in first and "score" in first
    # The seeded prompt (seq=1) is among the matches.
    matched_seqs = {m["seq_id"] for m in result["matches"]}
    assert 1 in matched_seqs


def test_context_injection_is_causally_linked(tmp_journal):
    """user_prompt → context_injection → llm_inference: the injection
    chains from the user prompt, and the inference chains from the
    injection (so the injected context is on the causal path)."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
    )
    s.send("rust borrow checker explainer please")
    s.send("rust borrow checker please")

    events = _events(tmp_journal)
    # Map seq_id → event for triggered_by lookups.
    by_seq = {e["seq_id"]: e for e in events}

    injections = _by_name(events, "context_injection")
    assert len(injections) == 1
    inj = injections[0]
    inj_seq = inj["seq_id"]

    # The user_prompt for the SECOND turn is the injection's trigger.
    trig = inj["metadata"]["triggered_by"]
    assert trig is not None
    trigger_ev = by_seq[trig]
    assert trigger_ev["event"]["tool_name"] == "user_prompt"
    assert trigger_ev["event"]["args"]["prompt"] == "rust borrow checker please"

    # The llm_inference that follows must chain from the injection, not
    # straight from the user_prompt — the model "saw" the injected context.
    llms_after = [
        e for e in events
        if e["event"]["tool_name"] == "llm_inference"
        and e["seq_id"] > inj_seq
    ]
    assert llms_after, "expected an llm_inference after the injection"
    assert llms_after[0]["metadata"]["triggered_by"] == inj_seq


def test_no_injection_event_when_preamble_absent(tmp_journal):
    """No match → no context_injection event (no audit noise)."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
    )
    # Two disjoint prompts: nothing to recall, so no preamble.
    s.send("alpha beta gamma")
    s.send("delta epsilon zeta")

    events = _events(tmp_journal)
    assert _by_name(events, "context_injection") == []


def test_no_injection_event_when_auto_context_off(tmp_journal):
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=False,
    )
    s.send("rust borrow checker explainer")
    s.send("rust borrow checker again")

    events = _events(tmp_journal)
    assert _by_name(events, "context_injection") == []


def test_user_prompt_stays_pristine_with_injection_event(tmp_journal):
    """The original-prompt invariant still holds: the user_prompt event
    records ONLY what the user typed, even though the injected context is
    now its own first-class event."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
    )
    s.send("rust borrow checker explainer please")
    s.send("rust borrow checker please")

    events = _events(tmp_journal)
    # Sanity: the injection path actually fired (otherwise this test would
    # pass vacuously).
    assert _by_name(events, "context_injection")
    for e in _by_name(events, "user_prompt"):
        prompt = e["event"]["args"]["prompt"]
        assert "Relevant prior conversation" not in prompt
        assert "auto-recalled" not in prompt

"""Tests for KorgChat v0.4.0.

These run against the actual korg_bridge extension — no mocks on the
ledger side. The MockResponder gives us deterministic LLM output so the
assertions are stable.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from korgchat import ChatSession, MockResponder, __version__
from korgchat.__main__ import main as cli_main


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


# ── Smoke ──────────────────────────────────────────────────────────────────


def test_version_string():
    assert __version__ == "0.4.0"


def test_session_construct(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    assert s.turns == 0
    assert tmp_journal.parent.exists()


# ── Single-turn shape ──────────────────────────────────────────────────────


def test_single_turn_writes_two_events(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    turn = s.send("hello korg")

    assert turn.user_seq == 1
    assert turn.assistant_seq == 2
    assert turn.assistant_text  # non-empty deterministic reply

    events = _events(tmp_journal)
    assert len(events) == 2
    assert events[0]["event"]["tool_name"] == "user_prompt"
    assert events[1]["event"]["tool_name"] == "llm_inference"
    assert events[0]["metadata"]["triggered_by"] is None
    assert events[1]["metadata"]["triggered_by"] == turn.user_seq


def test_root_event_id_shared_across_turn(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("hi")
    events = _events(tmp_journal)
    root = events[0]["metadata"]["root_event_id"]
    assert events[1]["metadata"]["root_event_id"] == root


# ── Multi-turn causal chain ────────────────────────────────────────────────


def test_three_turn_chain(tmp_journal):
    """Turn N's user_prompt chains to turn (N-1)'s llm_inference."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    t1 = s.send("turn one")
    t2 = s.send("turn two")
    t3 = s.send("turn three")

    # 6 events: 3 user_prompts interleaved with 3 llm_inferences.
    events = _events(tmp_journal)
    assert [e["seq_id"] for e in events] == [1, 2, 3, 4, 5, 6]

    # Cross-turn linkage: turn-2 user chains to turn-1 llm, turn-3 user
    # chains to turn-2 llm.
    assert events[0]["metadata"]["triggered_by"] is None        # root user
    assert events[1]["metadata"]["triggered_by"] == 1           # llm ← user
    assert events[2]["metadata"]["triggered_by"] == 2           # next-user ← prior-llm
    assert events[3]["metadata"]["triggered_by"] == 3           # llm ← user
    assert events[4]["metadata"]["triggered_by"] == 4           # next-user ← prior-llm
    assert events[5]["metadata"]["triggered_by"] == 5           # llm ← user

    # Whole conversation shares one root_event_id.
    roots = {e["metadata"]["root_event_id"] for e in events}
    assert len(roots) == 1


def test_history_accumulates(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("a")
    s.send("b")
    history = s.history
    assert len(history) == 2
    assert history[0].user_prompt == "a"
    assert history[1].user_prompt == "b"


# ── Persistence ────────────────────────────────────────────────────────────


def test_resume_picks_up_where_we_left_off(tmp_journal):
    """Closing and reopening a session continues the causal chain."""
    s1 = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s1.send("first turn")
    s1.send("second turn")
    del s1

    s2 = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    # Resumed session sees the existing seq counter through the bridge.
    t = s2.send("third turn after resume")
    events = _events(tmp_journal)

    # 6 events total: 4 from s1, 2 from s2.
    assert [e["seq_id"] for e in events] == [1, 2, 3, 4, 5, 6]
    # The post-resume user_prompt must chain to the last llm_inference of s1.
    assert events[4]["metadata"]["triggered_by"] == 4
    assert t.user_seq == 5
    assert t.assistant_seq == 6


# ── Input validation ──────────────────────────────────────────────────────


def test_empty_prompt_rejected(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    with pytest.raises(ValueError):
        s.send("")
    with pytest.raises(ValueError):
        s.send("   ")


# ── Mock determinism ───────────────────────────────────────────────────────


def test_mock_responder_deterministic():
    r = MockResponder()
    a, _, _ = r.respond([], "test prompt")
    b, _, _ = r.respond([], "test prompt")
    assert a == b


def test_mock_responder_varies_by_input():
    r = MockResponder()
    a, _, _ = r.respond([], "input one")
    b, _, _ = r.respond([], "input two")
    # Not guaranteed to differ for two inputs (5 buckets), but we'll pick
    # two we know map to different buckets in this hashing scheme.
    found_different = False
    for prompt in ["a", "bb", "ccc", "dddd", "eeeee", "ffffff"]:
        r2, _, _ = r.respond([], prompt)
        if r2 != a:
            found_different = True
            break
    assert found_different


# ── CLI entry point ────────────────────────────────────────────────────────


def test_cli_mock_three_turns_via_stdin(tmp_journal, monkeypatch, capsys):
    """Drive the CLI with three prompts piped to stdin, then EOF."""
    stdin = io.StringIO("hello\nhow are you\nbye\n")
    monkeypatch.setattr("sys.stdin", stdin)

    rc = cli_main(["--mock", "--journal", str(tmp_journal)])
    assert rc == 0

    events = _events(tmp_journal)
    # 3 turns × 2 events = 6 events.
    assert len(events) == 6
    out = capsys.readouterr().out
    assert "KorgChat" in out
    assert "Korg:" in out
    # Three "recorded" lines confirm three full turns landed.
    assert out.count("[recorded:") == 3


def test_cli_quit_command(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO("first\n/quit\nshould-not-be-recorded\n")
    monkeypatch.setattr("sys.stdin", stdin)

    rc = cli_main(["--mock", "--journal", str(tmp_journal)])
    assert rc == 0

    events = _events(tmp_journal)
    # Only one turn recorded — /quit exited before the next prompt.
    assert len(events) == 2

"""Tests for v0.5.1 /summarize.

Two layers:
  1. SummarizeEngine unit tests against a real ChatSession driving the
     MockResponder. The mock detects the summary-prompt marker and
     returns a templated digest reflecting the actual event mix, so
     assertions can check both the prompt construction and the response
     shape without a network call.
  2. CLI tests that drive the /summarize slash command via stdin.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from korgchat import ChatSession, MockResponder, Reply, ToolUse
from korgchat.__main__ import main as cli_main
from korgchat.summary import SUMMARY_PROMPT_MARKER, SummarizeEngine


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


# ── Engine ────────────────────────────────────────────────────────────────


def test_summarize_empty_scope_returns_placeholder(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    eng = SummarizeEngine(s)
    summary = eng.summarize_branch()
    assert summary.event_count == 0
    assert summary.text == "(no events to summarise in this scope)"
    assert not summary.truncated


def test_summarize_branch_default_is_current(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("first turn")
    s.send("second turn")
    eng = SummarizeEngine(s)
    summary = eng.summarize_branch()
    assert "branch 'main'" in summary.scope_descriptor
    # All 4 events up to journal-latest are in scope.
    assert summary.event_count == 4
    # Mock digest is structurally honest about counts.
    assert "2 user prompt" in summary.text
    assert "2 assistant" in summary.text


def test_summarize_named_branch(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("main turn 1")           # seq 1,2 on main
    s.fork_here("experiment")
    s.send("exp turn 1")            # seq 3,4 on experiment
    s.send("exp turn 2")            # seq 5,6 on experiment (tip)
    s.checkout("main")
    s.send("main turn 2")           # seq 7,8 on main

    eng = SummarizeEngine(s)
    # experiment's tip is 6 → events ≤ 6 = the 6-event prefix.
    exp_summary = eng.summarize_branch("experiment")
    assert "experiment" in exp_summary.scope_descriptor
    assert exp_summary.event_count == 6

    # main's tip = journal latest (8) → all 8 events in scope. v0.5.1
    # accepts the over-include from sibling branches as a known shortcut.
    main_summary = eng.summarize_branch("main")
    assert "main" in main_summary.scope_descriptor
    assert main_summary.event_count == 8


def test_summarize_tool_calls_render_in_prompt(tmp_journal):
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(
            replies=[
                Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 2, "b": 3})]),
                Reply(text="five"),
            ]
        ),
    )
    s.send("add some numbers")
    # Use a fresh free-mode mock to actually exercise summary detection.
    s.responder = MockResponder()
    eng = SummarizeEngine(s)
    summary = eng.summarize_branch()
    # Tool count surfaces in the mock digest body.
    assert "1 tool invocation" in summary.text


def test_summarize_since_scope(tmp_journal):
    """Events outside the since window must be excluded from the prompt."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("ancient turn")
    s.send("recent turn")

    # Rewrite the first turn's events to look 400 days old.
    raw = json.loads(tmp_journal.read_text())
    ancient = (
        (datetime.now(tz=timezone.utc) - timedelta(days=400))
        .isoformat()
        .replace("+00:00", "Z")
    )
    for i in (0, 1):
        raw[i]["event"]["timestamp"] = ancient
    tmp_journal.write_text(json.dumps(raw, indent=2))

    eng = SummarizeEngine(s)
    summary = eng.summarize_since(timedelta(days=30))
    # Only 2 events (the recent turn) should be in scope.
    assert summary.event_count == 2
    assert "last 30d" in summary.scope_descriptor


def test_summarize_topic_scope(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("question about rust ownership")
    s.send("question about python decorators")
    s.send("more on rust lifetimes")

    eng = SummarizeEngine(s)
    summary = eng.summarize_topic("rust")
    # The "rust" topic should pick up the two rust-related user_prompts and
    # any assistant replies that contain "rust" in their text (mock haikus
    # don't, so just the two user_prompts).
    assert "topic 'rust'" in summary.scope_descriptor
    assert summary.event_count >= 2


def test_summarize_limit_truncates_to_recent(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    for i in range(10):
        s.send(f"turn {i}")
    eng = SummarizeEngine(s)
    summary = eng.summarize_branch(limit=4)
    assert summary.truncated is True
    assert summary.event_count == 4


def test_summarize_save_writes_event(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("a discussion of rust lifetimes")
    eng = SummarizeEngine(s)
    summary = eng.summarize_branch(save=True)
    assert summary.seq_id is not None
    # The new event should be findable via /recall.
    events = _events(tmp_journal)
    saved = next(e for e in events if e["event"]["tool_name"] == "summary")
    assert saved["event"]["result"]["text"] == summary.text
    assert saved["event"]["source_agent"] == "agent:korgchat-summarizer"


def test_prompt_marker_present_in_responder_call(tmp_journal):
    """The summary prompt must carry the marker so MockResponder routes it
    to the digest branch instead of the haiku bucket."""
    captured_prompts: list[str] = []

    class _CapturingMock(MockResponder):
        def respond(self, **kw):
            captured_prompts.append(kw["prompt"])
            return super().respond(**kw)

    s = ChatSession(journal_path=tmp_journal, responder=_CapturingMock())
    s.send("warmup")
    eng = SummarizeEngine(s)
    eng.summarize_branch()
    # The most recent prompt should be the summary prompt — leading marker.
    assert captured_prompts[-1].startswith(SUMMARY_PROMPT_MARKER)
    assert "=== EVENTS ===" in captured_prompts[-1]


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_summarize_default_branch(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "first turn\n"
        "second turn\n"
        "/summarize\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[summarize] branch 'main'" in out
    assert "user prompt" in out or "assistant" in out


def test_cli_summarize_save_round_trips_with_recall(tmp_journal, monkeypatch, capsys):
    """A /summarize --save makes the summary findable via /recall."""
    stdin = io.StringIO(
        "tell me about rust\n"
        "/summarize --save\n"
        "/recall rust\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    # The save line tells us the seq_id was assigned.
    assert "saved as seq=" in out
    # /recall should pick up the summary event (since the rust prompt
    # itself + the saved summary mention the scope label).
    assert "match" in out


def test_cli_summarize_since(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "first\n"
        "/summarize --since 24h\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    # _format_duration collapses 24h to 1d; either label is acceptable.
    assert "last 1d" in out or "last 24h" in out


def test_cli_summarize_topic(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "rust ownership\n"
        "python decorators\n"
        "/summarize --topic rust\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "topic 'rust'" in out


def test_cli_summarize_mutually_exclusive_scopes(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "warmup\n"
        "/summarize main --topic rust\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "choose exactly one scope" in out


def test_cli_help_lists_summarize(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("/help\n/quit\n"))
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/summarize" in out
    assert "--topic" in out

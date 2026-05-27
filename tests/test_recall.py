"""Tests for the v0.4.3 RecallEngine + /recall slash command.

The engine reads the on-disk journal directly; tests build a realistic
journal by driving an actual ChatSession with scripted MockResponder
replies so the on-disk shape (including assistant_text via bridge v0.3.2)
matches what production would write.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from korgchat import (
    ChatSession,
    MockResponder,
    RecallEngine,
    Reply,
    ToolUse,
    format_matches,
)
from korgchat.__main__ import main as cli_main


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _build_session(journal: Path, scripted: list[Reply]) -> ChatSession:
    return ChatSession(journal_path=journal, responder=MockResponder(scripted))


# ── Empty / edge cases ────────────────────────────────────────────────────


def test_missing_journal_returns_empty(tmp_path):
    eng = RecallEngine(tmp_path / "never.json", mode='substring')
    assert eng.search("anything") == []


def test_empty_query_returns_empty(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("hello")
    eng = RecallEngine(tmp_journal, mode='substring')
    assert eng.search("") == []
    assert eng.search("   ") == []


def test_no_match_returns_empty(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("hello korg")
    eng = RecallEngine(tmp_journal, mode='substring')
    assert eng.search("absolutely_not_present") == []


# ── Basic hits ────────────────────────────────────────────────────────────


def test_finds_user_prompt(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("tell me about rust ownership")
    s.send("now tell me about python decorators")
    eng = RecallEngine(tmp_journal, mode='substring')
    hits = eng.search("rust")
    assert len(hits) == 1
    assert hits[0].kind == "user_prompt"
    assert "rust" in hits[0].snippet


def test_finds_assistant_text(tmp_journal):
    """v0.4.3 stores Reply.text on llm_inference events via bridge 0.3.2."""
    s = _build_session(
        tmp_journal,
        scripted=[
            Reply(text="the answer is foo bar baz", prompt_tokens=4, completion_tokens=6),
        ],
    )
    s.send("what is it?")
    eng = RecallEngine(tmp_journal, mode='substring')
    hits = eng.search("foo")
    assert len(hits) == 1
    assert hits[0].kind == "llm_inference"
    assert "foo" in hits[0].snippet


def test_finds_tool_call_by_name(tmp_journal):
    """`/recall add` should find an add tool call even without args matching."""
    s = _build_session(
        tmp_journal,
        scripted=[
            Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 1, "b": 2})]),
            Reply(text="done"),
        ],
    )
    s.send("compute")
    eng = RecallEngine(tmp_journal, mode='substring')
    hits = eng.search("add")
    assert any(h.kind == "add" for h in hits)


def test_and_semantics_requires_all_terms(tmp_journal):
    """Multi-term query is AND — every term must appear in the event."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust borrow checker")
    s.send("python decorators")
    eng = RecallEngine(tmp_journal, mode='substring')
    # Both terms in the first prompt → match
    assert len(eng.search("rust borrow")) == 1
    # "rust" yes, "decorators" only in the OTHER prompt → no event has both
    assert eng.search("rust decorators") == []


def test_case_insensitive(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("Ledgers Are Cool")
    eng = RecallEngine(tmp_journal, mode='substring')
    assert len(eng.search("ledgers")) == 1
    assert len(eng.search("LEDGERS")) == 1
    assert len(eng.search("LeDgErS")) == 1


# ── Filters ───────────────────────────────────────────────────────────────


def test_kind_filter_user_prompt_only(tmp_journal):
    s = _build_session(
        tmp_journal,
        scripted=[Reply(text="ledger holds the rust history", prompt_tokens=4, completion_tokens=5)],
    )
    s.send("ask about rust history")
    eng = RecallEngine(tmp_journal, mode='substring')
    # Both events contain "rust" — kind filter picks one.
    user_only = eng.search("rust", kind="user_prompt")
    llm_only = eng.search("rust", kind="llm_inference")
    assert len(user_only) == 1 and user_only[0].kind == "user_prompt"
    assert len(llm_only) == 1 and llm_only[0].kind == "llm_inference"


def test_kind_filter_tool_call_excludes_chat_events(tmp_journal):
    """--kind tool_call must filter out user_prompt and llm_inference."""
    s = _build_session(
        tmp_journal,
        scripted=[
            Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 1, "b": 2})]),
            Reply(tool_uses=[ToolUse(id="t2", name="echo", input={"input": "hi"})]),
            Reply(text="done"),
        ],
    )
    # Use a prompt containing both tool names so the user_prompt event also
    # textually matches both queries below — this proves the kind filter
    # is what excludes it, not the search term missing.
    s.send("please run add then echo")

    eng = RecallEngine(tmp_journal, mode='substring')
    # Without filter, query "add" hits the user_prompt + the add tool_call.
    no_filter_add = eng.search("add")
    assert any(h.kind == "user_prompt" for h in no_filter_add)
    assert any(h.kind == "add" for h in no_filter_add)

    # With --kind tool_call, the user_prompt should be gone.
    tool_only_add = eng.search("add", kind="tool_call")
    assert all(h.kind not in ("user_prompt", "llm_inference") for h in tool_only_add)
    assert any(h.kind == "add" for h in tool_only_add)

    # Same shape for echo.
    tool_only_echo = eng.search("echo", kind="tool_call")
    assert all(h.kind not in ("user_prompt", "llm_inference") for h in tool_only_echo)
    assert any(h.kind == "echo" for h in tool_only_echo)


def test_since_filter_excludes_old_events(tmp_journal):
    """Events older than `since` are filtered out. We mock by editing the
    on-disk journal's timestamps because that's what production loads from."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("first about rust")
    s.send("second about rust too")

    # Rewrite the first event's timestamp to look ancient.
    raw = json.loads(tmp_journal.read_text())
    ancient = (datetime.now(tz=timezone.utc) - timedelta(days=400)).isoformat().replace("+00:00", "Z")
    raw[0]["event"]["timestamp"] = ancient
    tmp_journal.write_text(json.dumps(raw, indent=2))

    eng = RecallEngine(tmp_journal, mode='substring')
    recent_only = eng.search("rust", since=timedelta(days=30))
    # Both prompts say "rust" but the first is now 400d old → excluded.
    assert len(recent_only) == 1
    assert "second" in recent_only[0].snippet


def test_limit_caps_results(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    for i in range(5):
        s.send(f"turn {i} mentions ledger")
    eng = RecallEngine(tmp_journal, mode='substring')
    all_hits = eng.search("ledger")
    assert len(all_hits) == 5
    capped = eng.search("ledger", limit=2)
    assert len(capped) == 2


# ── Ranking ──────────────────────────────────────────────────────────────


def test_more_hits_outranks_fewer(tmp_journal):
    """An event with 3 occurrences of 'rust' beats an event with 1."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust")
    s.send("rust rust rust borrow")
    eng = RecallEngine(tmp_journal, mode='substring')
    hits = eng.search("rust")
    # Top-ranked should be the densely-rusty one.
    assert hits[0].snippet.count("rust") >= 2


def test_recency_tiebreak(tmp_journal):
    """Equal hit counts: the newer (higher seq_id) event wins."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("first ledger entry")
    s.send("second ledger entry")
    eng = RecallEngine(tmp_journal, mode='substring')
    hits = eng.search("ledger")
    assert hits[0].seq_id > hits[1].seq_id


# ── Output formatting ────────────────────────────────────────────────────


def test_format_matches_no_results():
    assert "no matches" in format_matches([], query="x")


def test_format_matches_renders_each_hit(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust borrow checker explainer")
    eng = RecallEngine(tmp_journal, mode='substring')
    out = format_matches(eng.search("rust"), query="rust")
    assert "seq=" in out
    assert "user_prompt" in out
    assert "rust" in out.lower()


# ── CLI integration ──────────────────────────────────────────────────────


def test_cli_recall_finds_prior_turn(tmp_journal, monkeypatch, capsys):
    """End-to-end: a chat turn, then a /recall on the same session."""
    stdin = io.StringIO(
        "rust is fun\n"
        "/recall rust\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[recall" in out  # matches both "[recall]" and "[recall · <mode>]"
    assert "match" in out
    assert "rust is fun" in out


def test_cli_recall_no_match_reports_cleanly(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "hello there\n"
        # Force substring mode — semantic recall may find a weak cosine
        # hit on any English query, which is correct behaviour but
        # incompatible with a "no matches" assertion. The semantic path
        # is tested separately in test_semantic_recall.py.
        "/recall --mode substring absolutely_not_in_the_chat\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[recall] no matches" in out


def test_cli_help_command(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO("/help\n/quit\n")
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/recall" in out
    assert "/help" in out


def test_cli_unknown_slash_command(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO("/nonsense\n/quit\n")
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unknown command" in out
    assert "/help" in out


def test_cli_recall_kind_filter_runs(tmp_journal, monkeypatch, capsys):
    """/recall --kind ... should parse and run without error."""
    stdin = io.StringIO(
        "tell me about rust\n"
        "/recall --kind llm_inference rust\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[recall" in out  # matches both "[recall]" and "[recall · <mode>]"

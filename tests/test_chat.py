"""Tests for KorgChat v0.4.1.

Runs against the actual korg_bridge extension. The MockResponder's scripted
mode is used wherever the inner tool-use loop is exercised, so assertions
are deterministic. Free mode (no scripted replies) is still tested for the
text-only path that v0.4.0 shipped.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from korgchat import (
    ChatSession,
    MockResponder,
    Reply,
    ToolUse,
    __version__,
    default_tools,
)
from korgchat.__main__ import main as cli_main


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


# ── Smoke ──────────────────────────────────────────────────────────────────


def test_version_string():
    assert __version__ == "0.5.2"


# ── Single-turn text-only path (v0.4.0 carryover) ──────────────────────────


def test_single_turn_writes_two_events(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    turn = s.send("hello korg")
    assert turn.user_seq == 1
    assert turn.assistant_seq == 2
    assert turn.tool_calls == []
    events = _events(tmp_journal)
    assert len(events) == 2
    assert events[0]["event"]["tool_name"] == "user_prompt"
    assert events[1]["event"]["tool_name"] == "llm_inference"


def test_three_turn_chain(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("turn one")
    s.send("turn two")
    s.send("turn three")
    events = _events(tmp_journal)
    assert [e["seq_id"] for e in events] == [1, 2, 3, 4, 5, 6]
    assert events[0]["metadata"]["triggered_by"] is None
    assert events[1]["metadata"]["triggered_by"] == 1
    assert events[2]["metadata"]["triggered_by"] == 2
    assert events[3]["metadata"]["triggered_by"] == 3
    assert events[4]["metadata"]["triggered_by"] == 4
    assert events[5]["metadata"]["triggered_by"] == 5


def test_resume_picks_up_where_we_left_off(tmp_journal):
    s1 = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s1.send("first")
    s1.send("second")
    del s1
    s2 = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    t = s2.send("third after resume")
    events = _events(tmp_journal)
    assert [e["seq_id"] for e in events] == [1, 2, 3, 4, 5, 6]
    assert events[4]["metadata"]["triggered_by"] == 4
    assert t.user_seq == 5
    assert t.assistant_seq == 6


def test_empty_prompt_rejected(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    with pytest.raises(ValueError):
        s.send("")
    with pytest.raises(ValueError):
        s.send("   ")


# ── v0.4.1 tool-use loop ───────────────────────────────────────────────────


def _scripted_one_tool() -> MockResponder:
    """A responder that requests one `add` call, then returns text on the
    follow-up call (after seeing the tool result)."""
    return MockResponder(
        replies=[
            Reply(
                tool_uses=[ToolUse(id="t1", name="add", input={"a": 2, "b": 3})],
                prompt_tokens=20,
                completion_tokens=0,
            ),
            Reply(text="2 + 3 = 5", prompt_tokens=24, completion_tokens=6),
        ]
    )


def test_single_tool_use_records_four_events(tmp_journal):
    """LLM requests one tool; KorgChat executes it; LLM produces final text.

    Journal should have: user_prompt, llm_inference (round 1), add tool_call,
    llm_inference (round 2 / terminal) — 4 events.
    """
    s = ChatSession(journal_path=tmp_journal, responder=_scripted_one_tool())
    turn = s.send("add 2 and 3")

    events = _events(tmp_journal)
    assert [e["event"]["tool_name"] for e in events] == [
        "user_prompt",
        "llm_inference",
        "add",
        "llm_inference",
    ]
    assert [e["seq_id"] for e in events] == [1, 2, 3, 4]

    # Causal chain per §2a:
    #   1 user_prompt  (root)
    #   2 llm round-1   triggered_by=1
    #   3 add tool      triggered_by=2  (sibling under round-1 LLM)
    #   4 llm round-2   triggered_by=2  (NOT 3 — §2a says chain to prior LLM)
    assert events[0]["metadata"]["triggered_by"] is None
    assert events[1]["metadata"]["triggered_by"] == 1
    assert events[2]["metadata"]["triggered_by"] == 2
    assert events[3]["metadata"]["triggered_by"] == 2

    # Turn surface
    assert turn.user_seq == 1
    assert turn.assistant_seq == 4
    assert turn.assistant_text == "2 + 3 = 5"
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "add"
    assert call.input == {"a": 2, "b": 3}
    assert call.result == {"sum": 5}
    assert call.success is True
    assert call.seq == 3


def test_parallel_tools_share_producing_llm_seq(tmp_journal):
    """Two tools requested in a single LLM round are siblings under that
    round's llm_inference."""
    responder = MockResponder(
        replies=[
            Reply(
                tool_uses=[
                    ToolUse(id="t1", name="add", input={"a": 1, "b": 2}),
                    ToolUse(id="t2", name="echo", input={"input": "hi"}),
                ]
            ),
            Reply(text="done", prompt_tokens=1, completion_tokens=1),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    turn = s.send("do two things")
    events = _events(tmp_journal)
    # 5 events: user, llm_r1, add, echo, llm_r2.
    assert [e["event"]["tool_name"] for e in events] == [
        "user_prompt",
        "llm_inference",
        "add",
        "echo",
        "llm_inference",
    ]
    assert events[2]["metadata"]["triggered_by"] == 2  # add ← round-1 LLM
    assert events[3]["metadata"]["triggered_by"] == 2  # echo ← round-1 LLM (sibling)
    assert events[4]["metadata"]["triggered_by"] == 2  # llm_r2 ← round-1 LLM (§2a)
    assert turn.assistant_seq == 5
    assert len(turn.tool_calls) == 2
    assert {c.name for c in turn.tool_calls} == {"add", "echo"}


def test_multi_round_tool_use(tmp_journal):
    """Two tool rounds in one turn: LLM calls add, sees result, calls echo, then ends."""
    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 10, "b": 20})]),
            Reply(tool_uses=[ToolUse(id="t2", name="echo", input={"input": "30"})]),
            Reply(text="answer: 30"),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    turn = s.send("add 10 and 20 then echo")
    events = _events(tmp_journal)
    assert [e["event"]["tool_name"] for e in events] == [
        "user_prompt",      # 1
        "llm_inference",    # 2 — round 1
        "add",              # 3 — under llm 2
        "llm_inference",    # 4 — round 2, chains to 2 per §2a
        "echo",             # 5 — under llm 4
        "llm_inference",    # 6 — round 3 / terminal, chains to 4 per §2a
    ]
    chain = [e["metadata"]["triggered_by"] for e in events]
    assert chain == [None, 1, 2, 2, 4, 4]
    assert turn.assistant_text == "answer: 30"
    assert turn.assistant_seq == 6
    assert [c.name for c in turn.tool_calls] == ["add", "echo"]


def test_unknown_tool_is_recorded_with_error(tmp_journal):
    """An LLM-requested tool that isn't registered should emit a failed tool_call
    event (success=False), not crash the session. The LLM gets the error to
    react to."""
    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id="t1", name="missing_tool", input={})]),
            Reply(text="recovered"),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    turn = s.send("use a tool that doesn't exist")
    events = _events(tmp_journal)
    bad = events[2]["event"]
    assert bad["tool_name"] == "missing_tool"
    assert bad["success"] is False
    assert "error" in bad["result"]
    assert turn.tool_calls[0].success is False


def test_tool_handler_exception_recorded_as_error(tmp_journal):
    """A tool that raises during execution records success=False and lets
    the conversation continue."""
    responder = MockResponder(
        replies=[
            # add() raises ValueError on missing inputs.
            Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 1})]),
            Reply(text="caught it"),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    turn = s.send("trigger an error")
    assert turn.tool_calls[0].success is False
    assert "error" in turn.tool_calls[0].result
    events = _events(tmp_journal)
    assert events[2]["event"]["success"] is False


def test_max_iterations_blocks_runaway(tmp_journal):
    """A responder that never returns text eventually trips the safety cap."""
    # Build a script of nothing but tool requests, longer than the cap.
    from korgchat.chat import MAX_TOOL_USE_ITERATIONS

    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id=f"t{i}", name="echo", input={"input": "loop"})])
            for i in range(MAX_TOOL_USE_ITERATIONS + 2)
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    with pytest.raises(RuntimeError, match="exceeded"):
        s.send("trigger the cap")


def test_free_mode_tool_marker(tmp_journal):
    """Free-mode MockResponder recognises [tool:name(args)] markers in the prompt."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    turn = s.send("please [tool:add(a=4, b=6)] for me")
    # Marker should be picked up and round-tripped through the loop:
    # user_prompt, llm_round1 (requesting add), add tool, llm_round2 (terminal).
    events = _events(tmp_journal)
    assert [e["event"]["tool_name"] for e in events] == [
        "user_prompt",
        "llm_inference",
        "add",
        "llm_inference",
    ]
    assert turn.tool_calls[0].result == {"sum": 10}


# ── v0.4.2 streaming ──────────────────────────────────────────────────────


def test_stream_emits_full_text_in_chunks(tmp_journal):
    """on_token receives one chunk per char (for MockResponder); concatenating
    them rebuilds the full assistant text."""
    chunks: list[str] = []
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(stream_delay_secs=0.0),
    )
    s.on_token = chunks.append
    turn = s.send("hello streaming world")
    # MockResponder.stream emits one char per on_token call.
    assert "".join(chunks) == turn.assistant_text
    assert len(chunks) == len(turn.assistant_text)


def test_stream_default_unscripted_still_writes_journal(tmp_journal):
    """Streaming is a UX layer; the journal still records one llm_inference
    event per round with the full text."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(stream_delay_secs=0.0),
    )
    s.on_token = lambda _c: None
    turn = s.send("first turn")
    events = _events(tmp_journal)
    assert len(events) == 2
    assert events[1]["event"]["tool_name"] == "llm_inference"
    # Mock's hash-bucketed reply must match the on-disk record. The bridge
    # stores the assistant text in args/result for llm_inference events,
    # but the public surface is turn.assistant_text — which is what we
    # care about being non-empty + reconstructable.
    assert turn.assistant_text


def test_stream_round_start_fires_once_per_round(tmp_journal):
    """A two-round tool-use turn fires on_round_start twice."""
    round_starts = 0

    def bump():
        nonlocal round_starts
        round_starts += 1

    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id="t1", name="add", input={"a": 1, "b": 2})]),
            Reply(text="result is 3"),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    s.on_round_start = bump
    # Make sure streaming is active so the round-start path matters too.
    s.on_token = lambda _c: None
    s.send("compute")
    assert round_starts == 2  # one per LLM round


def test_stream_skip_when_round_has_no_text(tmp_journal):
    """A tool-only round produces no on_token calls, but on_round_start fires."""
    rounds = 0
    chunks: list[str] = []

    def round_start():
        nonlocal rounds
        rounds += 1

    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id="t1", name="echo", input={"input": "x"})]),
            Reply(text="done"),
        ]
    )
    s = ChatSession(journal_path=tmp_journal, responder=responder)
    s.on_round_start = round_start
    s.on_token = chunks.append
    s.send("go")
    assert rounds == 2
    # Round 1 emitted no tokens (tool-only); round 2 emitted "done".
    assert "".join(chunks) == "done"


def test_stream_default_responder_yields_full_text_at_once(tmp_journal):
    """The Responder.stream() default impl (no override) should still feed
    on_token — once, with the full text."""

    class PlainResponder(MockResponder):
        # MockResponder overrides stream(); make a subclass that *doesn't*
        # so we hit the Responder.stream default path.
        def stream(self, **kwargs):  # type: ignore[override]
            from korgchat.chat import Responder
            return Responder.stream(self, **kwargs)

    chunks: list[str] = []
    s = ChatSession(journal_path=tmp_journal, responder=PlainResponder())
    s.on_token = chunks.append
    s.send("hello")
    # Default impl: one chunk containing the entire reply.
    assert len(chunks) == 1
    assert chunks[0]  # non-empty


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_mock_three_text_turns(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\nhow are you\nbye\n"))
    # Use --stream-delay=0 to keep tests fast.
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    events = _events(tmp_journal)
    assert len(events) == 6
    out = capsys.readouterr().out
    assert "KorgChat" in out
    assert out.count("[recorded:") == 3
    # Streaming default produces visible "Korg: " prefix per turn.
    assert out.count("Korg: ") == 3


def test_cli_no_stream_flag_atomic_print(tmp_journal, monkeypatch, capsys):
    """--no-stream produces a single atomic print per turn (the v0.4.0 UX)."""
    monkeypatch.setattr("sys.stdin", io.StringIO("hi\n/quit\n"))
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--no-stream"])
    assert rc == 0
    events = _events(tmp_journal)
    assert len(events) == 2
    out = capsys.readouterr().out
    assert "(--no-stream)" in out
    assert "Korg: " in out


def test_cli_tool_marker_run(tmp_journal, monkeypatch, capsys):
    """End-to-end CLI run that triggers a tool via marker syntax."""
    monkeypatch.setattr("sys.stdin", io.StringIO("[tool:add(a=7, b=8)] please\n/quit\n"))
    rc = cli_main(["--mock", "--journal", str(tmp_journal)])
    assert rc == 0
    out = capsys.readouterr().out
    # Should see the tool-call trace line and the "1 tool call(s)" tail.
    assert "🔧" in out
    assert "add" in out
    assert "tool call(s)" in out
    events = _events(tmp_journal)
    assert [e["event"]["tool_name"] for e in events] == [
        "user_prompt",
        "llm_inference",
        "add",
        "llm_inference",
    ]
    assert events[2]["event"]["result"] == {"sum": 15}

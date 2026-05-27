"""Structured dry-run of the AnthropicResponder code path.

Verifies every shape and protocol decision short of the actual network call.
The anthropic SDK is installed and imported normally; only the `Anthropic()`
client is patched so `client.messages.create` / `client.messages.stream`
return controlled fakes.

This catches the class of bug where AnthropicResponder builds wrong-shape
requests or mis-parses response blocks — bugs that would otherwise only
show up when a paid API key is set and a real call is made. With the
mocks in place, the same code path executes deterministically in ~1ms.

If/when ANTHROPIC_API_KEY is available, the same call sites can be
exercised against the real API by removing the patches — these tests
serve as the contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import anthropic  # ensure the real module loads — we patch its surface below

from korgchat import (
    AnthropicResponder,
    ChatSession,
    Reply,
    Tool,
    ToolRegistry,
    ToolUse,
)


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


# ── Test fixtures: fake Message / ContentBlock objects ───────────────────


def _text_block(text: str) -> SimpleNamespace:
    """Mimic anthropic.types.TextBlock with the fields our code reads."""
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    """Mimic anthropic.types.ToolUseBlock."""
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _fake_message(*, text_blocks=(), tool_blocks=(), in_tokens=10, out_tokens=20):
    """Mimic a complete anthropic.types.Message."""
    return SimpleNamespace(
        content=[*text_blocks, *tool_blocks],
        usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
    )


class _FakeStream:
    """Mimic the anthropic streaming context manager.

    Iterating `.text_stream` yields each text chunk; `.get_final_message()`
    returns the assembled fake Message after the iteration completes.
    """

    def __init__(self, text_chunks: list[str], tool_blocks=()):
        self._text_chunks = text_chunks
        self._tool_blocks = tool_blocks
        self._full_text = "".join(text_chunks)

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    @property
    def text_stream(self):
        for c in self._text_chunks:
            yield c

    def get_final_message(self):
        return _fake_message(
            text_blocks=[_text_block(self._full_text)] if self._full_text else [],
            tool_blocks=list(self._tool_blocks),
        )


# ── _parse_response: response → Reply ────────────────────────────────────


def test_parse_response_text_only():
    msg = _fake_message(text_blocks=[_text_block("hello world")])
    r = AnthropicResponder._parse_response(msg)
    assert r.text == "hello world"
    assert r.tool_uses == []
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 20


def test_parse_response_tool_use_only():
    msg = _fake_message(
        tool_blocks=[_tool_use_block("toolu_X", "add", {"a": 2, "b": 3})]
    )
    r = AnthropicResponder._parse_response(msg)
    assert r.text == ""
    assert len(r.tool_uses) == 1
    tu = r.tool_uses[0]
    assert tu.id == "toolu_X"
    assert tu.name == "add"
    assert tu.input == {"a": 2, "b": 3}


def test_parse_response_text_then_tool():
    """Anthropic can interleave text with tool_use within one assistant message."""
    msg = _fake_message(
        text_blocks=[_text_block("Let me check.")],
        tool_blocks=[_tool_use_block("toolu_X", "echo", {"input": "x"})],
    )
    r = AnthropicResponder._parse_response(msg)
    assert r.text == "Let me check."
    assert len(r.tool_uses) == 1


def test_parse_response_concats_multiple_text_blocks():
    """Multiple TextBlock blocks in one Message → single text string."""
    msg = _fake_message(text_blocks=[
        _text_block("part one "),
        _text_block("part two"),
    ])
    r = AnthropicResponder._parse_response(msg)
    assert r.text == "part one part two"


def test_parse_response_missing_usage():
    """Older SDK responses may not have .usage; defaults to 0."""
    msg = SimpleNamespace(content=[_text_block("hi")])  # no usage
    r = AnthropicResponder._parse_response(msg)
    assert r.text == "hi"
    assert r.prompt_tokens == 0
    assert r.completion_tokens == 0


# ── _parse_final_message reuses _parse_response ──────────────────────────


def test_parse_final_message_same_as_parse_response():
    msg = _fake_message(text_blocks=[_text_block("final")])
    a = AnthropicResponder._parse_response(msg)
    b = AnthropicResponder._parse_final_message(msg)
    assert a == b


# ── Construction guards ──────────────────────────────────────────────────


def test_anthropic_responder_model_attr():
    r = AnthropicResponder(model="claude-opus-4-7", max_tokens=512)
    assert r.model == "claude-opus-4-7"
    assert r._max_tokens == 512


def test_anthropic_responder_raises_when_sdk_missing(monkeypatch):
    """If anthropic isn't importable, construction must produce a clear error
    pointing at --mock — not a bare ImportError on a deep stack frame."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("simulated: anthropic not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="--mock"):
        AnthropicResponder()


# ── Stateless respond() — message construction shape ───────────────────


def test_respond_builds_simple_user_message():
    r = AnthropicResponder()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_message(
        text_blocks=[_text_block("ok")]
    )

    with patch("anthropic.Anthropic", return_value=fake_client):
        reply = r.respond(history=[], prompt="hello", prior_tool_results=None, tools=None)

    assert reply.text == "ok"
    call = fake_client.messages.create.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == 2048
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    # No tools registry → `tools` key not added.
    assert "tools" not in kwargs


def test_respond_includes_tools_when_registry_non_empty():
    r = AnthropicResponder()
    reg = ToolRegistry([
        Tool(
            name="search",
            description="grep the journal",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            handler=lambda _a: {"ok": True},
        ),
    ])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_message(text_blocks=[_text_block("done")])

    with patch("anthropic.Anthropic", return_value=fake_client):
        r.respond(history=[], prompt="find ledger entries", prior_tool_results=None, tools=reg)

    kwargs = fake_client.messages.create.call_args.kwargs
    assert "tools" in kwargs
    assert kwargs["tools"][0]["name"] == "search"
    assert kwargs["tools"][0]["description"] == "grep the journal"
    assert kwargs["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"


def test_respond_serialises_history_as_user_assistant_pairs(tmp_journal):
    """Prior turns should appear as alternating user/assistant messages.

    The production code passes _anthropic_buf into messages.create() by
    reference, then mutates it after the call returns (appending the
    assistant reply for the next turn). MagicMock records args by reference
    too, so a vanilla `call_args.kwargs["messages"]` reads the post-mutation
    state. We snapshot at call time via side_effect to assert on what the
    API actually saw.
    """
    captured: list[list[dict]] = []

    def capture(**kwargs):
        # Deep-snapshot the messages — content can be a list of dicts that
        # ChatSession would mutate (tool_use blocks), so json round-trip
        # is the safest copy.
        captured.append(json.loads(json.dumps(kwargs["messages"])))
        return _fake_message(text_blocks=[_text_block("reply")])

    s = ChatSession(journal_path=tmp_journal, responder=AnthropicResponder())
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = capture

    with patch("anthropic.Anthropic", return_value=fake_client):
        s.send("hello number one")
        s.send("hello number two")

    # Two API calls — first with just the new prompt, second with prior
    # turn's history + the new prompt.
    assert len(captured) == 2
    assert captured[0] == [{"role": "user", "content": "hello number one"}]
    assert captured[1] == [
        {"role": "user", "content": "hello number one"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "hello number two"},
    ]


# ── ChatSession._anthropic_continue (non-streaming inner loop) ──────────


def test_anthropic_tool_use_round_trip(tmp_journal):
    """A single tool-use round end-to-end: the inner loop opens the API
    twice (once for tool request, once for terminal text) and serialises
    tool_result blocks on the user message between them."""
    s = ChatSession(journal_path=tmp_journal, responder=AnthropicResponder())

    fake_client = MagicMock()
    # First call → tool_use; second call → terminal text.
    fake_client.messages.create.side_effect = [
        _fake_message(
            tool_blocks=[_tool_use_block("toolu_AAA", "add", {"a": 1, "b": 2})]
        ),
        _fake_message(text_blocks=[_text_block("the answer is 3")]),
    ]

    with patch("anthropic.Anthropic", return_value=fake_client):
        turn = s.send("compute 1+2")

    # 4 journal events: user_prompt, llm_inference(round1), add tool, llm_inference(terminal)
    with tmp_journal.open() as f:
        events = json.load(f)
    assert len(events) == 4
    names = [e["event"]["tool_name"] for e in events]
    assert names == ["user_prompt", "llm_inference", "add", "llm_inference"]
    assert turn.assistant_text == "the answer is 3"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "add"
    assert turn.tool_calls[0].result == {"sum": 3}

    # Inspect the second API call's messages list: must contain the
    # tool_use block from the assistant + the tool_result block from the user.
    second_call = fake_client.messages.create.call_args_list[1]
    messages = second_call.kwargs["messages"]
    # [user prompt, assistant tool_use, user tool_result]
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "toolu_AAA"
        for b in messages[1]["content"]
    )
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_AAA"
    # The tool_result content is JSON of the executed tool's output.
    assert json.loads(messages[2]["content"][0]["content"]) == {"sum": 3}


# ── ChatSession._anthropic_continue_stream (streaming inner loop) ────────


def test_anthropic_streaming_path_yields_chunks_and_records_event(tmp_journal):
    """Streaming path: text_stream chunks fire on_token, final message
    parses into a Reply, and one llm_inference event is recorded with the
    full text."""
    s = ChatSession(journal_path=tmp_journal, responder=AnthropicResponder())
    chunks: list[str] = []
    s.on_token = chunks.append

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeStream(
        text_chunks=["hello", " streaming", " world"]
    )

    with patch("anthropic.Anthropic", return_value=fake_client):
        turn = s.send("stream me")

    assert chunks == ["hello", " streaming", " world"]
    assert turn.assistant_text == "hello streaming world"

    with tmp_journal.open() as f:
        events = json.load(f)
    # 2 events: user_prompt + llm_inference (carrying the full text).
    assert len(events) == 2
    assert events[1]["event"]["tool_name"] == "llm_inference"
    assert events[1]["event"]["result"].get("text") == "hello streaming world"


def test_anthropic_streaming_passes_tools_to_api(tmp_journal):
    reg = ToolRegistry([
        Tool(name="echo",
             description="copy input back",
             input_schema={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
             handler=lambda a: {"echoed": a["input"]}),
    ])
    s = ChatSession(
        journal_path=tmp_journal,
        responder=AnthropicResponder(),
        tools=reg,
    )
    s.on_token = lambda _c: None

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = _FakeStream(text_chunks=["done"])

    with patch("anthropic.Anthropic", return_value=fake_client):
        s.send("hi")

    kwargs = fake_client.messages.stream.call_args.kwargs
    assert "tools" in kwargs
    assert kwargs["tools"][0]["name"] == "echo"

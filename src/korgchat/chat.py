"""Core chat session that records every turn into the Korg ledger.

v0.4.1 adds tool use: the LLM can request tool invocations, KorgChat
executes them, and feeds the results back for the LLM's next thought.

Causal chain for a multi-step turn (one user input, N tool calls, final text):

    seq=U   user_prompt          triggered_by=<prior_llm or None>
    seq=L1  llm_inference        triggered_by=U
    seq=T1  tool_call(name=add)  triggered_by=L1
    seq=T2  tool_call(name=echo) triggered_by=L1  ← sibling of T1
    seq=L2  llm_inference        triggered_by=L1  ← per spec §2a: LLM chains
                                                    to prior LLM, NOT to T2
    seq=T3  tool_call            triggered_by=L2
    seq=L3  llm_inference        triggered_by=L2  (terminal text)

The next turn's user_prompt chains to L3 (the last `llm_inference` of the
prior turn) — matching the multi-turn semantics from v0.4.0.

Per-turn safety cap: MAX_TOOL_USE_ITERATIONS prevents infinite loops if a
responder keeps requesting tools without ever returning text.
"""

from __future__ import annotations

import abc
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import korg_bridge

from korgchat.branches import MAIN_BRANCH, BranchStore
from korgchat.tools import ToolRegistry, default_tools

# Auto-context import is lazy inside _build_auto_context() — pulls in
# the embeddings stack (fastembed when available) which we don't want
# to load until the session actually uses it.


# ── Data shapes ────────────────────────────────────────────────────────────


@dataclass
class ToolUse:
    """A request from the LLM to invoke a tool."""

    id: str  # unique within this turn — used to pair with ToolResult
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a ToolUse, ready to feed back to the LLM."""

    id: str  # matches the ToolUse.id this is responding to
    output: Any  # JSON-serialisable
    is_error: bool = False


@dataclass
class Reply:
    """One response from a Responder.respond() call.

    Either `text` is non-empty (terminal — turn is done) OR `tool_uses` is
    non-empty (we have tools to execute, then call respond() again with the
    results in `tool_results`). Both at once is allowed too — Anthropic
    sometimes interleaves text with tool calls.
    """

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_uses)


@dataclass
class ToolCall:
    """A tool call that was actually executed during a turn.

    Carries the journal seq_id so callers (CLI, tests) can correlate with the
    on-disk event.
    """

    seq: int
    name: str
    input: dict[str, Any]
    result: Any
    success: bool
    duration_ms: int


@dataclass
class Turn:
    """A complete user→[LLM+tools]*→final-text exchange, with seq_ids
    pointing at the journal events written during the turn."""

    user_seq: int
    user_prompt: str
    assistant_seq: int  # seq of the terminal llm_inference that produced text
    assistant_text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    duration_ms: int = 0


# Safety cap. A model that keeps requesting tools without ever returning
# text — either buggy or stuck in a thought loop — gets terminated with a
# clear error rather than spinning forever on the user's prompt.
MAX_TOOL_USE_ITERATIONS = 8


# ── Responders ─────────────────────────────────────────────────────────────


class Responder(abc.ABC):
    """A pluggable backend that turns a chat history + tool results into a Reply."""

    @property
    def model(self) -> str:  # informational — recorded on the llm_inference event
        return self.__class__.__name__

    @abc.abstractmethod
    def respond(
        self,
        *,
        history: list[Turn],
        prompt: str,
        prior_tool_results: list[ToolResult] | None = None,
        tools: ToolRegistry | None = None,
    ) -> Reply:
        """Produce the next Reply.

        On the first call within a turn, `prompt` is the user's new input and
        `prior_tool_results` is empty.

        On subsequent calls within the same turn (the inner tool-use loop),
        `prompt` is still the original user input, but `prior_tool_results`
        carries the results of every tool call from the previous Reply.

        Returning a Reply with `wants_tools=False` ends the turn.
        """

    def stream(
        self,
        *,
        history: list[Turn],
        prompt: str,
        prior_tool_results: list[ToolResult] | None = None,
        tools: ToolRegistry | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Reply:
        """Streaming variant. Calls `on_token(chunk)` as text becomes
        available, returns the same Reply shape as `respond()` at the end.

        Default implementation = block on respond(), then emit the full text
        as a single chunk. Subclasses with real streaming protocols
        (Anthropic, MockResponder with delay) override this for incremental
        delivery.

        The journal still records one atomic `llm_inference` event per call
        with the full text — streaming is a CLI/UX feature only.
        """
        reply = self.respond(
            history=history,
            prompt=prompt,
            prior_tool_results=prior_tool_results,
            tools=tools,
        )
        if on_token is not None and reply.text:
            on_token(reply.text)
        return reply


class MockResponder(Responder):
    """Deterministic responder for offline use, CI, and tests.

    Two modes:

    1. **Scripted mode** (constructor receives `replies=[Reply(...), ...]`):
       successive calls to `respond()` pop replies off the queue in order.
       Used in tests to make tool-use loops deterministic.

    2. **Free mode** (no scripted replies): picks a canned text reply based
       on a hash of the prompt. Also recognises a literal marker syntax in
       the prompt — `[tool:add(a=2, b=3)]` — and returns a Reply requesting
       that tool. After all tools execute, falls back to a canned reply.
    """

    REPLIES = [
        "pages turn forward / each entry signed by the past / time leaves no escape",
        "merkle roots whisper / consensus a slow drumline / blockchain heart beats on",
        "korg sees the moves / every tool call signed and chained / nothing left to chance",
        "rewind is a verb / undo lives inside the log / past becomes present",
        "between input lines / there is the silent ledger / quietly counting",
    ]

    _TOOL_MARKER_RE = re.compile(r"\[tool:([a-zA-Z_][a-zA-Z0-9_]*)\(([^\]]*?)\)\]")

    def __init__(
        self,
        replies: list[Reply] | None = None,
        *,
        stream_delay_secs: float = 0.0,
    ) -> None:
        self._scripted = list(replies) if replies is not None else None
        # Pause between simulated token chunks during stream(). 0 keeps tests
        # fast and deterministic; CLI runs default to a small visible delay so
        # the user actually sees a streaming effect with the mock responder.
        self._stream_delay_secs = stream_delay_secs

    @property
    def model(self) -> str:
        return "mock:deterministic"

    def respond(
        self,
        *,
        history: list[Turn],
        prompt: str,
        prior_tool_results: list[ToolResult] | None = None,
        tools: ToolRegistry | None = None,
    ) -> Reply:
        if self._scripted is not None:
            if not self._scripted:
                # Failsafe: scripted queue ran out — return a canned text reply
                # so the test sees a deterministic terminal state rather than
                # an exception.
                return Reply(text="(mock script exhausted)", prompt_tokens=0, completion_tokens=0)
            return self._scripted.pop(0)

        # Free mode. If we're being called with tool results, the previous
        # respond() requested tools; now we just need to produce text.
        if prior_tool_results:
            text = self._summarize_tool_results(prior_tool_results)
            return Reply(
                text=text,
                prompt_tokens=len(re.findall(r"\S+", prompt)),
                completion_tokens=len(re.findall(r"\S+", text)),
            )

        # v0.5.1: special-case /summarize prompts so the mock CLI experience
        # produces something that *looks* like a digest instead of a haiku.
        # Real (Anthropic) responders see the same marker as harmless
        # preamble and produce a real summary.
        from korgchat.summary import SUMMARY_PROMPT_MARKER

        if SUMMARY_PROMPT_MARKER in prompt:
            text = self._mock_summary(prompt)
            return Reply(
                text=text,
                prompt_tokens=len(re.findall(r"\S+", prompt)),
                completion_tokens=len(re.findall(r"\S+", text)),
            )

        # Look for tool markers in the prompt: "[tool:add(a=2, b=3)]"
        tool_uses = []
        for m in self._TOOL_MARKER_RE.finditer(prompt):
            tool_name = m.group(1)
            input_str = m.group(2)
            try:
                input_dict = self._parse_kv(input_str)
            except ValueError:
                continue
            tool_uses.append(
                ToolUse(id=f"toolu_{uuid.uuid4().hex[:12]}", name=tool_name, input=input_dict)
            )

        if tool_uses:
            return Reply(
                tool_uses=tool_uses,
                prompt_tokens=len(re.findall(r"\S+", prompt)),
                completion_tokens=0,
            )

        # No tools requested → canned text reply.
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        text = self.REPLIES[h % len(self.REPLIES)]
        prompt_tokens = len(re.findall(r"\S+", prompt)) + sum(
            len(re.findall(r"\S+", t.user_prompt + " " + t.assistant_text))
            for t in history
        )
        completion_tokens = len(re.findall(r"\S+", text))
        return Reply(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @staticmethod
    def _parse_kv(s: str) -> dict[str, Any]:
        """Parse `a=2, b=3, name="foo"` into {"a":2,"b":3,"name":"foo"}.

        Numbers parse as int/float, quoted strings strip quotes, anything else
        stays as a raw string."""
        out: dict[str, Any] = {}
        for pair in re.split(r",\s*", s.strip()):
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"bad kv pair: {pair!r}")
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (
                v.startswith("'") and v.endswith("'")
            ):
                out[k] = v[1:-1]
            else:
                try:
                    out[k] = int(v)
                except ValueError:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
        return out

    @staticmethod
    def _mock_summary(prompt: str) -> str:
        """Produce a deterministic, structurally-honest "summary" for mock
        mode. We extract counts of event types from the prompt body so the
        digest reflects the actual scope, not a canned haiku."""
        body = prompt.split("=== EVENTS ===", 1)[1] if "=== EVENTS ===" in prompt else ""
        body = body.split("=== END EVENTS ===", 1)[0]

        users = body.count("] user:")
        assistants = body.count("] assistant:")
        # Tool lines look like `] tool foo(...) → [ok] ...`
        tools = body.count("] tool ")

        # Pull the scope label from the prompt for a more useful first line.
        scope = "the selected events"
        for line in prompt.splitlines():
            if line.startswith("Scope:"):
                scope = line[len("Scope:"):].strip().rstrip(".")
                break

        parts = [
            f"Summary of {scope}.",
            f"Saw {users} user prompt(s), {assistants} assistant reply(ies), "
            f"and {tools} tool invocation(s).",
        ]
        if users == 0 and assistants == 0 and tools == 0:
            parts.append("Nothing substantive happened in this scope.")
        else:
            parts.append(
                "Conversation flowed without notable interruptions or errors; "
                "no open threads detected by the mock summarizer."
            )
        return " ".join(parts)

    @staticmethod
    def _summarize_tool_results(results: list[ToolResult]) -> str:
        parts = []
        for r in results:
            if r.is_error:
                parts.append(f"{r.id}: ERROR ({r.output!r})")
            else:
                parts.append(f"{r.id} → {json.dumps(r.output, sort_keys=True)}")
        return " | ".join(parts)

    def stream(
        self,
        *,
        history: list[Turn],
        prompt: str,
        prior_tool_results: list[ToolResult] | None = None,
        tools: ToolRegistry | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Reply:
        """Simulate a token-by-token stream by emitting one character at a
        time. `stream_delay_secs` (constructor param) controls the gap
        between chunks — 0 for tests, ~0.005 for visible CLI streaming."""
        reply = self.respond(
            history=history,
            prompt=prompt,
            prior_tool_results=prior_tool_results,
            tools=tools,
        )
        if on_token is not None and reply.text:
            for ch in reply.text:
                on_token(ch)
                if self._stream_delay_secs > 0:
                    time.sleep(self._stream_delay_secs)
        return reply


class AnthropicResponder(Responder):
    """Live Anthropic-backed responder with tool-use support.

    Lazily imports the anthropic SDK so `korgchat --mock` doesn't need the
    package installed.
    """

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 2048):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install korgchat[anthropic]` "
                "or use --mock for offline mode."
            ) from e
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def respond(
        self,
        *,
        history: list[Turn],
        prompt: str,
        prior_tool_results: list[ToolResult] | None = None,
        tools: ToolRegistry | None = None,
    ) -> Reply:
        import anthropic

        client = anthropic.Anthropic()

        # Build the messages list from history.
        messages: list[dict[str, Any]] = []
        for t in history:
            messages.append({"role": "user", "content": t.user_prompt})
            messages.append({"role": "assistant", "content": t.assistant_text})

        if prior_tool_results is None:
            # First call this turn — user is sending a brand-new prompt.
            messages.append({"role": "user", "content": prompt})
        else:
            # Mid-turn continuation. We must have asked the API for tools on
            # a previous call; resend the original user prompt + the assistant
            # tool_use block + the user tool_result block. The caller is
            # responsible for keeping the conversation messages_state straight
            # — for v0.4.1 we keep that bookkeeping inside ChatSession by
            # passing the accumulated messages-so-far via prior_tool_results
            # encoded as a side channel. To keep this responder stateless we
            # rebuild from the chat-level dataclasses.
            #
            # Simplification used here: we re-send the user prompt followed by
            # a synthesised tool_result-only user message. This loses the
            # exact tool_use blocks from the prior assistant message, so
            # Anthropic will see "tool_result for an unknown tool_use_id"
            # and likely complain. For v0.4.1 we route this through
            # ChatSession's internal `_anthropic_messages` cache instead;
            # see the note in send().
            raise RuntimeError(
                "AnthropicResponder.respond(prior_tool_results=...) requires "
                "ChatSession's stateful path — wire via _continue_anthropic()."
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if tools is not None and len(tools) > 0:
            kwargs["tools"] = tools.to_anthropic_tools()

        resp = client.messages.create(**kwargs)
        return self._parse_response(resp)

    @staticmethod
    def _parse_response(resp: Any) -> Reply:
        """Convert an Anthropic Message into our Reply shape."""
        text_parts = []
        tool_uses = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append(
                    ToolUse(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input) if block.input else {},
                    )
                )
        usage = getattr(resp, "usage", None)
        return Reply(
            text="".join(text_parts),
            tool_uses=tool_uses,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )

    @staticmethod
    def _parse_final_message(final_msg: Any) -> Reply:
        """Same shape as _parse_response but for messages.stream()'s
        get_final_message(). The two API surfaces return very similar objects;
        we keep them as two adapters in case Anthropic drifts them later."""
        return AnthropicResponder._parse_response(final_msg)


# ── Session ────────────────────────────────────────────────────────────────


@dataclass
class ChatSession:
    """A multi-turn KorgChat conversation backed by korg_bridge.

    v0.4.1 wires a tool-use inner loop: a single .send() may write many
    journal events as the LLM and tools alternate.
    """

    journal_path: Path
    responder: Responder
    tools: ToolRegistry = field(default_factory=default_tools)
    source_agent: str = "agent:korgchat@0.5.3"
    on_tool_call: Callable[[ToolCall], None] | None = None
    # v0.4.2: streaming callbacks. When `on_token` is set, the session routes
    # each LLM round through Responder.stream() so the caller sees text chunks
    # as they arrive. `on_round_start` fires once per LLM round (before the
    # first token is requested) so a CLI can print a fresh "Korg: " prefix.
    on_token: Callable[[str], None] | None = None
    on_round_start: Callable[[], None] | None = None
    # v0.5.3: when True, send() runs a semantic /recall on every user
    # prompt and prepends a preamble of relevant prior events to the
    # responder request. The user_prompt event in the journal still
    # records the ORIGINAL prompt — auto-context lives in the LLM call,
    # not in the audit log.
    auto_context: bool = False
    # Fired with (preamble_text, match_count) whenever an auto-context
    # preamble is injected. CLI uses it to print a "🧠 [auto-context] …"
    # indicator. None → no preamble injected this turn.
    on_context_injected: Callable[[str, int], None] | None = None
    _bridge: korg_bridge.Bridge = field(init=False)
    _branches: BranchStore = field(init=False)
    # v0.5.0: which named branch is currently active. "main" = the implicit
    # trunk (its tip is the journal's latest seq); anything else is a
    # bookmark in `.korg/branches.json`. New turns chain triggered_by from
    # whichever branch's tip is current.
    current_branch: str = MAIN_BRANCH
    _history: list[Turn] = field(default_factory=list, init=False)
    _last_llm_seq: int | None = field(default=None, init=False)
    # Stateful Anthropic message buffer for mid-turn continuations.
    # Reset at the start of each send().
    _anthropic_buf: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.journal_path = Path(self.journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._bridge = korg_bridge.Bridge(str(self.journal_path))
        # Branches sidecar lives next to the journal so the two travel
        # together as a single conversation artifact.
        self._branches = BranchStore(self.journal_path.parent / "branches.json")
        self._last_llm_seq = self._resolve_branch_tip(self.current_branch)

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    @property
    def turns(self) -> int:
        return len(self._history)

    # ── Branch management (v0.5.0) ─────────────────────────────────────

    @property
    def branches(self) -> BranchStore:
        return self._branches

    def _resolve_branch_tip(self, name: str) -> int | None:
        """Return the seq_id new turns should chain triggered_by from when
        operating on this branch. Main = journal's latest overall seq."""
        if name == MAIN_BRANCH:
            return self._bridge.last_seq_id() or None
        return self._branches.get(name).tip_seq

    def fork_here(self, name: str) -> None:
        """Create a new branch bookmark at the current `_last_llm_seq` and
        switch to it. Empty-journal fork is rejected (nothing to fork from)."""
        fork_seq = self._last_llm_seq
        if fork_seq is None or fork_seq == 0:
            raise ValueError(
                "cannot fork: no events on this conversation yet. Send at "
                "least one message before forking."
            )
        self._branches.create(name, fork_seq=fork_seq)
        self.current_branch = name
        # Clear in-memory history when switching branches — a fresh REPL
        # context per branch keeps prompts focused and avoids leaking the
        # other branch's tail into prompts.
        self._history = []
        self._anthropic_buf = []

    def checkout(self, name: str) -> int | None:
        """Switch the active branch. New turns chain from the named branch's
        tip (or the journal's latest overall seq for `main`)."""
        if name != MAIN_BRANCH and name not in self._branches:
            raise KeyError(f"branch {name!r} does not exist")
        self.current_branch = name
        self._last_llm_seq = self._resolve_branch_tip(name)
        self._history = []
        self._anthropic_buf = []
        return self._last_llm_seq

    def send(self, prompt: str) -> Turn:
        """Send a prompt; loop through any tool-use rounds; return the Turn."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        turn_start = time.monotonic()

        # 1. Record the user_prompt.
        if self._last_llm_seq is None:
            user_seq = self._bridge.record_user_prompt(prompt)
        else:
            user_seq = self._bridge.record_tool_call(
                source_agent="human:korgchat-user",
                tool_name="user_prompt",
                args={"prompt": prompt},
                result={"success": True},
                success=True,
                duration_ms=0,
                triggered_by=self._last_llm_seq,
            )

        # v0.5.3: optionally augment the responder's view of the prompt
        # with relevant prior conversation, picked by semantic /recall.
        # The journal already has the ORIGINAL prompt at `user_seq`; the
        # `effective_prompt` only affects what the LLM sees.
        effective_prompt = prompt
        if self.auto_context:
            preamble = self._build_auto_context(prompt, exclude_seq=user_seq)
            if preamble:
                effective_prompt = f"{preamble}\n\n— current user message —\n{prompt}"

        # Reset the per-turn Anthropic message buffer.
        self._anthropic_buf = self._build_history_messages()
        self._anthropic_buf.append({"role": "user", "content": effective_prompt})

        # 2. Inner loop: ask responder, execute any tools, repeat until text.
        tool_calls: list[ToolCall] = []
        # `producing_llm_seq` is the most recent llm_inference seq — its
        # children (tool_calls) are siblings under it, and the NEXT
        # llm_inference will chain back to it per spec §2a.
        producing_llm_seq: int | None = None
        # `prior_llm_seq` is what each new llm_inference will use for
        # triggered_by. For the first round this is `user_seq`; for round 2+
        # it's the *previous* llm_inference (§2a).
        prior_llm_seq: int = user_seq

        for iteration in range(MAX_TOOL_USE_ITERATIONS):
            if self.on_round_start is not None:
                self.on_round_start()
            # Pass the (potentially auto-context-augmented) prompt to the
            # responder; mid-loop tool continuations don't re-augment.
            reply = self._ask_responder(effective_prompt if iteration == 0 else prompt)

            t_inf = int((time.monotonic() - turn_start) * 1000) if iteration == 0 else 0
            # v0.4.3: pass assistant_text so /recall can search reply content.
            # Empty text (e.g. tool-only rounds) records None so the result
            # object stays minimal for those events.
            llm_seq = self._bridge.record_llm_call(
                model=self.responder.model,
                prompt_tokens=int(reply.prompt_tokens),
                completion_tokens=int(reply.completion_tokens),
                duration_ms=t_inf,
                triggered_by=prior_llm_seq,
                source_agent=self.source_agent,
                assistant_text=reply.text if reply.text else None,
            )
            producing_llm_seq = llm_seq

            if not reply.wants_tools:
                # Terminal: model returned text. Record the assistant message
                # in the message buffer so any future turn sees it as history.
                if reply.text:
                    self._anthropic_buf.append(
                        {"role": "assistant", "content": reply.text}
                    )
                duration_ms = int((time.monotonic() - turn_start) * 1000)
                turn = Turn(
                    user_seq=user_seq,
                    user_prompt=prompt,
                    assistant_seq=llm_seq,
                    assistant_text=reply.text,
                    tool_calls=tool_calls,
                    duration_ms=duration_ms,
                )
                self._history.append(turn)
                self._last_llm_seq = llm_seq
                # v0.5.0: when on a non-main branch, advance the branch's
                # tip so the next checkout resumes from the right place.
                # No-op for main (tip = journal latest, which already moved).
                if self.current_branch != MAIN_BRANCH:
                    self._branches.update_tip(self.current_branch, llm_seq)
                return turn

            # Tools requested. Push the assistant tool_use blocks onto the
            # message buffer so a follow-up Anthropic call sees them.
            self._anthropic_buf.append(
                {
                    "role": "assistant",
                    "content": [
                        # Text emitted before the tool calls (if any) goes first
                        # so Anthropic's expected ordering is preserved.
                        *([{"type": "text", "text": reply.text}] if reply.text else []),
                        *[
                            {
                                "type": "tool_use",
                                "id": tu.id,
                                "name": tu.name,
                                "input": tu.input,
                            }
                            for tu in reply.tool_uses
                        ],
                    ],
                }
            )

            # Execute each tool, record one tool_call event each, build
            # ToolResult list for the next responder call.
            results: list[ToolResult] = []
            for tu in reply.tool_uses:
                tool_seq, call = self._execute_tool(tu, producing_llm_seq=llm_seq)
                tool_calls.append(call)
                results.append(
                    ToolResult(id=tu.id, output=call.result, is_error=not call.success)
                )
                if self.on_tool_call is not None:
                    self.on_tool_call(call)

            # Push tool_result blocks onto the message buffer as a single user
            # message (Anthropic protocol).
            self._anthropic_buf.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.id,
                            "content": json.dumps(r.output, sort_keys=True),
                            "is_error": r.is_error,
                        }
                        for r in results
                    ],
                }
            )

            # Next iteration: llm_inference chains to *this* llm (§2a).
            prior_llm_seq = llm_seq
            # Stash results so the next _ask_responder pass can pick them up
            # in mock mode (Anthropic mode rebuilds from _anthropic_buf).
            self._pending_results = results

        # If we fall off the end, the model exceeded MAX_TOOL_USE_ITERATIONS.
        raise RuntimeError(
            f"tool-use loop exceeded {MAX_TOOL_USE_ITERATIONS} iterations "
            f"without a terminal text reply (last seq={producing_llm_seq})"
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _ask_responder(self, prompt: str) -> Reply:
        """Dispatch to the responder. Routes Anthropic mode through the
        stateful message buffer; mock/other modes get the simpler interface.

        When `on_token` is set on the session, takes the streaming code path
        so text chunks reach the caller as they're produced. The journal
        still gets one atomic llm_inference event per round."""
        pending = getattr(self, "_pending_results", None)
        streaming = self.on_token is not None
        try:
            if isinstance(self.responder, AnthropicResponder):
                if streaming:
                    return self._anthropic_continue_stream(prompt)
                return self._anthropic_continue(prompt)
            if streaming:
                return self.responder.stream(
                    history=self._history,
                    prompt=prompt,
                    prior_tool_results=pending,
                    tools=self.tools,
                    on_token=self.on_token,
                )
            return self.responder.respond(
                history=self._history,
                prompt=prompt,
                prior_tool_results=pending,
                tools=self.tools,
            )
        finally:
            self._pending_results = None

    def _anthropic_continue(self, prompt: str) -> Reply:
        """Call Anthropic with the current `_anthropic_buf` and parse Reply."""
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": self.responder.model,
            "max_tokens": getattr(self.responder, "_max_tokens", 2048),
            "messages": self._anthropic_buf,
        }
        if len(self.tools) > 0:
            kwargs["tools"] = self.tools.to_anthropic_tools()
        resp = client.messages.create(**kwargs)
        return AnthropicResponder._parse_response(resp)

    def _anthropic_continue_stream(self, prompt: str) -> Reply:
        """Streaming variant: call `client.messages.stream()` and fire
        `self.on_token` for every text_delta. The accumulated final message
        is parsed into the Reply we return so the rest of send() works
        unchanged.

        Only text deltas stream; tool_use blocks are emitted by Anthropic
        as complete units inside the assistant message and are surfaced via
        the final Reply, not as token chunks.
        """
        import anthropic

        client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": self.responder.model,
            "max_tokens": getattr(self.responder, "_max_tokens", 2048),
            "messages": self._anthropic_buf,
        }
        if len(self.tools) > 0:
            kwargs["tools"] = self.tools.to_anthropic_tools()

        on_token = self.on_token  # local alias — set above _ask_responder check
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if on_token is not None and text:
                    on_token(text)
            final = stream.get_final_message()
        return AnthropicResponder._parse_final_message(final)

    def _build_auto_context(self, prompt: str, *, exclude_seq: int) -> str | None:
        """v0.5.3: ask the AutoContextEngine for a preamble for this prompt.

        The engine is lazily imported + constructed so sessions without
        auto_context never pay the embedding-stack import cost.
        """
        from korgchat.context import AutoContextEngine

        engine = getattr(self, "_auto_ctx_engine", None)
        if engine is None:
            engine = AutoContextEngine(self)
            self._auto_ctx_engine = engine
        preamble = engine.build_preamble(prompt, exclude_seqs={exclude_seq})
        if preamble and self.on_context_injected is not None:
            # Count match lines (each starts with two-space "  • ").
            n = preamble.count("\n  • ")
            self.on_context_injected(preamble, n)
        return preamble

    def _build_history_messages(self) -> list[dict[str, Any]]:
        """Convert prior Turns into Anthropic-shape user/assistant messages."""
        msgs: list[dict[str, Any]] = []
        for t in self._history:
            msgs.append({"role": "user", "content": t.user_prompt})
            msgs.append({"role": "assistant", "content": t.assistant_text})
        return msgs

    def _execute_tool(
        self, tu: ToolUse, *, producing_llm_seq: int
    ) -> tuple[int, ToolCall]:
        """Execute a single tool_use and record an AgentToolCall event."""
        start = time.monotonic()
        success = True
        try:
            tool = self.tools.get(tu.name)
            result_obj: Any = tool.call(tu.input)
        except KeyError:
            result_obj = {"error": f"unknown tool: {tu.name!r}"}
            success = False
        except Exception as exc:  # noqa: BLE001 — surface any tool error
            result_obj = {"error": str(exc), "exception_type": type(exc).__name__}
            success = False
        duration_ms = int((time.monotonic() - start) * 1000)

        seq = self._bridge.record_tool_call(
            source_agent=self.source_agent,
            tool_name=tu.name,
            args=tu.input,
            result=result_obj,
            success=success,
            duration_ms=duration_ms,
            triggered_by=producing_llm_seq,
        )
        call = ToolCall(
            seq=seq,
            name=tu.name,
            input=tu.input,
            result=result_obj,
            success=success,
            duration_ms=duration_ms,
        )
        return seq, call

    def __repr__(self) -> str:
        return (
            f"<ChatSession journal={self.journal_path} turns={self.turns} "
            f"responder={self.responder.model!r} tools={self.tools.names()}>"
        )


def select_responder(use_mock: bool) -> Responder:
    """CLI helper: build the appropriate responder."""
    if use_mock:
        return MockResponder()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Use --mock for offline mode, or export "
            "ANTHROPIC_API_KEY=sk-... to enable live mode."
        )
    return AnthropicResponder()

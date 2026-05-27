"""Core chat session that records every turn into the Korg ledger.

Architecture:

    user input  ──>  ChatSession.send(prompt)
                         │
                         ├── record_user_prompt(prompt)   → seq U
                         │   triggered_by = last_llm_seq  (chains to prior turn)
                         │
                         ├── responder.respond(history)   → text
                         │
                         ├── record_llm_call(model, tokens, ms, triggered_by=U)
                         │                                  → seq L
                         │
                         └── update last_llm_seq = L

The bridge handles the lock + atomic write under the hood; ChatSession is
just a thin Python orchestrator. Replacing MockResponder with
AnthropicResponder is a one-line swap.
"""

from __future__ import annotations

import abc
import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import korg_bridge


# A turn pair: the user_prompt seq and the llm_inference seq it produced.
@dataclass
class Turn:
    user_seq: int
    user_prompt: str
    assistant_seq: int
    assistant_text: str
    duration_ms: int


# ── Responders ─────────────────────────────────────────────────────────────


class Responder(abc.ABC):
    """A pluggable backend that turns a chat history into the next reply."""

    @property
    def model(self) -> str:  # informational — recorded on the llm_inference event
        return self.__class__.__name__

    @abc.abstractmethod
    def respond(self, history: list[Turn], prompt: str) -> tuple[str, int, int]:
        """Return (text, prompt_tokens, completion_tokens). Approximations are fine."""


class MockResponder(Responder):
    """Deterministic responder for offline use, CI, and tests.

    Picks a canned reply based on a hash of the prompt so the same input
    always produces the same output — useful for diffing journals across runs.
    """

    REPLIES = [
        "pages turn forward / each entry signed by the past / time leaves no escape",
        "merkle roots whisper / consensus a slow drumline / blockchain heart beats on",
        "korg sees the moves / every tool call signed and chained / nothing left to chance",
        "rewind is a verb / undo lives inside the log / past becomes present",
        "between input lines / there is the silent ledger / quietly counting",
    ]

    @property
    def model(self) -> str:
        return "mock:deterministic"

    def respond(self, history: list[Turn], prompt: str) -> tuple[str, int, int]:
        # Hash chooses a reply; the (history-length, prompt-length) influences
        # token counts so the on-disk numbers are stable but vary per turn.
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        idx = h % len(self.REPLIES)
        text = self.REPLIES[idx]
        # Cheap "tokens" approximation: split on whitespace.
        prompt_tokens = len(re.findall(r"\S+", prompt)) + sum(
            len(re.findall(r"\S+", t.user_prompt + " " + t.assistant_text))
            for t in history
        )
        completion_tokens = len(re.findall(r"\S+", text))
        return text, prompt_tokens, completion_tokens


class AnthropicResponder(Responder):
    """Live Anthropic-backed responder. Requires ANTHROPIC_API_KEY in env.

    Imports the anthropic SDK lazily so KorgChat in --mock mode never needs
    the package installed."""

    def __init__(self, model: str = "claude-opus-4-7", max_tokens: int = 1024):
        try:
            import anthropic  # noqa: F401 (presence check)
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

    def respond(self, history: list[Turn], prompt: str) -> tuple[str, int, int]:
        import anthropic

        client = anthropic.Anthropic()
        messages = []
        for t in history:
            messages.append({"role": "user", "content": t.user_prompt})
            messages.append({"role": "assistant", "content": t.assistant_text})
        messages.append({"role": "user", "content": prompt})

        resp = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return text, prompt_tokens, completion_tokens


# ── Session ────────────────────────────────────────────────────────────────


@dataclass
class ChatSession:
    """A multi-turn KorgChat conversation backed by korg_bridge.

    Construct one ChatSession per logical conversation. The journal path can
    be shared with other writers (korgex agent runs, korg-server browsing) —
    the bridge's file lock serialises concurrent writes.
    """

    journal_path: Path
    responder: Responder
    source_agent: str = "agent:korgchat@0.4.0"
    _bridge: korg_bridge.Bridge = field(init=False)
    _history: list[Turn] = field(default_factory=list, init=False)
    # last_llm_seq is what the next user_prompt links to via triggered_by.
    # None on the very first turn (root user_prompt has triggered_by=None).
    _last_llm_seq: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.journal_path = Path(self.journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._bridge = korg_bridge.Bridge(str(self.journal_path))
        # If the journal already has events (resuming a conversation), pick
        # up where we left off so the new turn chains causally.
        self._last_llm_seq = self._bridge.last_seq_id() or None

    @property
    def history(self) -> list[Turn]:
        return list(self._history)

    @property
    def turns(self) -> int:
        return len(self._history)

    def send(self, prompt: str) -> Turn:
        """Record a user prompt, invoke the responder, record the llm_inference."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        # Record the user_prompt. Root turn (first ever) has triggered_by=None;
        # subsequent turns chain to the prior assistant response so a forensic
        # walk reaches the very first user_prompt as the conversation root.
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

        # Invoke the responder.
        t0 = time.monotonic()
        text, prompt_tokens, completion_tokens = self.responder.respond(
            self._history, prompt
        )
        duration_ms = int((time.monotonic() - t0) * 1000)

        # Record the llm_inference.
        assistant_seq = self._bridge.record_llm_call(
            model=self.responder.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            triggered_by=user_seq,
            source_agent=self.source_agent,
        )

        turn = Turn(
            user_seq=user_seq,
            user_prompt=prompt,
            assistant_seq=assistant_seq,
            assistant_text=text,
            duration_ms=duration_ms,
        )
        self._history.append(turn)
        self._last_llm_seq = assistant_seq
        return turn

    def __repr__(self) -> str:
        return (
            f"<ChatSession journal={self.journal_path} turns={self.turns} "
            f"responder={self.responder.model!r}>"
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

"""KorgChat /summarize — feed scoped events to the LLM and get a digest.

This is the first feature that combines the ledger + search + the LLM.
Three orthogonal ways to scope what gets summarised:

  branch          (default: current)    — every event on a named branch
  --since DUR     (timedelta)           — every event from the last N
  --topic QUERY   (str)                 — events matching a recall query

A summary is just a chat turn with a structured prompt. By default the
result is ephemeral (printed once, not recorded). Pass `--save` to write
the digest back into the journal as a `summary` event so future `/recall`
calls can find it — turning the summary into a first-class artifact.

Default cap: 50 most-recent events in the scope. The model can read more
than that, but a 50-event window is the right shape for "what did we
just talk about" — bigger requires explicit `--limit`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from korgchat.branches import MAIN_BRANCH
from korgchat.recall import RecallEngine

if TYPE_CHECKING:
    from korgchat.chat import ChatSession


SUMMARY_PROMPT_MARKER = "[korgchat-summarize-v1]"


DEFAULT_LIMIT = 50


@dataclass
class Summary:
    """One summarise() result."""

    text: str
    scope_descriptor: str   # human-readable label of what got summarised
    event_count: int        # how many events actually went into the prompt
    truncated: bool         # True if the scope had more events than the limit
    seq_id: int | None = None  # populated only when --save wrote it back


class SummarizeEngine:
    """Build summarise prompts for a session's journal and run them through
    the session's responder. Stateless across calls — re-reads the journal
    each time so concurrent writers (korgex, another KorgChat process) are
    reflected immediately."""

    def __init__(self, session: "ChatSession"):
        self.session = session
        self.journal_path = Path(session.journal_path)
        self._recall = RecallEngine(self.journal_path)

    # ── Public scopes ─────────────────────────────────────────────────

    def summarize_branch(
        self,
        branch_name: str | None = None,
        *,
        limit: int = DEFAULT_LIMIT,
        save: bool = False,
    ) -> Summary:
        """Summarise every event on a named branch. None → current branch."""
        target = branch_name or self.session.current_branch
        events = self._events_in_branch(target)
        descriptor = f"branch '{target}' ({len(events)} events)"
        return self._summarize_events(
            events, descriptor=descriptor, limit=limit, save=save
        )

    def summarize_since(
        self,
        since: timedelta,
        *,
        limit: int = DEFAULT_LIMIT,
        save: bool = False,
    ) -> Summary:
        """Summarise every event whose timestamp is newer than now-since."""
        events = self._events_since(since)
        descriptor = f"last {_format_duration(since)} ({len(events)} events)"
        return self._summarize_events(
            events, descriptor=descriptor, limit=limit, save=save
        )

    def summarize_topic(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        save: bool = False,
    ) -> Summary:
        """Summarise every event matching a /recall-style query."""
        matches = self._recall.search(query, limit=10_000)  # don't truncate
        events = [m.raw_event for m in matches]
        descriptor = f"topic {query!r} ({len(events)} events)"
        return self._summarize_events(
            events, descriptor=descriptor, limit=limit, save=save
        )

    # ── Scope collectors ──────────────────────────────────────────────

    def _events_in_branch(self, branch_name: str) -> list[dict]:
        """Return every event up to (and including) the branch's tip seq.

        Why "all events ≤ tip" rather than a strict triggered_by chain walk:

        * §2a says round-N's llm_inference chains to round-(N-1)'s llm_inference,
          NOT to any tool call between them. A strict ancestry walk therefore
          drops every tool call from the digest, which is unhelpful — "what
          happened?" should include the tools that ran.
        * For multi-branch conversations: the simple cap-by-seq rule includes
          events from sibling branches that forked between the conversation
          root and this branch's tip. That's a known approximation. Users who
          want a tighter scope can use --topic instead. v0.5.2 can refine.
        """
        if branch_name == MAIN_BRANCH:
            tip = self.session._bridge.last_seq_id() or 0
        else:
            try:
                tip = self.session.branches.get(branch_name).tip_seq
            except KeyError:
                return []
        if tip == 0:
            return []
        return [ev for ev in self._all_events() if ev.get("seq_id", 0) <= tip]

    def _events_since(self, since: timedelta) -> list[dict]:
        cutoff = datetime.now(tz=timezone.utc) - since
        out = []
        for ev in self._all_events():
            ts = ev.get("event", {}).get("timestamp")
            evt_time = _parse_ts(ts)
            if evt_time is not None and evt_time >= cutoff:
                out.append(ev)
        return out

    def _events_by_seq(self) -> dict[int, dict]:
        return {ev.get("seq_id"): ev for ev in self._all_events() if "seq_id" in ev}

    def _all_events(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        try:
            with self.journal_path.open() as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    # ── Prompt assembly + responder dispatch ──────────────────────────

    def _summarize_events(
        self,
        events: list[dict],
        *,
        descriptor: str,
        limit: int,
        save: bool,
    ) -> Summary:
        truncated = len(events) > limit
        # Keep the most recent N — those are usually what the user means.
        scoped = events[-limit:] if truncated else events

        if not scoped:
            return Summary(
                text="(no events to summarise in this scope)",
                scope_descriptor=descriptor,
                event_count=0,
                truncated=False,
            )

        prompt = _build_summary_prompt(scoped, descriptor=descriptor, truncated=truncated)
        reply = self.session.responder.respond(
            history=[],   # summaries don't have conversational context themselves
            prompt=prompt,
            prior_tool_results=None,
            tools=None,   # no tool-use during a summary call
        )

        summary = Summary(
            text=reply.text,
            scope_descriptor=descriptor,
            event_count=len(scoped),
            truncated=truncated,
        )

        if save and reply.text:
            seq = self.session._bridge.record_tool_call(
                source_agent="agent:korgchat-summarizer",
                tool_name="summary",
                args={"scope": descriptor},
                result={"text": reply.text},
                success=True,
                duration_ms=0,
                triggered_by=self.session._last_llm_seq,
            )
            summary.seq_id = seq

        return summary


# ── Prompt builder ────────────────────────────────────────────────────


def _build_summary_prompt(
    events: list[dict],
    *,
    descriptor: str,
    truncated: bool,
) -> str:
    """Build a single str prompt that lays out the events and asks for a
    digest. The leading marker lets MockResponder render a templated reply
    for visible mock UX without confusing real Anthropic calls (which see
    the marker as harmless preamble)."""
    lines: list[str] = [
        SUMMARY_PROMPT_MARKER,
        f"You are asked to summarise a slice of a conversation log.",
        f"Scope: {descriptor}.",
    ]
    if truncated:
        lines.append(
            f"(Older events were omitted; you're seeing the most recent {len(events)}.)"
        )
    lines.append("")
    lines.append("=== EVENTS ===")
    for ev in events:
        lines.append(_render_event_line(ev))
    lines.append("=== END EVENTS ===")
    lines.append("")
    lines.append(
        "Write a tight digest of what was discussed, what tools ran "
        "(if any), what conclusions or decisions were reached, and any open "
        "threads the user might want to pick up. Don't echo the raw events; "
        "synthesize them. 4–8 sentences, plain prose."
    )
    return "\n".join(lines)


def _render_event_line(ev: dict) -> str:
    """One-line summary of an event for the prompt body."""
    body = ev.get("event", {})
    tool = body.get("tool_name", "?")
    seq = ev.get("seq_id", "?")
    ts = (body.get("timestamp") or "")[:19]
    if tool == "user_prompt":
        text = body.get("args", {}).get("prompt", "")
        return f"[seq={seq} {ts}] user: {_truncate(text, 240)}"
    if tool == "llm_inference":
        text = body.get("result", {}).get("text", "")
        if text:
            return f"[seq={seq} {ts}] assistant: {_truncate(text, 240)}"
        return f"[seq={seq} {ts}] assistant: (no text — tool-only round)"
    if tool == "summary":
        text = body.get("result", {}).get("text", "")
        scope = body.get("args", {}).get("scope", "?")
        return f"[seq={seq} {ts}] PRIOR SUMMARY ({scope}): {_truncate(text, 200)}"
    if tool == "context_injection":
        # Auto-context recall — meta, not a tool the user invoked. Render it
        # distinctly so digests don't miscount it as a tool call.
        result = body.get("result", {})
        query = body.get("args", {}).get("query", "")
        n = result.get("match_count", 0)
        return (
            f"[seq={seq} {ts}] context: auto-recalled {n} prior event(s) "
            f"for {_truncate(json.dumps(query), 80)}"
        )
    if tool in ("tool_schema_snapshot", "tool_validation"):
        # Audit-trail meta-events bracketing a real tool call. Keep them out
        # of the "tool invocation" tally — they describe a call, they aren't
        # one. Render on their own line shape (no leading "tool ").
        target = body.get("args", {}).get("tool_name", "?")
        if tool == "tool_schema_snapshot":
            h = (body.get("result", {}).get("schema_hash") or "")[:12]
            return f"[seq={seq} {ts}] schema-snapshot {target} (hash={h})"
        result = body.get("result", {})
        verdict = "valid" if result.get("valid") else "INVALID"
        return f"[seq={seq} {ts}] schema-validation {target} → {verdict}"
    # Generic tool call.
    args = body.get("args", {})
    result = body.get("result", {})
    success = body.get("success", True)
    status = "ok" if success else "ERR"
    args_repr = _truncate(json.dumps(args, sort_keys=True), 80)
    result_repr = _truncate(json.dumps(result, sort_keys=True), 80)
    return f"[seq={seq} {ts}] tool {tool}({args_repr}) → [{status}] {result_repr}"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _format_duration(d: timedelta) -> str:
    total = int(d.total_seconds())
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days:
        return f"{days}d"
    if hours:
        return f"{hours}h"
    if mins:
        return f"{mins}m"
    return f"{total}s"

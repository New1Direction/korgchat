"""KorgChat CLI entry point — `korgchat`."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

from korgchat import __version__
from korgchat.chat import ChatSession, MockResponder, ToolCall, select_responder
from korgchat.recall import RecallEngine, format_matches


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="korgchat",
        description=(
            "Chat with an LLM. Every turn is recorded in the Korg cognitive "
            "ledger via korg-bridge."
        ),
    )
    p.add_argument(
        "--journal",
        default=os.environ.get("KORGCHAT_JOURNAL", ".korg/journal.json"),
        help="Path to the ledger journal file. Default: .korg/journal.json "
             "(also honors KORGCHAT_JOURNAL env var).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic mock responder. No API key required, no "
             "network call. Useful for CI, offline use, and reproducibility.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"korgchat {__version__}",
    )
    p.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Auto-exit after N turns. If omitted, runs until EOF (Ctrl-D) or "
             "the user types /quit.",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token-by-token streaming output. The full reply prints "
             "atomically at the end of each turn. The journal is unaffected — "
             "still one llm_inference event per LLM round either way.",
    )
    p.add_argument(
        "--stream-delay",
        type=float,
        default=0.005,
        help="(--mock only) Delay between simulated tokens, seconds. Default "
             "0.005 gives a visible streaming effect; set to 0 for instant output.",
    )
    return p


def _print_banner(session: ChatSession, mock: bool, streaming: bool) -> None:
    border = "─" * 60
    print(border)
    print(f" KorgChat {__version__}")
    print(f" journal:    {session.journal_path}")
    mode_tag = "  (deterministic)" if mock else ""
    stream_tag = "" if streaming else "  (--no-stream)"
    print(f" responder:  {session.responder.model}{mode_tag}{stream_tag}")
    print(f" tools:      {', '.join(session.tools.names()) or '(none)'}")
    print(f" exit:       /quit, /exit, or Ctrl-D")
    print(border)


def _print_tool_call(call: ToolCall) -> None:
    """Live trace line for one tool execution."""
    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in call.input.items())
    status = "ok" if call.success else "ERR"
    result_repr = (
        json.dumps(call.result) if call.success
        else f"{call.result.get('error', call.result)}"
    )
    # Newline before so the line doesn't trail a streamed Korg: line.
    print(f"\n  🔧 [{status}] {call.name}({args_str}) → {result_repr}  "
          f"(seq={call.seq}, {call.duration_ms}ms)")


# ── Slash-command dispatch ────────────────────────────────────────────────

# Sentinels returned by slash-command handlers so the REPL knows what to do.
_SLASH_HANDLED = object()   # command consumed the line, keep looping
_SLASH_EXIT    = object()   # command asked the REPL to exit (code 0)


def _cmd_help(_args: str, _session: ChatSession) -> object:
    print(
        "\nKorgChat slash commands:\n"
        "  /help                 — this list\n"
        "  /recall <query>       — search prior turns (substring, AND of terms)\n"
        "  /recall --kind K Q    — filter by event kind: user_prompt | llm_inference | tool_call\n"
        "  /recall --since 7d Q  — only events from the last N days (or 24h, 30m, ...)\n"
        "  /recall --limit N Q   — cap matches (default 10)\n"
        "  /quit, /exit          — leave the REPL\n"
    )
    return _SLASH_HANDLED


_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")


def _parse_duration(s: str) -> timedelta | None:
    """Accept '30m', '24h', '7d', '90s', or '1.5h'. None on bad input."""
    m = _DURATION_RE.match(s.strip().lower())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    return {
        "s": timedelta(seconds=val),
        "m": timedelta(minutes=val),
        "h": timedelta(hours=val),
        "d": timedelta(days=val),
    }[unit]


def _cmd_recall(args: str, session: ChatSession) -> object:
    """Parse `/recall [--kind K] [--since DUR] [--limit N] <query>` and run."""
    kind: str | None = None
    since: timedelta | None = None
    limit = 10
    tokens = args.split()
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--kind" and i + 1 < len(tokens):
            kind = tokens[i + 1]
            i += 2
        elif t == "--since" and i + 1 < len(tokens):
            d = _parse_duration(tokens[i + 1])
            if d is None:
                print(f"[recall] bad --since duration: {tokens[i + 1]!r}; expected e.g. 7d, 24h, 30m")
                return _SLASH_HANDLED
            since = d
            i += 2
        elif t == "--limit" and i + 1 < len(tokens):
            try:
                limit = max(1, int(tokens[i + 1]))
            except ValueError:
                print(f"[recall] bad --limit: {tokens[i + 1]!r}; expected an integer")
                return _SLASH_HANDLED
            i += 2
        else:
            positional.append(t)
            i += 1

    query = " ".join(positional).strip()
    if not query:
        print("[recall] usage: /recall [--kind K] [--since DUR] [--limit N] <query>")
        return _SLASH_HANDLED

    engine = RecallEngine(session.journal_path)
    matches = engine.search(query, kind=kind, since=since, limit=limit)
    print("\n" + format_matches(matches, query=query))
    return _SLASH_HANDLED


_SLASH_COMMANDS = {
    "/help": _cmd_help,
    "/recall": _cmd_recall,
}


def _maybe_handle_slash(prompt: str, session: ChatSession) -> object | None:
    """Return _SLASH_HANDLED / _SLASH_EXIT if the prompt was a slash command,
    or None to let it fall through to the responder."""
    if not prompt.startswith("/"):
        return None
    head, _, rest = prompt.partition(" ")
    head_lower = head.lower()
    if head_lower in {"/quit", "/exit"}:
        return _SLASH_EXIT
    handler = _SLASH_COMMANDS.get(head_lower)
    if handler is None:
        print(f"[korgchat] unknown command {head!r} — try /help")
        return _SLASH_HANDLED
    return handler(rest, session)


class _Streamer:
    """Stdout streamer that prints a fresh `Korg: ` prefix on the first
    token of each LLM round, then flushes each subsequent chunk inline.

    Wires to ChatSession.on_round_start + ChatSession.on_token. Tracking
    `_fresh_round` per round keeps tool-only rounds from printing an
    empty `Korg: ` line."""

    def __init__(self) -> None:
        self._fresh_round = True

    def on_round_start(self) -> None:
        self._fresh_round = True

    def on_token(self, chunk: str) -> None:
        if self._fresh_round:
            print("\nKorg: ", end="", flush=True)
            self._fresh_round = False
        print(chunk, end="", flush=True)


def _interactive_loop(
    session: ChatSession, max_turns: int | None, streaming: bool
) -> int:
    """Read-eval-print loop. Returns the process exit code."""
    session.on_tool_call = _print_tool_call
    if streaming:
        streamer = _Streamer()
        session.on_round_start = streamer.on_round_start
        session.on_token = streamer.on_token

    while True:
        if max_turns is not None and session.turns >= max_turns:
            print(f"\n[korgchat] reached --turns={max_turns}; exiting")
            return 0
        try:
            prompt = input("\nYou: ")
        except EOFError:
            print()  # newline so the shell prompt doesn't share a line
            return 0
        except KeyboardInterrupt:
            print("\n[korgchat] interrupted")
            return 130

        prompt = prompt.strip()
        if not prompt:
            continue

        # Slash-command dispatch — runs before we touch the responder.
        slash_result = _maybe_handle_slash(prompt, session)
        if slash_result is _SLASH_EXIT:
            return 0
        if slash_result is _SLASH_HANDLED:
            continue

        try:
            turn = session.send(prompt)
        except Exception as e:  # noqa: BLE001 — surface anything from responder
            print(f"\n[korgchat] error: {e}", file=sys.stderr)
            return 1

        # In streaming mode the assistant text has already been printed
        # incrementally. In non-streaming mode print it once here.
        if not streaming and turn.assistant_text:
            print(f"\nKorg: {turn.assistant_text}")

        tail = (
            f", {len(turn.tool_calls)} tool call(s)"
            if turn.tool_calls else ""
        )
        print(
            f"\n[recorded: turn {session.turns}, "
            f"seq={turn.user_seq} (user) → seq={turn.assistant_seq} (assistant){tail}]"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        responder = select_responder(args.mock)
    except RuntimeError as e:
        print(f"[korgchat] {e}", file=sys.stderr)
        return 2

    # If we're in mock + streaming mode, give the mock a non-zero
    # per-character delay so the streaming UX is actually visible.
    streaming = not args.no_stream
    if args.mock and streaming and args.stream_delay > 0:
        responder = MockResponder(stream_delay_secs=args.stream_delay)

    session = ChatSession(
        journal_path=Path(args.journal),
        responder=responder,
    )
    _print_banner(session, mock=args.mock, streaming=streaming)
    return _interactive_loop(session, args.turns, streaming=streaming)


if __name__ == "__main__":
    sys.exit(main())

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
from korgchat.summary import SummarizeEngine


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
    branch_count = len(session.branches.list())
    extra = f"  (+{branch_count} branch{'es' if branch_count != 1 else ''})" if branch_count else ""
    print(f" branch:     {session.current_branch}{extra}")
    print(f" exit:       /quit, /exit, or Ctrl-D  (try /help for commands)")
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
        "  /help                   — this list\n"
        "  /recall <query>         — search prior turns (substring, AND of terms)\n"
        "  /recall --kind K Q      — filter by event kind: user_prompt | llm_inference | tool_call\n"
        "  /recall --since 7d Q    — only events from the last N days (24h, 30m, ...)\n"
        "  /recall --limit N Q     — cap matches (default 10)\n"
        "  /summarize              — digest the current branch via the LLM\n"
        "  /summarize <branch>     — digest a named branch (or 'main')\n"
        "  /summarize --since DUR  — digest events from the last N (7d, 24h, ...)\n"
        "  /summarize --topic Q    — digest events matching a /recall query\n"
        "  /summarize --save       — also record the digest as a 'summary' event (findable via /recall)\n"
        "  /branches               — list named conversation branches (with current marker)\n"
        "  /fork <name>            — bookmark this point as a branch and switch to it\n"
        "  /checkout <name|main>   — switch the active branch; new turns resume from its tip\n"
        "  /branch-delete <name>   — drop a branch bookmark (the events themselves stay in the journal)\n"
        "  /branch-rename <old> <new> — rename a branch\n"
        "  /quit, /exit            — leave the REPL\n"
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


def _cmd_branches(_args: str, session: ChatSession) -> object:
    """List all branches with a `← current` marker on the active one."""
    from korgchat.branches import MAIN_BRANCH
    branches = session.branches.list()
    cur = session.current_branch
    main_marker = "  ← current" if cur == MAIN_BRANCH else ""
    print(f"\n[branches] active = {cur!r}")
    print(f"  {MAIN_BRANCH:<24} (trunk){main_marker}")
    if not branches:
        print(f"  (no named branches — use /fork <name> to create one)")
    else:
        for b in branches:
            marker = "  ← current" if b.name == cur else ""
            short_ts = b.created_at[:19].replace("T", " ")
            print(
                f"  {b.name:<24} fork@{b.fork_seq}  tip@{b.tip_seq}  "
                f"({short_ts}){marker}"
            )
    return _SLASH_HANDLED


def _cmd_fork(args: str, session: ChatSession) -> object:
    """`/fork <name>` — create a branch bookmark here, switch to it."""
    name = args.strip()
    if not name:
        print("[fork] usage: /fork <name>")
        return _SLASH_HANDLED
    try:
        session.fork_here(name)
    except (ValueError, KeyError) as e:
        print(f"[fork] {e}")
        return _SLASH_HANDLED
    b = session.branches.get(name)
    print(f"\n[fork] branch {name!r} created at seq={b.fork_seq}, now active")
    return _SLASH_HANDLED


def _cmd_checkout(args: str, session: ChatSession) -> object:
    """`/checkout <name|main>` — switch active branch."""
    from korgchat.branches import MAIN_BRANCH
    name = args.strip()
    if not name:
        print("[checkout] usage: /checkout <name|main>")
        return _SLASH_HANDLED
    try:
        tip = session.checkout(name)
    except KeyError as e:
        print(f"[checkout] {e}")
        return _SLASH_HANDLED
    tip_repr = "empty" if tip is None or tip == 0 else f"seq={tip}"
    print(f"\n[checkout] now on {name!r} ({tip_repr})")
    if name != MAIN_BRANCH:
        print(
            f"  (next turn will chain triggered_by from {tip_repr}; "
            f"in-memory history cleared)"
        )
    return _SLASH_HANDLED


def _cmd_branch_delete(args: str, session: ChatSession) -> object:
    name = args.strip()
    if not name:
        print("[branch-delete] usage: /branch-delete <name>")
        return _SLASH_HANDLED
    if name == session.current_branch:
        print(
            f"[branch-delete] refusing to delete the active branch — "
            f"`/checkout main` first."
        )
        return _SLASH_HANDLED
    try:
        b = session.branches.delete(name)
    except (KeyError, ValueError) as e:
        print(f"[branch-delete] {e}")
        return _SLASH_HANDLED
    print(
        f"\n[branch-delete] removed {b.name!r} (events from seq={b.fork_seq} "
        f"to {b.tip_seq} stay in the journal)"
    )
    return _SLASH_HANDLED


def _cmd_branch_rename(args: str, session: ChatSession) -> object:
    parts = args.split()
    if len(parts) != 2:
        print("[branch-rename] usage: /branch-rename <old> <new>")
        return _SLASH_HANDLED
    old, new = parts
    try:
        session.branches.rename(old, new)
    except (KeyError, ValueError) as e:
        print(f"[branch-rename] {e}")
        return _SLASH_HANDLED
    # Update session pointer if we renamed the current branch.
    if session.current_branch == old:
        session.current_branch = new
    print(f"\n[branch-rename] {old!r} → {new!r}")
    return _SLASH_HANDLED


def _cmd_summarize(args: str, session: ChatSession) -> object:
    """/summarize [branch] [--since DUR] [--topic Q] [--limit N] [--save]"""
    from korgchat.summary import DEFAULT_LIMIT

    tokens = args.split()
    since = None
    topic = None
    limit = DEFAULT_LIMIT
    save = False
    positional: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--since" and i + 1 < len(tokens):
            d = _parse_duration(tokens[i + 1])
            if d is None:
                print(f"[summarize] bad --since {tokens[i + 1]!r}; try 7d, 24h, 30m")
                return _SLASH_HANDLED
            since = d
            i += 2
        elif t == "--topic" and i + 1 < len(tokens):
            topic = tokens[i + 1]
            i += 2
        elif t == "--limit" and i + 1 < len(tokens):
            try:
                limit = max(1, int(tokens[i + 1]))
            except ValueError:
                print(f"[summarize] bad --limit {tokens[i + 1]!r}; expected int")
                return _SLASH_HANDLED
            i += 2
        elif t == "--save":
            save = True
            i += 1
        else:
            positional.append(t)
            i += 1

    # Mutually-exclusive scope selectors (positional branch wins if multiple).
    selectors = sum(1 for x in (positional, since, topic) if x)
    if selectors > 1:
        print(
            "[summarize] choose exactly one scope: a branch name, "
            "--since DUR, or --topic QUERY"
        )
        return _SLASH_HANDLED

    engine = SummarizeEngine(session)
    try:
        if positional:
            summary = engine.summarize_branch(positional[0], limit=limit, save=save)
        elif since is not None:
            summary = engine.summarize_since(since, limit=limit, save=save)
        elif topic is not None:
            summary = engine.summarize_topic(topic, limit=limit, save=save)
        else:
            # Default = current branch.
            summary = engine.summarize_branch(limit=limit, save=save)
    except Exception as e:  # noqa: BLE001 — surface anything from the responder
        print(f"[summarize] error: {e}")
        return _SLASH_HANDLED

    print(f"\n[summarize] {summary.scope_descriptor}")
    if summary.truncated:
        print(f"  (showing the most recent {summary.event_count} events; older omitted)")
    if summary.seq_id is not None:
        print(f"  (saved as seq={summary.seq_id}; findable via /recall)")
    print()
    print(summary.text)
    return _SLASH_HANDLED


_SLASH_COMMANDS = {
    "/help": _cmd_help,
    "/recall": _cmd_recall,
    "/summarize": _cmd_summarize,
    "/branches": _cmd_branches,
    "/fork": _cmd_fork,
    "/checkout": _cmd_checkout,
    "/branch-delete": _cmd_branch_delete,
    "/branch-rename": _cmd_branch_rename,
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

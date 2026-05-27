"""KorgChat CLI entry point — `korgchat`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from korgchat import __version__
from korgchat.chat import ChatSession, ToolCall, select_responder


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
    return p


def _print_banner(session: ChatSession, mock: bool) -> None:
    border = "─" * 60
    print(border)
    print(f" KorgChat {__version__}")
    print(f" journal:    {session.journal_path}")
    print(f" responder:  {session.responder.model}{'  (deterministic)' if mock else ''}")
    print(f" tools:      {', '.join(session.tools.names()) or '(none)'}")
    print(f" exit:       /quit, /exit, or Ctrl-D")
    print(border)


def _print_tool_call(call: ToolCall) -> None:
    """Live trace line for one tool execution."""
    args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in call.input.items())
    status = "ok" if call.success else "ERR"
    result_repr = json.dumps(call.result) if call.success else f"{call.result.get('error', call.result)}"
    print(f"  🔧 [{status}] {call.name}({args_str}) → {result_repr}  (seq={call.seq}, {call.duration_ms}ms)")


def _interactive_loop(session: ChatSession, max_turns: int | None) -> int:
    """Read-eval-print loop. Returns the process exit code."""
    session.on_tool_call = _print_tool_call
    while True:
        if max_turns is not None and session.turns >= max_turns:
            print(f"[korgchat] reached --turns={max_turns}; exiting")
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
        if prompt.lower() in {"/quit", "/exit"}:
            return 0

        try:
            turn = session.send(prompt)
        except Exception as e:  # noqa: BLE001 — surface anything from responder
            print(f"\n[korgchat] error: {e}", file=sys.stderr)
            return 1

        if turn.assistant_text:
            print(f"\nKorg: {turn.assistant_text}")
        tail = (
            f", {len(turn.tool_calls)} tool call(s)"
            if turn.tool_calls
            else ""
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

    session = ChatSession(
        journal_path=Path(args.journal),
        responder=responder,
    )
    _print_banner(session, mock=args.mock)
    return _interactive_loop(session, args.turns)


if __name__ == "__main__":
    sys.exit(main())

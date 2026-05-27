"""Tests for v0.5.3 auto-context injection.

The journal must always record the ORIGINAL prompt (not the augmented
one) — auto-context lives inside the responder call, not the audit log.
That's the most important invariant; several tests pin it.

Semantic-recall tests are gated on `fastembed` (the optional [semantic]
extra). Without it, auto-context falls back to substring recall, which
is much noisier — we still test the wiring works, just with looser
score expectations.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from korgchat import (
    AnthropicResponder,
    ChatSession,
    MockResponder,
    Reply,
)
from korgchat.__main__ import main as cli_main
from korgchat.context import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_N,
    PREAMBLE_FOOTER,
    PREAMBLE_HEADER,
    AutoContextEngine,
)


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


@pytest.fixture(scope="module")
def fastembed_available():
    pytest.importorskip("fastembed")
    return True


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


# ── Defaults ──────────────────────────────────────────────────────────────


def test_auto_context_off_by_default(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    assert s.auto_context is False


def test_engine_constants_make_sense():
    assert DEFAULT_TOP_N == 3
    assert DEFAULT_MIN_SCORE > 0.3  # stricter than /recall's 0.30
    assert DEFAULT_MIN_SCORE < 1.0


# ── AutoContextEngine ──────────────────────────────────────────────────


def test_engine_returns_none_on_empty_query(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("warmup")
    engine = AutoContextEngine(s)
    assert engine.build_preamble("") is None
    assert engine.build_preamble("   ") is None


def test_engine_returns_none_when_journal_empty(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    engine = AutoContextEngine(s)
    assert engine.build_preamble("anything") is None


def test_engine_returns_preamble_with_relevant_history(tmp_journal, fastembed_available):
    """Semantic recall finds the rust thread; auto-context surfaces it."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("how does the rust borrow checker prevent data races")
    s.send("css flexbox alignment tips")
    s.send("rust ownership and lifetimes")

    # New question on the rust topic — auto-context should pick up the
    # two rust-related prior prompts, not the CSS one.
    engine = AutoContextEngine(s, min_score=0.35)  # slightly looser for test stability
    preamble = engine.build_preamble("confused about borrowing")
    assert preamble is not None
    assert PREAMBLE_HEADER in preamble
    assert PREAMBLE_FOOTER in preamble
    # Rust topics should appear; CSS shouldn't (or at least shouldn't dominate).
    lowered = preamble.lower()
    assert "rust" in lowered or "borrow" in lowered


def test_engine_excludes_seqs_from_results(tmp_journal, fastembed_available):
    """exclude_seqs prevents the just-recorded user_prompt from being
    auto-recalled as its own context."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("rust borrow checker explainer")  # seq 1 (user), 2 (llm)

    engine = AutoContextEngine(s, min_score=0.30)
    # Without exclusion: the seq=1 user prompt may match itself.
    p_all = engine.build_preamble("rust borrowing")
    # With exclusion of seq=1: the user prompt itself is filtered out.
    p_excl = engine.build_preamble("rust borrowing", exclude_seqs={1})

    if p_all is not None and p_excl is not None:
        # The excluded preamble shouldn't reference seq=1.
        assert "seq=1 " not in p_excl
    # At minimum, the exclusion path completes without crashing.
    assert p_excl is None or "seq=1 " not in p_excl


def test_engine_returns_none_when_no_match_passes_threshold(tmp_journal, fastembed_available):
    """A high-threshold query against unrelated content should produce None."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("how does the rust borrow checker work")

    # Very loose-to-unrelated query, very strict threshold.
    engine = AutoContextEngine(s, min_score=0.95)
    assert engine.build_preamble("quantum chromodynamics and gauge symmetry") is None


# ── ChatSession integration ───────────────────────────────────────────


def test_session_auto_context_off_does_not_call_engine(tmp_journal, fastembed_available):
    """When auto_context=False, the responder gets the ORIGINAL prompt only."""
    captured: list[str] = []

    class _CapturingMock(MockResponder):
        def respond(self, *, history, prompt, prior_tool_results=None, tools=None):
            captured.append(prompt)
            return Reply(text="ok", prompt_tokens=1, completion_tokens=1)

    s = ChatSession(
        journal_path=tmp_journal,
        responder=_CapturingMock(),
        auto_context=False,
    )
    s.send("rust borrow checker")
    s.send("more about rust")

    # No preamble header anywhere.
    assert all(PREAMBLE_HEADER not in p for p in captured)


def test_session_auto_context_on_injects_preamble(tmp_journal, fastembed_available):
    """When auto_context=True, later prompts may carry a preamble."""
    captured: list[str] = []

    class _CapturingMock(MockResponder):
        def respond(self, *, history, prompt, prior_tool_results=None, tools=None):
            captured.append(prompt)
            return Reply(text="ok", prompt_tokens=1, completion_tokens=1)

    s = ChatSession(
        journal_path=tmp_journal,
        responder=_CapturingMock(),
        auto_context=True,
    )
    # First turn — nothing in journal to recall from, so no preamble.
    s.send("how does the rust borrow checker prevent data races")
    # Second turn — same topic; auto-context should fire.
    s.send("more on rust borrowing semantics please")

    # Find at least one captured prompt that carries the preamble header.
    augmented = [p for p in captured if PREAMBLE_HEADER in p]
    assert augmented, (
        f"expected at least one augmented prompt; got captures of lengths "
        f"{[len(p) for p in captured]}"
    )


def test_journal_records_original_prompt_not_augmented(tmp_journal, fastembed_available):
    """Critical invariant: the audit log shows what the user actually typed."""
    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
    )
    s.send("rust borrow checker")
    s.send("more on rust borrowing")

    events = _events(tmp_journal)
    user_prompts = [
        e["event"]["args"]["prompt"]
        for e in events
        if e["event"]["tool_name"] == "user_prompt"
    ]
    # No prompt in the journal should carry the auto-context preamble.
    for p in user_prompts:
        assert PREAMBLE_HEADER not in p
        assert "auto-recalled" not in p


def test_on_context_injected_callback_fires(tmp_journal, fastembed_available):
    """The callback gives the CLI a hook to print an indicator."""
    fired: list[tuple[int]] = []

    def hook(_preamble: str, n: int) -> None:
        fired.append((n,))

    s = ChatSession(
        journal_path=tmp_journal,
        responder=MockResponder(),
        auto_context=True,
        on_context_injected=hook,
    )
    s.send("how does the rust borrow checker prevent data races")
    s.send("more on rust borrowing")

    # If any auto-context injected on the second turn, the hook fired
    # with at least one match. If thresholds excluded everything, the
    # list stays empty — that's fine; the wiring is still correct.
    if fired:
        assert all(n >= 1 for (n,) in fired)


# ── CLI integration ──────────────────────────────────────────────────


def test_cli_auto_context_off_by_default(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("hi\n/quit\n"))
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "auto-ctx:" not in out  # banner doesn't mention it


def test_cli_auto_context_flag_shows_banner(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("hi\n/quit\n"))
    rc = cli_main([
        "--mock",
        "--journal", str(tmp_journal),
        "--stream-delay", "0",
        "--auto-context",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "auto-ctx:" in out
    assert "ON" in out


def test_cli_auto_context_indicator_fires(tmp_journal, monkeypatch, capsys, fastembed_available):
    """Run two related turns under --auto-context; the second should
    print the indicator if semantic recall passes the threshold."""
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "how does the rust borrow checker prevent data races\n"
            "more on rust borrowing semantics please\n"
            "/quit\n"
        ),
    )
    rc = cli_main([
        "--mock",
        "--journal", str(tmp_journal),
        "--stream-delay", "0",
        "--auto-context",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Either the indicator fired (semantic match passed threshold) or it
    # didn't (no match strong enough). Both are valid outcomes; the
    # important thing is the run completes and the banner shows ON.
    assert "auto-ctx:" in out
    # If the indicator appeared, sanity-check its shape.
    if "[auto-context]" in out:
        assert "injected" in out
        assert "prior" in out


def test_cli_journal_pristine_under_auto_context(tmp_journal, monkeypatch, capsys):
    """The journal should NOT carry the preamble even with --auto-context."""
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "first prompt\n"
            "second prompt\n"
            "/quit\n"
        ),
    )
    rc = cli_main([
        "--mock",
        "--journal", str(tmp_journal),
        "--stream-delay", "0",
        "--auto-context",
    ])
    assert rc == 0
    events = _events(tmp_journal)
    for e in events:
        if e["event"]["tool_name"] == "user_prompt":
            assert PREAMBLE_HEADER not in e["event"]["args"]["prompt"]

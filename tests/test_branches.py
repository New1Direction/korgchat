"""Tests for v0.5.0 conversation branches.

Two layers:
  1. `BranchStore` unit tests over `.korg/branches.json` (CRUD + atomic save)
  2. `ChatSession` integration: fork → diverge → checkout → resume

The journal events themselves don't change — branches are purely a
client-side bookmarking layer on top of the existing triggered_by chain.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from korgchat import Branch, BranchStore, ChatSession, MAIN_BRANCH, MockResponder
from korgchat.__main__ import main as cli_main


# ── BranchStore (storage layer) ───────────────────────────────────────────


@pytest.fixture
def empty_store(tmp_path: Path) -> BranchStore:
    return BranchStore(tmp_path / "branches.json")


def test_missing_file_is_empty_store(empty_store):
    assert empty_store.list() == []
    assert empty_store.names() == []
    assert "anything" not in empty_store


def test_create_and_retrieve(empty_store):
    b = empty_store.create("alpha", fork_seq=10)
    assert b.name == "alpha"
    assert b.fork_seq == 10
    assert b.tip_seq == 10
    assert "alpha" in empty_store
    assert empty_store.get("alpha") == b
    assert empty_store.names() == ["alpha"]


def test_create_rejects_duplicate(empty_store):
    empty_store.create("a", fork_seq=1)
    with pytest.raises(ValueError, match="already exists"):
        empty_store.create("a", fork_seq=5)


def test_create_rejects_reserved_main(empty_store):
    with pytest.raises(ValueError, match="reserved"):
        empty_store.create(MAIN_BRANCH, fork_seq=1)


def test_create_validates_name(empty_store):
    for bad in ["", "a/b", "with space", "dot.dot", "x" * 65]:
        with pytest.raises(ValueError, match="invalid|reserved|1.{1,3}64"):
            empty_store.create(bad, fork_seq=0)


def test_update_tip_advances(empty_store):
    empty_store.create("alpha", fork_seq=10)
    empty_store.update_tip("alpha", 25)
    assert empty_store.get("alpha").tip_seq == 25


def test_update_tip_rejects_regression(empty_store):
    empty_store.create("alpha", fork_seq=10)
    with pytest.raises(ValueError, match="cannot precede"):
        empty_store.update_tip("alpha", 5)


def test_update_tip_idempotent_skip(empty_store):
    """Setting the tip to its current value should not be an error."""
    empty_store.create("alpha", fork_seq=10)
    empty_store.update_tip("alpha", 10)  # no-op; doesn't crash
    assert empty_store.get("alpha").tip_seq == 10


def test_update_tip_main_is_noop(empty_store):
    """main has no sidecar entry; updating its tip is silently allowed."""
    empty_store.update_tip(MAIN_BRANCH, 999)  # no error
    assert MAIN_BRANCH not in empty_store


def test_rename(empty_store):
    empty_store.create("alpha", fork_seq=10)
    empty_store.rename("alpha", "beta")
    assert "alpha" not in empty_store
    assert "beta" in empty_store
    assert empty_store.get("beta").fork_seq == 10


def test_delete(empty_store):
    empty_store.create("alpha", fork_seq=10)
    deleted = empty_store.delete("alpha")
    assert deleted.name == "alpha"
    assert "alpha" not in empty_store


def test_persistence_across_instances(tmp_path):
    """A fresh BranchStore on the same path sees the previously-written data."""
    path = tmp_path / "branches.json"
    s1 = BranchStore(path)
    s1.create("a", fork_seq=10)
    s1.create("b", fork_seq=20)
    s1.update_tip("a", 15)

    s2 = BranchStore(path)
    assert sorted(s2.names()) == ["a", "b"]
    assert s2.get("a").tip_seq == 15
    assert s2.get("b").fork_seq == 20


def test_corrupt_file_recovers_empty(tmp_path):
    """A malformed JSON file logs a warning but doesn't crash construction."""
    path = tmp_path / "branches.json"
    path.write_text("{ this is not valid json")
    s = BranchStore(path)
    assert s.list() == []


def test_save_is_atomic(empty_store, tmp_path):
    """No `.tmp` files left behind in the parent dir after a successful save."""
    empty_store.create("alpha", fork_seq=1)
    siblings = list(tmp_path.iterdir())
    # The persistent branches.json + zero tmp files.
    assert any(s.name == "branches.json" for s in siblings)
    assert not any(s.name.endswith(".tmp") for s in siblings)


# ── ChatSession integration ───────────────────────────────────────────────


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def _events(journal: Path) -> list[dict]:
    with journal.open() as f:
        return json.load(f)


def test_session_starts_on_main(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    assert s.current_branch == MAIN_BRANCH
    assert s.branches.list() == []


def test_fork_requires_at_least_one_event(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    with pytest.raises(ValueError, match="cannot fork"):
        s.fork_here("alpha")


def test_fork_creates_branch_at_current_seq(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("turn 1")
    # After turn 1, _last_llm_seq = 2 (user_prompt @ 1 → llm_inference @ 2).
    s.fork_here("alpha")
    assert s.current_branch == "alpha"
    b = s.branches.get("alpha")
    assert b.fork_seq == 2
    assert b.tip_seq == 2


def test_turns_on_branch_advance_its_tip(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("turn on main")
    s.fork_here("alpha")
    s.send("turn on alpha 1")
    s.send("turn on alpha 2")
    # Each turn writes 2 events; alpha's tip should be the latest.
    assert s.branches.get("alpha").tip_seq == 6


def test_checkout_resumes_from_branch_tip(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("turn 1")  # seq 1,2 on main
    s.fork_here("alpha")
    s.send("on alpha A")  # seq 3,4 on alpha
    s.send("on alpha B")  # seq 5,6 on alpha
    s.checkout(MAIN_BRANCH)
    # After checkout main, in-memory history is cleared.
    assert s.turns == 0
    # The next turn should chain triggered_by from the journal's latest
    # overall seq (which is 6, since alpha events live in the same journal).
    t = s.send("back on main")
    # The new user_prompt should chain to seq=6 (last overall in journal).
    events = _events(tmp_journal)
    assert events[-2]["metadata"]["triggered_by"] == 6  # user_prompt of new turn
    assert t.user_seq == 7
    assert t.assistant_seq == 8


def test_checkout_to_branch_resumes_from_branch_tip(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("turn 1")        # 1, 2 on main
    s.fork_here("alpha")    # alpha forks @ 2
    s.send("alpha A")       # 3, 4 — alpha tip=4
    s.checkout(MAIN_BRANCH)
    s.send("main A")        # 5, 6 — main moves forward
    s.checkout("alpha")
    # alpha's tip is still 4 (not 6) — the main-side events don't affect it.
    assert s.current_branch == "alpha"
    t = s.send("alpha B")
    events = _events(tmp_journal)
    # The new user_prompt's triggered_by should be alpha's tip (4), not 6.
    new_user_event = events[t.user_seq - 1]  # seq is 1-indexed
    assert new_user_event["metadata"]["triggered_by"] == 4


def test_checkout_unknown_branch_raises(tmp_journal):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    with pytest.raises(KeyError):
        s.checkout("nope")


# ── CLI integration ──────────────────────────────────────────────────────


def test_cli_help_lists_branch_commands(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("/help\n/quit\n"))
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    for cmd in ("/branches", "/fork", "/checkout", "/branch-delete"):
        assert cmd in out


def test_cli_fork_and_branches_listing(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "hello\n"
        "/fork experiment\n"
        "/branches\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "branch 'experiment' created" in out
    # /branches should mark experiment as current.
    assert "experiment" in out
    # The trunk row should mention main.
    assert "main" in out


def test_cli_fork_diverge_then_checkout_main(tmp_journal, monkeypatch, capsys):
    """End-to-end: fork, take a turn on the branch, switch back to main,
    take another turn, verify the journal has both threads."""
    stdin = io.StringIO(
        "turn one\n"
        "/fork experiment\n"
        "turn two on experiment\n"
        "/checkout main\n"
        "turn three on main\n"
        "/branches\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out

    # 3 chat turns recorded; one of them on the branch.
    assert out.count("[recorded:") == 3

    events = _events(tmp_journal)
    assert len(events) == 6  # 3 turns × 2 events each

    # Branch sidecar should record the experiment branch with tip=4
    # (turn 2 created events 3+4, both on experiment).
    branches_path = tmp_journal.parent / "branches.json"
    assert branches_path.exists()
    branches_data = json.loads(branches_path.read_text())
    assert len(branches_data["branches"]) == 1
    exp = branches_data["branches"][0]
    assert exp["name"] == "experiment"
    assert exp["fork_seq"] == 2
    assert exp["tip_seq"] == 4

    # The third turn (back on main) should chain from seq=4 (the most
    # recent event in the journal at that point).
    assert events[4]["metadata"]["triggered_by"] == 4  # turn 3 user_prompt
    assert events[5]["metadata"]["triggered_by"] == 5  # turn 3 llm_inference


def test_cli_branch_delete(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "warmup\n"
        "/fork doomed\n"
        "/checkout main\n"
        "/branch-delete doomed\n"
        "/branches\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "removed 'doomed'" in out
    assert "no named branches" in out


def test_cli_cannot_delete_active_branch(tmp_journal, monkeypatch, capsys):
    """Refuse to delete the currently-checked-out branch."""
    stdin = io.StringIO(
        "warmup\n"
        "/fork active\n"
        "/branch-delete active\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "refusing to delete the active branch" in out


def test_cli_branch_rename(tmp_journal, monkeypatch, capsys):
    stdin = io.StringIO(
        "warmup\n"
        "/fork oldname\n"
        "/branch-rename oldname newname\n"
        "/branches\n"
        "/quit\n"
    )
    monkeypatch.setattr("sys.stdin", stdin)
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "oldname" in out  # rename log line
    assert "newname" in out

"""Conversation branches — named bookmarks into the journal's triggered_by DAG.

A branch is just a name + a `tip_seq`. When you `/fork rust-attempt` at
seq=42, KorgChat records a bookmark:

    {name: "rust-attempt", fork_seq: 42, tip_seq: 42, created_at: "..."}

The journal itself doesn't change — events are still chained via
`triggered_by`. The branch label is purely client-side: KorgChat reads
the branch's tip when you `/checkout`, then new turns chain from there.
Switching back to `main` resumes from the journal's latest overall seq.

The shape is intentionally lightweight: no `branch_id` UUIDs on events,
no bridge change, no schema bump. The DAG that emerges from concurrent
branches lives in the existing `triggered_by` chain — branches are
just labels that say "here's where I was typing."

Storage: `<journal_dir>/branches.json` next to `journal.json`. Atomic
writes via tmp + rename so a crash mid-save can't corrupt the index.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# The name reserved for the trunk. Never appears in branches.json; main's
# tip is always the latest event in the journal overall.
MAIN_BRANCH = "main"


@dataclass
class Branch:
    name: str
    fork_seq: int       # seq we forked from — invariant; never changes
    tip_seq: int        # head of this branch — moves forward each turn
    created_at: str     # ISO-8601 UTC timestamp


class BranchStore:
    """CRUD over the `.korg/branches.json` sidecar.

    Stateful (caches the loaded list in memory) so callers don't pay disk
    I/O on every list() call during a hot REPL session. Always re-reads
    from disk on construction, and writes atomically.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._branches: dict[str, Branch] = {}
        self._load()

    # ── public API ────────────────────────────────────────────────────

    def list(self) -> list[Branch]:
        """Every named branch, ordered by creation time (oldest first)."""
        return sorted(self._branches.values(), key=lambda b: b.created_at)

    def names(self) -> list[str]:
        return sorted(self._branches.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._branches

    def get(self, name: str) -> Branch:
        if name == MAIN_BRANCH:
            raise KeyError(
                f"{MAIN_BRANCH!r} is the implicit trunk — not stored in the "
                f"sidecar. Read its tip from the journal directly."
            )
        if name not in self._branches:
            raise KeyError(f"branch {name!r} does not exist")
        return self._branches[name]

    def create(self, name: str, *, fork_seq: int) -> Branch:
        """Create a new branch at `fork_seq`. The tip starts equal to
        fork_seq — both move as turns are added. Raises if the name is
        already taken or is the reserved `main`."""
        if name == MAIN_BRANCH:
            raise ValueError(f"{MAIN_BRANCH!r} is the reserved trunk name")
        if not _is_valid_name(name):
            raise ValueError(
                f"branch name {name!r} must be 1–64 chars of "
                f"[A-Za-z0-9_-] (no slashes, spaces, or dots)"
            )
        if name in self._branches:
            raise ValueError(f"branch {name!r} already exists")
        if fork_seq < 0:
            raise ValueError(f"fork_seq must be >= 0, got {fork_seq}")

        b = Branch(
            name=name,
            fork_seq=int(fork_seq),
            tip_seq=int(fork_seq),
            created_at=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self._branches[name] = b
        self._save()
        return b

    def update_tip(self, name: str, tip_seq: int) -> None:
        """Bump the branch's head to a new seq. Used after every turn that
        lands on a non-main branch."""
        if name == MAIN_BRANCH:
            # Main's tip is the journal's latest seq — nothing to persist.
            return
        if name not in self._branches:
            raise KeyError(f"branch {name!r} does not exist")
        if tip_seq < self._branches[name].fork_seq:
            raise ValueError(
                f"tip_seq {tip_seq} cannot precede fork_seq "
                f"{self._branches[name].fork_seq}"
            )
        # Only persist when the tip actually changed; saves a write per turn
        # on no-op cases (slash commands, /recall, etc).
        if self._branches[name].tip_seq == tip_seq:
            return
        self._branches[name].tip_seq = int(tip_seq)
        self._save()

    def rename(self, old: str, new: str) -> Branch:
        if old == MAIN_BRANCH:
            raise ValueError(f"cannot rename {MAIN_BRANCH!r}")
        if new == MAIN_BRANCH:
            raise ValueError(f"cannot rename to reserved {MAIN_BRANCH!r}")
        if not _is_valid_name(new):
            raise ValueError(f"new branch name {new!r} invalid")
        if old not in self._branches:
            raise KeyError(f"branch {old!r} does not exist")
        if new in self._branches:
            raise ValueError(f"branch {new!r} already exists")
        b = self._branches.pop(old)
        b.name = new
        self._branches[new] = b
        self._save()
        return b

    def delete(self, name: str) -> Branch:
        if name == MAIN_BRANCH:
            raise ValueError(f"cannot delete {MAIN_BRANCH!r}")
        if name not in self._branches:
            raise KeyError(f"branch {name!r} does not exist")
        b = self._branches.pop(name)
        self._save()
        return b

    # ── internal ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            self._branches = {}
            return
        try:
            with self.path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt sidecar is recoverable — log to stderr, start fresh.
            import sys
            sys.stderr.write(
                f"WARNING: branches.json at {self.path} is malformed; starting fresh.\n"
            )
            self._branches = {}
            return

        branches: dict[str, Branch] = {}
        for raw in data.get("branches", []):
            try:
                branches[raw["name"]] = Branch(
                    name=raw["name"],
                    fork_seq=int(raw["fork_seq"]),
                    tip_seq=int(raw["tip_seq"]),
                    created_at=raw["created_at"],
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed entries rather than crash the whole load.
                continue
        self._branches = branches

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "branches": [asdict(b) for b in self.list()],
        }
        # Atomic write: tmp file in the same directory, fsync, rename.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".branches-", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise


def _is_valid_name(name: str) -> bool:
    """Branch names must avoid characters that would be awkward in CLI
    arguments or file paths. 1–64 chars of [A-Za-z0-9_-]."""
    if not name or len(name) > 64:
        return False
    return all(c.isalnum() or c in "-_" for c in name)

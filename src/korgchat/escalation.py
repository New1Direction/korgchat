"""Escalation harvest — the second compounding loop.

When the pay gate **ESCALATEs** (genuine ambiguity goldseel couldn't resolve,
or the model was unreachable), the case is logged here. A human later resolves
it (approve / reject + why). Resolved escalations export in goldseel's training
format and feed the next retrain — so the cases the model *couldn't* handle
become the cases it *learns*.

Two loops make the system compound:
  * the ontology (`korgchat.ontology`) compounds **knowledge** — known
    recipients resolve deterministically, the known set grows.
  * this log compounds **judgment** — ambiguous cases a human had to judge
    become labeled data, so the model needs the human less over time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EscalationLog:
    """A JSONL log of escalations and their human resolutions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._entries: list[dict[str, Any]] = []
        if self.path.exists():
            self._entries = [
                json.loads(line) for line in self.path.read_text().splitlines() if line.strip()
            ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(json.dumps(e) + "\n" for e in self._entries))

    def record(
        self,
        *,
        intent: str,
        mandate_summary: dict[str, Any],
        redemption: dict[str, Any],
        reason: str,
        amount_usd: float,
        recipient: str,
        mandate_hash: str,
    ) -> str:
        """Log an escalation as pending. Idempotent on (intent, redemption, amount)."""
        eid = hashlib.sha256(
            json.dumps([intent, redemption, amount_usd], sort_keys=True).encode()
        ).hexdigest()[:12]
        if any(e["id"] == eid and e["status"] == "pending" for e in self._entries):
            return eid  # already pending; don't duplicate
        self._entries.append(
            {
                "id": eid,
                "status": "pending",
                "case": {
                    "intent": intent,
                    "mandate_summary": mandate_summary,
                    "redemption": redemption,
                },
                "context": {
                    "amount_usd": amount_usd,
                    "recipient": recipient,
                    "mandate_hash": mandate_hash,
                    "reason": reason,
                },
                "resolution": None,
            }
        )
        self._save()
        return eid

    def pending(self) -> list[dict[str, Any]]:
        return [e for e in self._entries if e["status"] == "pending"]

    def resolve(self, eid: str, verdict: str, reasoning: str = "", by: str = "human") -> dict[str, Any]:
        """Record a human's resolution of a pending escalation."""
        if verdict not in ("approve", "reject"):
            raise ValueError("resolve verdict must be 'approve' or 'reject'")
        for e in self._entries:
            if e["id"] == eid and e["status"] == "pending":
                e["status"] = "resolved"
                e["resolution"] = {"verdict": verdict, "reasoning": reasoning, "by": by}
                self._save()
                return e
        raise KeyError(f"no pending escalation {eid!r}")

    def export_training_cases(self) -> list[dict[str, Any]]:
        """Resolved escalations as goldseel training cases (the harvest)."""
        cases = []
        for e in self._entries:
            if e["status"] == "resolved" and e["resolution"]:
                c = e["case"]
                res = e["resolution"]
                cases.append(
                    {
                        "intent": c["intent"],
                        "mandate_summary": c["mandate_summary"],
                        "redemption": c["redemption"],
                        "expected_verdict": res["verdict"],
                        "expected_reasoning": res["reasoning"]
                        or f"Human-resolved escalation: {res['verdict']}.",
                        "_archetype": "harvested-escalation",
                    }
                )
        return cases

    def write_training_jsonl(self, path: str | Path) -> int:
        """Write the harvest to a jsonl ready to merge into the next training set."""
        cases = self.export_training_cases()
        Path(path).write_text("".join(json.dumps(c) + "\n" for c in cases))
        return len(cases)

    def stats(self) -> dict[str, int]:
        pending = sum(1 for e in self._entries if e["status"] == "pending")
        resolved = len(self._entries) - pending
        return {"total": len(self._entries), "pending": pending, "resolved": resolved}

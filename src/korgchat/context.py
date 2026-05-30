"""Auto-context injection (v0.5.3) — local-first ambient memory.

Before every responder call (when `ChatSession.auto_context` is on), this
module runs a semantic `/recall` on the user's prompt, picks the most
relevant prior events, formats them as a preamble, and prepends to the
prompt the responder sees. The user_prompt event recorded in the journal
is the ORIGINAL prompt — auto-context lives in the responder request,
not in the audit log.

Tradeoffs:

* Threshold is **0.40** vs `/recall`'s 0.30. Auto-injection is more
  aggressive than user-typed search — every turn pays the cost — so the
  bar is higher.
* Top-N defaults to **3**. Beyond that, the preamble starts dominating
  the prompt budget and confuses the model with too many unrelated
  threads.
* Returns `None` when no event passes the threshold. The responder
  doesn't see a preamble at all in that case (no "based on prior
  context" noise when there isn't any).
* Excludes the just-recorded user_prompt itself from results (avoids
  echoing the prompt back).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from korgchat.recall import Match, RecallEngine

if TYPE_CHECKING:
    from korgchat.chat import ChatSession


DEFAULT_TOP_N = 3
DEFAULT_MIN_SCORE = 0.40
PREAMBLE_HEADER = "Relevant prior conversation (auto-recalled from this user's history):"
PREAMBLE_FOOTER = (
    "Use this context to inform your reply when relevant. "
    "Don't quote it back to the user verbatim."
)


@dataclass
class ContextInjection:
    """The result of an auto-context recall: the rendered preamble plus the
    structured matches that produced it.

    `build_context()` returns this so callers (ChatSession) can both inject
    the preamble into the responder request AND record a first-class
    `context_injection` ledger event capturing exactly which prior events
    (seq_id + score) were surfaced. Without the structured matches the
    injected context would stay a "ghost" — visible to the model but
    invisible to the audit log.
    """

    query: str               # the prompt that drove the recall
    preamble: str            # the formatted text prepended to the responder request
    matches: list[Match]     # the prior events that qualified, best-first
    top_n: int
    min_score: float
    mode: str                # which recall path actually ran ("semantic" / "substring")

    @property
    def match_count(self) -> int:
        return len(self.matches)

    def matches_as_records(self) -> list[dict]:
        """Flatten matches into JSON-serialisable dicts for the ledger
        event's `result.matches`. Deliberately minimal — seq_id, score,
        kind, timestamp — so a replay can re-point at the source events
        without duplicating their full bodies (those are already on disk)."""
        return [
            {
                "seq_id": m.seq_id,
                "score": round(float(m.score), 6),
                "kind": m.kind,
                "timestamp": m.timestamp,
            }
            for m in self.matches
        ]


@dataclass
class AutoContextEngine:
    """Build a "system context" preamble for a prompt by semantic recall.

    Stateful only in that it holds a long-lived `RecallEngine` (whose own
    state is the embedding cache). Safe to construct once per session.
    """

    session: "ChatSession"
    top_n: int = DEFAULT_TOP_N
    min_score: float = DEFAULT_MIN_SCORE
    _recall: RecallEngine = field(init=False)

    def __post_init__(self) -> None:
        # mode="auto" means: semantic if fastembed is installed, else
        # substring. We don't error out when fastembed is missing — auto
        # context just becomes substring-driven (much noisier; that's the
        # user's fault for asking).
        self._recall = RecallEngine(self.session.journal_path, mode="auto")

    def build_context(
        self, prompt: str, *, exclude_seqs: set[int] | None = None
    ) -> ContextInjection | None:
        """Run the recall, filter to qualifying matches, and return both the
        rendered preamble and the structured matches — or None if nothing
        passes the threshold.

        This is the primitive `build_preamble()` is built on; ChatSession
        uses it directly so it can record a `context_injection` ledger
        event alongside the (otherwise invisible) prompt augmentation.
        """
        if not prompt or not prompt.strip():
            return None

        # Over-fetch and filter — semantic recall's own 0.30 threshold lets
        # weak hits through that we want to drop for auto-injection.
        hits = self._recall.search(prompt, limit=self.top_n * 3)
        if not hits:
            return None

        excluded = exclude_seqs or set()
        qualified = [
            h for h in hits
            if h.score >= self.min_score and h.seq_id not in excluded
        ][: self.top_n]
        if not qualified:
            return None

        preamble = self._render_preamble(qualified)
        return ContextInjection(
            query=prompt,
            preamble=preamble,
            matches=qualified,
            top_n=self.top_n,
            min_score=self.min_score,
            mode=self._recall.last_mode,
        )

    def build_preamble(self, prompt: str, *, exclude_seqs: set[int] | None = None) -> str | None:
        """Return a formatted preamble string or None if no matches qualify.

        Thin wrapper over `build_context()` for callers that only need the
        text and don't care about the structured matches."""
        ctx = self.build_context(prompt, exclude_seqs=exclude_seqs)
        return ctx.preamble if ctx is not None else None

    @staticmethod
    def _render_preamble(matches: list[Match]) -> str:
        lines = [PREAMBLE_HEADER]
        for h in matches:
            short_ts = (h.timestamp or "")[:10]  # YYYY-MM-DD
            lines.append(
                f"  • [{h.kind} seq={h.seq_id} {short_ts} score={h.score:.2f}] "
                f"{h.snippet}"
            )
        lines.append("")
        lines.append(PREAMBLE_FOOTER)
        return "\n".join(lines)

    @property
    def mode(self) -> str:
        """Which path the underlying RecallEngine actually used most
        recently. Useful for the CLI to display 'semantic' vs 'substring'
        in the auto-context indicator."""
        return self._recall.last_mode

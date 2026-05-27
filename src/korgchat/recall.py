"""KorgChat /recall — local-first search over the Korg ledger.

The journal at `.korg/journal.json` is the single source of truth. This
module loads it on demand, walks the event array, and returns matching
turns ranked by (match count × recency).

v0.4.3 ships substring matching with AND-of-terms semantics — predictable,
no external dependencies, no embedding model. Semantic recall (embeddings
+ ANN search) is a v0.5.x follow-up; for the alpha we want zero magic so
it's obvious to the user what `/recall foo` will find.

Searchable fields per event tool_name:

  user_prompt    — args.prompt
  llm_inference  — result.text          (added in bridge v0.3.2)
  <tool name>    — args (any string in the dict) + result (any string)

Filters:

  kind=    "user_prompt" | "llm_inference" | "tool_call" | "<exact name>"
           ("tool_call" matches any tool_name that isn't user_prompt or
           llm_inference; an exact name matches that specific tool.)
  since=   timedelta — only events newer than now-since
  limit=   int (default 10)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from korgchat.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingCache,
    EmbeddingDependencyMissing,
    EmbeddingEngine,
    batch_cosine,
    text_for_event,
)


SNIPPET_CONTEXT_CHARS = 60

# Cosine threshold below which a "semantic" hit is dropped from results.
# 0.30 is loose-but-not-noisy with BAAI/bge-small-en-v1.5 — empirical;
# tighten in v0.5.3 if false-positive rate becomes an issue.
SEMANTIC_MIN_SCORE = 0.30

RecallMode = Literal["auto", "semantic", "substring"]


@dataclass
class Match:
    """One hit in a recall search."""

    seq_id: int
    timestamp: str            # ISO 8601 from event.timestamp
    kind: str                 # "user_prompt" | "llm_inference" | "<tool name>"
    snippet: str              # short window of text around the first hit
    score: float              # higher = better (count × recency bonus)
    raw_event: dict[str, Any] = field(default_factory=dict)


class RecallEngine:
    """Search over the events on disk in `journal_path`.

    A fresh engine reads the file once per .search() call — there's no
    persistent index for the substring path. Semantic mode keeps a
    `.korg/embeddings.json` sidecar updated incrementally so only new
    events need embedding on each call.
    """

    def __init__(
        self,
        journal_path: Path | str,
        *,
        mode: RecallMode = "auto",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.journal_path = Path(journal_path)
        self.mode: RecallMode = mode
        self.embedding_model = embedding_model
        self._engine: EmbeddingEngine | None = None
        self._cache: EmbeddingCache | None = None
        # last_mode reflects which path actually ran on the most recent
        # .search() — useful for callers (CLI) to label the output and
        # for tests asserting on auto-fallback behaviour.
        self.last_mode: str = mode

    # ── Public API ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        since: timedelta | None = None,
        limit: int = 10,
    ) -> list[Match]:
        if self.mode == "semantic":
            return self._search_semantic(query, kind=kind, since=since, limit=limit)
        if self.mode == "auto":
            try:
                return self._search_semantic(
                    query, kind=kind, since=since, limit=limit
                )
            except EmbeddingDependencyMissing:
                # Silent fallback — auto mode is "use the best available."
                # The CLI inspects last_mode for the output label.
                self.last_mode = "substring (fastembed not installed)"
                return self._search_substring(
                    query, kind=kind, since=since, limit=limit
                )
        return self._search_substring(query, kind=kind, since=since, limit=limit)

    # ── Substring path (v0.4.3) ───────────────────────────────────────

    def _search_substring(
        self,
        query: str,
        *,
        kind: str | None,
        since: timedelta | None,
        limit: int,
    ) -> list[Match]:
        self.last_mode = "substring"
        terms = _tokenize(query)
        if not terms:
            return []

        events = self._load_events()
        if not events:
            return []

        cutoff: datetime | None = None
        if since is not None:
            cutoff = datetime.now(tz=timezone.utc) - since

        results: list[Match] = []
        for e in events:
            tool_name = e.get("event", {}).get("tool_name", "")
            if not _kind_match(kind, tool_name):
                continue

            ts_str = e.get("event", {}).get("timestamp")
            ev_time = _parse_timestamp(ts_str)
            if cutoff is not None and ev_time is not None and ev_time < cutoff:
                continue

            haystack = _searchable_text(e)
            if not haystack:
                continue

            hits = _count_hits(haystack, terms)
            if hits == 0:
                continue

            snippet = _make_snippet(haystack, terms, SNIPPET_CONTEXT_CHARS)
            recency_bonus = _recency_bonus(ev_time)
            results.append(
                Match(
                    seq_id=int(e.get("seq_id", 0)),
                    timestamp=ts_str or "",
                    kind=tool_name,
                    snippet=snippet,
                    score=hits + recency_bonus,
                    raw_event=e,
                )
            )

        results.sort(key=lambda m: (-m.score, -m.seq_id))
        return results[:limit]

    # ── Semantic path (v0.5.2) ────────────────────────────────────────

    def _search_semantic(
        self,
        query: str,
        *,
        kind: str | None,
        since: timedelta | None,
        limit: int,
    ) -> list[Match]:
        """Cosine-rank events against the query embedding.

        Maintains `.korg/embeddings.json` next to the journal. On each call:
        load all events → compute embeddings for any new ones → save cache
        → embed query → cosine-rank → filter by kind/since/threshold → top N.
        """
        if not query.strip():
            return []
        events = self._load_events()
        if not events:
            self.last_mode = "semantic"
            return []

        # Lazy-build engine + cache on first semantic call.
        if self._engine is None:
            self._engine = EmbeddingEngine(model_name=self.embedding_model)
        if self._cache is None:
            self._cache = EmbeddingCache(
                self._embeddings_path(),
                model_name=self.embedding_model,
            )

        # Index events by seq for fast post-rank lookup.
        events_by_seq = {
            int(e["seq_id"]): e
            for e in events
            if isinstance(e.get("seq_id"), int)
        }
        all_seqs = sorted(events_by_seq.keys())

        # Embed any events that aren't in the cache yet.
        missing = self._cache.missing_seqs(all_seqs)
        if missing:
            texts = [text_for_event(events_by_seq[s]) for s in missing]
            vectors = self._engine.embed_texts(texts)
            self._cache.put_many(zip(missing, vectors))
            self._cache.save()

        # Embed the query.
        try:
            qvec = self._engine.embed_one(query)
        except EmbeddingDependencyMissing:
            raise  # propagate so auto mode can fall back
        if not qvec:
            return []

        # Rank everything.
        cache_seqs, cache_vectors = self._cache.all_vectors()
        scores = batch_cosine(qvec, cache_vectors)

        cutoff: datetime | None = None
        if since is not None:
            cutoff = datetime.now(tz=timezone.utc) - since

        candidates: list[Match] = []
        for seq, score in zip(cache_seqs, scores):
            if score < SEMANTIC_MIN_SCORE:
                continue
            ev = events_by_seq.get(seq)
            if ev is None:
                continue
            tool_name = ev.get("event", {}).get("tool_name", "")
            if not _kind_match(kind, tool_name):
                continue
            ts_str = ev.get("event", {}).get("timestamp")
            ev_time = _parse_timestamp(ts_str)
            if cutoff is not None and ev_time is not None and ev_time < cutoff:
                continue

            # Snippet for semantic mode is just the head of the searchable
            # text (no query terms to highlight). Keep the renderer
            # consistent with substring mode so the CLI table layout is
            # identical.
            haystack = _searchable_text(ev)
            snippet = haystack[: 2 * SNIPPET_CONTEXT_CHARS]
            if len(haystack) > 2 * SNIPPET_CONTEXT_CHARS:
                snippet += "…"

            candidates.append(
                Match(
                    seq_id=seq,
                    timestamp=ts_str or "",
                    kind=tool_name,
                    snippet=snippet,
                    score=float(score),
                    raw_event=ev,
                )
            )

        candidates.sort(key=lambda m: (-m.score, -m.seq_id))
        self.last_mode = "semantic"
        return candidates[:limit]

    def _embeddings_path(self) -> Path:
        return self.journal_path.parent / "embeddings.json"

    # ── Helpers ────────────────────────────────────────────────────────

    def _load_events(self) -> list[dict]:
        if not self.journal_path.exists():
            return []
        try:
            with self.journal_path.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            return []
        return []


# ── Module-level helpers (no class state needed) ────────────────────────


def _tokenize(query: str) -> list[str]:
    """Whitespace-split, lowercase, drop empties. AND semantics — every
    term must appear in the haystack for the event to match."""
    return [t for t in query.lower().split() if t]


def _searchable_text(event: dict) -> str:
    """Collect every string the user might want to grep across an event.

    For user_prompt: args.prompt
    For llm_inference: result.text (v0.3.2+)
    For tool_calls: every string in args + result (recursively)
    """
    body = event.get("event", {})
    tool_name = body.get("tool_name", "")
    parts: list[str] = []

    if tool_name == "user_prompt":
        prompt = body.get("args", {}).get("prompt")
        if isinstance(prompt, str):
            parts.append(prompt)
    elif tool_name == "llm_inference":
        text = body.get("result", {}).get("text")
        if isinstance(text, str):
            parts.append(text)
    else:
        # Arbitrary tool — scrape every string from args + result.
        _flatten_strings(body.get("args"), parts)
        _flatten_strings(body.get("result"), parts)

    # Always include tool_name itself so `/recall add` finds add tool calls
    # even when the args don't textually mention "add".
    parts.append(tool_name)
    return "\n".join(parts).lower()


def _flatten_strings(obj: Any, out: list[str]) -> None:
    """Recursive — appends every string leaf from a nested dict/list to `out`."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_strings(v, out)
    # Numbers/bools/None contribute nothing to text search.


def _count_hits(haystack: str, terms: Iterable[str]) -> int:
    """All terms must appear (AND); score = total occurrences of terms."""
    total = 0
    for t in terms:
        n = haystack.count(t)
        if n == 0:
            return 0  # AND semantics — bail early
        total += n
    return total


def _make_snippet(haystack: str, terms: Iterable[str], context: int) -> str:
    """Return a window around the first match of any term, ellipsised."""
    first = -1
    for t in terms:
        i = haystack.find(t)
        if i != -1 and (first == -1 or i < first):
            first = i
    if first == -1:
        return haystack[: 2 * context]

    start = max(0, first - context)
    end = min(len(haystack), first + context)
    snippet = haystack[start:end].replace("\n", " ").strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(haystack) else ""
    return f"{prefix}{snippet}{suffix}"


def _kind_match(filter_: str | None, tool_name: str) -> bool:
    if filter_ is None:
        return True
    if filter_ == "tool_call":
        # "anything that isn't a chat message"
        return tool_name not in ("user_prompt", "llm_inference")
    return tool_name == filter_


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # CapabilityEvent::AgentToolCall::timestamp is chrono::DateTime<Utc>
        # serialised as RFC3339 with a "Z" suffix.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _recency_bonus(ev_time: datetime | None) -> float:
    """A small (0.0–0.99) bonus that decays with age so ties favour
    newer events. Never large enough to flip a hit-count comparison."""
    if ev_time is None:
        return 0.0
    age = datetime.now(tz=timezone.utc) - ev_time
    days = max(age.total_seconds() / 86400.0, 0.0)
    # 1 / (1 + days) — 1.0 at age 0, 0.5 at 1 day, ~0.09 at 10 days.
    return 1.0 / (1.0 + days) * 0.99


# ── CLI rendering ──────────────────────────────────────────────────────


def format_matches(matches: list[Match], *, query: str) -> str:
    """Render a Match list as a multi-line, terminal-friendly summary."""
    if not matches:
        return f"[recall] no matches for {query!r}"
    lines = [f"[recall] {len(matches)} match(es) for {query!r}:"]
    for m in matches:
        short_ts = (m.timestamp or "")[:19].replace("T", " ")
        lines.append(
            f"  seq={m.seq_id:<4}  {short_ts:<19}  {m.kind:<14}  {m.snippet}"
        )
    return "\n".join(lines)

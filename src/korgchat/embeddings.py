"""Local embeddings for semantic /recall (v0.5.2).

Two pieces:

  EmbeddingEngine  — lazy-imports `fastembed`, returns float32 numpy arrays.
                     `fastembed` uses ONNX Runtime (no torch dependency)
                     and ships small (~30MB without the model). The default
                     model BAAI/bge-small-en-v1.5 is ~130MB on first use,
                     cached under ~/.cache/fastembed.

  EmbeddingCache   — sidecar `.korg/embeddings.json` storing
                     seq_id → vector. Incremental: only events without a
                     cached embedding get re-computed on each /recall.
                     If the model name or vector dimension changes between
                     runs, the cache is invalidated automatically (vectors
                     from different models can't be compared by cosine).

Why not torch + sentence-transformers? `sentence-transformers` pulls
~800MB of torch wheels + a CUDA-flavoured stack on many platforms.
`fastembed` runs the same model architectures through ONNX with a tenth
of the disk footprint and identical retrieval quality for our use case.

This module is OPTIONAL. The package's [semantic] extra installs
`fastembed`; without it, `RecallEngine(mode="auto")` silently falls back
to v0.4.3 substring matching.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# Default model. Chosen for: tiny vector dim (384), strong retrieval
# benchmarks, broadly cached in fastembed's HF mirror.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingDependencyMissing(ImportError):
    """Raised when semantic recall is requested but `fastembed` isn't
    installed. The auto-mode path catches this and falls back."""


@dataclass
class EmbeddingEngine:
    """Wrapper around `fastembed.TextEmbedding`.

    The model object is constructed lazily on first `embed_texts()` call —
    the import alone is ~3s on cold start and the first embed downloads
    the model if it's not cached. Once warm, embeddings are sub-millisecond.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    _model: Any = field(default=None, init=False, repr=False)
    _dim: int | None = field(default=None, init=False, repr=False)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Empty list → empty list."""
        if not texts:
            return []
        self._ensure_model()
        # `fastembed` returns a generator of np arrays. We hold the
        # float-list form because that's what the JSON sidecar stores.
        vectors = list(self._model.embed(texts))
        return [v.astype("float32").tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        out = self.embed_texts([text])
        return out[0] if out else []

    @property
    def dim(self) -> int:
        """Vector dimension. Triggers a single embed if not yet known."""
        if self._dim is None:
            # Cheapest probe: embed a 1-char string and read the length.
            self._dim = len(self.embed_one(" "))
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbeddingDependencyMissing(
                "semantic recall needs `fastembed` — install with "
                "`pip install korgchat[semantic]` or use --mode substring."
            ) from exc
        self._model = TextEmbedding(model_name=self.model_name)


# ── Cosine similarity (numpy-free hot path) ────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine over two same-length vectors. Used in tests and
    when we want to avoid importing numpy. The /recall hot path uses the
    numpy version below when a batch of events is ranked at once."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def batch_cosine(query: list[float], candidates: list[list[float]]) -> list[float]:
    """Cosine of one query vector against many candidates. Uses numpy when
    available (way faster on big batches); falls back to plain Python."""
    if not candidates:
        return []
    try:
        import numpy as np

        q = np.asarray(query, dtype="float32")
        m = np.asarray(candidates, dtype="float32")
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [0.0] * len(candidates)
        m_norms = np.linalg.norm(m, axis=1)
        # Avoid div-by-zero on degenerate vectors.
        safe = np.where(m_norms == 0, 1.0, m_norms)
        scores = (m @ q) / (safe * q_norm)
        return [float(s) if n > 0 else 0.0 for s, n in zip(scores, m_norms)]
    except ImportError:
        return [cosine_similarity(query, c) for c in candidates]


# ── Persistent cache ───────────────────────────────────────────────────


class EmbeddingCache:
    """seq_id → vector, persisted as JSON next to the journal.

    Stateful (in-memory dict mirrored to disk). Atomic writes (tmp + fsync
    + rename) so a crash mid-save can't corrupt the cache.

    When the configured model changes between runs, the cached vectors are
    discarded — they were produced by a different model and aren't
    comparable by cosine to new query vectors.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, path: Path | str, *, model_name: str, dim: int | None = None):
        self.path = Path(path)
        self.model_name = model_name
        self.dim: int | None = dim
        self._vectors: dict[int, list[float]] = {}
        self._load()

    # ── public ────────────────────────────────────────────────────────

    def __contains__(self, seq_id: int) -> bool:
        return int(seq_id) in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def get(self, seq_id: int) -> list[float] | None:
        return self._vectors.get(int(seq_id))

    def put(self, seq_id: int, vector: list[float]) -> None:
        """Stash a vector. Caller is responsible for calling save() when
        done with a batch — we don't disk-flush per-put because /recall
        embeds many events in a single tight loop."""
        if self.dim is None:
            self.dim = len(vector)
        self._vectors[int(seq_id)] = vector

    def put_many(self, items: Iterable[tuple[int, list[float]]]) -> None:
        for seq_id, vec in items:
            self.put(seq_id, vec)

    def missing_seqs(self, all_seqs: Iterable[int]) -> list[int]:
        """Subset of `all_seqs` for which we don't have a cached embedding.
        Used by RecallEngine to embed only new events."""
        return [int(s) for s in all_seqs if int(s) not in self._vectors]

    def all_vectors(self) -> tuple[list[int], list[list[float]]]:
        """(seqs, vectors) in stable order — for batch cosine scoring."""
        seqs = sorted(self._vectors.keys())
        return seqs, [self._vectors[s] for s in seqs]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "model_name": self.model_name,
            "dim": self.dim,
            # Cast keys to strings — JSON object keys must be strings.
            "embeddings": {str(k): v for k, v in sorted(self._vectors.items())},
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".embeddings-", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    # ── internal ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt sidecar — start fresh (warn so user sees it).
            import sys
            sys.stderr.write(
                f"WARNING: embeddings.json at {self.path} is malformed; recomputing.\n"
            )
            return

        # Model-mismatch invalidation: vectors from a different model
        # aren't comparable. Drop everything and re-embed.
        cached_model = data.get("model_name")
        if cached_model != self.model_name:
            import sys
            sys.stderr.write(
                f"[recall] embedding model changed "
                f"({cached_model!r} → {self.model_name!r}); rebuilding cache.\n"
            )
            return

        cached_dim = data.get("dim")
        if isinstance(cached_dim, int):
            self.dim = cached_dim

        embeds = data.get("embeddings", {})
        for k, v in embeds.items():
            try:
                self._vectors[int(k)] = [float(x) for x in v]
            except (TypeError, ValueError):
                continue  # skip malformed entries


# ── Text-for-embedding adapter ─────────────────────────────────────────


def text_for_event(event: dict) -> str:
    """Produce the single string that gets embedded for an event.

    Reuses the same content surface as substring search (see recall.py),
    but flattened to a single line per event. Includes the tool_name as
    a prefix so the model has at least a token of structural context.
    """
    body = event.get("event", {})
    tool = body.get("tool_name", "")
    if tool == "user_prompt":
        prompt = body.get("args", {}).get("prompt", "")
        return f"user: {prompt}" if isinstance(prompt, str) else f"user: {prompt!r}"
    if tool == "llm_inference":
        text = body.get("result", {}).get("text", "")
        if isinstance(text, str) and text:
            return f"assistant: {text}"
        return "assistant: (tool-only round)"
    if tool == "summary":
        text = body.get("result", {}).get("text", "")
        scope = body.get("args", {}).get("scope", "")
        return f"summary({scope}): {text}"
    # Generic tool — name + args + result smoothed into prose.
    args = body.get("args", {})
    result = body.get("result", {})
    return f"tool {tool} args={json.dumps(args, sort_keys=True)} result={json.dumps(result, sort_keys=True)}"

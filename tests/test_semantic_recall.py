"""Tests for v0.5.2 semantic /recall.

Two layers:
  1. EmbeddingCache unit tests — load/save/invalidate.
  2. End-to-end semantic ranking with the real fastembed model.

The fastembed import + first model load is ~3s + a one-time download.
Subsequent calls are fast (<10ms per text). We gate the semantic path
on `pytest.importorskip("fastembed")` so the suite still runs on
machines without the optional dep installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from korgchat import ChatSession, MockResponder, RecallEngine, Reply
from korgchat.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingCache,
    EmbeddingDependencyMissing,
    EmbeddingEngine,
    batch_cosine,
    cosine_similarity,
    text_for_event,
)


# ── Cosine helpers ────────────────────────────────────────────────────


def test_cosine_self_is_one():
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # mismatched dims


def test_batch_cosine_matches_pairwise_python():
    q = [1.0, 0.0, 0.0]
    cands = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]
    scores = batch_cosine(q, cands)
    expected = [cosine_similarity(q, c) for c in cands]
    for s, e in zip(scores, expected):
        assert s == pytest.approx(e, abs=1e-5)


def test_batch_cosine_empty_candidates():
    assert batch_cosine([1.0, 0.0], []) == []


# ── text_for_event ────────────────────────────────────────────────────


def test_text_for_event_user_prompt():
    ev = {
        "seq_id": 1,
        "event": {
            "tool_name": "user_prompt",
            "args": {"prompt": "explain rust lifetimes"},
        },
    }
    assert "rust lifetimes" in text_for_event(ev)
    assert text_for_event(ev).startswith("user:")


def test_text_for_event_llm_inference_with_text():
    ev = {
        "seq_id": 2,
        "event": {
            "tool_name": "llm_inference",
            "result": {"text": "the borrow checker prevents data races"},
        },
    }
    assert "borrow checker" in text_for_event(ev)
    assert text_for_event(ev).startswith("assistant:")


def test_text_for_event_llm_inference_tool_only():
    ev = {
        "seq_id": 2,
        "event": {"tool_name": "llm_inference", "result": {"completion_tokens": 0}},
    }
    assert "tool-only" in text_for_event(ev)


def test_text_for_event_generic_tool():
    ev = {
        "seq_id": 3,
        "event": {
            "tool_name": "add",
            "args": {"a": 1, "b": 2},
            "result": {"sum": 3},
        },
    }
    out = text_for_event(ev)
    assert out.startswith("tool add ")
    assert '"a"' in out
    assert "sum" in out


def test_text_for_event_summary():
    ev = {
        "seq_id": 5,
        "event": {
            "tool_name": "summary",
            "args": {"scope": "branch 'main' (8 events)"},
            "result": {"text": "We discussed lifetimes and then ran add()."},
        },
    }
    out = text_for_event(ev)
    assert out.startswith("summary(")
    assert "lifetimes" in out


# ── EmbeddingCache ────────────────────────────────────────────────────


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "embeddings.json"


def test_cache_empty_when_file_missing(cache_path):
    c = EmbeddingCache(cache_path, model_name="m")
    assert len(c) == 0
    assert c.get(1) is None
    assert 1 not in c


def test_cache_put_and_get(cache_path):
    c = EmbeddingCache(cache_path, model_name="m")
    c.put(1, [0.1, 0.2, 0.3])
    c.put(2, [0.4, 0.5, 0.6])
    assert len(c) == 2
    assert c.get(1) == [0.1, 0.2, 0.3]
    assert c.dim == 3


def test_cache_save_load_round_trip(cache_path):
    c = EmbeddingCache(cache_path, model_name="m")
    c.put(1, [0.1, 0.2, 0.3])
    c.put(2, [0.4, 0.5, 0.6])
    c.save()

    c2 = EmbeddingCache(cache_path, model_name="m")
    assert sorted(c2._vectors.keys()) == [1, 2]
    assert c2.get(2) == [0.4, 0.5, 0.6]
    assert c2.dim == 3


def test_cache_invalidates_on_model_change(cache_path, capsys):
    c = EmbeddingCache(cache_path, model_name="model-A")
    c.put(1, [0.1, 0.2, 0.3])
    c.save()

    # Reopen with a different model name — cache should drop.
    c2 = EmbeddingCache(cache_path, model_name="model-B")
    assert len(c2) == 0
    err = capsys.readouterr().err
    assert "embedding model changed" in err


def test_cache_missing_seqs_computes_diff(cache_path):
    c = EmbeddingCache(cache_path, model_name="m")
    c.put(1, [0.0])
    c.put(3, [0.0])
    assert c.missing_seqs([1, 2, 3, 4]) == [2, 4]


def test_cache_corrupt_file_warns_and_starts_empty(cache_path, capsys):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{ not valid json")
    c = EmbeddingCache(cache_path, model_name="m")
    assert len(c) == 0
    err = capsys.readouterr().err
    assert "malformed" in err


def test_cache_atomic_save_leaves_no_tmp(cache_path, tmp_path):
    c = EmbeddingCache(cache_path, model_name="m")
    c.put(1, [0.1])
    c.save()
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# ── EmbeddingDependencyMissing fallback ───────────────────────────────


def test_engine_raises_when_fastembed_missing(monkeypatch):
    """Auto-mode catches this; semantic-mode lets it propagate."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    eng = EmbeddingEngine()
    with pytest.raises(EmbeddingDependencyMissing):
        eng.embed_one("hello")


# ── Real semantic ranking (gated on fastembed availability) ──────────


@pytest.fixture(scope="module")
def fastembed_available():
    pytest.importorskip("fastembed")
    return True


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "journal.json"


def test_semantic_ranks_concept_above_unrelated(tmp_journal, fastembed_available):
    """The headline promise: queries match by meaning, not just keywords."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("how does the borrow checker prevent data races")
    s.send("python decorators wrap functions")
    s.send("rust ownership and lifetimes")
    s.send("css flexbox alignment tricks")

    eng = RecallEngine(tmp_journal, mode="semantic")
    hits = eng.search("confused about borrowing in rust", limit=4)

    # All four user_prompts should be considered (some may fall below the
    # 0.30 threshold and be filtered). Whatever survives, the rust topics
    # must outrank the CSS one.
    user_hits = [h for h in hits if h.kind == "user_prompt"]
    rust_seqs = {1, 5}  # turn 1 (borrow checker) + turn 3 (ownership/lifetimes)
    css_seq = 7         # turn 4 (CSS flexbox)
    rust_scores = [h.score for h in user_hits if h.seq_id in rust_seqs]
    css_scores = [h.score for h in user_hits if h.seq_id == css_seq]
    assert rust_scores, "expected at least one rust-topic hit"
    # If CSS made it past the threshold, every rust score should still beat it.
    if css_scores:
        assert max(rust_scores) > max(css_scores)


def test_semantic_caches_embeddings_incrementally(tmp_journal, fastembed_available):
    """First call embeds everything; second call sees the cache and only
    embeds new events."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("first")
    s.send("second")

    eng = RecallEngine(tmp_journal, mode="semantic")
    eng.search("anything")
    cache_path = tmp_journal.parent / "embeddings.json"
    assert cache_path.exists()
    first_cache = json.loads(cache_path.read_text())
    assert len(first_cache["embeddings"]) == 4  # 2 turns × 2 events

    # Add a turn; same engine should only embed the new event.
    s.send("third")
    # New engine on the same dir picks up the existing cache.
    eng2 = RecallEngine(tmp_journal, mode="semantic")
    eng2.search("anything")
    second_cache = json.loads(cache_path.read_text())
    assert len(second_cache["embeddings"]) == 6


def test_semantic_last_mode_is_set(tmp_journal, fastembed_available):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("hi")
    eng = RecallEngine(tmp_journal, mode="semantic")
    eng.search("hello")
    assert eng.last_mode == "semantic"


def test_auto_mode_falls_back_to_substring_when_dep_missing(
    tmp_journal, monkeypatch
):
    """If fastembed isn't importable, auto mode silently uses substring."""
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("the rust borrow checker is strict")

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    eng = RecallEngine(tmp_journal, mode="auto")
    hits = eng.search("rust")
    assert "substring" in eng.last_mode
    # And substring did find the hit on "rust".
    assert hits and any(h.kind == "user_prompt" for h in hits)


def test_semantic_respects_kind_filter(tmp_journal, fastembed_available):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    s.send("tell me about rust lifetimes")

    eng = RecallEngine(tmp_journal, mode="semantic")
    hits = eng.search("ownership", kind="user_prompt")
    assert all(h.kind == "user_prompt" for h in hits)


def test_semantic_respects_limit(tmp_journal, fastembed_available):
    s = ChatSession(journal_path=tmp_journal, responder=MockResponder())
    for i in range(6):
        s.send(f"turn {i} about rust ownership and borrowing")

    eng = RecallEngine(tmp_journal, mode="semantic")
    hits = eng.search("rust", limit=3)
    assert len(hits) <= 3


# ── CLI integration ───────────────────────────────────────────────────


import io
from korgchat.__main__ import main as cli_main


def test_cli_recall_mode_substring_flag_works(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "tell me about rust\n"
            "/recall --mode substring rust\n"
            "/quit\n"
        ),
    )
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[recall · substring]" in out


def test_cli_recall_mode_semantic_flag_works(tmp_journal, monkeypatch, capsys, fastembed_available):
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "rust ownership and lifetimes\n"
            "/recall --mode semantic borrowing\n"
            "/quit\n"
        ),
    )
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[recall · semantic]" in out


def test_cli_recall_bad_mode_rejected(tmp_journal, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            "warmup\n"
            "/recall --mode neuro-fuzzy nonsense\n"
            "/quit\n"
        ),
    )
    rc = cli_main(["--mock", "--journal", str(tmp_journal), "--stream-delay", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "bad --mode" in out

"""Recipient-category ontology — the gate's deterministic knowledge floor.

A controlled vocabulary of recipient categories with **synonyms** and an
**is-a hierarchy**, plus a **vendor registry** (domain/name -> categories). The
gate resolves a payment against the mandate's allowed/denied category *classes*
here FIRST; only genuinely unknown recipients fall through to the model. This
is what makes `ml-inference ≡ ai-inference ≡ llm-inference` (all *is-a*
`ai-compute`) a structural fact instead of something a 3B model has to
re-derive — and occasionally get wrong.

**It compounds.** `learn()` writes a newly-classified recipient back into the
registry (optionally persisted to disk), so the known set grows monotonically:
the more decisions the system makes, the fewer reach the (fallible, pay-per-
call) model, and the more consistent it gets. That's a data network effect —
each decision makes the next one cheaper and surer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── is-a hierarchy: leaf category -> parent class ──────────────────────────
HIERARCHY: dict[str, str] = {
    # ai-compute
    "ml-inference": "ai-compute",
    "llm-inference": "ai-compute",
    "gpu-compute": "ai-compute",
    "training-compute": "ai-compute",
    "fine-tuning": "ai-compute",
    "embeddings": "ai-compute",
    "vector-db": "ai-compute",
    # infra
    "cloud-hosting": "infra",
    "object-storage": "infra",
    "cdn": "infra",
    "bandwidth": "infra",
    "serverless": "infra",
    # data
    "data-api": "data",
    "market-data": "data",
    "search-api": "data",
    "web-scraping": "data",
    # software / saas (neutral)
    "saas": "software",
    "api-credits": "software",
    "dev-tools": "software",
    # prohibited (default-deny class)
    "gambling": "prohibited",
    "adult": "prohibited",
    "crypto-trading": "prohibited",
    "weapons": "prohibited",
    "drugs": "prohibited",
    "darknet": "prohibited",
}

# ── synonyms -> canonical leaf category ────────────────────────────────────
SYNONYMS: dict[str, str] = {
    "ai-inference": "ml-inference",
    "ai inference": "ml-inference",
    "inference": "ml-inference",
    "model-inference": "ml-inference",
    "ai-compute-inference": "ml-inference",
    "llm": "llm-inference",
    "llm-api": "llm-inference",
    "gpu": "gpu-compute",
    "gpu-rental": "gpu-compute",
    "compute": "gpu-compute",
    "training": "training-compute",
    "finetuning": "fine-tuning",
    "embedding": "embeddings",
    "vectors": "vector-db",
    "vector-database": "vector-db",
    "hosting": "cloud-hosting",
    "vps": "cloud-hosting",
    "storage": "object-storage",
    "blob-storage": "object-storage",
    "lambda": "serverless",
    "casino": "gambling",
    "betting": "gambling",
    "sportsbook": "gambling",
    "poker": "gambling",
    "porn": "adult",
    "nsfw": "adult",
    "crypto": "crypto-trading",
    "trading": "crypto-trading",
    "exchange": "crypto-trading",
    "defi": "crypto-trading",
}

# ── seed vendor registry: domain/name (lowercased) -> categories ───────────
SEED_VENDORS: dict[str, list[str]] = {
    "api.openai.com": ["ml-inference"],
    "openai": ["ml-inference"],
    "api.anthropic.com": ["ml-inference"],
    "anthropic": ["ml-inference"],
    "api.x.ai": ["ml-inference"],
    "replicate.com": ["ml-inference"],
    "api.together.xyz": ["ml-inference"],
    "huggingface.co": ["ml-inference", "embeddings"],
    "vast.ai": ["gpu-compute"],
    "runpod.io": ["gpu-compute"],
    "lambdalabs.com": ["gpu-compute"],
    "modal.com": ["serverless", "gpu-compute"],
    "aws.amazon.com": ["cloud-hosting"],
    "cloud.google.com": ["cloud-hosting"],
    "cloudflare.com": ["cdn", "cloud-hosting"],
    "vercel.com": ["serverless", "cloud-hosting"],
    "pinecone.io": ["vector-db"],
    "bet365.com": ["gambling"],
    "draftkings.com": ["gambling"],
    "binance.com": ["crypto-trading"],
    "coinbase.com": ["crypto-trading"],
    "kraken.com": ["crypto-trading"],
    "pornhub.com": ["adult"],
}


def canonical(category: str) -> str:
    """Resolve a category string to its canonical leaf via synonyms."""
    c = str(category).lower().strip()
    return SYNONYMS.get(c, c)


def parent_class(category: str) -> str:
    """The is-a parent class of a category (or itself if it's already a class)."""
    c = canonical(category)
    return HIERARCHY.get(c, c)


def expand(category: str) -> set[str]:
    """A category plus its parent class — the set to test membership against."""
    c = canonical(category)
    return {c, parent_class(c)}


class CategoryOntology:
    """Deterministic recipient classification + the learning loop.

    Pass ``store_path`` to persist learned vendor mappings across runs (the
    compounding knowledge base). Without it, learning is in-memory only.
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.vendors: dict[str, list[str]] = {k: list(v) for k, v in SEED_VENDORS.items()}
        self.store_path = Path(store_path) if store_path else None
        self.learned_count = 0
        if self.store_path and self.store_path.exists():
            try:
                learned = json.loads(self.store_path.read_text())
                for k, v in learned.items():
                    self.vendors[k.lower()] = sorted(set(self.vendors.get(k.lower(), [])) | set(v))
            except (json.JSONDecodeError, OSError):
                pass

    # ── classification ────────────────────────────────────────────────────
    def categories_for(self, redemption: dict[str, Any]) -> set[str]:
        """Canonical categories for a redemption: explicit ones + registry lookup."""
        cats = {canonical(c) for c in (redemption.get("recipient_categories") or [])}
        for key in (redemption.get("recipient_domain"), redemption.get("recipient_name")):
            if key and str(key).lower() in self.vendors:
                cats.update(canonical(c) for c in self.vendors[str(key).lower()])
        return cats

    def resolve(
        self,
        redemption: dict[str, Any],
        *,
        allow: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> dict[str, Any]:
        """Deterministic floor verdict against allowed/denied category classes.

        ``allow`` / ``deny`` may contain leaf categories or parent classes.
        Returns floor ``DENY`` / ``ALLOW`` / ``UNKNOWN`` (defer to the model).
        """
        allow_set = {canonical(a) for a in (allow or [])}
        deny_set = {canonical(d) for d in (deny or [])}
        cats = self.categories_for(redemption)
        if not cats:
            return {"floor": "UNKNOWN", "reasons": ["recipient not classified by ontology"], "categories": []}

        denied = sorted(c for c in cats if expand(c) & deny_set)
        if denied:
            return {
                "floor": "DENY",
                "reasons": [f"category {denied} is in a denied class"],
                "categories": sorted(cats),
            }
        if allow_set and all(expand(c) & allow_set for c in cats):
            return {
                "floor": "ALLOW",
                "reasons": [f"all categories {sorted(cats)} within allowed classes"],
                "categories": sorted(cats),
            }
        return {
            "floor": "UNKNOWN",
            "reasons": ["category not clearly within allowed or denied classes"],
            "categories": sorted(cats),
        }

    # ── the compounding loop ──────────────────────────────────────────────
    def learn(self, redemption: dict[str, Any], categories: list[str]) -> str | None:
        """Remember a recipient's categories so future payments resolve deterministically.

        Caches the *classification* (domain -> categories), never a verdict —
        the verdict is always re-derived from the mandate. Returns the key
        learned, or None if there was nothing to key on.
        """
        key = (redemption.get("recipient_domain") or redemption.get("recipient_name") or "")
        key = str(key).lower().strip()
        cats = sorted({canonical(c) for c in categories if c})
        if not key or not cats:
            return None
        merged = sorted(set(self.vendors.get(key, [])) | set(cats))
        if merged != self.vendors.get(key):
            self.vendors[key] = merged
            self.learned_count += 1
            self._persist()
        return key

    def _persist(self) -> None:
        if not self.store_path:
            return
        # persist only the delta vs the seed (the learned knowledge)
        learned = {k: v for k, v in self.vendors.items() if v != SEED_VENDORS.get(k)}
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(learned, indent=1, sort_keys=True))
        except OSError:
            pass

    def stats(self) -> dict[str, int]:
        return {"vendors_known": len(self.vendors), "learned_this_session": self.learned_count}

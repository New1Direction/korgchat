"""Recipient-category ontology: the deterministic knowledge floor + learning loop."""

import json

from korgchat.ontology import CategoryOntology, canonical, expand, parent_class


def test_synonyms_and_hierarchy():
    assert canonical("ai-inference") == "ml-inference"
    assert canonical("AI Inference") == "ml-inference"
    assert canonical("casino") == "gambling"
    assert parent_class("ml-inference") == "ai-compute"
    assert parent_class("ai-inference") == "ai-compute"  # resolved via synonym first
    assert parent_class("gambling") == "prohibited"
    assert "ai-compute" in expand("llm-inference")


def test_resolve_allow_deny_unknown():
    ont = CategoryOntology()
    allow, deny = ["ai-compute"], ["prohibited"]

    # known-good vendor (registry) -> ALLOW
    assert ont.resolve({"recipient_domain": "api.openai.com"}, allow=allow, deny=deny)["floor"] == "ALLOW"
    # synonym category resolves to an allowed class
    assert ont.resolve({"recipient_categories": ["ai-inference"]}, allow=allow, deny=deny)["floor"] == "ALLOW"
    # prohibited class -> DENY (even via synonym)
    assert ont.resolve({"recipient_categories": ["casino"]}, allow=allow, deny=deny)["floor"] == "DENY"
    # unknown vendor, no category -> defer to the model
    assert ont.resolve({"recipient_domain": "mystery.io"}, allow=allow, deny=deny)["floor"] == "UNKNOWN"
    # a known category outside the allowed (and not denied) classes -> defer
    assert ont.resolve({"recipient_categories": ["market-data"]}, allow=allow, deny=deny)["floor"] == "UNKNOWN"


def test_learn_grows_registry_and_persists(tmp_path):
    store = tmp_path / "ontology.json"
    ont = CategoryOntology(store_path=store)
    before = ont.stats()["vendors_known"]

    key = ont.learn({"recipient_domain": "newvendor.io"}, ["gpu-compute"])
    assert key == "newvendor.io"
    assert ont.stats()["vendors_known"] == before + 1
    # now resolves deterministically
    assert ont.resolve({"recipient_domain": "newvendor.io"}, allow=["ai-compute"])["floor"] == "ALLOW"

    # persisted -> a fresh ontology pointed at the same store reloads it
    reloaded = CategoryOntology(store_path=store)
    assert "newvendor.io" in reloaded.vendors
    assert reloaded.resolve({"recipient_domain": "newvendor.io"}, allow=["ai-compute"])["floor"] == "ALLOW"


def test_persist_stores_only_the_learned_delta(tmp_path):
    store = tmp_path / "ontology.json"
    ont = CategoryOntology(store_path=store)
    # re-learning a seed vendor with the same categories is a no-op (no growth, not persisted)
    ont.learn({"recipient_domain": "api.openai.com"}, ["ml-inference"])
    ont.learn({"recipient_domain": "brandnew.ai"}, ["ml-inference"])
    if store.exists():
        learned = json.loads(store.read_text())
        assert "api.openai.com" not in learned  # unchanged seed isn't written
        assert learned.get("brandnew.ai") == ["ml-inference"]

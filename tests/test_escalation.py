"""Escalation harvest — the second compounding loop."""

import pytest

from korgchat.escalation import EscalationLog
from korgchat.gate import goldseel_pay_tool, payment_mandate


class FakeGate:
    def __init__(self, verdict):
        self.verdict = verdict

    def evaluate(self, *a):
        return {"verdict": self.verdict, "reasoning": "ambiguous, defer to a human"}


def _record(log, dom="mystery.io"):
    return log.record(
        intent="AI inference only",
        mandate_summary={"spend_cap_remaining": "50.00 USDC"},
        redemption={"recipient_domain": dom, "amount_usdc": "12.00", "recipient_categories": None},
        reason="ambiguous",
        amount_usd=12,
        recipient=dom,
        mandate_hash="abc",
    )


def test_record_resolve_export_persist(tmp_path):
    log = EscalationLog(tmp_path / "esc.jsonl")
    eid = _record(log)
    assert log.stats() == {"total": 1, "pending": 1, "resolved": 0}

    # idempotent: same case doesn't double-log while pending
    assert _record(log) == eid
    assert log.stats()["total"] == 1

    log.resolve(eid, "approve", "human verified it is a legit inference vendor")
    assert log.stats() == {"total": 1, "pending": 0, "resolved": 1}

    cases = log.export_training_cases()
    assert len(cases) == 1
    c = cases[0]
    assert c["expected_verdict"] == "approve"
    assert c["intent"] == "AI inference only"
    assert c["_archetype"] == "harvested-escalation"
    assert "human verified" in c["expected_reasoning"]

    # persistence round-trip
    reloaded = EscalationLog(tmp_path / "esc.jsonl")
    assert reloaded.stats()["resolved"] == 1
    assert reloaded.export_training_cases()[0]["expected_verdict"] == "approve"


def test_resolve_rejects_bad_verdict(tmp_path):
    log = EscalationLog(tmp_path / "e.jsonl")
    eid = _record(log)
    with pytest.raises(ValueError):
        log.resolve(eid, "maybe")


def test_pay_tool_logs_escalation_then_harvests(tmp_path):
    log = EscalationLog(tmp_path / "esc.jsonl")
    # unknown recipient (no category) + a gate that escalates -> ESCALATE -> logged
    tool = goldseel_pay_tool(
        payment_mandate("AI inference only.", 50), gate=FakeGate("escalate"), escalation_log=log
    )
    r = tool.call(
        {"amount_usd": 9, "recipient_domain": "ambiguous-vendor.io", "resource_description": "general services"}
    )
    assert r["decision"] == "ESCALATE"
    assert r["escalation_id"]
    assert log.stats()["pending"] == 1

    # a human resolves it -> it becomes a labeled training case for the next retrain
    log.resolve(r["escalation_id"], "reject", "not actually an AI vendor")
    harvest = log.export_training_cases()
    assert len(harvest) == 1
    assert harvest[0]["expected_verdict"] == "reject"
    # carries goldseel's dollar-format inputs, ready to train on
    assert harvest[0]["redemption"]["amount_usdc"] == "9.00"


def test_unreachable_model_also_escalates_and_logs(tmp_path):
    log = EscalationLog(tmp_path / "esc.jsonl")
    tool = goldseel_pay_tool(
        payment_mandate("AI inference only.", 50), gate=FakeGate("skip"), escalation_log=log
    )
    r = tool.call({"amount_usd": 5, "recipient_domain": "down-vendor.io"})
    assert r["decision"] == "ESCALATE"  # model unreachable -> defer, never auto-approve
    assert log.stats()["pending"] == 1

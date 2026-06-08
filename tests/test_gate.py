"""goldseel-gated `pay` tool.

The gate is injectable, so these run offline with a fake judge. The live
endpoint test is opt-in (set ``KORGCHAT_GOLDSEEL_LIVE=1``).
"""

import json
import os

import pytest

from korgchat.chat import ChatSession, MockResponder, Reply, ToolUse
from korgchat.gate import GoldseelGate, goldseel_pay_tool, payment_mandate
from korgchat.tools import default_tools

INTENT = "Pay only for AI inference / GPU compute. No gambling, adult, or crypto-trading."


class FakeGate:
    def __init__(self, verdict: str, reasoning: str = "fake") -> None:
        self.verdict = verdict
        self.reasoning = reasoning
        self.calls: list = []

    def evaluate(self, intent, mandate_summary, redemption):
        self.calls.append((intent, mandate_summary, redemption))
        return {"verdict": self.verdict, "reasoning": self.reasoning}


def _tool(verdict, cap=50.0):
    gate = FakeGate(verdict)
    return goldseel_pay_tool(payment_mandate(INTENT, cap), gate=gate), gate


def test_accept_within_cap_and_approved():
    tool, gate = _tool("approve", cap=50)
    r = tool.call(
        {"amount_usd": 12, "recipient_name": "OpenAI", "recipient_categories": ["ml-inference"]}
    )
    assert r["decision"] == "ACCEPT"
    assert r["settled"] is True
    assert r["remaining_after"] == 38.0
    assert r["mandate_hash"]
    assert len(gate.calls) == 1
    assert gate.calls[0][1]["spend_cap_remaining"] == 50.0  # goldseel saw the right cap


def test_remaining_cap_decrements_across_calls():
    tool, _ = _tool("approve", cap=50)
    tool.call({"amount_usd": 30, "recipient_name": "OpenAI"})
    r2 = tool.call({"amount_usd": 25, "recipient_name": "OpenAI"})  # 30+25 > 50
    assert r2["decision"] == "REJECT"
    assert any("cap" in x for x in r2["reasons"])


def test_reject_when_goldseel_rejects():
    tool, _ = _tool("reject")
    # an UNKNOWN recipient (no ontology match) actually reaches the model
    r = tool.call({"amount_usd": 10, "recipient_domain": "unknownshop.io"})
    assert r["decision"] == "REJECT"
    assert r["decided_by"] == "goldseel"
    assert r["settled"] is False
    assert any("goldseel" in x for x in r["reasons"])


def test_over_cap_does_not_spend_a_goldseel_call():
    tool, gate = _tool("approve", cap=50)
    r = tool.call({"amount_usd": 100, "recipient_name": "OpenAI"})
    assert r["decision"] == "REJECT"
    assert gate.calls == []  # deterministic floor short-circuited the paid call


def test_escalate_when_goldseel_unreachable():
    tool, _ = _tool("skip")  # "skip" == unreachable
    r = tool.call({"amount_usd": 10, "recipient_name": "OpenAI"})
    assert r["decision"] == "ESCALATE"
    assert r["settled"] is False


def test_pay_decision_is_recorded_in_the_ledger(tmp_path):
    jpath = str(tmp_path / "journal.json")
    registry = default_tools()
    registry.register(goldseel_pay_tool(payment_mandate(INTENT, 50), gate=FakeGate("approve")))
    responder = MockResponder(
        replies=[
            Reply(
                tool_uses=[
                    ToolUse(
                        id="p1",
                        name="pay",
                        input={
                            "amount_usd": 12,
                            "recipient_name": "OpenAI",
                            "recipient_categories": ["ml-inference"],
                        },
                    )
                ]
            ),
            Reply(text="paid"),
        ]
    )
    ChatSession(journal_path=jpath, responder=responder, tools=registry).send(
        "pay OpenAI $12 for inference"
    )
    events = json.load(open(jpath))
    pay_event = next(
        e
        for e in events
        if e["event"]["tool_name"] == "pay" and "decision" in (e["event"].get("result") or {})
    )
    result = pay_event["event"]["result"]
    assert result["decision"] == "ACCEPT"
    assert result["mandate_hash"]
    assert result["goldseel"]["verdict"] == "approve"


def _ont_tool(verdict, cap=100.0):
    gate = FakeGate(verdict)
    mandate = payment_mandate(INTENT, cap, allow_classes=["ai-compute"], deny_classes=["prohibited"])
    return goldseel_pay_tool(mandate, gate=gate), gate


def test_ontology_allow_bypasses_the_model():
    # the model would (wrongly) reject — but a known-good recipient never reaches it
    tool, gate = _ont_tool("reject")
    r = tool.call(
        {"amount_usd": 12, "recipient_domain": "api.openai.com", "recipient_categories": ["ml-inference"]}
    )
    assert r["decision"] == "ACCEPT"
    assert r["decided_by"] == "ontology"
    assert r["floor"] == "ALLOW"
    assert gate.calls == []  # false-reject is structurally impossible here


def test_ontology_deny_bypasses_the_model():
    # the model would (wrongly) approve — but a prohibited recipient is blocked first
    tool, gate = _ont_tool("approve")
    r = tool.call(
        {"amount_usd": 10, "recipient_domain": "bet365.com", "recipient_categories": ["gambling"]}
    )
    assert r["decision"] == "REJECT"
    assert r["decided_by"] == "ontology"
    assert r["floor"] == "DENY"
    assert gate.calls == []


def test_unknown_recipient_consults_the_model():
    tool, gate = _ont_tool("approve")
    r = tool.call({"amount_usd": 5, "recipient_domain": "mystery.io"})
    assert r["decided_by"] == "goldseel"
    assert len(gate.calls) == 1


def test_compounding_learns_a_new_vendor():
    tool, gate = _ont_tool("approve")
    r1 = tool.call({"amount_usd": 5, "recipient_domain": "newgpu.io", "recipient_categories": ["gpu-compute"]})
    assert r1["decided_by"] == "ontology"
    assert r1["learned"] == "newgpu.io"
    # second call has NO explicit category — resolved from the learned registry
    r2 = tool.call({"amount_usd": 5, "recipient_domain": "newgpu.io"})
    assert r2["decided_by"] == "ontology"
    assert gate.calls == []  # the model was never needed


@pytest.mark.skipif(
    os.environ.get("KORGCHAT_GOLDSEEL_LIVE") != "1",
    reason="set KORGCHAT_GOLDSEEL_LIVE=1 to hit the live goldseel Modal endpoint",
)
def test_live_goldseel_endpoint():
    gate = GoldseelGate()
    verdict = gate.evaluate(
        INTENT,
        {"spend_cap_remaining": 80, "use_counter_remaining": None, "expiry": None, "recipient_policy": None},
        {
            "recipient_name": "Bet365",
            "recipient_domain": "bet365.com",
            "recipient_categories": ["gambling"],
            "vendor_known": True,
            "amount_usdc": 20_000_000,
            "resource_description": "casino deposit",
        },
    )
    assert verdict["verdict"] in ("approve", "reject", "skip")

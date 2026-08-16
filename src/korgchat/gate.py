"""goldseel-gated ``pay`` tool for KorgChat.

goldseel is an owned mandate-enforcement model (served on Modal, serverless):
given an INTENT (what the human authorized), a MANDATE_SUMMARY (spend cap,
expiry, recipient policy), and a REDEMPTION (the proposed payment), it returns
``approve`` / ``reject``. The ``pay`` tool consults it *before* authorizing a
payment, layered over a deterministic spend-cap floor, and maps the outcome to
a three-way decision:

  * **REJECT**   — a deterministic check failed, or goldseel rejected it.
  * **ESCALATE** — goldseel was unreachable (defer to a human; fail-safe, never
                   auto-approve when the owned model is down).
  * **ACCEPT**   — within cap and goldseel approved.

The decision + the goldseel verdict + the mandate hash are returned in the tool
result, which KorgChat hash-chains into the korg ledger — so *what an agent was
allowed to spend, and why,* is provable.

Settlement itself is out of scope here (the x402 on-chain path lives in the
quaestor demo); ``pay`` records the authorization decision and, on ACCEPT,
marks a simulated settlement.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from .escalation import EscalationLog
from .ontology import CategoryOntology
from .tools import Tool

# goldseel on Modal (serverless, native shape: {intent, mandate_summary, redemption} -> {verdict, reasoning}).
DEFAULT_GOLDSEEL_URL = "https://kabukich0--goldseel-endpoint-goldseel-evaluate.modal.run"


class Gate(Protocol):
    """Anything that can judge a redemption against a mandate."""

    def evaluate(
        self, intent: str, mandate_summary: dict[str, Any], redemption: dict[str, Any]
    ) -> dict[str, Any]: ...


class GoldseelGate:
    """Calls the goldseel Modal endpoint (native shape). Never raises.

    Returns ``{"verdict": "approve"|"reject"|"skip", "reasoning": str}``;
    ``"skip"`` means goldseel was unreachable (the caller should escalate).
    """

    def __init__(self, url: str | None = None, *, timeout: float = 35.0) -> None:
        self.url = url or os.environ.get("GOLDSEEL_URL") or DEFAULT_GOLDSEEL_URL
        self.timeout = timeout

    def evaluate(
        self, intent: str, mandate_summary: dict[str, Any], redemption: dict[str, Any]
    ) -> dict[str, Any]:
        body = json.dumps(
            {"intent": intent, "mandate_summary": mandate_summary, "redemption": redemption}
        ).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            return {"verdict": "skip", "reasoning": f"goldseel unreachable: {e}"}
        verdict = str(data.get("verdict", "")).lower()
        if "escalate" in verdict:
            normalized = "escalate"
        elif "approve" in verdict:
            normalized = "approve"
        elif "reject" in verdict:
            normalized = "reject"
        else:
            normalized = "skip"
        return {"verdict": normalized, "reasoning": str(data.get("reasoning", ""))}


def payment_mandate(
    intent: str,
    spend_cap_usd: float,
    *,
    allow_classes: list[str] | None = None,
    deny_classes: list[str] | None = None,
    recipient_policy: Any = None,
) -> dict[str, Any]:
    """Build a payment mandate.

    ``allow_classes`` / ``deny_classes`` are ontology category classes (e.g.
    ``["ai-compute"]`` / ``["prohibited"]``) used by the deterministic floor.
    ``deny_classes`` defaults to ``["prohibited"]``. ``intent`` is the free-text
    purpose used by goldseel for recipients the ontology can't resolve.
    """
    return {
        "intent": intent,
        "spend_cap_usd": float(spend_cap_usd),
        "allow_classes": list(allow_classes) if allow_classes else [],
        "deny_classes": list(deny_classes) if deny_classes is not None else ["prohibited"],
        "recipient_policy": recipient_policy,
    }


def _mandate_hash(mandate: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(mandate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def goldseel_pay_tool(
    mandate: dict[str, Any],
    *,
    gate: Gate | None = None,
    ontology: CategoryOntology | None = None,
    escalation_log: EscalationLog | None = None,
    learn: bool = True,
    name: str = "pay",
    simulate: bool = True,
) -> Tool:
    """A ``pay`` :class:`~korgchat.tools.Tool` gated by the ontology + goldseel.

    ``mandate`` is a :func:`payment_mandate`. The deterministic category
    ontology resolves known recipients (ALLOW/DENY) without a model call; only
    genuine unknowns reach goldseel. With ``learn=True`` newly-classified
    recipients are written back to the ontology so future calls resolve
    deterministically — the system compounds. The tool tracks remaining cap
    across calls within the session.
    """
    judge: Gate = gate or GoldseelGate()
    ont: CategoryOntology = ontology or CategoryOntology()
    mandate_hash = _mandate_hash(mandate)
    state = {"remaining": float(mandate["spend_cap_usd"])}

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        amount = args.get("amount_usd")
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("pay: 'amount_usd' must be a positive number")
        amount = float(amount)
        recipient = args.get("recipient_name") or args.get("recipient_domain") or "unknown"

        redemption = {
            "recipient_domain": args.get("recipient_domain"),
            "recipient_name": args.get("recipient_name"),
            "recipient_categories": args.get("recipient_categories"),
            "vendor_known": bool(args.get("recipient_name") or args.get("recipient_domain")),
            "amount_usdc": int(round(amount * 1e6)),
            "resource_description": args.get("resource_description"),
        }
        # the dollar-denominated view goldseel was trained on
        gs_redemption = {**redemption, "amount_usdc": f"{amount:.2f}"}
        gs_summary: dict[str, Any] | None = None

        reasons: list[str] = []
        decided_by = None
        floor = {"floor": "SKIPPED", "categories": []}
        verdict = {"verdict": "skip", "reasoning": "not evaluated"}
        escalate = False

        # 1. deterministic spend-cap floor
        if amount > state["remaining"]:
            reasons.append(
                f"amount ${amount:.2f} exceeds remaining mandate cap ${state['remaining']:.2f}"
            )
            decided_by = "deterministic-cap"

        # 2. ontology floor — known recipients resolve without a model call
        if not reasons:
            floor = ont.resolve(
                redemption, allow=mandate.get("allow_classes"), deny=mandate.get("deny_classes")
            )
            if floor["floor"] == "DENY":
                reasons.append(f"ontology: {floor['reasons'][0]}")
                decided_by = "ontology"
            elif floor["floor"] == "ALLOW":
                decided_by = "ontology"  # deterministic accept — goldseel not consulted
            else:
                # 3. genuine unknown -> consult the owned model (pay-per-call).
                # goldseel was trained on DOLLAR-denominated amounts (e.g. "12.00"),
                # NOT on-chain micros — send the dollar view so it reads the cap right.
                gs_summary = {
                    # match goldseel's training distribution: dollar cap, a POSITIVE
                    # use-counter (None reads as "exhausted -> reject"), no expiry.
                    "spend_cap_remaining": f"{state['remaining']:.2f} USDC",
                    "use_counter_remaining": 999,
                    "expiry_iso": None,
                    "recipient_policy": mandate.get("recipient_policy") or "any",
                }
                verdict = judge.evaluate(mandate["intent"], gs_summary, gs_redemption)
                decided_by = "goldseel"
                if verdict["verdict"] == "reject":
                    reasons.append(f"goldseel: {verdict['reasoning']}")
                # model defers OR is unreachable -> defer to a human
                escalate = verdict["verdict"] in ("escalate", "skip")

        # compounding: cache any explicit classification so the next call is deterministic
        learned_key = None
        if learn and redemption.get("recipient_categories"):
            learned_key = ont.learn(redemption, redemption["recipient_categories"])

        decision = "REJECT" if reasons else ("ESCALATE" if escalate else "ACCEPT")
        settled = False
        if decision == "ACCEPT":
            state["remaining"] -= amount
            settled = simulate

        # harvest: log escalations so a human's later resolution becomes training data
        escalation_id = None
        if decision == "ESCALATE" and escalation_log is not None:
            escalation_id = escalation_log.record(
                intent=mandate["intent"],
                mandate_summary=gs_summary or {},
                redemption=gs_redemption,
                reason=verdict.get("reasoning") or "goldseel deferred or unreachable",
                amount_usd=amount,
                recipient=recipient,
                mandate_hash=mandate_hash,
            )

        return {
            "decision": decision,
            "amount_usd": amount,
            "recipient": recipient,
            "decided_by": decided_by,
            "floor": floor["floor"],
            "categories": floor.get("categories", []),
            "reasons": reasons,
            "goldseel": verdict,
            "remaining_after": round(state["remaining"], 6),
            "mandate_hash": mandate_hash,
            "learned": learned_key,
            "vendors_known": ont.stats()["vendors_known"],
            "escalation_id": escalation_id,
            "settled": settled,
            "settlement": "simulated" if settled else None,
        }

    return Tool(
        name=name,
        description=(
            "Authorize a payment, gated by the goldseel mandate-enforcement "
            "model. Returns a decision (ACCEPT / REJECT / ESCALATE) and the "
            "goldseel verdict. A deterministic spend-cap is enforced first; "
            "goldseel then judges the payment against the authorized intent. "
            "If goldseel is unreachable the payment ESCALATEs to a human "
            "(never auto-approved). The decision is recorded to the ledger."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "amount_usd": {"type": "number", "description": "Amount to pay in USD."},
                "recipient_name": {"type": "string", "description": "Payee name."},
                "recipient_domain": {"type": "string", "description": "Payee domain."},
                "recipient_categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Vendor categories, e.g. ['ml-inference'].",
                },
                "resource_description": {
                    "type": "string",
                    "description": "What the payment is for.",
                },
            },
            "required": ["amount_usd"],
        },
        handler=handler,
    )

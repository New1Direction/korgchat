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
        normalized = "approve" if "approve" in verdict else "reject" if "reject" in verdict else "skip"
        return {"verdict": normalized, "reasoning": str(data.get("reasoning", ""))}


def payment_mandate(
    intent: str, spend_cap_usd: float, *, recipient_policy: Any = None
) -> dict[str, Any]:
    """Build a payment mandate: the human-authorized intent + a spend cap."""
    return {
        "intent": intent,
        "spend_cap_usd": float(spend_cap_usd),
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
    name: str = "pay",
    simulate: bool = True,
) -> Tool:
    """A ``pay`` :class:`~korgchat.tools.Tool` gated by goldseel.

    ``mandate`` is a :func:`payment_mandate`. The tool tracks remaining cap
    across calls within the session.
    """
    judge: Gate = gate or GoldseelGate()
    mandate_hash = _mandate_hash(mandate)
    state = {"remaining": float(mandate["spend_cap_usd"])}

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        amount = args.get("amount_usd")
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("pay: 'amount_usd' must be a positive number")
        amount = float(amount)
        recipient = args.get("recipient_name") or args.get("recipient_domain") or "unknown"

        reasons: list[str] = []
        # deterministic floor: never exceed the signed cap
        if amount > state["remaining"]:
            reasons.append(
                f"amount ${amount:.2f} exceeds remaining mandate cap ${state['remaining']:.2f}"
            )

        # Consult goldseel only if the deterministic floor passed — the owned
        # model is pay-per-call, so don't spend a call on an already-doomed pay.
        verdict = {"verdict": "skip", "reasoning": "not evaluated (deterministic reject)"}
        escalate = False
        if not reasons:
            mandate_summary = {
                "spend_cap_remaining": state["remaining"],
                "use_counter_remaining": None,
                "expiry": None,
                "recipient_policy": mandate.get("recipient_policy"),
            }
            redemption = {
                "recipient_domain": args.get("recipient_domain"),
                "recipient_name": args.get("recipient_name"),
                "recipient_categories": args.get("recipient_categories"),
                "vendor_known": bool(args.get("recipient_name") or args.get("recipient_domain")),
                "amount_usdc": int(round(amount * 1e6)),
                "resource_description": args.get("resource_description"),
            }
            verdict = judge.evaluate(mandate["intent"], mandate_summary, redemption)
            if verdict["verdict"] == "reject":
                reasons.append(f"goldseel: {verdict['reasoning']}")
            escalate = verdict["verdict"] == "skip"  # owned model down -> defer to a human

        decision = "REJECT" if reasons else ("ESCALATE" if escalate else "ACCEPT")
        settled = False
        if decision == "ACCEPT":
            state["remaining"] -= amount
            settled = simulate

        return {
            "decision": decision,
            "amount_usd": amount,
            "recipient": recipient,
            "reasons": reasons,
            "goldseel": verdict,
            "remaining_after": round(state["remaining"], 6),
            "mandate_hash": mandate_hash,
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

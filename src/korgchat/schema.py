"""Tool-schema hashing + conformance checking (task #7).

A recorded conversation is only replayable if you also know the *contract*
each tool call was made against. Tool schemas drift — a `required` field is
added, a type tightens — and once that happens an old journal's tool calls
become un-judgeable: you can see the args the model sent, but not the
`input_schema` it was shown. So KorgChat snapshots the schema (with a
content hash) before each call and validates the call against it after.

Two pieces live here:

  * `schema_hash(schema)` — a deterministic, canonical sha256 of a JSON
    Schema dict. Canonicalization matches korg-ledger@v1: JSON with sorted
    keys, compact `(",", ":")` separators, and non-ASCII `\\uXXXX`-escaped
    (`ensure_ascii=True`). Logically-identical schemas hash identically;
    any change to the schema changes the hash. That divergence is the
    signal a replay uses to detect "this tool's contract changed."

  * `validate_input(schema, args)` — a minimal JSON-Schema conformance
    check covering the subset KorgChat's tools actually use: `type:object`
    with `properties` (per-field `type`) and `required`. Returns a
    `ValidationResult` with a `valid` flag and human-readable `violations`.

We intentionally do NOT depend on the `jsonschema` package. KorgChat's
core keeps a deliberately thin dependency surface (anthropic + fastembed
are both optional, lazily imported); a tiny self-contained validator for
the handful of schema features in play is the better fit and keeps the
ledger path importable everywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# JSON Schema `type` keyword → the Python types that satisfy it.
# `bool` is excluded from "integer"/"number" because in JSON Schema a
# boolean is not a number, even though `bool` is an `int` subclass in
# Python — we special-case it below.
_JSON_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


def canonical_json(obj: Any) -> str:
    """Canonical JSON encoding per korg-ledger@v1: sorted keys, compact
    separators, non-ASCII escaped. Byte-for-byte stable across runs."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def schema_hash(schema: dict[str, Any]) -> str:
    """sha256 hex of the canonical JSON encoding of `schema`.

    Deterministic and key-order-invariant: re-snapshotting an unchanged
    schema yields the same hash; any change to the schema yields a
    different one.
    """
    return hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()


@dataclass
class ValidationResult:
    """Verdict from `validate_input`. `valid` is the headline; `violations`
    explains why not (empty when valid)."""

    valid: bool
    violations: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {"valid": self.valid, "violations": list(self.violations)}


def validate_input(schema: dict[str, Any], args: dict[str, Any]) -> ValidationResult:
    """Check `args` against a (subset of) JSON Schema `schema`.

    Supported keywords: top-level `type` (only meaningfully `object`),
    `properties` with per-field `type`, and `required`. Unknown keywords
    are ignored (lenient) rather than treated as violations — this is a
    conformance *audit*, not a strict gatekeeper, and we'd rather under-
    than over-report on schema features we don't model.
    """
    violations: list[str] = []

    if not isinstance(args, dict):
        return ValidationResult(
            valid=False,
            violations=[f"input must be an object, got {type(args).__name__}"],
        )

    # Required fields present?
    for key in schema.get("required", []) or []:
        if key not in args:
            violations.append(f"missing required field {key!r}")

    # Declared property types satisfied?
    props = schema.get("properties", {}) or {}
    for key, value in args.items():
        if key not in props:
            # Extra fields aren't a hard error here (additionalProperties is
            # not modelled); the snapshot already froze the declared shape.
            continue
        declared = props[key].get("type")
        if declared is None:
            continue
        if not _type_ok(declared, value):
            got = type(value).__name__
            violations.append(
                f"field {key!r} expected type {declared!r}, got {got}"
            )

    return ValidationResult(valid=not violations, violations=violations)


def _type_ok(declared: Any, value: Any) -> bool:
    """True if `value` satisfies the declared JSON Schema `type`. `declared`
    may be a single type name or a list of acceptable names (JSON Schema
    allows a `type` array — value matches if it satisfies any)."""
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        check = _JSON_TYPE_CHECKS.get(name)
        if check is None:
            # Unknown declared type → don't claim a violation we can't prove.
            return True
        if check(value):
            return True
    return False

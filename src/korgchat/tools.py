"""Built-in tool registry for KorgChat v0.4.1.

Three deterministic tools shipped by default:

  * `echo`     — return whatever input you got. Useful for testing the
                 round-trip and as a no-op the model can call to "show its
                 work" without side effects.
  * `add`      — add two numbers and return the sum. Deterministic numeric
                 op; safest possible non-trivial tool.
  * `get_time` — return the current Unix timestamp (or a frozen value
                 supplied at registry-construction time).

The registry is intentionally minimal — v0.4.1's goal is to prove the
tool-use *loop*, not to ship a tool library. Users embedding KorgChat as a
library can construct their own `ToolRegistry` with whatever tools their
product needs and pass it into `ChatSession`.

Each tool's `input_schema` is a JSON Schema dict that `AnthropicResponder`
forwards directly to the Anthropic API's `tools` parameter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A single callable the LLM can invoke."""

    name: str
    description: str
    # JSON Schema describing the `input` dict the tool expects.
    input_schema: dict[str, Any]
    # Python callable: receives the parsed input dict, returns a result.
    # Result must be JSON-serialisable.
    handler: Callable[[dict[str, Any]], Any]

    def call(self, args: dict[str, Any]) -> Any:
        return self.handler(args)

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Shape Anthropic's `messages.create(tools=[...])` accepts."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Maps tool name → Tool. Supports lookup, listing, and Anthropic-schema export."""

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_schema() for t in self._tools.values()]


# ── Builtins ───────────────────────────────────────────────────────────────


def _echo_handler(args: dict[str, Any]) -> dict[str, Any]:
    return {"echoed": args.get("input", "")}


def _add_handler(args: dict[str, Any]) -> dict[str, Any]:
    # Reject non-numeric inputs loudly so the model sees an actual error
    # instead of a silent default-to-zero.
    for k in ("a", "b"):
        if k not in args:
            raise ValueError(f"add: missing required input {k!r}")
        if not isinstance(args[k], (int, float)):
            raise TypeError(
                f"add: input {k!r} must be a number, got {type(args[k]).__name__}"
            )
    return {"sum": args["a"] + args["b"]}


def _get_time_handler_factory(frozen_value: float | None):
    """Closes over an optional frozen value so tests stay deterministic."""

    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        ts = frozen_value if frozen_value is not None else time.time()
        return {"unix_seconds": ts}

    return _handler


def default_tools(*, frozen_time: float | None = None) -> ToolRegistry:
    """Return a fresh registry pre-populated with the v0.4.1 builtin set.

    Pass `frozen_time` to get a deterministic `get_time` (useful in tests
    that want reproducible journals).
    """
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="Return whatever input you got. Useful for "
                "testing or for the model to show its work without side effects.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Any string."},
                    },
                    "required": ["input"],
                },
                handler=_echo_handler,
            ),
            Tool(
                name="add",
                description="Add two numbers and return their sum.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
                handler=_add_handler,
            ),
            Tool(
                name="get_time",
                description="Return the current Unix timestamp.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
                handler=_get_time_handler_factory(frozen_time),
            ),
        ]
    )

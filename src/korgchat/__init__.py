"""KorgChat — the first chat product built on the Korg cognitive ledger."""

from korgchat.chat import (
    AnthropicResponder,
    ChatSession,
    MockResponder,
    Reply,
    Responder,
    ToolCall,
    ToolResult,
    ToolUse,
    Turn,
    MAX_TOOL_USE_ITERATIONS,
)
from korgchat.tools import Tool, ToolRegistry, default_tools

__version__ = "0.4.1"

__all__ = [
    "__version__",
    # session + replies
    "ChatSession",
    "Reply",
    "Turn",
    "MAX_TOOL_USE_ITERATIONS",
    # responders
    "Responder",
    "MockResponder",
    "AnthropicResponder",
    # tool-use shapes
    "ToolUse",
    "ToolResult",
    "ToolCall",
    # tool registry
    "Tool",
    "ToolRegistry",
    "default_tools",
]

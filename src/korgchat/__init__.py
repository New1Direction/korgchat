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
from korgchat.branches import MAIN_BRANCH, Branch, BranchStore
from korgchat.recall import Match, RecallEngine, format_matches
from korgchat.tools import Tool, ToolRegistry, default_tools

__version__ = "0.5.0"

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
    # recall (v0.4.3)
    "Match",
    "RecallEngine",
    "format_matches",
    # branches (v0.5.0)
    "Branch",
    "BranchStore",
    "MAIN_BRANCH",
]

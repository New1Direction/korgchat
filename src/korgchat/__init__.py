"""KorgChat — the first chat product built on the Korg cognitive ledger."""

from korgchat.chat import ChatSession, MockResponder, AnthropicResponder, Responder

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "ChatSession",
    "MockResponder",
    "AnthropicResponder",
    "Responder",
]

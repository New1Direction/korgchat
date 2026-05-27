# KorgChat

The first chat product built on the Korg cognitive ledger.

Every turn of a KorgChat conversation is recorded as an `AgentToolCall` event
in the same `.korg/journal.json` that powers korgex's agent loop, the MCP
session browser, and the Ctrl-R rewind in korg-tui. The conversation is
written **synchronously, in-process** via the `korg_bridge` PyO3 extension —
no HTTP server required.

```
You: write a one-line haiku about ledgers
Korg: pages turn forward / each entry signed by the past / time leaves no escape

[recorded: turn 1, seq=1 (user) → seq=2 (assistant)]

You: another one
Korg: blockchain heart beating / consensus a slow drumline / merkle roots hold true

[recorded: turn 2, seq=3 (user) → seq=4 (assistant)]
```

After the conversation ends, the ledger can be:
- replayed with `korg-tui` and `Ctrl-R` to rewind
- served over MCP via `mcp_server.py` for browsing in other clients
- consumed by `korgex` as causal context for a follow-up coding task

## Install

```bash
# 1. Install korg-bridge from the Korg workspace
cd /path/to/Korg/crates/korg-bridge
maturin develop

# 2. Install korgchat
cd /path/to/KorgChat
pip install -e .

# 3. (optional) Anthropic SDK for live LLM mode
pip install -e .[anthropic]
export ANTHROPIC_API_KEY=sk-...
```

## Run

```bash
# Deterministic mock mode — no API key, no network call:
korgchat --mock

# Live mode (Anthropic):
korgchat

# Pick a custom journal location:
korgchat --journal ./my-conversation.json --mock
```

## Causal chain

KorgChat writes 2 events per turn:

```
turn 1:  seq=1  user_prompt          triggered_by=None
         seq=2  llm_inference        triggered_by=1
turn 2:  seq=3  user_prompt          triggered_by=2   ← chains to prior LLM round
         seq=4  llm_inference        triggered_by=3
```

The `user_prompt → llm_inference` shape mirrors what korgex emits, so a
single ledger can host both interactive chat and autonomous agent runs
without losing causal coherence.

## License

MIT.

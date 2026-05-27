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

KorgChat writes ≥2 events per turn (more when the model invokes tools):

```
turn 1 (text only):
  seq=1  user_prompt          triggered_by=None
  seq=2  llm_inference        triggered_by=1

turn 2 (with tool use):
  seq=3  user_prompt          triggered_by=2     ← chains to prior turn's LLM
  seq=4  llm_inference        triggered_by=3      (round 1: LLM asks for `add`)
  seq=5  add tool_call        triggered_by=4      (sibling under round-1 LLM)
  seq=6  llm_inference        triggered_by=4      (round 2: LLM answers,
                                                   chains to round-1 per spec §2a,
                                                   NOT to the tool call at seq=5)
```

The `user_prompt → llm_inference → tool_call*` shape mirrors what korgex
emits, so a single ledger can host both interactive chat and autonomous
agent runs without losing causal coherence.

## Tools (v0.4.1)

KorgChat ships three deterministic built-in tools:

| Name       | Input            | Output                  |
|------------|------------------|-------------------------|
| `echo`     | `{input: str}`   | `{echoed: str}`         |
| `add`      | `{a, b: number}` | `{sum: number}`         |
| `get_time` | `{}`             | `{unix_seconds: float}` |

In `--mock` mode you can trigger a tool deterministically with the marker
syntax `[tool:NAME(arg=value, ...)]` in your prompt:

```
You: please [tool:add(a=2, b=3)] for me
  🔧 [ok] add(a=2, b=3) → {"sum": 5}  (seq=3, 0ms)
Korg: toolu_… → {"sum": 5}
```

Embedding KorgChat as a library? Pass your own `ToolRegistry`:

```python
from korgchat import ChatSession, MockResponder, Tool, ToolRegistry

reg = ToolRegistry([
    Tool(name="read_file", description="...",
         input_schema={"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
         handler=lambda args: {"content": open(args["path"]).read()}),
])

session = ChatSession(
    journal_path=".korg/journal.json",
    responder=MockResponder(),
    tools=reg,
)
```

The safety cap `MAX_TOOL_USE_ITERATIONS` (8) terminates any model that
keeps requesting tools without ever returning text.

## License

MIT.

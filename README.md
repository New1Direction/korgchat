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

# Disable streaming (atomic per-turn print, v0.4.0 UX):
korgchat --mock --no-stream

# Tune mock-mode streaming speed (default 0.005s/char):
korgchat --mock --stream-delay 0.05      # slow & visible
korgchat --mock --stream-delay 0         # instant
```

## Streaming (v0.4.2)

By default, every assistant text reply streams to stdout character-by-character
as it's produced. With `AnthropicResponder` this uses the SDK's
`messages.stream()` and renders tokens as they arrive from the API; with
`MockResponder` it emits one character at a time with a configurable delay
so the streaming effect is visible offline.

The journal contract is unchanged: every LLM round still produces exactly
one `llm_inference` event containing the full reply text. Streaming is a
CLI/UX layer, not a protocol change.

Embedders can attach streaming callbacks to any `ChatSession`:

```python
chunks = []
session.on_round_start = lambda: print("\nKorg: ", end="", flush=True)
session.on_token       = lambda c: (chunks.append(c), print(c, end="", flush=True))
session.send("hello")
# chunks reconstructs the full text the model produced.
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

## Semantic recall — search by meaning, not just words (v0.5.2)

`/recall` now defaults to semantic mode when the optional `fastembed`
extra is installed. Queries match by concept: "confused about borrowing"
finds turns that *discuss* the rust borrow checker even if the literal
word "borrowing" doesn't appear.

```
You: /recall --mode substring borrowing
[recall] no matches for 'borrowing'

You: /recall --mode semantic borrowing
[recall · semantic] 8 match(es) for 'borrowing':
  seq=1  user_prompt     how does the borrow checker prevent data races
  seq=5  user_prompt     rust ownership and lifetimes
  ...
```

Setup (optional — without it, `/recall` falls back to v0.4.3 substring):

```bash
pip install korgchat[semantic]
```

First search downloads the embedding model (~130MB, cached under
`~/.cache/fastembed`). Subsequent embeds are sub-millisecond.

Modes:

| `--mode`     | Behavior                                                                  |
|--------------|---------------------------------------------------------------------------|
| `auto` (default) | Use semantic if `fastembed` is installed; otherwise fall back to substring. |
| `semantic`   | Embedding-backed cosine ranking. Raises if `fastembed` is missing.        |
| `substring`  | The v0.4.3 keyword path. AND-of-terms, case-insensitive.                  |

Output header shows which path ran: `[recall · semantic]` vs `[recall · substring]`.

Embeddings live in `.korg/embeddings.json` next to the journal. Only
new events get embedded on each `/recall` call (incremental). Changing
the model name invalidates the cache automatically — vectors from
different models aren't comparable by cosine.

## Summarize — ask the LLM to digest prior conversation (v0.5.1)

`/summarize` feeds a scoped slice of the journal back to the model and
prints a digest. The first feature where the ledger, search, and the
LLM work together.

```
You: tell me about rust ownership
Korg: ...
You: [tool:add(a=12, b=30)] please
  🔧 [ok] add(a=12, b=30) → {"sum": 42}
Korg: ...
You: /summarize
[summarize] branch 'main' (4 events)

Summary of branch 'main' (4 events). Saw 1 user prompt(s), 2 assistant
reply(ies), and 1 tool invocation(s). Conversation flowed without
notable interruptions or errors; no open threads detected.
```

| Form                              | Scope                                                |
|-----------------------------------|------------------------------------------------------|
| `/summarize`                      | The current branch (default)                         |
| `/summarize <branch>`             | A named branch or `main`                             |
| `/summarize --since DUR`          | Events from the last N (`7d`, `24h`, `30m`, …)       |
| `/summarize --topic Q`            | Events matching a /recall-style query                |
| `/summarize --limit N`            | Cap the events sent to the model (default 50)        |
| `/summarize --save`               | Also record the digest as a `summary` event in the journal so it's findable via `/recall` later |

The summary call is ephemeral by default — no event is written to the
journal unless you pass `--save`. With `AnthropicResponder` you get a
real prose digest; with `MockResponder` you get a structurally-honest
template (counts of user/assistant/tool events + scope label) so the
CLI experience stays useful offline.

## Branches — try different threads side-by-side (v0.5.0)

A conversation branch is a bookmark into the journal: a name + the seq
you forked from + the head you've grown it to. Switching between
branches resumes from the right point so you can explore alternatives
without losing the original thread.

```
You: tell me about rust ownership
Korg: ...
You: /fork rust-deep-dive
[fork] branch 'rust-deep-dive' created at seq=2, now active

You: rust lifetimes are tricky
Korg: ...
You: /checkout main
[checkout] now on 'main' (seq=6)

You: back to a different topic
Korg: ...
You: /branches
[branches] active = 'main'
  main             (trunk)  ← current
  rust-deep-dive   fork@2  tip@6  (2026-05-27 09:44:29)

You: /checkout rust-deep-dive
[checkout] now on 'rust-deep-dive' (seq=6)
  (next turn will chain triggered_by from seq=6; in-memory history cleared)
```

| Command                          | Effect                                                        |
|----------------------------------|---------------------------------------------------------------|
| `/branches`                      | List named branches with a `← current` marker on the active one |
| `/fork <name>`                   | Bookmark this point as a branch and switch to it              |
| `/checkout <name|main>`          | Resume from the named branch's tip; clears in-memory history   |
| `/branch-delete <name>`          | Drop a bookmark. The events themselves stay in the journal     |
| `/branch-rename <old> <new>`     | Rename a bookmark                                              |

How it works: branches are stored in `.korg/branches.json` next to the
journal. The journal events themselves are unchanged — they're still
chained via `triggered_by`. A branch is just a saved seq_id where the
next turn should chain from. The natural DAG that results means main
and a fork share ancestry up to the fork point, then diverge.

Names must be 1–64 chars of `[A-Za-z0-9_-]`. `main` is reserved for the
implicit trunk.

## Recall — search your conversations (v0.4.3)

Every event KorgChat writes — user prompts, model replies, tool calls, tool
results — is searchable from inside the chat itself. No cloud, no embedding
model, no SDK setup: substring-grep over the local journal, AND-of-terms.

```
You: tell me about rust ownership
Korg: ...

You: /recall rust
[recall] 2 match(es) for 'rust':
  seq=1     2026-05-27 09:04   user_prompt     tell me about rust ownership…
  seq=2     2026-05-27 09:04   llm_inference   …rust borrow checker tracks…

You: /recall --kind tool_call add
[recall] 1 match(es) for 'add':
  seq=7     2026-05-27 09:04   add             add

You: /recall --since 24h "borrow checker"
You: /recall --limit 3 ledger
```

Flags:

| Flag         | Effect                                                          |
|--------------|-----------------------------------------------------------------|
| `--kind K`   | Filter to `user_prompt` / `llm_inference` / `tool_call` / `<name>` |
| `--since DUR`| `30m`, `24h`, `7d`, `1.5h` — only events newer than that         |
| `--limit N`  | Cap returned matches (default 10)                                |

`/help` lists every slash command.

What's distinctive vs ChatGPT/Claude.ai sidebar search: this works against
the *complete* event log including model replies and tool calls (not just
prompt titles), uses an open file format (`.korg/journal.json`), and runs
entirely local-first. The same data is consumable by `korg-server` MCP
browsing and `korg-tui` Ctrl-R rewind without a second copy.

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

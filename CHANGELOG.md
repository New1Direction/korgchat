# Changelog

All notable changes to KorgChat are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `.gitignore` extended for transient HTML artifacts.

## [0.5.3] — 2026-05-27

### Added
- **Auto-context injection (ambient memory).** With `--auto-context`, every new prompt triggers a semantic `/recall` against the journal; the top relevant prior events get formatted as a preamble and prepended to the LLM's view of the prompt. The journal still records the **original** prompt — the augmented version lives only in the LLM request, never in the audit log. This closes the "ChatGPT memory but local, visible, and auditable" loop.
- `AutoContextEngine` with tunable `top_n` and `min_score`. Default `min_score=0.40` (stricter than `/recall`'s 0.30); default `top_n=3`.
- `on_context_injected` callback for embedders.

## [0.5.2] — 2026-05-27

### Added
- **Semantic `/recall`.** Queries now match by concept, not just keyword. "Confused about borrowing" finds turns that *discuss* the rust borrow checker even if the literal word "borrowing" doesn't appear.
- New optional `[semantic]` extra: `pip install korgchat[semantic]` pulls `fastembed` (ONNX, no torch dependency).
- Default embedding model: `BAAI/bge-small-en-v1.5` (384-dim, ~130MB cached under `~/.cache/fastembed`).
- `/recall --mode auto|semantic|substring`. Auto uses semantic if `fastembed` is installed, else falls back to substring.
- Embedding cache lives at `.korg/embeddings.json`; incremental on each `/recall`; invalidates automatically on model change.

## [0.5.1] — 2026-05-27

### Added
- **`/summarize` command.** Feeds a scoped slice of the journal back to the model and prints a digest. The first feature where the ledger, search, and the LLM work together.
- Scopes: `/summarize` (current branch), `/summarize <branch>`, `/summarize --since DUR`, `/summarize --topic <query>`, `/summarize --limit N`, `/summarize --save` (records the digest back into the journal as a `summary` event).
- `SUMMARIZE_PROMPT_MARKER` routes mock responder to a structurally-honest digest template offline.

## [0.5.0] — 2026-05-27

### Added
- **Conversation branches.** A branch is a bookmark — a saved seq_id where the next turn chains from. Switching branches resumes from the right point so you can explore alternatives without losing the original thread.
- Slash commands: `/branches`, `/fork <name>`, `/checkout <name|main>`, `/branch-delete <name>`, `/branch-rename <old> <new>`.
- Branches live in `.korg/branches.json` next to the journal; the journal events themselves are unchanged.
- `main` is reserved for the implicit trunk. Names must be 1–64 chars of `[A-Za-z0-9_-]`.
- Structured dry-run test of `AnthropicResponder` paths.

## [0.4.3] — 2026-05-27

### Added
- **`/recall` substring search inside the chat.** AND-of-terms, case-insensitive, against the local journal — no cloud, no embedding model, no SDK setup.
- Flags: `--kind` (filter by event type), `--since DUR` (newer than 30m / 24h / 7d / 1.5h), `--limit N` (default 10).
- Searches user prompts, model replies, tool calls, and tool results — not just titles.

## [0.4.2] — 2026-05-26

### Added
- **Streaming output.** Assistant text streams to stdout character-by-character as it's produced. `AnthropicResponder` uses the SDK's `messages.stream()`; `MockResponder` emits with a configurable delay so the effect is visible offline.
- `ChatSession.on_round_start` + `on_token` callbacks for embedders.

### Changed
- Journal contract unchanged: every LLM round still produces exactly one `llm_inference` event containing the full reply text. Streaming is a CLI/UX layer, not a protocol change.

## [0.4.1] — 2026-05-26

### Added
- **Tool use inside the chat.** Three deterministic built-in tools (`echo`, `add`, `get_time`) usable from any responder.
- In `--mock` mode, `[tool:NAME(arg=value, ...)]` marker syntax in the user prompt deterministically triggers tool calls.
- Embeddable `ToolRegistry` API for custom tools.
- Safety cap `MAX_TOOL_USE_ITERATIONS = 8` terminates any model stuck in a tool loop without text.

## [0.4.0] — 2026-05-26

### Added
- **Initial alpha release.** The first chat product on the Korg cognitive ledger.
- Every turn recorded as an `AgentToolCall` event in `.korg/journal.json` via the `korg_bridge` PyO3 extension — synchronous, in-process, no HTTP server required.
- The journal is browsable in `korg-tui` (Ctrl-R rewind), serveable over MCP, and consumable by `korgex` as causal context for follow-up agent runs.
- Modes: `--mock` (deterministic, no API key) and live Anthropic.
- Causal chain follows spec §2a: `llm_inference` triggered_by always points to the prior `llm_inference`, not at intervening tool calls.

[Unreleased]: https://github.com/New1Direction/korgchat/compare/v0.5.3...HEAD
[0.5.3]: https://github.com/New1Direction/korgchat/releases/tag/v0.5.3
[0.5.2]: https://github.com/New1Direction/korgchat/releases/tag/v0.5.2
[0.5.1]: https://github.com/New1Direction/korgchat/releases/tag/v0.5.1
[0.5.0]: https://github.com/New1Direction/korgchat/releases/tag/v0.5.0
[0.4.3]: https://github.com/New1Direction/korgchat/releases/tag/v0.4.3
[0.4.2]: https://github.com/New1Direction/korgchat/releases/tag/v0.4.2
[0.4.1]: https://github.com/New1Direction/korgchat/releases/tag/v0.4.1
[0.4.0]: https://github.com/New1Direction/korgchat/releases/tag/v0.4.0

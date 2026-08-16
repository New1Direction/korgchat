# Changelog

All notable changes to KorgChat are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Sandboxed `bash` tool with verifiable exec (`--sandbox`).** New `korgchat.sandbox` module adds a `bash` tool backed by [just-bash](https://github.com/vercel-labs/just-bash) — a JS reimplementation of bash + ~90 coreutils over an in-memory filesystem, run as a persistent Node sidecar. The shell physically cannot reach the host filesystem or network (no network/python/js by default). Every `exec` returns `fs_hash`, a hash of the full virtual-filesystem state after the command; because each tool call is hash-chained into the ledger, the agent's shell session is **tamper-evident and replayable** — the same commands from a fresh sandbox reproduce the same hashes. Exports `SandboxClient` (stdio JSON-RPC), `bash_tool()`, and `tools_with_sandbox()`; enable in the CLI with `--sandbox` (requires Node ≥18 and `npm install` in `sandbox/`).
- **Mandate-gated shell (`--mandate-allow`).** The sandboxed `bash` tool can be constrained to a command allowlist. Enforced two ways: just-bash only registers the allowed commands (physical), and each line is parsed before exec so a disallowed or dynamically-named command (`$CMD`) is rejected — **fail-closed**. Every call carries a verdict (`{decision, reasons, commands_used, mandate_hash}`) recorded to the ledger, so what the agent was *allowed* to run is itself provable. New `shell_mandate(allow, deny)`; `SandboxClient(mandate=...)` / `.configure()`; `tools_with_sandbox(mandate=...)`; CLI `--mandate-allow ls,cat,grep,...`.
- **goldseel-gated `pay` tool.** New `korgchat.gate` module: a `pay` tool that authorizes a payment through the owned **goldseel** mandate-enforcement model (served on Modal, serverless). A deterministic spend-cap floor runs first — and short-circuits the pay-per-call model on an over-cap payment; goldseel then judges the payment against the authorized intent. Maps to a three-way decision: **REJECT** (cap or goldseel), **ESCALATE** (goldseel unreachable → defer to a human, *never* auto-approve), **ACCEPT** (within cap and approved). The decision + verdict + mandate hash are recorded to the ledger, so what an agent was allowed to spend is provable. New `GoldseelGate` (injectable), `payment_mandate()`, `goldseel_pay_tool()`. Offline tests use a fake judge; a live endpoint test is opt-in (`KORGCHAT_GOLDSEEL_LIVE=1`).
- **Recipient-category ontology — the gate's deterministic knowledge floor (`korgchat.ontology`).** A controlled vocabulary of recipient categories with **synonyms** and an **is-a hierarchy** (`ml-inference` ≡ `ai-inference` ≡ `llm-inference`, all *is-a* `ai-compute`; `gambling`/`adult`/`crypto-trading` *is-a* `prohibited`), plus a seeded **vendor registry**. The `pay` tool now resolves *known* recipients deterministically — **ALLOW/DENY without a model call** — and only genuine unknowns reach goldseel, making the `ml-inference ≠ ai-inference` false-reject *structurally impossible*. **It compounds:** `learn()` writes newly-classified recipients back to the registry (optionally persisted), so the known set grows monotonically — more decisions → fewer model calls → more consistent outcomes (a data network effect). The `pay` result records `decided_by` (ontology vs goldseel), the floor verdict, and what was learned. `payment_mandate()` gains `allow_classes` / `deny_classes` (default deny `["prohibited"]`).
- **Escalation harvest — the second compounding loop (`korgchat.escalation`).** When the pay gate **ESCALATEs** (goldseel defers, or is unreachable), the case is logged; a human resolves it (approve/reject + why); resolved escalations export in goldseel's training format and feed the *next* retrain. So the cases the model *couldn't* handle become the ones it *learns* — the ontology compounds **knowledge**, this compounds **judgment**. `EscalationLog` (`record` / `pending` / `resolve` / `export_training_cases` / `write_training_jsonl`); `goldseel_pay_tool(escalation_log=...)` logs on ESCALATE and returns an `escalation_id`; `GoldseelGate` now recognizes the model's `escalate` verdict (it was collapsing to `skip`).
- **Auto-context injection is now a first-class ledger event.** Previously the recall-augmented preamble the model actually saw was a *ghost* — the journal recorded only the user's original prompt. Now, whenever auto-context injects a preamble, a `context_injection` event is written capturing the preamble text, the recall query, and the matched `seq_id`s + scores, causally chained `user_prompt → context_injection → llm_inference`. The user_prompt event still records only what the user typed; the injected context is a separate, auditable, replayable event. New `AutoContextEngine.build_context()` returns a `ContextInjection` (preamble + structured matches); `build_preamble()` is now a thin wrapper over it.
- **Tool-schema snapshot + conformance events.** Every tool execution is now bracketed by two events: a `tool_schema_snapshot` *before* the call (the declared `input_schema`, `description`, and a deterministic `schema_hash`) and a `tool_validation` *after* (did the call's input conform to the declared schema? did the call succeed?). A replayed conversation stays meaningful even after a tool's schema changes — the contract it ran against is frozen on the ledger, and a stale call is detectable. New `korgchat.schema` module: `schema_hash()` (canonical sha256, byte-for-byte aligned with `korg-ledger@v1` canonicalization) and a dependency-free `validate_input()`.

### Changed
- `_render_event_line` in `/summarize` renders `context_injection`, `tool_schema_snapshot`, and `tool_validation` as distinct meta-event lines so digests don't miscount them as user-invoked tool calls.
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

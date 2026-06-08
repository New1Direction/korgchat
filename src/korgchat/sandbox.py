"""Sandboxed ``bash`` tool for KorgChat, backed by a just-bash sidecar.

The sidecar (``sandbox/sidecar.mjs``) is a persistent Node process running
`just-bash <https://github.com/vercel-labs/just-bash>`_ — a JS reimplementation
of bash + ~90 coreutils over an in-memory filesystem. It physically cannot
reach the host filesystem or network, so the model can run real shell commands
safely.

Every ``exec`` returns ``fs_hash`` — a hash of the full virtual-filesystem
state after the command. Because KorgChat records each tool call into the korg
ledger (hash-chained), embedding ``fs_hash`` in the tool result makes the
agent's shell session **tamper-evident and replayable**: the same commands
from a fresh sandbox reproduce the same hashes.

Usage::

    from korgchat import ChatSession
    from korgchat.sandbox import tools_with_sandbox

    session = ChatSession(journal_path=..., responder=..., tools=tools_with_sandbox())
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from .tools import Tool, ToolRegistry, default_tools

# sandbox.py lives at src/korgchat/sandbox.py; the sidecar sits at <root>/sandbox/.
_SIDECAR = Path(__file__).resolve().parent.parent.parent / "sandbox" / "sidecar.mjs"

DEFAULT_TIMEOUT_MS = 10_000


class SandboxError(RuntimeError):
    """Raised when the sandbox sidecar is unavailable or an exec fails to run."""


class SandboxClient:
    """Manages a persistent just-bash sidecar over stdio JSON-RPC.

    The Node process is spawned lazily on first use and reused, so the
    in-memory filesystem persists across commands within a session. Requests
    are serialized under a lock (one request, one response line).
    """

    def __init__(
        self,
        *,
        node: str | None = None,
        sidecar: Path | None = None,
        default_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        mandate: dict[str, Any] | None = None,
    ) -> None:
        self._node = node or shutil.which("node")
        self._sidecar = Path(sidecar) if sidecar else _SIDECAR
        self._default_timeout_ms = default_timeout_ms
        self._mandate = mandate
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._id = 0

    def _ensure(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if not self._node:
            raise SandboxError(
                "node not found on PATH; install Node >=18 to use the bash sandbox"
            )
        if not self._sidecar.exists():
            raise SandboxError(
                f"sandbox sidecar not found at {self._sidecar}; "
                f"run `npm install` in {self._sidecar.parent}"
            )
        self._proc = subprocess.Popen(
            [self._node, str(self._sidecar)],
            cwd=str(self._sidecar.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Apply the mandate to the fresh process before any exec (lock already held).
        if self._mandate is not None:
            self._id += 1
            self._send_recv(
                {"id": self._id, "op": "configure", "mandate": self._mandate}, self._proc
            )
        return self._proc

    def _send_recv(
        self, payload: dict[str, Any], proc: subprocess.Popen[str]
    ) -> dict[str, Any]:
        """Write one request + read one response line. Caller holds the lock."""
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (BrokenPipeError, ValueError) as e:  # pragma: no cover - process died
            raise SandboxError(f"sandbox sidecar communication failed: {e}") from e
        if not line:
            raise SandboxError("sandbox sidecar closed unexpectedly")
        return json.loads(line)

    def _rpc(self, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._ensure()
            self._id += 1
            return self._send_recv({"id": self._id, **req}, proc)

    def ping(self) -> dict[str, Any]:
        return self._rpc({"op": "ping"})

    def exec(
        self, command: str, *, timeout_ms: int | None = None, cwd: str | None = None
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "op": "exec",
            "cmd": command,
            "timeoutMs": timeout_ms or self._default_timeout_ms,
        }
        if cwd:
            req["cwd"] = cwd
        res = self._rpc(req)
        if not res.get("ok"):
            raise SandboxError(res.get("error", "sandbox exec failed"))
        return res

    def reset(self) -> dict[str, Any]:
        """Discard the virtual filesystem and start a fresh sandbox (keeps the mandate)."""
        return self._rpc({"op": "reset"})

    def configure(self, mandate: dict[str, Any] | None) -> dict[str, Any]:
        """Set the command mandate (``{"allow": [...], "deny": [...]}``).

        The allowlist is enforced physically (only those commands are
        registered in the sandbox) and as a pre-exec verdict. Resets the
        virtual filesystem.
        """
        self._mandate = mandate
        return self._rpc({"op": "configure", "mandate": mandate})

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def __enter__(self) -> "SandboxClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def bash_tool(client: SandboxClient | None = None, *, name: str = "bash") -> Tool:
    """A ``bash`` :class:`~korgchat.tools.Tool` backed by the just-bash sandbox.

    The result includes ``fs_hash`` (hash of the full virtual-FS state after the
    command), which KorgChat chains into the ledger — making the shell session
    replayable and tamper-evident.
    """
    sandbox = client or SandboxClient()

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("bash: 'command' must be a non-empty string")
        timeout_ms = args.get("timeout_ms")
        res = sandbox.exec(
            command,
            timeout_ms=timeout_ms if isinstance(timeout_ms, int) else None,
        )
        out: dict[str, Any] = {
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            "exit_code": res.get("exit_code"),
            "fs_hash": res.get("fs_hash"),
            "fs_files": res.get("fs_files"),
        }
        # Mandate verdict (present only when a mandate is configured).
        if "gate" in res:
            out["gate"] = res["gate"]
        if res.get("blocked"):
            out["blocked"] = True
        return out

    return Tool(
        name=name,
        description=(
            "Run a shell command in a sandboxed bash environment with an "
            "in-memory filesystem and NO host or network access. Supports "
            "standard unix commands (ls, cat, grep, sed, awk, find, sort, "
            "wc, jq, ...), pipes, redirects, variables, and loops. The "
            "filesystem persists across calls within a session. Returns "
            "stdout, stderr, exit_code, and fs_hash (a hash of the resulting "
            "filesystem state, used for verifiable replay)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional per-command timeout in milliseconds.",
                },
            },
            "required": ["command"],
        },
        handler=handler,
    )


def shell_mandate(allow: list[str], deny: list[str] | None = None) -> dict[str, Any]:
    """Build a shell mandate: an allowlist of commands (plus optional denylist).

    The allowlist is the security boundary — only those commands are registered
    in the sandbox. ``deny`` overrides ``allow`` for finer policy.
    """
    mandate: dict[str, Any] = {"allow": list(allow)}
    if deny:
        mandate["deny"] = list(deny)
    return mandate


def tools_with_sandbox(
    *,
    frozen_time: float | None = None,
    client: SandboxClient | None = None,
    mandate: dict[str, Any] | None = None,
) -> ToolRegistry:
    """The default builtins plus the sandboxed ``bash`` tool.

    Pass ``mandate`` (e.g. from :func:`shell_mandate`) to gate the shell to an
    allowlist of commands; the verdict for each call is carried in the tool
    result and recorded to the ledger.
    """
    if client is None:
        client = SandboxClient(mandate=mandate)
    elif mandate is not None:
        client.configure(mandate)
    registry = default_tools(frozen_time=frozen_time)
    registry.register(bash_tool(client))
    return registry

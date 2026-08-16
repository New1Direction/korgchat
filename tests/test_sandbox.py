"""Sandboxed `bash` tool + verifiable ledger-exec.

These tests need Node and the just-bash sidecar (run `npm install` in
``sandbox/``). They skip gracefully when that isn't available so the rest of
the suite stays green on a bare checkout.
"""

import json
import shutil

import pytest

from korgchat.chat import ChatSession, MockResponder, Reply, ToolUse
from korgchat.sandbox import SandboxClient, bash_tool, shell_mandate, tools_with_sandbox


def _sandbox_available() -> bool:
    if not shutil.which("node"):
        return False
    try:
        SandboxClient().ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sandbox_available(),
    reason="node + just-bash sandbox unavailable (run `npm install` in sandbox/)",
)


def test_exec_persists_and_hash_is_deterministic():
    sb = SandboxClient()
    try:
        run = bash_tool(sb).call
        r1 = run({"command": "echo persisted > /w.txt; cat /w.txt"})
        assert r1["stdout"] == "persisted\n"
        assert r1["exit_code"] == 0
        assert len(r1["fs_hash"]) == 64

        # the in-memory filesystem persists across tool calls within a session
        r2 = run({"command": "wc -c < /w.txt"})
        assert r2["stdout"].strip() == "10"
        assert r2["fs_hash"] == r1["fs_hash"]  # read-only: state unchanged

        # mutating then restoring state returns to the exact prior hash
        r3 = run({"command": "rm /w.txt"})
        assert r3["fs_hash"] != r1["fs_hash"]

        # replay from a fresh sandbox reproduces the same hash (deterministic)
        sb.reset()
        r4 = run({"command": "echo persisted > /w.txt; cat /w.txt"})
        assert r4["fs_hash"] == r1["fs_hash"]
    finally:
        sb.close()


def test_sandbox_cannot_reach_host_filesystem():
    sb = SandboxClient()
    try:
        # the host's real /etc/passwd must not be visible inside the sandbox
        out = bash_tool(sb).call({"command": "cat /etc/passwd 2>&1 || true"})
        assert "root:" not in out["stdout"]
    finally:
        sb.close()


def test_bash_exec_is_recorded_in_the_ledger_with_fs_hash(tmp_path):
    jpath = str(tmp_path / "journal.json")
    responder = MockResponder(
        replies=[
            Reply(
                tool_uses=[
                    ToolUse(
                        id="t1",
                        name="bash",
                        input={"command": "echo verifiable > /proof.txt; wc -c < /proof.txt"},
                    )
                ]
            ),
            Reply(text="done"),
        ]
    )
    sb = SandboxClient()
    try:
        session = ChatSession(
            journal_path=jpath,
            responder=responder,
            tools=tools_with_sandbox(client=sb),
        )
        turn = session.send("write a file and count its bytes")
        assert len(turn.tool_calls) == 1

        events = json.load(open(jpath))

        # the bash exec landed in the ledger, carrying its fs_hash
        bash_event = next(
            e
            for e in events
            if e["event"]["tool_name"] == "bash"
            and "fs_hash" in (e["event"].get("result") or {})
        )
        result = bash_event["event"]["result"]
        assert result["stdout"].strip() == "11"  # len("verifiable\n")
        assert result["exit_code"] == 0
        assert len(result["fs_hash"]) == 64

        # the exec is hash-chained into the ledger (genesis -> ... continuity)
        prev = "0" * 64
        for e in events:
            assert e["prev_hash"] == prev, f"chain breaks at seq {e['seq_id']}"
            assert e.get("entry_hash"), f"seq {e['seq_id']} not chained"
            prev = e["entry_hash"]
    finally:
        sb.close()


def test_mandate_allowlist_gates_commands():
    sb = SandboxClient(mandate=shell_mandate(["echo", "cat", "ls"], deny=["rm"]))
    try:
        run = bash_tool(sb).call

        ok = run({"command": "echo hi > /a.txt; cat /a.txt"})
        assert ok["gate"]["decision"] == "ACCEPT"
        assert ok["stdout"] == "hi\n"
        assert not ok.get("blocked")

        # explicitly denied command is blocked, and the exec never runs
        denied = run({"command": "rm /a.txt"})
        assert denied["gate"]["decision"] == "REJECT"
        assert denied.get("blocked") is True
        assert denied["exit_code"] == 126
        assert any("rm" in r for r in denied["gate"]["reasons"])
        assert run({"command": "cat /a.txt"})["stdout"] == "hi\n"  # file untouched

        # command outside the allowlist is rejected
        assert run({"command": "curl http://x"})["gate"]["decision"] == "REJECT"

        # a dynamically-named command fails closed
        assert run({"command": "C=ls; $C /"})["gate"]["decision"] == "REJECT"
    finally:
        sb.close()


def test_mandate_verdict_is_recorded_in_the_ledger(tmp_path):
    jpath = str(tmp_path / "journal.json")
    responder = MockResponder(
        replies=[
            Reply(tool_uses=[ToolUse(id="t1", name="bash", input={"command": "rm -rf /"})]),
            Reply(text="that was blocked"),
        ]
    )
    sb = SandboxClient(mandate=shell_mandate(["echo", "ls", "cat"]))
    try:
        ChatSession(
            journal_path=jpath,
            responder=responder,
            tools=tools_with_sandbox(client=sb),
        ).send("delete everything")
        events = json.load(open(jpath))
        bash_event = next(
            e
            for e in events
            if e["event"]["tool_name"] == "bash" and "gate" in (e["event"].get("result") or {})
        )
        result = bash_event["event"]["result"]
        assert result["gate"]["decision"] == "REJECT"
        assert result.get("blocked") is True
        assert result["gate"]["mandate_hash"]  # the mandate it ran against is on the ledger
    finally:
        sb.close()

// KorgChat sandbox sidecar — a persistent just-bash shell driven over stdio.
//
// One Bash instance lives for the process lifetime, so the in-memory
// filesystem is shared across exec calls (an agent session), while each
// exec gets fresh shell state (env/cwd) per just-bash semantics.
//
// Protocol: newline-delimited JSON, one request -> one response.
//   -> {"id":1,"op":"ping"}
//   <- {"id":1,"ok":true,"version":"..."}
//   -> {"id":2,"op":"configure","mandate":{"allow":["ls","cat","grep"]}}
//   <- {"id":2,"ok":true,"mandate_hash":"...","fs_hash":"...","fs_files":N}
//   -> {"id":3,"op":"exec","cmd":"echo hi > a.txt; cat a.txt","timeoutMs":10000}
//   <- {"id":3,"ok":true,"gate":{"decision":"ACCEPT",...},"stdout":"hi\n",
//        "stderr":"","exit_code":0,"fs_hash":"...","fs_files":N}
//   -> {"id":4,"op":"reset"}            // fresh sandbox (keeps the mandate)
//
// Safe by default: InMemoryFs, no network, no python, no js-exec. The shell
// physically cannot reach the host filesystem or network. A mandate adds a
// command allowlist enforced two ways: just-bash only registers the allowed
// commands (physical), AND each line is parsed before exec so a disallowed or
// dynamically-named command is rejected with a verdict (fail-closed).

import { Bash, parse } from "just-bash";
import { createHash } from "node:crypto";
import { createInterface } from "node:readline";

const VERSION = "korgchat-sandbox/2 (just-bash 3.0.1)";

// mandate: { allow?: string[], deny?: string[] } | null   (null = unrestricted)
let mandate = null;
let mandateHash = null;

function newBash() {
  const opts = {};
  if (mandate && Array.isArray(mandate.allow) && mandate.allow.length) {
    opts.commands = mandate.allow; // physical: only these commands are registered
  }
  return new Bash(opts);
}
let bash = newBash();

const sha256 = (data) => createHash("sha256").update(data).digest("hex");

// Stable JSON (sorted keys) so a mandate hashes the same regardless of order.
function stable(v) {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(stable).join(",") + "]";
  return "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + stable(v[k])).join(",") + "}";
}

const DYNAMIC = "<dynamic>";

// Recursively collect the command names a parsed line invokes. A command whose
// name is not a single static literal (e.g. `$CMD`) is reported as DYNAMIC so
// the mandate can fail closed on it.
function collectCommands(node, out) {
  if (!node || typeof node !== "object") return;
  if (Array.isArray(node)) {
    for (const n of node) collectCommands(n, out);
    return;
  }
  if (node.type === "SimpleCommand" && node.name) {
    const parts = node.name.parts;
    if (Array.isArray(parts) && parts.length === 1 && parts[0].type === "Literal") {
      out.add(String(parts[0].value));
    } else {
      out.add(DYNAMIC);
    }
  }
  for (const k of Object.keys(node)) collectCommands(node[k], out);
}

function gateVerdict(cmd) {
  if (!mandate) return { decision: "ACCEPT", reasons: [], commands_used: [], mandate_hash: null };
  let names;
  try {
    const out = new Set();
    collectCommands(parse(cmd), out);
    names = [...out];
  } catch (e) {
    return {
      decision: "REJECT",
      reasons: [`unparseable command: ${(e && e.message) || e}`],
      commands_used: [],
      mandate_hash: mandateHash,
    };
  }
  const allow = mandate.allow;
  const deny = mandate.deny || [];
  const reasons = [];
  for (const n of names) {
    if (n === DYNAMIC) reasons.push("dynamic command name (variable/expansion) not permitted under mandate");
    else if (deny.includes(n)) reasons.push(`command '${n}' is explicitly denied`);
    else if (allow && allow.length && !allow.includes(n)) reasons.push(`command '${n}' not in mandate allowlist`);
  }
  return {
    decision: reasons.length ? "REJECT" : "ACCEPT",
    reasons,
    commands_used: names,
    mandate_hash: mandateHash,
  };
}

// Walk the virtual filesystem into a deterministic manifest and hash it.
async function fsHash(root = "/") {
  const fs = bash.fs;
  const lines = [];
  async function walk(dir) {
    let names;
    try {
      names = await fs.readdir(dir);
    } catch {
      return;
    }
    for (const name of [...names].sort()) {
      const p = dir === "/" ? "/" + name : dir + "/" + name;
      let st;
      try {
        st = await fs.stat(p);
      } catch {
        continue;
      }
      if (st.isDirectory) {
        lines.push("d " + p);
        await walk(p);
      } else if (st.isSymbolicLink) {
        lines.push("l " + p);
      } else if (st.isFile) {
        try {
          const buf = await fs.readFileBuffer(p);
          lines.push("f " + p + " " + buf.length + " " + sha256(buf));
        } catch {
          lines.push("f " + p + " ?");
        }
      }
    }
  }
  await walk(root);
  return { hash: sha256(lines.join("\n")), files: lines.length };
}

async function handle(req) {
  const { id, op } = req;
  try {
    if (op === "ping") return { id, ok: true, version: VERSION };
    if (op === "configure") {
      mandate = req.mandate && typeof req.mandate === "object" ? req.mandate : null;
      mandateHash = mandate ? sha256(stable(mandate)) : null;
      bash = newBash();
      const f = await fsHash();
      return { id, ok: true, mandate_hash: mandateHash, fs_hash: f.hash, fs_files: f.files };
    }
    if (op === "reset") {
      bash = newBash(); // keeps the current mandate
      const f = await fsHash();
      return { id, ok: true, mandate_hash: mandateHash, fs_hash: f.hash, fs_files: f.files };
    }
    if (op === "exec") {
      const cmd = String(req.cmd ?? "");
      const gate = gateVerdict(cmd);
      if (gate.decision === "REJECT") {
        const f = await fsHash(req.fsRoot ?? "/");
        return {
          id,
          ok: true,
          blocked: true,
          gate,
          stdout: "",
          stderr: "blocked by mandate: " + gate.reasons.join("; ") + "\n",
          exit_code: 126,
          fs_hash: f.hash,
          fs_files: f.files,
        };
      }
      const opts = {};
      if (req.cwd) opts.cwd = req.cwd;
      if (req.env && typeof req.env === "object") opts.env = req.env;
      opts.signal = AbortSignal.timeout(req.timeoutMs ?? 10000);
      const r = await bash.exec(cmd, opts);
      const f = await fsHash(req.fsRoot ?? "/");
      return {
        id,
        ok: true,
        gate,
        stdout: r.stdout,
        stderr: r.stderr,
        exit_code: r.exitCode,
        fs_hash: f.hash,
        fs_files: f.files,
      };
    }
    return { id, ok: false, error: `unknown op: ${op}` };
  } catch (e) {
    return { id, ok: false, error: String((e && e.message) || e) };
  }
}

// Serialize handling so concurrent lines never interleave async FS state.
let queue = Promise.resolve();
const rl = createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const text = line.trim();
  if (!text) return;
  queue = queue.then(async () => {
    let req;
    try {
      req = JSON.parse(text);
    } catch {
      process.stdout.write(JSON.stringify({ id: null, ok: false, error: "bad json" }) + "\n");
      return;
    }
    const res = await handle(req);
    process.stdout.write(JSON.stringify(res) + "\n");
  });
});
rl.on("close", () => process.exit(0));

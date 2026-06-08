// KorgChat sandbox sidecar — a persistent just-bash shell driven over stdio.
//
// One Bash instance lives for the process lifetime, so the in-memory
// filesystem is shared across exec calls (an agent session), while each
// exec gets fresh shell state (env/cwd) per just-bash semantics.
//
// Protocol: newline-delimited JSON, one request -> one response.
//   -> {"id":1,"op":"ping"}
//   <- {"id":1,"ok":true,"version":"..."}
//   -> {"id":2,"op":"exec","cmd":"echo hi > a.txt; cat a.txt","timeoutMs":10000}
//   <- {"id":2,"ok":true,"stdout":"hi\n","stderr":"","exit_code":0,
//        "fs_hash":"<sha256 of the whole virtual FS state>","fs_files":N}
//   -> {"id":3,"op":"reset"}            // fresh sandbox
//
// Safe by default: InMemoryFs, no network, no python, no js-exec. The shell
// physically cannot reach the host filesystem or network.

import { Bash } from "just-bash";
import { createHash } from "node:crypto";
import { createInterface } from "node:readline";

const VERSION = "korgchat-sandbox/1 (just-bash 3.0.1)";

function newBash() {
  // All capability flags omitted => safest defaults (no net/python/js, InMemoryFs).
  return new Bash();
}
let bash = newBash();

const sha256 = (data) => createHash("sha256").update(data).digest("hex");

// Walk the virtual filesystem into a deterministic manifest and hash it.
// The hash is a pure function of full FS state => same commands from genesis
// reproduce the same hash (replayable / tamper-evident when chained).
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
    if (op === "reset") {
      bash = newBash();
      const f = await fsHash();
      return { id, ok: true, fs_hash: f.hash, fs_files: f.files };
    }
    if (op === "exec") {
      const opts = {};
      if (req.cwd) opts.cwd = req.cwd;
      if (req.env && typeof req.env === "object") opts.env = req.env;
      opts.signal = AbortSignal.timeout(req.timeoutMs ?? 10000);
      const r = await bash.exec(String(req.cmd ?? ""), opts);
      const f = await fsHash(req.fsRoot ?? "/");
      return {
        id,
        ok: true,
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

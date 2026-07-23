// Sequential stream reuse — drive >=100 short streams through ONE reused
// context/stream (reset between each) and prove: byte-identical token output
// every iteration (no stale KV / token history / event sequence), a clean
// reset->created transition each time, and a VRAM/RSS plateau over the tail.
// The lifecycle edge cases are exercised and reported by the same binary mode.

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const iterations = Math.max(1, Number(process.env.VOXTRAL_SEQUENTIAL_STREAMS || 100));
const VRAM_TAIL_LIMIT = 64 * 1024 * 1024; // bytes of tolerated VRAM jitter over the tail
const RSS_TAIL_LIMIT = 64 * 1024;         // KiB of tolerated RSS drift over the tail

const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run test:sequential-stream-reuse",
  requestedIterations: iterations,
};

try {
  const run = await runStreamSession({
    config,
    planName: "sequential-reuse",
    audioPath: config.remoteSmokeAudio,
    sequentialStreams: iterations,
    sequentialSamples: 48000,
    skipParity: true,
    timeoutMs: 1_200_000,
  });

  const checks = {
    ok: run.ok === true,
    allTokensConsistent: run.allTokensConsistent === true,
    allResetPristine: run.allResetPristine === true,
    allResetOk: run.allResetOk === true,
    allEdgesPass: run.allEdgesPass === true,
    vramPlateau: (run.vramTailRangeBytes ?? Infinity) <= VRAM_TAIL_LIMIT,
    rssPlateau: (run.rssTailRangeKiB ?? Infinity) <= RSS_TAIL_LIMIT,
  };
  const pass = Object.values(checks).every(Boolean);

  summary.iterations = run.iterations;
  summary.tokensPerStream = run.tokensPerStream;
  summary.checks = checks;
  summary.vram = { firstBytes: run.firstVramBytes, lastBytes: run.lastVramBytes, tailRangeBytes: run.vramTailRangeBytes, tailWindow: run.tailWindow };
  summary.rss = { firstKiB: run.firstRssKiB, lastKiB: run.lastRssKiB, tailRangeKiB: run.rssTailRangeKiB };
  summary.edges = run.edges;
  summary.exitCode = pass ? 0 : 1;
  if (!pass) process.exitCode = 1;

  console.log(`[seq-reuse] ${pass ? "PASS" : "FAIL"} iters=${run.iterations} tokens/stream=${run.tokensPerStream}`);
  console.log(`[seq-reuse] vram tail range=${run.vramTailRangeBytes}B rss tail range=${run.rssTailRangeKiB}KiB (window ${run.tailWindow})`);
  console.log(`[seq-reuse] checks: ${JSON.stringify(checks)}`);
  console.log(`[seq-reuse] edges: ${(run.edges || []).map((e) => `${e.name}=${e.pass}`).join(", ")}`);
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[seq-reuse] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config, testName: "sequential-stream-reuse", backend: "Vulkan",
    command: summary.command, result: summary,
  });
  console.log(`[seq-reuse] summary: ${artifact.directory}`);
}

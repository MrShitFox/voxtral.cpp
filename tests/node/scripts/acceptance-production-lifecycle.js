// Production lifecycle acceptance — exercise the documented stream state machine
// end to end: reset-based reuse, finish idempotency, feed-after-finish rejection,
// cancel mid-stream, reset-after-terminal, destroy under backpressure, and that
// the reference (non-incremental) decode path still completes. Correctness gates
// only; performance is covered by the benchmark scripts.

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const iterations = Math.max(3, Number(process.env.VOXTRAL_LIFECYCLE_ITERS || 12));

const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:production-lifecycle",
  iterations,
};

try {
  // Reset-based reuse + lifecycle edge cases (same binary mode).
  const life = await runStreamSession({
    config, planName: "production-lifecycle",
    audioPath: config.remoteSmokeAudio,
    sequentialStreams: iterations, sequentialSamples: 48000,
    skipParity: true, timeoutMs: 600_000,
  });

  // Reference (non-incremental) decode path still works. The finish-only oracle
  // is not a production incremental stream, so it does not support (and must not
  // be asked for) the incremental warmup.
  const ref = await runStreamSession({
    config, planName: "reference-mode",
    audioPath: config.remoteSmokeAudio, mode: "80ms",
    warmup: false, skipParity: true,
    env: { VOXTRAL_STREAM_DECODER: "reference" },
    timeoutMs: 180_000,
  });

  const edgeByName = Object.fromEntries((life.edges || []).map((e) => [e.name, e.pass]));
  const checks = {
    reuseTokensConsistent: life.allTokensConsistent === true,
    resetPristine: life.allResetPristine === true,
    resetOk: life.allResetOk === true,
    finishTwiceIdempotent: edgeByName.finish_twice_idempotent === true,
    feedAfterFinishRejected: edgeByName.feed_after_finish_rejected === true,
    cancelSetsCancelled: edgeByName.cancel_sets_cancelled === true,
    finishAfterCancelOk: edgeByName.finish_after_cancel_ok === true,
    resetAfterCancelCreated: edgeByName.reset_after_cancel_created === true,
    destroyUnderBackpressureNoCrash: edgeByName.destroy_under_backpressure_no_crash === true,
    resetFromCreatedIdempotent: edgeByName.reset_from_created_idempotent === true,
    referenceModeCompletes: ref.state === "completed" && (ref.tokens?.length || 0) > 0,
  };
  const pass = Object.values(checks).every(Boolean);

  summary.checks = checks;
  summary.edges = life.edges;
  summary.reuse = {
    iterations: life.iterations,
    tokensPerStream: life.tokensPerStream,
    vramTailRangeBytes: life.vramTailRangeBytes,
    rssTailRangeKiB: life.rssTailRangeKiB,
  };
  summary.referenceTokens = ref.tokens?.length ?? 0;
  summary.exitCode = pass ? 0 : 1;
  if (!pass) process.exitCode = 1;

  console.log(`[lifecycle] ${pass ? "PASS" : "FAIL"}`);
  console.log(`[lifecycle] checks: ${JSON.stringify(checks, null, 2)}`);
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[lifecycle] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config, testName: "production-lifecycle", backend: "Vulkan",
    command: summary.command, result: summary,
  });
  console.log(`[lifecycle] summary: ${artifact.directory}`);
}

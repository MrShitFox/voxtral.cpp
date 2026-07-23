import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  SESSION8_GATES,
  gate,
  gateLatency,
  gateSustained,
  prepareSession8,
  session8PrecisionEnvironment,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const seconds = Number(process.env.VOXTRAL_SOAK_SECONDS ?? 1800);
const invokedAs = process.argv.includes("--rollover")
  ? "test:realtime-rollover"
  : process.argv.includes("--soak")
    ? "test:realtime-soak-30m"
    : "acceptance:realtime-sustained";
const summary = {
  startedAt: new Date().toISOString(),
  command: `npm run ${invokedAs}`,
  requiredDurationSeconds: 1800,
  requestedDurationSeconds: seconds,
  steps: [],
};

function tokenWindow(run, absolutePosition, radius = 12) {
  const events = (run.events ?? []).filter((event) => event.type === "token");
  const index = events.findIndex((event) => event.decoderPosition === absolutePosition);
  if (index < 0) return [];
  return events.slice(Math.max(0, index - radius), index + radius + 1).map((event) => ({
    sequence: event.sequence,
    decoderPosition: event.decoderPosition,
    token: event.token,
    special: event.special,
  }));
}

function gateLatencyMatrix(cold, warmModel, warmGraphs) {
  gateLatency(warmGraphs, "warm-graphs eligibility latency");
  // Cold-process/model lifecycle remains reported, but model load, context
  // creation and shader warmup are not ASR eligibility hard gates.
  gate(cold.firstTokenMs > 0 && cold.firstVisibleTextMs > 0,
    "cold-process latency markers unavailable");
  gate(warmModel.firstTokenMs > 0 && warmModel.firstVisibleTextMs > 0,
    "warm-model latency markers unavailable");
}

try {
  gate(Number.isInteger(seconds) && seconds >= 1800,
    `30-minute acceptance cannot be shortened (requested ${seconds}s)`);
  const matrix = await loadLatestPrecisionMatrix(config);
  const selected = matrix.result.productionDecision.selected;
  gate(selected, "precision matrix has no selected production variant");
  const productionEnv = session8PrecisionEnvironment(selected);
  summary.precisionMatrixArtifact = matrix.directory;
  summary.selectedPrecision = selected;
  summary.encoderKv = matrix.result.variants[selected].encoderKv;
  summary.decoderKv = matrix.result.variants[selected].decoderKv;
  summary.rx6600VramTotalBytes = matrix.result.rx6600VramTotalBytes;
  await prepareSession8(summary, { config });

  const sustained = await runStreamSession({
    config,
    planName: `session8-sustained-${seconds}s`,
    syntheticSeconds: seconds,
    realtimeMs: 80,
    warmup: true,
    skipParity: true,
    monitorMemory: true,
    maxTotalSamples: (seconds + 60) * 16_000,
    env: productionEnv,
    // The process budget also includes deterministic synthesis of 28.8M PCM
    // samples, model/context creation and graph warmup before the 30-minute
    // paced interval starts. Keep those lifecycle costs outside the realtime
    // gates, but leave enough wall-clock headroom to receive the final JSON.
    timeoutMs: (seconds + 600) * 1000,
  });
  summary.sustained = summarizeRun(sustained);
  summary.rollover = {
    ...summary.sustained.decoderKvRing,
    tokenWindow: tokenWindow(sustained, sustained.decoderFirstWrapAbsolutePosition),
  };
  summary.targets = {
    rtfTargetMet: sustained.pipelineRtf <= SESSION8_GATES.pipelineRtfTarget,
    computeHeadroomTargetMet: sustained.computeHeadroomRatio >= 0.20,
    finishTargetMet: sustained.finishLatencyMs < SESSION8_GATES.finishTargetMs,
  };
  gate(sustained.audioDurationMs >= 1_800_000,
    `sustained audio duration ${sustained.audioDurationMs}ms < 1800000ms`);
  gateSustained(sustained, "30-minute paced stream", {
    requireWrap: true,
    precisionVariant: selected,
    peakVramLimitBytes: summary.rx6600VramTotalBytes,
  });

  // The same binary reports three non-overlapping latency regimes. Each call is
  // a fresh process; warmup is explicit only for the warm-graphs measurement.
  const cold = await runStreamSession({
    config,
    planName: "session8-latency-cold-process",
    mode: "80ms",
    realtimeMs: 80,
    skipParity: true,
    env: { ...productionEnv, MESA_SHADER_CACHE_DISABLE: "true" },
    timeoutMs: 180_000,
  });
  const warmModel = await runStreamSession({
    config,
    planName: "session8-latency-warm-model",
    mode: "80ms",
    realtimeMs: 80,
    skipParity: true,
    env: productionEnv,
    timeoutMs: 180_000,
  });
  const warmGraphs = await runStreamSession({
    config,
    planName: "session8-latency-warm-graphs",
    mode: "80ms",
    realtimeMs: 80,
    warmup: true,
    skipParity: true,
    env: productionEnv,
    timeoutMs: 180_000,
  });
  summary.latency = {
    coldShaderDiskCacheDisabled: true,
    coldProcess: summarizeRun(cold, { includeTokens: false }).latency,
    warmModel: summarizeRun(warmModel, { includeTokens: false }).latency,
    warmGraphs: summarizeRun(warmGraphs, { includeTokens: false }).latency,
  };
  gateLatencyMatrix(cold, warmModel, warmGraphs);
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: `session8-${invokedAs.replaceAll(":", "-")}`,
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    textArtifacts: summary.sustained ? {
      "transcript.txt": summary.sustained.transcript,
      "token-ids.txt": summary.sustained.tokens.join("\n"),
      "backlog.csv": [
        "stage,p50_ms,p95_ms,p99_ms,max_ms,final_ms,slope_ms_per_s,deadline_misses",
        `total,${summary.sustained.backlog.p50Ms},${summary.sustained.backlog.p95Ms},${summary.sustained.backlog.p99Ms},${summary.sustained.backlog.maxMs},${summary.sustained.backlog.finalMs},${summary.sustained.backlog.slopeMsPerSec},${summary.sustained.backlog.deadlineMisses}`,
        ...["encoder", "adapter", "decoder"].map((stage) => {
          const m = summary.sustained[`${stage}Backlog`];
          return `${stage},${m.p50Ms},${m.p95Ms},${m.p99Ms},${m.maxMs},${m.finalMs},${m.slopeMsPerSec},${m.deadlineMisses}`;
        }),
      ].join("\n"),
    } : {},
  });
  console.log(`[${invokedAs}] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[${invokedAs}] error: ${summary.error}`);
}

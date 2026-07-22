// Acceptance gate for the selected realtime encoder scheduler (4 logical / 32
// physical) against the retained 128/128 throughput baseline.
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { createChunkPlan } from "../helpers/chunks.js";
import { runProcess } from "../helpers/exec.js";
import {
  checkRemoteConnection,
  normalizeFixtureOnGpu,
  syncFixture,
  syncSources,
} from "../helpers/remote.js";
import { planCounts, runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const nodeCwd = new URL("..", import.meta.url).pathname;
const longFixture = process.env.VOXTRAL_LONG_AUDIO
  ? path.resolve(process.env.VOXTRAL_LONG_AUDIO)
  : path.join(config.localRepo, "voxTest2min.m4a");
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:encoder-realtime",
  production: { logical: 4, physical: 32 },
  baseline: { logical: 128, physical: 128 },
  steps: [],
  short: [],
  long: [],
};

const envFor = (logical, physical) => ({
  VOXTRAL_ENC_KV_LOGICAL_BATCH: String(logical),
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: String(physical),
  VOXTRAL_ENCODER_TELEMETRY: "1",
});

function gate(condition, message) {
  if (!condition) throw new Error(message);
}

function tokenBoundaryCsv(tokens, fixture, encoderFrames) {
  if (!Array.isArray(tokens) || tokens.length === 0 || !fixture || !encoderFrames) return "";
  const durationMs = Number(fixture.durationMs);
  const capacity = 878;
  const lines = ["tokenIndex,tokenId,approxAudioMs,approxEncoderFrame,approxRingWriteEpoch"];
  for (let index = 0; index < tokens.length; index += 1) {
    // Adapter/decoder are still finish-only, so this is explicitly a
    // proportional boundary-audit locator, not a semantic word alignment.
    const fraction = tokens.length === 1 ? 0 : index / (tokens.length - 1);
    const audioMs = durationMs * fraction;
    const encoderFrame = Math.min(encoderFrames - 1, Math.floor(encoderFrames * fraction));
    lines.push(`${index},${tokens[index]},${audioMs.toFixed(3)},${encoderFrame},${Math.floor(encoderFrame / capacity)}`);
  }
  return lines.join("\n");
}

function compact(result, label) {
  return {
    label,
    state: result.state,
    parityChecked: result.parityChecked,
    encoderSha256: result.encoderSha256,
    encoderMaxAbsDeltaVsBatch: result.encoderMaxAbsDeltaVsBatch,
    melMaxAbsDeltaVsBatch: result.melMaxAbsDeltaVsBatch,
    tokens: result.tokens,
    text: result.text,
    encoderUniqueFrames: result.encoderUniqueFrames,
    encoderTransformerFramesComputed: result.encoderTransformerFramesComputed,
    encoderWorkRatio: result.encoderWorkRatio,
    encoderLogicalFramesSubmitted: result.encoderLogicalFramesSubmitted,
    encoderPhysicalQueryRowsEvaluated: result.encoderPhysicalQueryRowsEvaluated,
    encoderPaddingRowsEvaluated: result.encoderPaddingRowsEvaluated,
    encoderPhysicalOverheadRatio: result.encoderPhysicalOverheadRatio,
    encoderGraphExecutions: result.encoderGraphExecutions,
    encoderKvAppends: result.encoderKvAppends,
    encoderWarmupFrames: result.encoderWarmupFrames,
    encoderKvWraps: result.encoderKvWraps,
    encoderKvEvictions: result.encoderKvEvictions,
    encoderKvCapacityFrames: result.encoderKvCapacityFrames,
    encoderFramesComputedDuringFinish: result.encoderFramesComputedDuringFinish,
    encoderFirstFrameAbsoluteMs: result.encoderFirstFrameAbsoluteMs,
    encoderFirstFrameResidenceMs: result.encoderFirstFrameResidenceMs,
    firstMelFrameAbsoluteMs: result.firstMelFrameAbsoluteMs,
    firstAdapterGroupAbsoluteMs: result.firstAdapterGroupAbsoluteMs,
    firstAdapterGroupResidenceMs: result.firstAdapterGroupResidenceMs,
    firstEightFrameGroupAbsoluteMs: result.firstEightFrameGroupAbsoluteMs,
    firstEightFrameGroupResidenceMs: result.firstEightFrameGroupResidenceMs,
    encoderResidenceP50Ms: result.encoderResidenceP50Ms,
    encoderResidenceP95Ms: result.encoderResidenceP95Ms,
    encoderResidenceP99Ms: result.encoderResidenceP99Ms,
    encoderResidenceMaxMs: result.encoderResidenceMaxMs,
    adapterGroupResidenceP50Ms: result.adapterGroupResidenceP50Ms,
    adapterGroupResidenceP95Ms: result.adapterGroupResidenceP95Ms,
    adapterGroupResidenceP99Ms: result.adapterGroupResidenceP99Ms,
    adapterGroupResidenceMaxMs: result.adapterGroupResidenceMaxMs,
    encoderComputeP95Ms: result.encoderComputeP95Ms,
    encoderComputeMaxMs: result.encoderComputeMaxMs,
    encoderComputeWarmMaxMs: result.encoderComputeWarmMaxMs,
    feedStartLatenessP95Ms: result.feedStartLatenessP95Ms,
    feedStartLatenessMaxMs: result.feedStartLatenessMaxMs,
    feedFinishLatenessP95Ms: result.feedFinishLatenessP95Ms,
    feedFinishLatenessMaxMs: result.feedFinishLatenessMaxMs,
    backlogP95Ms: result.backlogP95Ms,
    backlogP99Ms: result.backlogP99Ms,
    backlogMaxMs: result.backlogMaxMs,
    finalBacklogMs: result.finalBacklogMs,
    backlogGrowthSlopeMsPerSec: result.backlogGrowthSlopeMsPerSec,
    audioDurationMs: result.audioDurationMs,
    wallDurationMs: result.wallDurationMs,
    realtimeFactor: result.realtimeFactor,
    finishFrontendMs: result.finishFrontendMs,
    finishEncoderMs: result.finishEncoderMs,
    finishDecoderMs: result.finishDecoderMs,
    encoderKvAllocatedBytes: result.encoderKvAllocatedBytes,
    decoderKvAllocatedBytes: result.decoderKvAllocatedBytes,
    modelLoadedVramBytes: result.modelLoadedVramBytes,
    streamIdleVramBytes: result.streamIdleVramBytes,
    afterFinishVramBytes: result.afterFinishVramBytes,
    afterDestroyVramBytes: result.afterDestroyVramBytes,
    encoderMelRetainedBytes: result.encoderMelRetainedBytes,
    encoderMelPeakRetainedFrames: result.encoderMelPeakRetainedFrames,
    encoderOutputAccumulatedBytes: result.encoderOutputAccumulatedBytes,
    melHistoryRetained: result.melHistoryRetained,
    baselineVramBytes: result.baselineVramBytes,
    peakVramBytes: result.peakVramBytes,
    finalVramBytes: result.finalVramBytes,
    peakRssKiB: result.peakRssKiB,
  };
}

async function localStep(name, command, args, cwd = nodeCwd, timeoutMs = 900_000) {
  console.log(`[encoder-realtime] ${name}`);
  const result = await runProcess(command, args, {
    cwd,
    timeoutMs,
    onStdout: (chunk) => process.stdout.write(chunk),
    onStderr: (chunk) => process.stderr.write(chunk),
  });
  summary.steps.push({ name, exitCode: result.exitCode, wallMs: result.wallMs });
}

async function main() {
  await localStep("unit", "npm", ["run", "test:unit"]);
  await localStep("local-build", "npm", ["run", "build:local"], nodeCwd, 1_200_000);
  await localStep("cpp-unit", "ctest", ["--test-dir", config.localBuild, "--output-on-failure"], config.localRepo);
  await checkRemoteConnection({ config });
  await syncSources({ config });
  await buildRemoteVulkan({ config });

  // Fast permanent gate: all feed plans produce exactly the same encoder tensor,
  // tokens and transcript on the checked-in short spoken WAV.
  const plans = ["full", "80ms", "160ms", "480ms", "seeded-random:20260722"];
  const shortRuns = [];
  for (const mode of plans) {
    const result = await runStreamSession({
      config,
      planName: `realtime-short-${mode.replaceAll(":", "-")}`,
      mode,
      maxTokens: 0,
      env: envFor(4, 32),
      timeoutMs: 300_000,
    });
    gate(result.state === "completed", `${mode}: short stream failed`);
    gate(result.encoderMaxAbsDeltaVsBatch <= 1e-5, `${mode}: encoder tensor parity failed`);
    gate(result.encoderTransformerFramesComputed === result.encoderUniqueFrames, `${mode}: frame replay detected`);
    shortRuns.push(result);
    summary.short.push(compact(result, mode));
  }
  for (const result of shortRuns) {
    gate(result.encoderSha256 === shortRuns[0].encoderSha256, `${result.planName}: chunk-plan encoder divergence`);
    gate(JSON.stringify(result.tokens) === JSON.stringify(shortRuns[0].tokens), `${result.planName}: token divergence`);
    gate(result.text === shortRuns[0].text, `${result.planName}: transcript divergence`);
  }

  // Optional private long-form gate. Absence is an explicit skip rather than a
  // hidden failure; setting VOXTRAL_LONG_AUDIO makes absence fatal.
  let fixture;
  try {
    const transfer = await syncFixture(longFixture, { config });
    fixture = await normalizeFixtureOnGpu({ config });
    gate(transfer.localSha256 === fixture.sourceSha256 && transfer.remoteSha256 === fixture.sourceSha256,
      "fixture SHA-256 changed between local source, transfer and normalization");
    fixture.localSourceSha256 = transfer.localSha256;
    fixture.remoteSourceSha256 = transfer.remoteSha256;
  } catch (error) {
    if (process.env.VOXTRAL_LONG_AUDIO) throw error;
    summary.longSkipReason = `VOXTRAL_LONG_AUDIO not available: ${error.message}`;
    summary.exitCode = 0;
    return;
  }
  summary.fixture = fixture;

  const longRuns = [];
  for (const pace of [80, 160, 480]) {
    const result = await runStreamSession({
      config,
      planName: `spoken-production-paced-${pace}ms`,
      realtimeMs: pace,
      audioPath: fixture.wavPath,
      // Decode the complete finish-only reference so token/transcript identity
      // is proved for every paced chunk plan, not inferred from one token.
      maxTokens: 0,
      skipParity: true,
      env: envFor(4, 32),
      monitorMemory: pace === 80,
      timeoutMs: 360_000,
    });
    longRuns.push(result);
    summary.long.push(compact(result, `production-${pace}ms`));
    gate(result.encoderTransformerFramesComputed === result.encoderUniqueFrames, `${pace}ms: frame replay`);
    gate(result.finalBacklogMs < 20, `${pace}ms: final backlog ${result.finalBacklogMs} ms`);
    gate(result.backlogGrowthSlopeMsPerSec < 0.5, `${pace}ms: growing backlog`);
    gate(result.encoderKvWraps > 0 && result.encoderKvEvictions > 0, `${pace}ms: rollover not exercised`);
    gate(result.encoderMelPeakRetainedFrames < 300 && result.melHistoryRetained === false, `${pace}ms: Mel is not bounded`);
    gate(result.encoderComputeWarmMaxMs < 80, `${pace}ms: warm compute max ${result.encoderComputeWarmMaxMs} ms`);
  }

  // Irregular payloads use the same audio clock and one long-lived GPU process.
  // --realtime-ms enables pacing; the native harness derives each backlog
  // cadence from that plan row's own sample count.
  const randomCounts = planCounts(createChunkPlan(Buffer.alloc(fixture.sampleCount * 2), {
    strategy: "seeded-random",
    seed: 20260722,
    minSamples: 640,
    // Keep the random payload cadence within the realtime acceptance envelope
    // (40..160 ms); larger random blocks are a deliberate stress/throughput
    // sweep, not a low-latency production schedule.
    maxSamples: 2560,
  }));
  const randomResult = await runStreamSession({
    config,
    planName: "spoken-production-paced-seeded-random-20260722",
    counts: randomCounts,
    realtimeMs: 80,
    audioPath: fixture.wavPath,
    maxTokens: 0,
    skipParity: true,
    env: envFor(4, 32),
    timeoutMs: 360_000,
  });
  longRuns.push(randomResult);
  summary.long.push(compact(randomResult, "production-seeded-random-paced"));
  gate(randomResult.encoderTransformerFramesComputed === randomResult.encoderUniqueFrames, "random: frame replay");
  gate(randomResult.finalBacklogMs < 20, `random: final backlog ${randomResult.finalBacklogMs} ms`);
  gate(randomResult.backlogGrowthSlopeMsPerSec < 0.5, "random: growing backlog");
  gate(randomResult.encoderKvWraps > 0, "random: rollover not exercised");
  gate(randomResult.encoderMelPeakRetainedFrames < 300 && randomResult.melHistoryRetained === false,
    "random: Mel is not bounded");

  gate(longRuns[0].encoderResidenceP95Ms < 160, `80ms: frame residence p95 ${longRuns[0].encoderResidenceP95Ms} ms`);
  gate(longRuns[0].adapterGroupResidenceP95Ms < 160, `80ms: adapter residence p95 ${longRuns[0].adapterGroupResidenceP95Ms} ms`);
  gate(longRuns[1].adapterGroupResidenceP95Ms < 160, `160ms: adapter residence p95 ${longRuns[1].adapterGroupResidenceP95Ms} ms`);
  for (const result of longRuns) {
    gate(result.encoderSha256 === longRuns[0].encoderSha256, `${result.planName}: long encoder tensor divergence`);
    gate(JSON.stringify(result.tokens) === JSON.stringify(longRuns[0].tokens), `${result.planName}: long token divergence`);
    gate(result.text === longRuns[0].text, `${result.planName}: long transcript divergence`);
  }

  // Compute-only production and 128/128 throughput baseline prove that changing
  // scheduler latency does not change the tensor/tokens/transcript.
  const productionFull = await runStreamSession({
    config, planName: "spoken-production-full", mode: "full", audioPath: fixture.wavPath,
    // This run retains Mel only inside the test process and executes the
    // independent global batch encoder for the long-form tensor hard gate.
    maxTokens: 0, skipParity: false, monitorMemory: true, env: envFor(4, 32), timeoutMs: 420_000,
  });
  const baselineFull = await runStreamSession({
    config, planName: "spoken-baseline-128-128", mode: "full", audioPath: fixture.wavPath,
    maxTokens: 0, skipParity: true, monitorMemory: true, env: envFor(128, 128), timeoutMs: 360_000,
  });
  summary.long.push(compact(productionFull, "production-compute"));
  summary.long.push(compact(baselineFull, "baseline-compute"));
  gate(productionFull.parityChecked && productionFull.encoderMaxAbsDeltaVsBatch <= 1e-5,
    `long global-batch tensor parity ${productionFull.encoderMaxAbsDeltaVsBatch}`);
  gate(productionFull.encoderSha256 === baselineFull.encoderSha256, "production/baseline encoder tensor divergence");
  gate(JSON.stringify(productionFull.tokens) === JSON.stringify(baselineFull.tokens), "production/baseline token divergence");
  gate(productionFull.text === baselineFull.text, "production/baseline transcript divergence");
  for (const result of longRuns) {
    gate(JSON.stringify(result.tokens) === JSON.stringify(productionFull.tokens),
      `${result.planName}: full token sequence diverges from offline reference`);
    gate(result.text === productionFull.text,
      `${result.planName}: full transcript diverges from offline reference`);
  }
  summary.transcript = productionFull.text;
  summary.tokens = productionFull.tokens;
  summary.exitCode = 0;
}

try {
  await main();
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const textArtifacts = summary.transcript ? { "transcript.txt": summary.transcript } : {};
  const production = summary.long?.find((item) => item.label === "production-compute");
  const boundaries = tokenBoundaryCsv(summary.tokens, summary.fixture, production?.encoderUniqueFrames);
  if (boundaries) textArtifacts["token-boundaries.csv"] = boundaries;
  const artifact = await writeArtifactBundle({
    config,
    testName: "encoder-realtime-acceptance",
    backend: "Vulkan",
    command: "npm run acceptance:encoder-realtime",
    audioMetadata: summary.fixture,
    result: summary,
    textArtifacts,
  });
  console.log(`[encoder-realtime] summary: ${artifact.directory}`);
}

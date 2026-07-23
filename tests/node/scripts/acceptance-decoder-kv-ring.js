import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  exactTokens,
  gate,
  gateDeviceResident,
  gateKvMemory,
  gateRing,
  prepareSession8,
  session8PrecisionEnvironment,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:decoder-kv-ring",
  steps: [],
};

function tokenWindow(run, absolutePosition, radius = 8) {
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

function gateEvents(run, label) {
  const tokenEvents = (run.events ?? []).filter((event) => event.type === "token");
  for (let index = 0; index < tokenEvents.length; index += 1) {
    gate(tokenEvents[index].sequence === index + 1,
      `${label}: token event sequence reset/gap at ${index}`);
  }
  gate(tokenEvents.length === run.decoderSteps,
    `${label}: token events=${tokenEvents.length}, decoder steps=${run.decoderSteps}`);
}

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  const selected = matrix.result.productionDecision.selected;
  gate(selected, "precision matrix has no selected production variant");
  summary.precisionMatrixArtifact = matrix.directory;
  summary.selectedPrecision = selected;
  summary.encoderKv = matrix.result.variants[selected].encoderKv;
  summary.decoderKv = matrix.result.variants[selected].decoderKv;
  await prepareSession8(summary, { config });
  const common = {
    ...session8PrecisionEnvironment(selected),
    // The accepted spoken fixture reaches absolute decoder position 94. A
    // 64-slot test ring therefore wraps during real lexical tokens, while the
    // production allocation/capacity remains 8192 outside this explicit seam.
    VOXTRAL_DECODER_KV_TEST_CAPACITY: "64",
  };
  const reusable = await runStreamSession({
    config,
    planName: "decoder-ring-reusable-production",
    mode: "80ms",
    skipParity: true,
    monitorMemory: true,
    env: common,
    timeoutMs: 300_000,
  });
  const physical = await runStreamSession({
    config,
    planName: "decoder-ring-dynamic-physical",
    mode: "80ms",
    skipParity: true,
    env: { ...common, VOXTRAL_DECODER_STEP_GRAPH: "dynamic" },
    timeoutMs: 300_000,
  });
  const logical = await runStreamSession({
    config,
    planName: "decoder-ring-logical-oracle",
    mode: "80ms",
    skipParity: true,
    env: {
      ...common,
      VOXTRAL_DECODER_STEP_GRAPH: "dynamic",
      VOXTRAL_DECODER_RING_ATTENTION: "logical",
    },
    timeoutMs: 300_000,
  });

  // Persist every run before assertions so a parity failure remains fully
  // diagnosable instead of collapsing the artifact to one error string.
  summary.reusableProduction = summarizeRun(reusable);
  summary.physical = summarizeRun(physical);
  summary.logicalOracle = summarizeRun(logical);

  for (const [label, run] of [
    ["reusable production ring", reusable],
    ["physical ring", physical],
    ["logical ring oracle", logical],
  ]) {
    gateDeviceResident(run, label);
    gateKvMemory(run, label, { precisionVariant: selected });
    gateRing(run, label, { requireWrap: true });
  }
  gateEvents(reusable, "reusable production ring");
  gateEvents(physical, "physical ring");
  gateEvents(logical, "logical ring oracle");
  gate(exactTokens(reusable.tokens, physical.tokens) &&
       exactTokens(reusable.tokens, logical.tokens),
  "reusable/physical ring token sequence differs from logical-order oracle");
  gate(reusable.text === physical.text && reusable.text === logical.text,
    "reusable/physical ring transcript differs from logical-order oracle");
  gate(physical.decoderOldestAbsolutePosition > 0 &&
       physical.decoderNextAbsolutePosition > physical.decoderKvCapacity,
  "physical ring absolute positions did not advance through eviction");

  const firstWrap = reusable.decoderFirstWrapAbsolutePosition;
  summary.rollover = {
    firstWrapAbsolutePosition: firstWrap,
    firstWrapAudioMs: reusable.decoderFirstWrapAudioMs,
    wrapCount: reusable.decoderKvWraps,
    evictionCount: reusable.decoderKvEvictions,
    oldestAbsolutePosition: reusable.decoderOldestAbsolutePosition,
    lastAbsolutePosition: reusable.decoderNextAbsolutePosition - 1,
    bytesMoved: reusable.decoderKvBytesMoved,
    fullBufferMoves: reusable.decoderKvFullBufferMoves,
    preWrapP99Ms: reusable.decoderPreWrapP99Ms,
    wrapStepMs: reusable.decoderWrapStepMs,
    postWrapP99Ms: reusable.decoderPostWrapP99Ms,
    tokenWindow: tokenWindow(reusable, firstWrap),
  };
  summary.tokenParity = true;
  summary.transcriptParity = true;
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8-decoder-kv-ring",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    textArtifacts: summary.reusableProduction ? {
      "transcript.txt": summary.reusableProduction.transcript,
      "token-ids.txt": summary.reusableProduction.tokens.join("\n"),
    } : {},
  });
  console.log(`[decoder-kv-ring] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[decoder-kv-ring] error: ${summary.error}`);
}

// Strict end-to-end gate for the production single-stream path. Session 8
// deliberately has no regression ceilings: every published latency/backlog
// limit below is a hard assertion.
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { normalizeFixtureOnGpu, syncFixture } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  SESSION8_GATES,
  SESSION8_PRODUCTION_ENV,
  exactTokens,
  gate,
  gateDeviceResident,
  gateKvMemory,
  gateLatency,
  gateRing,
  prepareSession8,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const soak = process.argv.includes("--soak");
const summary = {
  startedAt: new Date().toISOString(),
  command: soak
    ? "npm run test:end-to-end-realtime:soak"
    : "npm run acceptance:end-to-end-realtime",
  steps: [],
};

function gateRealtime(r, label) {
  gateDeviceResident(r, label);
  gateKvMemory(r, label);
  gateRing(r, label);
  gate(r.decoderMode === "incremental", `${label}: decoder mode=${r.decoderMode}`);
  gate(r.pipelineRtf < SESSION8_GATES.pipelineRtfHard,
    `${label}: pipeline RTF ${r.pipelineRtf} >= ${SESSION8_GATES.pipelineRtfHard}`);
  gate(r.finalBacklogMs === 0, `${label}: final backlog ${r.finalBacklogMs} != 0`);
  gate(r.backlogGrowthSlopeMsPerSec <= 0,
    `${label}: backlog slope ${r.backlogGrowthSlopeMsPerSec} > 0`);
  gate(r.finishLatencyMs < SESSION8_GATES.finishMs,
    `${label}: finish ${r.finishLatencyMs}ms >= ${SESSION8_GATES.finishMs}ms`);
  gate(r.encoderFramesRecomputed === 0, `${label}: encoder replay=${r.encoderFramesRecomputed}`);
  gate(r.adapterGroupsCommitted === Math.floor(r.encoderFrames / 4),
    `${label}: adapter groups=${r.adapterGroupsCommitted}, encoder frames=${r.encoderFrames}`);
  const terminalEosSteps = r.decoderTokensEmitted < r.decoderSteps ? 1 : 0;
  gate(r.decoderSteps === r.decoderTokensEmitted + terminalEosSteps,
    `${label}: decoder replay/skip steps=${r.decoderSteps}, emitted=${r.decoderTokensEmitted}`);
  gate(r.tokensFlushedAtFinish <= 18,
    `${label}: finish tail ${r.tokensFlushedAtFinish} tokens > 18`);
}

try {
  await prepareSession8(summary, { config });
  if (soak) {
    const seconds = Number(process.env.VOXTRAL_SOAK_SECONDS ?? 600);
    gate(Number.isInteger(seconds) && seconds >= 600,
      `10-minute soak cannot be shortened (requested ${seconds}s)`);
    const run = await runStreamSession({
      config,
      planName: `strict-e2e-soak-${seconds}s`,
      syntheticSeconds: seconds,
      realtimeMs: 80,
      warmup: true,
      skipParity: true,
      monitorMemory: true,
      maxTotalSamples: (seconds + 60) * 16_000,
      env: SESSION8_PRODUCTION_ENV,
      timeoutMs: (seconds + 180) * 1000,
    });
    gateRealtime(run, "10-minute paced soak");
    gate(run.deadlineMissRate < SESSION8_GATES.deadlineMissRate,
      `10-minute paced soak: deadline miss rate ${run.deadlineMissRate}`);
    summary.soak = summarizeRun(run);
  } else {
    const localFixture = path.join(config.localRepo, "voxTest2min.m4a");
    const transfer = await syncFixture(localFixture, { config });
    const fixture = await normalizeFixtureOnGpu({ config });
    gate(fixture.sourceSha256 === transfer.localSha256,
      "normalized fixture provenance differs from transferred SHA-256");
    summary.fixture = {
      sha256: transfer.localSha256,
      wavSha256: fixture.canonicalWavSha256,
      pcmSha256: fixture.canonicalPcmSha256,
      durationMs: fixture.durationMs,
      sampleCount: fixture.sampleCount,
    };

    const paced = await runStreamSession({
      config,
      planName: "strict-e2e-spoken-2m",
      audioPath: fixture.wavPath,
      realtimeMs: 80,
      warmup: true,
      skipParity: true,
      monitorMemory: true,
      env: SESSION8_PRODUCTION_ENV,
      timeoutMs: 420_000,
    });
    summary.paced = summarizeRun(paced);
    gateRealtime(paced, "2-minute spoken paced stream");

    // Large-fixture quality gate: compare the complete paced FP16/4x4
    // production sequence with an independent F32/4x32 finish-only oracle.
    // The oracle is compute-only to avoid another two minutes of artificial
    // pacing; feed boundaries and the canonical PCM remain identical.
    const spokenReference = await runStreamSession({
      config,
      planName: "strict-spoken-f32-reference",
      audioPath: fixture.wavPath,
      mode: "80ms",
      env: {
        ...SESSION8_PRODUCTION_ENV,
        VOXTRAL_STREAM_DECODER: "reference",
        VOXTRAL_ENCODER_KV_TYPE: "f32",
        VOXTRAL_DECODER_KV_TYPE: "f32",
        VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
      },
      timeoutMs: 600_000,
    });
    gate(exactTokens(paced.tokens, spokenReference.tokens),
      "2-minute spoken FP16 production tokens differ from F32/4x32 oracle");
    gate(paced.text === spokenReference.text,
      "2-minute spoken FP16 production transcript differs from F32/4x32 oracle");
    summary.spokenParity = {
      productionTokens: paced.tokens.length,
      referenceTokens: spokenReference.tokens.length,
      transcript: paced.text,
      exact: true,
    };

    const plans = ["full", "80ms", "160ms", "480ms", "seeded-random:20260722"];
    summary.parity = [];
    let acceptedTokens = null;
    let acceptedTranscript = null;
    for (const mode of plans) {
      const name = mode.replaceAll(":", "-");
      const reference = await runStreamSession({
        config,
        planName: `strict-reference-${name}`,
        mode,
        env: { ...SESSION8_PRODUCTION_ENV, VOXTRAL_STREAM_DECODER: "reference" },
        timeoutMs: 300_000,
      });
      const production = await runStreamSession({
        config,
        planName: `strict-production-${name}`,
        mode,
        env: SESSION8_PRODUCTION_ENV,
        timeoutMs: 300_000,
      });
      gate(exactTokens(production.tokens, reference.tokens),
        `${mode}: token divergence from finish-only oracle`);
      gate(production.text === reference.text,
        `${mode}: transcript divergence from finish-only oracle`);
      gateRealtime(production, `${mode}: production`);
      if (acceptedTokens === null) {
        acceptedTokens = production.tokens;
        acceptedTranscript = production.text;
      } else {
        gate(exactTokens(production.tokens, acceptedTokens),
          `${mode}: cross-plan token divergence`);
        gate(production.text === acceptedTranscript,
          `${mode}: cross-plan transcript divergence`);
      }
      summary.parity.push({
        mode,
        nTokens: production.tokens.length,
        transcript: production.text,
        exact: true,
      });
    }
    // Keep the expensive correctness/parity matrix observable even when the
    // published latency budget fails. The suite still exits non-zero: moving
    // this assertion does not relax or reinterpret any gate.
    gateLatency(paced, "2-minute spoken paced stream");
  }
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: soak ? "strict-end-to-end-realtime-soak" : "strict-end-to-end-realtime",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    audioMetadata: summary.fixture ?? null,
    textArtifacts: summary.paced ? {
      "transcript.txt": summary.paced.transcript,
      "token-ids.txt": summary.paced.tokens.join("\n"),
    } : {},
  });
  console.log(`[end-to-end-realtime${soak ? ":soak" : ""}] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[end-to-end-realtime] error: ${summary.error}`);
}

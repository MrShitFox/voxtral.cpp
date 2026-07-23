import crypto from "node:crypto";
import fs from "node:fs";
import { writeFile } from "node:fs/promises";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { createChunkPlan } from "../helpers/chunks.js";
import { runProcess } from "../helpers/exec.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { deriveProductionDecision } from "../helpers/precision-cache.js";
import {
  divergenceRegions,
  semanticRisk,
  sha256Json,
  tokenRecords,
  transcriptMetrics,
  transcriptWordDiff,
} from "../helpers/quality.js";
import { normalizeFixtureOnGpu, runRemote } from "../helpers/remote.js";
import { planCounts, runStreamSession } from "../helpers/stream.js";
import {
  SESSION8_PRODUCTION_ENV,
  exactTokens,
  gate,
  prepareSession8,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  schemaVersion: 1,
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:precision-matrix",
  steps: [],
  fixtures: [],
  variants: {},
  fixtureResults: {},
  globalOracles: {},
};
const textArtifacts = {};
const latestPointer = path.join(config.artifactDir, "session8.1-latest-precision-matrix.txt");

const VARIANTS = Object.freeze([
  {
    id: "A",
    name: "f32-f32",
    encoderKv: "F32",
    decoderKv: "F32",
    encoderElementSize: 4,
    decoderElementSize: 4,
  },
  {
    id: "B",
    name: "encoder-fp16",
    encoderKv: "FP16",
    decoderKv: "F32",
    encoderElementSize: 2,
    decoderElementSize: 4,
  },
  {
    id: "C",
    name: "decoder-fp16",
    encoderKv: "F32",
    decoderKv: "FP16",
    encoderElementSize: 4,
    decoderElementSize: 2,
  },
  {
    id: "D",
    name: "dual-fp16",
    encoderKv: "FP16",
    decoderKv: "FP16",
    encoderElementSize: 2,
    decoderElementSize: 2,
  },
]);

function sha256File(filePath) {
  const hash = crypto.createHash("sha256");
  return new Promise((resolve, reject) => {
    const input = fs.createReadStream(filePath);
    input.on("data", (chunk) => hash.update(chunk));
    input.on("end", () => resolve(hash.digest("hex")));
    input.on("error", reject);
  });
}

async function fixtureSafety(localPath) {
  const tracked = await runProcess(
    "git", ["ls-files", "--error-unmatch", path.basename(localPath)],
    { cwd: config.localRepo, timeoutMs: 10_000 },
  ).then(() => true, () => false);
  const ignored = await runProcess(
    "git", ["check-ignore", "-q", path.basename(localPath)],
    { cwd: config.localRepo, timeoutMs: 10_000 },
  ).then(() => true, () => false);
  gate(!tracked, `${path.basename(localPath)} is tracked by Git`);
  gate(ignored, `${path.basename(localPath)} is not ignored`);
  return { tracked, ignored };
}

function variantEnvironment(variant, extra = {}) {
  return {
    ...SESSION8_PRODUCTION_ENV,
    VOXTRAL_ENCODER_KV_TYPE: variant.encoderKv.toLowerCase().replace("p", ""),
    VOXTRAL_DECODER_KV_TYPE: variant.decoderKv.toLowerCase().replace("p", ""),
    VOXTRAL_CAPTURE_OUTPUT_SHA: "1",
    ...extra,
  };
}

function compactRun(run, { includeTokens = false, includeTimeline = false } = {}) {
  const compact = summarizeRun(run, { includeTokens });
  if (!includeTokens) {
    delete compact.tokens;
    delete compact.transcript;
  }
  compact.tokenSha256 = sha256Json(run.tokens ?? []);
  compact.transcriptSha256 = crypto.createHash("sha256").update(run.text ?? "").digest("hex");
  compact.encoderElementSize = run.encoderKvElementSize;
  compact.decoderElementSize = run.decoderKvElementSize;
  compact.encoderPhysicalRows = run.encoderPhysicalQueryRows;
  compact.encoderLogicalRows = run.encoderLogicalBatchFrames;
  compact.coldOrWarm = run.warmupApplied ? "warm" : "cold";
  if (includeTimeline) compact.tokenTimeline = tokenRecords(run);
  return compact;
}

function validateCompletedRun(run, variant, label) {
  gate(run.state === "completed" && run.finishStatus === "ok",
    `${label}: state=${run.state} finish=${run.finishStatus}`);
  gate(run.evidence?.vulkanEnabled === true && run.evidence?.rx6600Detected === true,
    `${label}: missing RX 6600/Vulkan evidence`);
  gate(run.evidence?.cpuOnlyFallbackDetected !== true, `${label}: CPU-only fallback`);
  gate(run.encoderKvElementSize === variant.encoderElementSize,
    `${label}: encoder element size ${run.encoderKvElementSize}`);
  gate(run.decoderKvElementSize === variant.decoderElementSize,
    `${label}: decoder element size ${run.decoderKvElementSize}`);
  gate(run.encoderLogicalBatchFrames === 4 && run.encoderPhysicalQueryRows === 4,
    `${label}: encoder shape ${run.encoderLogicalBatchFrames}/${run.encoderPhysicalQueryRows}`);
  gate(run.eventsDropped === 0, `${label}: events dropped=${run.eventsDropped}`);
  gate(run.feedQueueFullReturns === 0 && run.backpressureObserved === false,
    `${label}: unexpected event-queue backpressure`);
  gate(run.decoderKvBytesMoved === 0 && run.decoderKvFullBufferMoves === 0,
    `${label}: decoder rollover moved data`);
  gate(run.encoderOutputAccumulatedBytes === 0,
    `${label}: host encoder accumulation=${run.encoderOutputAccumulatedBytes}`);
  gate(run.encoderOutputD2hBytes === 0 && run.adapterInputD2hBytes === 0 &&
       run.adapterOutputD2hBytes === 0 && run.logitsD2hBytes === 0,
  `${label}: production-path D2H is non-zero`);
  gate(run.encoderShaRows === run.encoderFrames,
    `${label}: encoder SHA rows ${run.encoderShaRows} != frames ${run.encoderFrames}`);
  gate(run.adapterShaRows === run.adapterGroupsCommitted,
    `${label}: adapter SHA rows ${run.adapterShaRows} != groups ${run.adapterGroupsCommitted}`);
  gate(/^[a-f0-9]{64}$/u.test(run.encoderSha256) &&
       /^[a-f0-9]{64}$/u.test(run.adapterSha256),
  `${label}: missing output SHA diagnostics`);
}

function exactRunParity(left, right, label) {
  gate(exactTokens(left.tokens, right.tokens), `${label}: token IDs are not deterministic`);
  gate(left.text === right.text, `${label}: transcript is not deterministic`);
  gate(left.encoderSha256 === right.encoderSha256,
    `${label}: encoder output SHA is not deterministic`);
  gate(left.adapterSha256 === right.adapterSha256,
    `${label}: adapter output SHA is not deterministic`);
}

function aggregateVariant(variantId) {
  const fixtureEntries = Object.values(summary.fixtureResults)
    .map((fixture) => fixture.variants[variantId]);
  const quality = fixtureEntries.map((entry) => entry.quality);
  const plans = fixtureEntries.flatMap((entry) => entry.plans);
  // The requested random-chunk plan is a chunk-invariance diagnostic, not a
  // realtime cadence. Only the explicitly named paced 80/160/480 plans
  // participate in sustained-realtime gates.
  const paced = plans.filter((plan) => plan.name.startsWith("paced-"));
  const wordEdits = quality.reduce((sum, item) => sum + item.wer.edits, 0);
  const wordUnits = quality.reduce((sum, item) => sum + item.wer.referenceUnits, 0);
  const charEdits = quality.reduce((sum, item) => sum + item.cer.edits, 0);
  const charUnits = quality.reduce((sum, item) => sum + item.cer.referenceUnits, 0);
  const tokenEdits = quality.reduce((sum, item) => sum + item.tokenDistance, 0);
  const tokenUnits = quality.reduce((sum, item) => sum + item.tokenReferenceCount, 0);
  const risks = quality.map((item) => item.semanticRisk);
  const numericalQualityPassed =
    wordEdits / Math.max(1, wordUnits) <= 0.005 &&
    charEdits / Math.max(1, charUnits) <= 0.0025 &&
    tokenEdits / Math.max(1, tokenUnits) <= 0.005;
  const hardQualityPassed = risks.every((risk) =>
    !risk.changedNumbers &&
    !risk.changedNegations &&
    !risk.sentenceCountChanged &&
    !risk.sustainedTokenDesynchronization);
  const realtimePassed = paced.every((run) =>
    run.pipelineRtf < 0.95 &&
    run.backlog.finalMs === 0 &&
    run.backlog.slopeMsPerSec <= 0 &&
    run.eventsDropped === 0);
  return {
    fixtures: fixtureEntries.length,
    wordEdits,
    wordReferenceUnits: wordUnits,
    wer: wordEdits / Math.max(1, wordUnits),
    charEdits,
    charReferenceUnits: charUnits,
    cer: charEdits / Math.max(1, charUnits),
    tokenEdits,
    tokenReferenceUnits: tokenUnits,
    tokenDivergence: tokenEdits / Math.max(1, tokenUnits),
    numericalQualityPassed,
    hardQualityPassed,
    deterministic: fixtureEntries.every((entry) => entry.deterministic),
    chunkPlanIndependent: fixtureEntries.every((entry) => entry.chunkPlanIndependent),
    realtimePassed,
    maxPipelineRtf: Math.max(...plans.map((run) => run.pipelineRtf)),
    maxFinalBacklogMs: Math.max(...paced.map((run) => run.backlog.finalMs)),
    maxBacklogSlopeMsPerSec: Math.max(...paced.map((run) => run.backlog.slopeMsPerSec)),
    peakVramBytes: Math.max(...plans.map((run) => run.memory.peakVramBytes ?? 0)),
    decoderKvBytes: Math.max(...plans.map((run) => run.memory.decoderKvBytes ?? 0)),
    encoderKvBytes: Math.max(...plans.map((run) => run.memory.encoderKvBytes ?? 0)),
    result: numericalQualityPassed && hardQualityPassed && realtimePassed ? "PASS" : "REJECT",
  };
}

async function runVariantPlan({ fixture, variant, plan, cold = false }) {
  const label = `${fixture.id}-${variant.id}-${plan.name}-${cold ? "cold" : "warm"}`;
  console.log(`[precision-matrix] START ${label}`);
  const run = await runStreamSession({
    config,
    planName: label,
    audioPath: fixture.wavPath,
    mode: plan.mode,
    counts: plan.counts,
    realtimeMs: plan.realtimeMs,
    warmup: !cold,
    skipParity: true,
    monitorMemory: true,
    env: variantEnvironment(variant, cold ? { MESA_SHADER_CACHE_DISABLE: "true" } : {}),
    timeoutMs: plan.realtimeMs > 0
      ? Math.ceil(fixture.durationMs + 300_000)
      : 1_200_000,
  });
  validateCompletedRun(run, variant, label);
  console.log(
    `[precision-matrix] DONE ${label} tokens=${run.tokens.length} ` +
    `RTF=${run.pipelineRtf.toFixed(4)} backlog=${run.finalBacklogMs} ` +
    `VRAM=${run.peakVramBytes}`,
  );
  return run;
}

try {
  await prepareSession8(summary, { config });

  const vram = await runRemote(
    "cat /sys/class/drm/card1/device/mem_info_vram_total",
    { config, timeoutMs: 30_000 },
  );
  summary.rx6600VramTotalBytes = Number(vram.stdout.trim());
  gate(Number.isFinite(summary.rx6600VramTotalBytes) && summary.rx6600VramTotalBytes > 0,
    "RX 6600 VRAM total is unavailable");

  const fixtureDefinitions = [
    {
      id: "voxTest2min",
      file: "voxTest2min.m4a",
      remoteSource: config.remoteFixture2min,
      expectedSha256: "d66181fd86a94d04156cb0d1e1aacaae3d2f97e131af0e38a6fcf29c04ab45a4",
    },
    {
      id: "voxTest4min",
      file: "voxTest4min.m4a",
      remoteSource: config.remoteFixture4min,
      expectedSha256: "e662476be717c53a50bee67a03c0463b5712447c29178952d95285dc887328c3",
    },
  ];

  for (const definition of fixtureDefinitions) {
    const localPath = path.join(config.localRepo, definition.file);
    gate(fs.statSync(localPath).isFile(), `${definition.file} is missing`);
    const localSha256 = await sha256File(localPath);
    gate(localSha256 === definition.expectedSha256,
      `${definition.file}: unexpected local SHA-256 ${localSha256}`);
    const safety = await fixtureSafety(localPath);
    const normalized = await normalizeFixtureOnGpu({
      config,
      sourcePath: definition.remoteSource,
      timeoutMs: 300_000,
    });
    gate(normalized.sourceSha256 === localSha256,
      `${definition.file}: remote source SHA-256 ${normalized.sourceSha256} != local ${localSha256}`);
    gate(normalized.sampleRate === 16_000 && normalized.channels === 1,
      `${definition.file}: canonical audio is not mono 16 kHz`);
    const fixture = {
      ...definition,
      localPath,
      localSha256,
      trackedByGit: safety.tracked,
      ignoredByGit: safety.ignored,
      ...normalized,
      plansTested: ["full", "paced-80ms", "paced-160ms", "paced-480ms", "seeded-random"],
      languageContent: "captured real-world speech; language/content classified from transcript in artifact",
    };
    summary.fixtures.push({
      id: fixture.id,
      file: fixture.file,
      durationMs: fixture.durationMs,
      sourceSha256: fixture.sourceSha256,
      canonicalWavSha256: fixture.canonicalWavSha256,
      canonicalPcmSha256: fixture.canonicalPcmSha256,
      sampleCount: fixture.sampleCount,
      plansTested: fixture.plansTested,
      trackedByGit: fixture.trackedByGit,
      ignoredByGit: fixture.ignoredByGit,
      remoteSource: fixture.remoteSource,
      canonicalWavPath: fixture.wavPath,
    });

    const randomCounts = planCounts(createChunkPlan(
      Buffer.alloc(fixture.sampleCount * 2),
      {
        strategy: "seeded-random",
        seed: 20260723,
        minSamples: 640,
        maxSamples: 7_680,
      },
    ));
    const plans = [
      { name: "full", mode: "full", realtimeMs: 0 },
      { name: "paced-80ms", mode: "80ms", realtimeMs: 80 },
      { name: "paced-160ms", mode: "160ms", realtimeMs: 160 },
      { name: "paced-480ms", mode: "480ms", realtimeMs: 480 },
      { name: "seeded-random", counts: randomCounts, realtimeMs: 0 },
    ];

    console.log(`[precision-matrix] START ${fixture.id}-global-f32-4x32-oracle`);
    const globalOracle = await runStreamSession({
      config,
      planName: `${fixture.id}-global-f32-4x32-oracle`,
      audioPath: fixture.wavPath,
      mode: "full",
      skipParity: true,
      monitorMemory: true,
      env: {
        ...SESSION8_PRODUCTION_ENV,
        VOXTRAL_STREAM_DECODER: "reference",
        VOXTRAL_ENCODER_KV_TYPE: "f32",
        VOXTRAL_DECODER_KV_TYPE: "f32",
        VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
        VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
      },
      timeoutMs: 1_200_000,
    });
    gate(globalOracle.state === "completed" && globalOracle.finishStatus === "ok",
      `${fixture.id}: global F32 4/32 oracle failed`);
    gate(globalOracle.encoderKvElementSize === 4 &&
         globalOracle.decoderKvElementSize === 4 &&
         globalOracle.encoderPhysicalQueryRows === 32,
    `${fixture.id}: invalid global F32 4/32 oracle configuration`);
    summary.globalOracles[fixture.id] = compactRun(globalOracle, { includeTokens: true });
    textArtifacts[`${fixture.id}-global-oracle-transcript.txt`] = globalOracle.text;
    console.log(`[precision-matrix] DONE ${fixture.id}-global-f32-4x32-oracle tokens=${globalOracle.tokens.length}`);

    const fixtureResult = {
      globalOracleTokenSha256: sha256Json(globalOracle.tokens),
      globalOracleTranscriptSha256:
        crypto.createHash("sha256").update(globalOracle.text).digest("hex"),
      variants: {},
    };
    let f32ProductionReference = null;

    for (const variant of VARIANTS) {
      const fullPlan = plans[0];
      const cold = await runVariantPlan({ fixture, variant, plan: fullPlan, cold: true });
      const warm = await runVariantPlan({ fixture, variant, plan: fullPlan, cold: false });
      exactRunParity(cold, warm, `${fixture.id}-${variant.id} cold/warm`);
      if (variant.id === "A") {
        gate(exactTokens(warm.tokens, globalOracle.tokens),
          `${fixture.id}: F32/F32 4/4 tokens differ from global F32 4/32 oracle`);
        gate(warm.text === globalOracle.text,
          `${fixture.id}: F32/F32 4/4 transcript differs from global F32 4/32 oracle`);
        f32ProductionReference = warm;
      }

      const planRuns = [warm];
      for (const plan of plans.slice(1)) {
        const run = await runVariantPlan({ fixture, variant, plan, cold: false });
        exactRunParity(warm, run, `${fixture.id}-${variant.id}-${plan.name}`);
        planRuns.push(run);
      }

      gate(f32ProductionReference !== null,
        `${fixture.id}: F32 production reference unavailable before ${variant.id}`);
      const referenceRecords = tokenRecords(f32ProductionReference);
      const candidateRecords = tokenRecords(warm);
      gate(referenceRecords.every((record, index) =>
        record.id === f32ProductionReference.tokens[index]),
      `${fixture.id}: F32 token event timeline does not match token IDs`);
      gate(candidateRecords.every((record, index) => record.id === warm.tokens[index]),
        `${fixture.id}-${variant.id}: token event timeline does not match token IDs`);
      const divergence = divergenceRegions(referenceRecords, candidateRecords);
      const transcript = transcriptMetrics(f32ProductionReference.text, warm.text);
      const risk = semanticRisk(f32ProductionReference.text, warm.text, divergence);
      const quality = {
        tokenDistance: divergence.tokenDistance,
        tokenReferenceCount: divergence.tokenReferenceCount,
        tokenCandidateCount: divergence.tokenCandidateCount,
        tokenDivergenceRate: divergence.tokenDivergenceRate,
        firstDivergence: divergence.firstDivergence,
        lastDivergence: divergence.lastDivergence,
        sustainedDesynchronization: divergence.sustainedDesynchronization,
        regions: divergence.regions,
        wer: transcript.wer,
        cer: transcript.cer,
        semanticRisk: risk,
      };

      const entry = {
        variant: variant.id,
        encoderKv: variant.encoderKv,
        decoderKv: variant.decoderKv,
        shape: "4/4",
        deterministic: true,
        chunkPlanIndependent: true,
        cold: compactRun(cold),
        representative: compactRun(warm, {
          includeTokens: true,
          includeTimeline: true,
        }),
        plans: planRuns.map((run, index) => ({
          name: plans[index].name,
          ...compactRun(run),
        })),
        quality,
      };
      fixtureResult.variants[variant.id] = entry;
      textArtifacts[`${fixture.id}-${variant.id}-transcript.txt`] = warm.text;
      textArtifacts[`${fixture.id}-${variant.id}-token-ids.txt`] =
        warm.tokens.join("\n");
      textArtifacts[`${fixture.id}-${variant.id}-transcript-diff.txt`] =
        transcriptWordDiff(f32ProductionReference.text, warm.text);
    }
    summary.fixtureResults[fixture.id] = fixtureResult;
  }

  for (const variant of VARIANTS) {
    summary.variants[variant.id] = {
      id: variant.id,
      encoderKv: variant.encoderKv,
      decoderKv: variant.decoderKv,
      shape: "4/4",
      ...aggregateVariant(variant.id),
    };
  }

  summary.productionDecision = deriveProductionDecision(summary);
  gate(summary.productionDecision.selected !== null,
    "no precision variant passed production gates");

  const divergenceLines = [];
  for (const [fixtureId, fixture] of Object.entries(summary.fixtureResults)) {
    for (const [variantId, entry] of Object.entries(fixture.variants)) {
      for (const region of entry.quality.regions) {
        divergenceLines.push(JSON.stringify({
          fixture: fixtureId,
          variant: variantId,
          timestampMs: region.timestampMs,
          referenceIndices: [region.referenceStart, region.referenceEnd],
          candidateIndices: [region.candidateStart, region.candidateEnd],
          referenceIds: region.referenceIds,
          candidateIds: region.candidateIds,
          referencePieces: region.referencePieces,
          candidatePieces: region.candidatePieces,
          contextBeforeReference: region.contextBeforeReference,
          contextAfterReference: region.contextAfterReference,
          contextBeforeCandidate: region.contextBeforeCandidate,
          contextAfterCandidate: region.contextAfterCandidate,
          localReferenceText: region.localReferenceText,
          localCandidateText: region.localCandidateText,
          classification: region.classification,
          meaningChanged: region.meaningChanged,
          reconverged: region.reconverged,
        }));
      }
    }
  }
  textArtifacts["divergence-analysis.txt"] =
    divergenceLines.length > 0 ? divergenceLines.join("\n") : "No token divergences.";

  const matrixRows = [
    "variant,encoder_kv,decoder_kv,token_divergence,wer,cer,max_rtf,max_final_backlog_ms,max_backlog_slope_ms_per_s,peak_vram_bytes,result",
    ...VARIANTS.map(({ id }) => {
      const item = summary.variants[id];
      return [
        id,
        item.encoderKv,
        item.decoderKv,
        item.tokenDivergence,
        item.wer,
        item.cer,
        item.maxPipelineRtf,
        item.maxFinalBacklogMs,
        item.maxBacklogSlopeMsPerSec,
        item.peakVramBytes,
        item.result,
      ].join(",");
    }),
  ];
  textArtifacts["precision-matrix.csv"] = matrixRows.join("\n");
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.stack ?? error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8.1-precision-matrix",
    backend: "Vulkan/RADV RX 6600",
    command: summary.command,
    result: summary,
    audioMetadata: summary.fixtures,
    textArtifacts,
  });
  if (summary.exitCode === 0) {
    await writeFile(latestPointer, `${artifact.directory}\n`);
  }
  console.log(
    `[precision-matrix] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`,
  );
  if (summary.error) console.error(`[precision-matrix] error: ${summary.error}`);
}

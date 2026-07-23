import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runProcess } from "../helpers/exec.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  exactTokens,
  gate,
  prepareSession8,
  session8PrecisionEnvironment,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:real-fixtures",
  steps: [],
  productionRepeats: {},
};
const textArtifacts = {};

function gateProductionRepeat(run, reference, fixtureId, repeat, expected, vramLimit) {
  const label = `${fixtureId}-${expected.id}-paced-80ms-repeat-${repeat}`;
  gate(run.state === "completed" && run.finishStatus === "ok",
    `${label}: state=${run.state} finish=${run.finishStatus}`);
  gate(run.evidence?.vulkanEnabled === true && run.evidence?.rx6600Detected === true,
    `${label}: missing Vulkan/RX 6600 evidence`);
  gate(run.evidence?.cpuOnlyFallbackDetected !== true, `${label}: CPU fallback`);
  gate(run.encoderKvElementSize === (expected.encoderKv === "FP16" ? 2 : 4) &&
       run.decoderKvElementSize === (expected.decoderKv === "FP16" ? 2 : 4),
  `${label}: wrong KV precision`);
  gate(run.encoderLogicalBatchFrames === 4 && run.encoderPhysicalQueryRows === 4,
    `${label}: encoder shape is not 4/4`);
  gate(exactTokens(run.tokens, reference.tokens), `${label}: token IDs differ`);
  gate(run.text === reference.transcript, `${label}: transcript differs`);
  gate(run.encoderSha256 === reference.encoderSha256,
    `${label}: encoder output SHA differs`);
  gate(run.adapterSha256 === reference.adapterSha256,
    `${label}: adapter output SHA differs`);
  gate(run.pipelineRtf < 0.95, `${label}: pipeline RTF ${run.pipelineRtf} >= 0.95`);
  gate(run.finalBacklogMs === 0, `${label}: final backlog ${run.finalBacklogMs} != 0`);
  gate(run.backlogGrowthSlopeMsPerSec <= 0,
    `${label}: backlog slope ${run.backlogGrowthSlopeMsPerSec} > 0`);
  gate(run.eventsDropped === 0, `${label}: events dropped=${run.eventsDropped}`);
  gate(run.feedQueueFullReturns === 0,
    `${label}: feed queue-full returns=${run.feedQueueFullReturns}`);
  gate(run.decoderKvBytesMoved === 0 && run.decoderKvFullBufferMoves === 0,
    `${label}: decoder KV rollover copied bytes`);
  gate(run.encoderOutputAccumulatedBytes === 0,
    `${label}: host encoder accumulation=${run.encoderOutputAccumulatedBytes}`);
  gate(run.encoderOutputD2hBytes === 0 && run.adapterInputD2hBytes === 0 &&
       run.adapterOutputD2hBytes === 0 && run.logitsD2hBytes === 0,
  `${label}: production-path D2H is non-zero`);
  gate(Number.isFinite(run.peakVramBytes) && run.peakVramBytes < vramLimit,
    `${label}: peak VRAM ${run.peakVramBytes} >= ${vramLimit}`);
}

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  summary.sourceArtifact = matrix.directory;
  gate(matrix.result.fixtures.length === 2, "precision artifact does not contain both fixtures");
  const expected = new Map([
    ["voxTest2min.m4a", "d66181fd86a94d04156cb0d1e1aacaae3d2f97e131af0e38a6fcf29c04ab45a4"],
    ["voxTest4min.m4a", "e662476be717c53a50bee67a03c0463b5712447c29178952d95285dc887328c3"],
  ]);
  for (const fixture of matrix.result.fixtures) {
    gate(expected.get(fixture.file) === fixture.sourceSha256,
      `${fixture.file}: unexpected source SHA-256 ${fixture.sourceSha256}`);
    gate(fixture.trackedByGit === false && fixture.ignoredByGit === true,
      `${fixture.file}: unsafe Git fixture state`);
    gate(fixture.plansTested.length === 5 &&
         ["full", "paced-80ms", "paced-160ms", "paced-480ms", "seeded-random"]
           .every((plan) => fixture.plansTested.includes(plan)),
    `${fixture.file}: incomplete chunk-plan matrix`);
    const fixtureResult = matrix.result.fixtureResults[fixture.id];
    for (const variantId of ["A", "B", "C", "D"]) {
      const variant = fixtureResult.variants[variantId];
      gate(variant.deterministic === true,
        `${fixture.id}-${variantId}: cold/warm nondeterminism`);
      gate(variant.chunkPlanIndependent === true,
        `${fixture.id}-${variantId}: chunk-plan-dependent output`);
      gate(variant.plans.length === 5,
        `${fixture.id}-${variantId}: only ${variant.plans.length} plans`);
    }
  }

  const trackedAudio = await runProcess("git", ["ls-files"], {
    cwd: config.localRepo,
    timeoutMs: 10_000,
  });
  const forbiddenTracked = trackedAudio.stdout.split(/\r?\n/u)
    .filter((name) => /voxTest(?:2|4)min/u.test(name));
  gate(forbiddenTracked.length === 0,
    `private/generated audio is tracked: ${forbiddenTracked.join(", ")}`);
  const staged = await runProcess("git", ["diff", "--cached", "--name-only"], {
    cwd: config.localRepo,
    timeoutMs: 10_000,
  });
  const forbiddenStaged = staged.stdout.split(/\r?\n/u)
    .filter((name) => /(?:voxTest(?:2|4)min|\.m4a$|\.wav$|\.pcm$)/u.test(name));
  gate(forbiddenStaged.length === 0,
    `private/generated audio is staged: ${forbiddenStaged.join(", ")}`);

  const selected = matrix.result.productionDecision.selected;
  const selectedVariant = matrix.result.variants[selected];
  gate(selectedVariant, `unknown selected precision ${selected}`);
  await prepareSession8(summary, { config });
  const productionEnv = session8PrecisionEnvironment(selected, {
    VOXTRAL_CAPTURE_OUTPUT_SHA: "1",
  });
  for (const fixture of matrix.result.fixtures) {
    const reference =
      matrix.result.fixtureResults[fixture.id].variants[selected].representative;
    const repeats = [];
    for (let repeat = 1; repeat <= 2; repeat += 1) {
      const label = `${fixture.id}-${selected}-paced-80ms-repeat-${repeat}`;
      console.log(`[real-fixtures] START ${label}`);
      const run = await runStreamSession({
        config,
        planName: label,
        audioPath: fixture.canonicalWavPath,
        mode: "80ms",
        realtimeMs: 80,
        warmup: true,
        skipParity: true,
        monitorMemory: true,
        env: productionEnv,
        timeoutMs: Math.ceil(fixture.durationMs + 300_000),
      });
      gateProductionRepeat(
        run,
        reference,
        fixture.id,
        repeat,
        { id: selected, ...selectedVariant },
        matrix.result.rx6600VramTotalBytes,
      );
      repeats.push(summarizeRun(run, { includeTokens: false }));
      textArtifacts[`${fixture.id}-${selected}-repeat-${repeat}-transcript.txt`] =
        run.text;
      textArtifacts[`${fixture.id}-${selected}-repeat-${repeat}-token-ids.txt`] =
        run.tokens.join("\n");
      console.log(
        `[real-fixtures] DONE ${label} tokens=${run.tokens.length} ` +
        `RTF=${run.pipelineRtf.toFixed(4)} backlog=${run.finalBacklogMs}`,
      );
    }
    summary.productionRepeats[fixture.id] = {
      selected,
      exactDeterminism: true,
      repeats,
    };
  }
  summary.fixtures = matrix.result.fixtures;
  summary.selected = selected;
  summary.fixtureSafety = {
    forbiddenTracked,
    forbiddenStaged,
    statement:
      "voxTest2min.m4a and voxTest4min.m4a were used only as local test fixtures. Neither file nor any generated WAV/PCM derivative is tracked or committed.",
  };
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.stack ?? error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8.1-real-fixtures",
    backend: "Vulkan/RADV RX 6600",
    command: summary.command,
    result: summary,
    audioMetadata: summary.fixtures,
    textArtifacts,
  });
  console.log(`[real-fixtures] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[real-fixtures] error: ${summary.error}`);
}

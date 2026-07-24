import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import {
  exactTokens,
  gateChecks,
  preparePublicApi,
  PUBLIC_PROFILE_ENV,
  publicRunChecks,
  runPublicApi,
} from "../helpers/public-api.js";
import { normalizeFixtureOnGpu } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:public-c-api",
  shortPlans: [],
  longFixtures: [],
};

function internalEventSignature(run) {
  const code = {
    token: "T",
    partial_text: "P",
    final_text: "F",
    error: "E",
    completed: "C",
  };
  return (run.events ?? []).map((event) => code[event.type] ?? "?").join("");
}

function parityChecks(internal, external) {
  const internalPartials =
    (internal.events ?? []).filter((event) => event.type === "partial_text").length;
  return {
    tokensExact: exactTokens(external.tokens, internal.tokens),
    transcriptByteExact: external.transcript === internal.text,
    eventSequenceExact:
      external.eventSignature === internalEventSignature(internal),
    encoderFramesExact:
      external.metrics.encoderFrames === internal.encoderFrames,
    adapterGroupsExact:
      external.metrics.adapterGroups === internal.adapterGroupsCommitted,
    decoderStepsExact:
      external.metrics.decoderSteps === internal.decoderSteps,
    tokenEventsExact:
      external.tokenEvents === internal.tokens.length,
    partialEventsExact:
      external.partialEvents === internalPartials,
    audioDurationExact:
      external.metrics.audioDurationMs === internal.audioDurationMs,
    decoderKvWrapsExact:
      external.metrics.decoderKvWraps === internal.decoderKvWraps,
    decoderKvEvictionsExact:
      external.metrics.decoderKvEvictions === internal.decoderKvEvictions,
    decoderKvMovesExact:
      external.metrics.decoderKvBytesMoved === internal.decoderKvBytesMoved,
    vulkanRx6600:
      internal.evidence?.vulkanEnabled === true &&
      internal.evidence?.rx6600Detected === true &&
      internal.evidence?.cpuOnlyFallbackDetected !== true,
  };
}

function realtimeChecks(run) {
  return {
    paced: run.paced === true,
    pipelineRtfBelowOne: run.metrics.pipelineRtf < 0.95,
    finalBacklogZero: run.metrics.backlogFinalMs === 0,
    backlogNonGrowing: run.metrics.backlogSlopeMsPerS <= 0,
    decoderKvMovesZero: run.metrics.decoderKvBytesMoved === 0,
  };
}

function publicSummary(run) {
  return {
    samples: run.samplesConsumed,
    tokens: run.tokenCount,
    transcript: run.transcript,
    eventSignature: run.eventSignature,
    metrics: run.metrics,
    feedCallMeanMs: run.feedCallMeanMs,
  };
}

try {
  const prepared = await preparePublicApi({ config });
  summary.buildWallMs = prepared.build.wallMs;

  const plans = [
    { name: "full", mode: "full", chunkSamples: 1_000_000_000 },
    { name: "80ms", mode: "80ms", chunkSamples: 1280 },
    { name: "160ms", mode: "160ms", chunkSamples: 2560 },
    { name: "480ms", mode: "480ms", chunkSamples: 7680 },
    {
      name: "seeded-random:20260722",
      mode: "seeded-random:20260722",
      chunkSamples: 7680,
      randomSeed: 20260722,
    },
  ];

  let acceptedTokens = null;
  let acceptedTranscript = null;
  for (const plan of plans) {
    const internal = await runStreamSession({
      config,
      planName: `public-api-oracle-${plan.name.replaceAll(":", "-")}`,
      mode: plan.mode,
      env: PUBLIC_PROFILE_ENV,
      timeoutMs: 360_000,
    });
    const external = await runPublicApi({
      config,
      chunkSamples: plan.chunkSamples,
      randomSeed: plan.randomSeed ?? null,
      timeoutMs: 360_000,
    });
    const contract = publicRunChecks(external);
    const parity = parityChecks(internal, external);
    const crossPlan = {
      tokensExact:
        acceptedTokens === null || exactTokens(external.tokens, acceptedTokens),
      transcriptExact:
        acceptedTranscript === null ||
        external.transcript === acceptedTranscript,
    };
    gateChecks(contract, `${plan.name} public contract`);
    gateChecks(parity, `${plan.name} internal/public parity`);
    gateChecks(crossPlan, `${plan.name} cross-plan parity`);
    acceptedTokens ??= external.tokens;
    acceptedTranscript ??= external.transcript;
    summary.shortPlans.push({
      plan: plan.name,
      contract,
      parity,
      crossPlan,
      public: publicSummary(external),
      internal: {
        tokens: internal.tokens.length,
        transcript: internal.text,
        encoderFrames: internal.encoderFrames,
        adapterGroups: internal.adapterGroupsCommitted,
        decoderSteps: internal.decoderSteps,
      },
    });
    console.log(
      `[public-c-api] short ${plan.name}: PASS ` +
      `tokens=${external.tokenCount}`,
    );
  }

  const fixtures = [
    {
      name: "voxTest2min.m4a",
      sourcePath: config.remoteFixture2min,
      timeoutMs: 600_000,
    },
    {
      name: "voxTest4min.m4a",
      sourcePath: config.remoteFixture4min,
      timeoutMs: 900_000,
    },
  ];
  for (const fixtureSpec of fixtures) {
    const fixture = await normalizeFixtureOnGpu({
      config,
      sourcePath: fixtureSpec.sourcePath,
    });
    const internal = await runStreamSession({
      config,
      planName: `public-api-${fixtureSpec.name}-internal`,
      audioPath: fixture.wavPath,
      realtimeMs: 80,
      warmup: true,
      skipParity: true,
      env: PUBLIC_PROFILE_ENV,
      timeoutMs: fixtureSpec.timeoutMs,
    });
    const external = await runPublicApi({
      config,
      audioPath: fixture.wavPath,
      chunkSamples: 1280,
      paced: true,
      timeoutMs: fixtureSpec.timeoutMs,
    });
    const contract = publicRunChecks(external);
    const parity = parityChecks(internal, external);
    const realtime = realtimeChecks(external);
    gateChecks(contract, `${fixtureSpec.name} public contract`);
    gateChecks(parity, `${fixtureSpec.name} internal/public parity`);
    gateChecks(realtime, `${fixtureSpec.name} public realtime`);
    summary.longFixtures.push({
      fixture: fixtureSpec.name,
      sourceSha256: fixture.sourceSha256,
      wavSha256: fixture.canonicalWavSha256,
      pcmSha256: fixture.canonicalPcmSha256,
      durationMs: fixture.durationMs,
      sampleCount: fixture.sampleCount,
      contract,
      parity,
      realtime,
      public: publicSummary(external),
      internal: {
        tokens: internal.tokens.length,
        transcript: internal.text,
        pipelineRtf: internal.pipelineRtf,
        finalBacklogMs: internal.finalBacklogMs,
        backlogSlopeMsPerS: internal.backlogGrowthSlopeMsPerSec,
      },
    });
    console.log(
      `[public-c-api] ${fixtureSpec.name}: PASS ` +
      `tokens=${external.tokenCount} RTF=${external.metrics.pipelineRtf}`,
    );
  }

  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-c-api] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const latest = summary.longFixtures.at(-1);
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-c-api",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    audioMetadata: latest ? {
      sourceSha256: latest.sourceSha256,
      preparedSha256: latest.wavSha256,
      sampleCount: latest.sampleCount,
      durationMs: latest.durationMs,
    } : null,
    textArtifacts: latest ? {
      "transcript.txt": latest.public.transcript,
    } : {},
  });
  console.log(
    `[public-c-api] ${summary.exitCode === 0 ? "PASS" : "FAIL"} ` +
    `summary: ${artifact.directory}`,
  );
}

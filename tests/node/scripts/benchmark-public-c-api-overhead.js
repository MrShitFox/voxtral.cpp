import path from "node:path";

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
import {
  normalizeFixtureOnGpu,
  runRemote,
  shellQuote,
} from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const repeats = Math.max(
  3,
  Number(process.env.VOXTRAL_PUBLIC_BENCH_REPEATS ?? 3),
);
const longRepeats = Math.max(
  1,
  Number(process.env.VOXTRAL_PUBLIC_BENCH_LONG_REPEATS ?? 1),
);
const skip30m = process.env.VOXTRAL_PUBLIC_BENCH_SKIP_30M === "1";
const VRAM_MEASUREMENT_JITTER_BYTES = 1024 * 1024;
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run benchmark:public-c-api-overhead",
  repeats,
  longRepeats,
  workloads: [],
};

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function increment(value, baseline) {
  if (!Number.isFinite(value) || !Number.isFinite(baseline)) return null;
  return Math.max(0, value - baseline);
}

async function compareWorkload(workload) {
  const pairs = [];
  for (let repeat = 0; repeat < workload.repeats; repeat += 1) {
    const internalOptions = {
      config,
      planName: `public-overhead-${workload.name}-internal-r${repeat}`,
      mode: "80ms",
      warmup: true,
      skipParity: true,
      monitorMemory: true,
      env: PUBLIC_PROFILE_ENV,
      timeoutMs: workload.timeoutMs,
    };
    const publicOptions = {
      config,
      chunkSamples: 1280,
      monitorMemory: true,
      timeoutMs: workload.timeoutMs,
    };
    if (workload.syntheticSeconds) {
      internalOptions.syntheticSeconds = workload.syntheticSeconds;
      publicOptions.syntheticSeconds = workload.syntheticSeconds;
    } else {
      internalOptions.audioPath = workload.audioPath;
      publicOptions.audioPath = workload.audioPath;
    }

    // Alternate launch order so clock/temperature drift does not consistently
    // favour either side of the adapter comparison.
    let internal;
    let external;
    if ((repeat & 1) === 0) {
      internal = await runStreamSession(internalOptions);
      external = await runPublicApi(publicOptions);
    } else {
      external = await runPublicApi(publicOptions);
      internal = await runStreamSession(internalOptions);
    }

    const contract = publicRunChecks(external);
    const parity = {
      tokensExact: exactTokens(external.tokens, internal.tokens),
      transcriptExact: external.transcript === internal.text,
      encoderFramesExact:
        external.metrics.encoderFrames === internal.encoderFrames,
      adapterGroupsExact:
        external.metrics.adapterGroups === internal.adapterGroupsCommitted,
      decoderStepsExact:
        external.metrics.decoderSteps === internal.decoderSteps,
      kvMovesExact:
        external.metrics.decoderKvBytesMoved === internal.decoderKvBytesMoved,
    };
    const measurements = {
      internalRtf:
        Number.isFinite(internal.pipelineRtf) && internal.pipelineRtf > 0,
      publicRtf:
        Number.isFinite(external.metrics.pipelineRtf) &&
        external.metrics.pipelineRtf > 0,
      internalMemory:
        Number.isFinite(internal.baselineVramBytes) &&
        Number.isFinite(internal.peakVramBytes) &&
        Number.isFinite(internal.peakRssKiB),
      publicMemory:
        Number.isFinite(external.baselineVramBytes) &&
        Number.isFinite(external.peakVramBytes) &&
        Number.isFinite(external.peakRssKiB),
      internalBacklog:
        Number.isFinite(internal.finalBacklogMs) &&
        Number.isFinite(internal.backlogGrowthSlopeMsPerSec),
      publicBacklog:
        Number.isFinite(external.metrics.backlogFinalMs) &&
        Number.isFinite(external.metrics.backlogSlopeMsPerS),
    };
    gateChecks(contract, `${workload.name} r${repeat} public contract`);
    gateChecks(parity, `${workload.name} r${repeat} parity`);
    gateChecks(measurements, `${workload.name} r${repeat} measurements`);
    pairs.push({
      repeat,
      contract,
      parity,
      measurements,
      internal: {
        pipelineRtf: internal.pipelineRtf,
        feedCallMeanMs: internal.feedLatencyMeanMs,
        baselineVramBytes: internal.baselineVramBytes,
        peakVramBytes: internal.peakVramBytes,
        peakRssKiB: internal.peakRssKiB,
        encoderOutputD2hBytes: internal.encoderOutputD2hBytes,
        adapterInputD2hBytes: internal.adapterInputD2hBytes,
        adapterOutputD2hBytes: internal.adapterOutputD2hBytes,
        decoderKvBytesMoved: internal.decoderKvBytesMoved,
        finalBacklogMs: internal.finalBacklogMs,
        backlogSlopeMsPerS: internal.backlogGrowthSlopeMsPerSec,
      },
      public: {
        pipelineRtf: external.metrics.pipelineRtf,
        feedCallMeanMs: external.feedCallMeanMs,
        baselineVramBytes: external.baselineVramBytes,
        peakVramBytes: external.peakVramBytes,
        peakRssKiB: external.peakRssKiB,
        decoderKvBytesMoved: external.metrics.decoderKvBytesMoved,
        finalBacklogMs: external.metrics.backlogFinalMs,
        backlogSlopeMsPerS: external.metrics.backlogSlopeMsPerS,
      },
      tokenCount: external.tokenCount,
      transcript: external.transcript,
    });
    console.log(
      `[public-overhead] ${workload.name} r${repeat}: ` +
      `internal RTF=${internal.pipelineRtf} ` +
      `public RTF=${external.metrics.pipelineRtf}`,
    );
  }

  const internalRtf = median(
    pairs.map((pair) => pair.internal.pipelineRtf),
  );
  const publicRtf = median(
    pairs.map((pair) => pair.public.pipelineRtf),
  );
  const rtfDelta = internalRtf > 0
    ? (publicRtf - internalRtf) / internalRtf
    : Infinity;
  const internalVram = median(pairs.map((pair) => increment(
    pair.internal.peakVramBytes,
    pair.internal.baselineVramBytes,
  )));
  const publicVram = median(pairs.map((pair) => increment(
    pair.public.peakVramBytes,
    pair.public.baselineVramBytes,
  )));
  const internalRss = median(
    pairs.map((pair) => pair.internal.peakRssKiB),
  );
  const publicRss = median(
    pairs.map((pair) => pair.public.peakRssKiB),
  );
  const checks = {
    rtfRegressionAtMostOnePercent: rtfDelta <= 0.01,
    peakVramUnchanged:
      publicVram - internalVram <= VRAM_MEASUREMENT_JITTER_BYTES,
    rssDeltaAtMostTwoMiB: publicRss - internalRss <= 2048,
    noNewEncoderD2h:
      pairs.every((pair) => pair.internal.encoderOutputD2hBytes === 0),
    noNewAdapterD2h:
      pairs.every((pair) =>
        pair.internal.adapterInputD2hBytes === 0 &&
        pair.internal.adapterOutputD2hBytes === 0),
    decoderKvMovesZero:
      pairs.every((pair) =>
        pair.internal.decoderKvBytesMoved === 0 &&
        pair.public.decoderKvBytesMoved === 0),
    finalBacklogZero:
      pairs.every((pair) =>
        pair.internal.finalBacklogMs === 0 &&
        pair.public.finalBacklogMs === 0),
    backlogNonGrowing:
      pairs.every((pair) =>
        pair.internal.backlogSlopeMsPerS <= 0 &&
        pair.public.backlogSlopeMsPerS <= 0),
  };
  gateChecks(checks, `${workload.name} performance`);
  return {
    name: workload.name,
    repeats: workload.repeats,
    durationMs: workload.durationMs,
    pairs,
    median: {
      internalRtf,
      publicRtf,
      rtfDelta,
      internalPeakVramIncrementBytes: internalVram,
      publicPeakVramIncrementBytes: publicVram,
      vramDeltaBytes: publicVram - internalVram,
      internalPeakRssKiB: internalRss,
      publicPeakRssKiB: publicRss,
      rssDeltaKiB: publicRss - internalRss,
      internalFeedCallMeanMs: median(
        pairs.map((pair) => pair.internal.feedCallMeanMs),
      ),
      publicFeedCallMeanMs: median(
        pairs.map((pair) => pair.public.feedCallMeanMs),
      ),
    },
    checks,
  };
}

try {
  await preparePublicApi({ config });

  const microBinary = path.posix.join(
    config.remoteBuild,
    "voxtral_public_api_call_benchmark",
  );
  const microProc = await runRemote(shellQuote(microBinary), {
    config,
    timeoutMs: 120_000,
  });
  const microLine = microProc.stdout
    .split(/\r?\n/u)
    .find((line) => line.trim().startsWith("{"));
  if (!microLine) throw new Error("adapter microbenchmark emitted no JSON");
  summary.adapterMicrobenchmark = JSON.parse(microLine);
  gateChecks({
    benchmarkOk: summary.adapterMicrobenchmark.ok === true,
    feedCpuOverheadAtMostPointOneMs:
      summary.adapterMicrobenchmark.adapterOverheadMs <= 0.1,
  }, "adapter microbenchmark");

  const fixture2min = await normalizeFixtureOnGpu({
    config,
    sourcePath: config.remoteFixture2min,
  });
  const fixture4min = await normalizeFixtureOnGpu({
    config,
    sourcePath: config.remoteFixture4min,
  });
  summary.fixtures = {
    twoMinute: {
      sourceSha256: fixture2min.sourceSha256,
      wavSha256: fixture2min.canonicalWavSha256,
      pcmSha256: fixture2min.canonicalPcmSha256,
    },
    fourMinute: {
      sourceSha256: fixture4min.sourceSha256,
      wavSha256: fixture4min.canonicalWavSha256,
      pcmSha256: fixture4min.canonicalPcmSha256,
    },
  };

  const workloads = [
    {
      name: "short",
      audioPath: config.remoteSmokeAudio,
      durationMs: 3580,
      repeats,
      timeoutMs: 360_000,
    },
    {
      name: "2-minute",
      audioPath: fixture2min.wavPath,
      durationMs: fixture2min.durationMs,
      repeats,
      timeoutMs: 720_000,
    },
    {
      name: "4-minute",
      audioPath: fixture4min.wavPath,
      durationMs: fixture4min.durationMs,
      repeats,
      timeoutMs: 1_200_000,
    },
  ];
  if (!skip30m) {
    workloads.push({
      name: "30-minute-synthetic",
      syntheticSeconds: 1800,
      durationMs: 1_800_000,
      repeats: longRepeats,
      timeoutMs: 2_400_000,
    });
  }
  for (const workload of workloads) {
    summary.workloads.push(await compareWorkload(workload));
  }
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-overhead] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-c-api-overhead",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
  });
  console.log(
    `[public-overhead] ${summary.exitCode === 0 ? "PASS" : "FAIL"} ` +
    `summary: ${artifact.directory}`,
  );
}

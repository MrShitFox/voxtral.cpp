import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  SESSION8_PRODUCTION_ENV,
  gate,
  prepareSession8,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run test:rss-rollover-plateau",
  minimumWraps: 40,
  plateauWindowWraps: 20,
  settlementDecoderSteps: 10,
  steps: [],
  runs: {},
};

function selectedEnvironment(selected) {
  const types = {
    A: ["f32", "f32"],
    B: ["f16", "f32"],
    C: ["f32", "f16"],
    D: ["f16", "f16"],
  }[selected];
  gate(types, `unknown production precision ${selected}`);
  return {
    ...SESSION8_PRODUCTION_ENV,
    VOXTRAL_ENCODER_KV_TYPE: types[0],
    VOXTRAL_DECODER_KV_TYPE: types[1],
    VOXTRAL_DECODER_KV_TEST_CAPACITY: "64",
  };
}

function linearRegression(points) {
  if (points.length < 2) return { slope: 0, intercept: points[0]?.y ?? 0, rSquared: 0 };
  const n = points.length;
  const sx = points.reduce((sum, point) => sum + point.x, 0);
  const sy = points.reduce((sum, point) => sum + point.y, 0);
  const sxx = points.reduce((sum, point) => sum + point.x * point.x, 0);
  const sxy = points.reduce((sum, point) => sum + point.x * point.y, 0);
  const denominator = n * sxx - sx * sx;
  const slope = denominator === 0 ? 0 : (n * sxy - sx * sy) / denominator;
  const intercept = (sy - slope * sx) / n;
  const mean = sy / n;
  const total = points.reduce((sum, point) => sum + (point.y - mean) ** 2, 0);
  const residual = points.reduce(
    (sum, point) => sum + (point.y - (intercept + slope * point.x)) ** 2,
    0,
  );
  return {
    slope,
    intercept,
    rSquared: total === 0 ? 1 : Math.max(0, 1 - residual / total),
  };
}

function plateauAnalysis(run, label) {
  const samples = run.rolloverMemory ?? [];
  gate(run.decoderKvWraps >= summary.minimumWraps,
    `${label}: decoder wraps ${run.decoderKvWraps} < ${summary.minimumWraps}`);
  gate(samples.length >= summary.minimumWraps,
    `${label}: only ${samples.length} per-wrap memory samples`);
  gate(samples.every((sample) =>
    sample.decoderKvBytesMoved === 0 && sample.decoderKvFullBufferMoves === 0),
  `${label}: rollover copied decoder KV`);
  gate(run.decoderKvBytesMoved === 0 && run.decoderKvFullBufferMoves === 0,
    `${label}: final decoder KV movement counters are non-zero`);

  const settled = samples.filter((sample) =>
    sample.settledCaptured && sample.settledAfterDecoderSteps >= 10);
  gate(settled.length >= summary.minimumWraps - 3,
    `${label}: only ${settled.length} wraps have >=10-step settlement`);
  const plateau = settled.slice(-summary.plateauWindowWraps);
  gate(plateau.length === summary.plateauWindowWraps,
    `${label}: insufficient tail plateau samples`);

  const rss = plateau.map((sample) => ({ x: sample.wrap, y: sample.settled.rssKiB }));
  const anonymous = plateau.map((sample) => ({
    x: sample.wrap,
    y: sample.settled.anonymousRssKiB,
  }));
  const vram = plateau.map((sample) => ({
    x: sample.wrap,
    y: sample.settled.vramBytes / 1024,
  }));
  const rssRegression = linearRegression(rss);
  const anonymousRegression = linearRegression(anonymous);
  const vramRegression = linearRegression(vram);
  const rssDeltaKiB = rss.at(-1).y - rss[0].y;
  const anonymousDeltaKiB = anonymous.at(-1).y - anonymous[0].y;
  const vramDeltaBytes = (vram.at(-1).y - vram[0].y) * 1024;
  const rssRangeKiB = Math.max(...rss.map((point) => point.y)) -
    Math.min(...rss.map((point) => point.y));
  const anonymousRangeKiB = Math.max(...anonymous.map((point) => point.y)) -
    Math.min(...anonymous.map((point) => point.y));
  const vramRangeBytes = (Math.max(...vram.map((point) => point.y)) -
    Math.min(...vram.map((point) => point.y))) * 1024;
  const approximateLinearRssGrowth =
    rssRegression.slope > 512 &&
    rssRegression.rSquared >= 0.6 &&
    rssDeltaKiB > 8 * 1024;
  const approximateLinearAnonymousGrowth =
    anonymousRegression.slope > 512 &&
    anonymousRegression.rSquared >= 0.6 &&
    anonymousDeltaKiB > 8 * 1024;
  const graphObjects = plateau.map((sample) => sample.graphObjects);
  const graphAllocations = plateau.map((sample) => sample.graphAllocations);
  const decoderAllocations = plateau.map((sample) => sample.decoderAllocations);

  gate(!approximateLinearRssGrowth,
    `${label}: approximately linear RSS growth slope=${rssRegression.slope} KiB/wrap r2=${rssRegression.rSquared}`);
  gate(!approximateLinearAnonymousGrowth,
    `${label}: approximately linear anonymous RSS growth slope=${anonymousRegression.slope} KiB/wrap r2=${anonymousRegression.rSquared}`);
  gate(Math.abs(rssDeltaKiB) <= 16 * 1024 && rssRangeKiB <= 32 * 1024,
    `${label}: RSS did not plateau delta=${rssDeltaKiB} KiB range=${rssRangeKiB} KiB`);
  gate(Math.abs(anonymousDeltaKiB) <= 16 * 1024 && anonymousRangeKiB <= 32 * 1024,
    `${label}: anonymous RSS did not plateau delta=${anonymousDeltaKiB} KiB range=${anonymousRangeKiB} KiB`);
  gate(Math.abs(vramDeltaBytes) <= 8 * 1024 * 1024 && vramRangeBytes <= 16 * 1024 * 1024,
    `${label}: VRAM did not plateau delta=${vramDeltaBytes} range=${vramRangeBytes}`);
  gate(Math.max(...graphObjects) === Math.min(...graphObjects),
    `${label}: graph objects grew across post-warmup wraps`);
  gate(Math.max(...graphAllocations) === Math.min(...graphAllocations),
    `${label}: graph allocations grew across post-warmup wraps`);
  gate(Math.max(...decoderAllocations) === Math.min(...decoderAllocations),
    `${label}: decoder allocations grew across post-warmup wraps`);

  return {
    wraps: run.decoderKvWraps,
    capturedWraps: samples.length,
    settledWraps: settled.length,
    plateauFromWrap: plateau[0].wrap,
    warmupWrapsExcluded: settled.length - plateau.length,
    plateauWindowWraps: plateau.length,
    rss: {
      regressionKiBPerWrap: rssRegression,
      deltaKiB: rssDeltaKiB,
      rangeKiB: rssRangeKiB,
      approximatelyLinearGrowth: approximateLinearRssGrowth,
    },
    anonymousRss: {
      regressionKiBPerWrap: anonymousRegression,
      deltaKiB: anonymousDeltaKiB,
      rangeKiB: anonymousRangeKiB,
      approximatelyLinearGrowth: approximateLinearAnonymousGrowth,
    },
    vram: {
      regressionKiBPerWrap: vramRegression,
      deltaBytes: vramDeltaBytes,
      rangeBytes: vramRangeBytes,
    },
    graphObjects: {
      first: graphObjects[0],
      last: graphObjects.at(-1),
      min: Math.min(...graphObjects),
      max: Math.max(...graphObjects),
    },
    graphAllocations: {
      first: graphAllocations[0],
      last: graphAllocations.at(-1),
      min: Math.min(...graphAllocations),
      max: Math.max(...graphAllocations),
    },
    decoderAllocations: {
      first: decoderAllocations[0],
      last: decoderAllocations.at(-1),
      min: Math.min(...decoderAllocations),
      max: Math.max(...decoderAllocations),
    },
    result: "PASS",
  };
}

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  summary.sourceArtifact = matrix.directory;
  summary.selected = matrix.result.productionDecision.selected;
  const fixture = matrix.result.fixtures.find((item) => item.id === "voxTest4min");
  gate(fixture, "4-minute normalized fixture is missing from precision artifact");
  summary.fixture = fixture;
  await prepareSession8(summary, { config });

  for (const [name, mallocTrimAfter] of [["without-trim", false], ["diagnostic-trim", true]]) {
    console.log(`[rss-rollover] START ${name}`);
    const run = await runStreamSession({
      config,
      planName: `session8.1-rss-${name}`,
      audioPath: fixture.canonicalWavPath,
      mode: "80ms",
      warmup: true,
      skipParity: true,
      monitorMemory: true,
      captureRolloverMemory: true,
      mallocTrimAfter,
      discardEventHistory: true,
      env: selectedEnvironment(summary.selected),
      timeoutMs: 900_000,
    });
    gate(run.state === "completed" && run.finishStatus === "ok",
      `${name}: stream failed`);
    gate(run.childExited === true && run.childExitRssKiB === 0,
      `${name}: child exit was not observed`);
    gate(run.eventHistoryRetained === false &&
         run.retainedEventHistoryCount === 0,
    `${name}: test consumer retained event history`);
    gate(run.memoryAttribution?.vulkanAllocationCountAvailable === false &&
         run.memoryAttribution?.hostAllocationCountAvailable === false,
    `${name}: allocation attribution availability was not explicit`);
    if (mallocTrimAfter) {
      gate(run.mallocTrimRequested === true &&
           run.mallocTrimAvailable === true &&
           run.mallocTrimApplied === true,
      `${name}: diagnostic malloc_trim was not applied after destroy`);
    } else {
      gate(run.mallocTrimRequested === false && run.mallocTrimApplied === false,
        `${name}: malloc_trim unexpectedly affected steady-state run`);
    }
    summary.runs[name] = {
      run: summarizeRun(run, { includeTokens: false }),
      memorySnapshots: run.memorySnapshots,
      rolloverMemory: run.rolloverMemory,
      analysis: null,
    };
    let analysis;
    try {
      analysis = plateauAnalysis(run, name);
      summary.runs[name].analysis = analysis;
    } catch (error) {
      summary.runs[name].analysis = {
        result: "FAIL",
        error: error.message,
      };
      throw error;
    }
    console.log(
      `[rss-rollover] DONE ${name} wraps=${analysis.wraps} ` +
      `rssSlope=${analysis.rss.regressionKiBPerWrap.slope.toFixed(2)} KiB/wrap`,
    );
  }

  summary.trimDiagnostic = {
    beforeKiB: summary.runs["diagnostic-trim"].memorySnapshots.afterDestroy.rssKiB,
    afterKiB: summary.runs["diagnostic-trim"].memorySnapshots.afterMallocTrim.rssKiB,
    releasedKiB:
      summary.runs["diagnostic-trim"].memorySnapshots.afterDestroy.rssKiB -
      summary.runs["diagnostic-trim"].memorySnapshots.afterMallocTrim.rssKiB,
    interpretation:
      "malloc_trim was called only after stream destruction as a diagnostic; it is not part of the production steady-state path.",
  };
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.stack ?? error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const rows = [
    "mode,wrap,absolute_position,rss_before_kib,rss_after_kib,rss_settled_kib,anonymous_settled_kib,vram_before_bytes,vram_after_bytes,vram_settled_bytes,graph_objects,graph_allocations,decoder_allocations,kv_bytes_moved,full_buffer_moves,settled_after_steps",
  ];
  for (const [mode, run] of Object.entries(summary.runs)) {
    for (const sample of run.rolloverMemory ?? []) {
      rows.push([
        mode,
        sample.wrap,
        sample.absolutePosition,
        sample.before.rssKiB,
        sample.after.rssKiB,
        sample.settled.rssKiB,
        sample.settled.anonymousRssKiB,
        sample.before.vramBytes,
        sample.after.vramBytes,
        sample.settled.vramBytes,
        sample.graphObjects,
        sample.graphAllocations,
        sample.decoderAllocations,
        sample.decoderKvBytesMoved,
        sample.decoderKvFullBufferMoves,
        sample.settledAfterDecoderSteps,
      ].join(","));
    }
  }
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8.1-rss-rollover-plateau",
    backend: "Vulkan/RADV RX 6600",
    command: summary.command,
    result: summary,
    textArtifacts: { "rss-rollover.csv": rows.join("\n") },
  });
  console.log(`[rss-rollover] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[rss-rollover] error: ${summary.error}`);
}

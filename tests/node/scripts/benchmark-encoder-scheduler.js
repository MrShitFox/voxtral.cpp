// Staged logical/physical encoder scheduler sweep.
//
// The executable is intentionally driven by one remote process per case. The
// process itself owns the complete PCM timeline and performs pacing, so no SSH
// round-trip occurs at chunk boundaries. The private M4A is transferred and
// normalized separately from source sync; only hashes and measurements enter
// the local artifact bundle.
import fs from "node:fs/promises";
import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { runProcess } from "../helpers/exec.js";
import {
  checkRemoteConnection,
  normalizeFixtureOnGpu,
  runRemote,
  shellQuote,
  syncFixture,
  syncSources,
} from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const longFixture = process.env.VOXTRAL_LONG_AUDIO
  ? path.resolve(process.env.VOXTRAL_LONG_AUDIO)
  : path.join(config.localRepo, "voxTest2min.m4a");
const startedAt = new Date().toISOString();
const summary = {
  startedAt,
  command: "npm run benchmark:encoder-scheduler",
  scheduler: "static",
  sourceSyncExcludesFixture: true,
  cases: [],
  fixture: null,
};

function silenceWav(samples) {
  const dataBytes = samples * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(16_000, 24);
  buf.writeUInt32LE(32_000, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(dataBytes, 40);
  return buf;
}

function caseEnv(logical, physical) {
  return {
    VOXTRAL_ENC_KV_LOGICAL_BATCH: String(logical),
    VOXTRAL_ENC_KV_PHYSICAL_ROWS: String(physical),
    VOXTRAL_ENCODER_TELEMETRY: "1",
  };
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function casesCsv(cases) {
  const columns = [
    "phase", "logical", "physical", "paceMs", "encoderResidenceP50Ms",
    "encoderResidenceP95Ms", "encoderResidenceP99Ms", "encoderResidenceMaxMs",
    "adapterGroupResidenceP95Ms", "encoderComputeP95Ms", "encoderComputeMaxMs",
    "backlogMaxMs", "finalBacklogMs", "backlogGrowthSlopeMsPerSec",
    "encoderPhysicalOverheadRatio", "realtimeFactor", "encoderMaxAbsDeltaVsBatch",
    "encoderTransformerFramesComputed", "encoderUniqueFrames", "encoderSha256",
  ];
  return [
    columns.map(csvCell).join(","),
    ...cases.map((item) => columns.map((column) => csvCell(item[column])).join(",")),
  ].join("\n");
}

function compactResult(result, logical, physical, phase, paceMs = 0) {
  const r = result;
  return {
    phase,
    logical,
    physical,
    paceMs,
    mode: r.mode,
    planName: r.planName,
    state: r.state,
    finishStatus: r.finishStatus,
    parityChecked: r.parityChecked,
    encoderSha256: r.encoderSha256,
    tokens: r.tokens,
    text: r.text,
    encoderMaxAbsDeltaVsBatch: r.encoderMaxAbsDeltaVsBatch,
    melMaxAbsDeltaVsBatch: r.melMaxAbsDeltaVsBatch,
    encoderUniqueFrames: r.encoderUniqueFrames,
    encoderTransformerFramesComputed: r.encoderTransformerFramesComputed,
    encoderWorkRatio: r.encoderWorkRatio,
    encoderLogicalFramesSubmitted: r.encoderLogicalFramesSubmitted,
    encoderPhysicalQueryRowsEvaluated: r.encoderPhysicalQueryRowsEvaluated,
    encoderPaddingRowsEvaluated: r.encoderPaddingRowsEvaluated,
    encoderPhysicalOverheadRatio: r.encoderPhysicalOverheadRatio,
    encoderGraphExecutions: r.encoderGraphExecutions,
    encoderKvAppends: r.encoderKvAppends,
    encoderWarmupFrames: r.encoderWarmupFrames,
    encoderKvWraps: r.encoderKvWraps,
    encoderKvEvictions: r.encoderKvEvictions,
    encoderKvCapacityFrames: r.encoderKvCapacityFrames,
    encoderFramesComputedDuringFinish: r.encoderFramesComputedDuringFinish,
    encoderFirstFrameAbsoluteMs: r.encoderFirstFrameAbsoluteMs,
    encoderFirstFrameResidenceMs: r.encoderFirstFrameResidenceMs,
    firstMelFrameAbsoluteMs: r.firstMelFrameAbsoluteMs,
    firstAdapterGroupAbsoluteMs: r.firstAdapterGroupAbsoluteMs,
    firstAdapterGroupResidenceMs: r.firstAdapterGroupResidenceMs,
    firstEightFrameGroupAbsoluteMs: r.firstEightFrameGroupAbsoluteMs,
    firstEightFrameGroupResidenceMs: r.firstEightFrameGroupResidenceMs,
    encoderResidenceP50Ms: r.encoderResidenceP50Ms,
    encoderResidenceP95Ms: r.encoderResidenceP95Ms,
    encoderResidenceP99Ms: r.encoderResidenceP99Ms,
    encoderResidenceMaxMs: r.encoderResidenceMaxMs,
    adapterGroupResidenceP50Ms: r.adapterGroupResidenceP50Ms,
    adapterGroupResidenceP95Ms: r.adapterGroupResidenceP95Ms,
    adapterGroupResidenceP99Ms: r.adapterGroupResidenceP99Ms,
    adapterGroupResidenceMaxMs: r.adapterGroupResidenceMaxMs,
    encoderComputeP50Ms: r.encoderComputeP50Ms,
    encoderComputeP95Ms: r.encoderComputeP95Ms,
    encoderComputeP99Ms: r.encoderComputeP99Ms,
    encoderComputeMaxMs: r.encoderComputeMaxMs,
    encoderComputeWarmMaxMs: r.encoderComputeWarmMaxMs,
    feedStartLatenessP95Ms: r.feedStartLatenessP95Ms,
    feedStartLatenessMaxMs: r.feedStartLatenessMaxMs,
    feedFinishLatenessP95Ms: r.feedFinishLatenessP95Ms,
    feedFinishLatenessMaxMs: r.feedFinishLatenessMaxMs,
    backlogP95Ms: r.backlogP95Ms,
    backlogP99Ms: r.backlogP99Ms,
    backlogMaxMs: r.backlogMaxMs,
    finalBacklogMs: r.finalBacklogMs,
    backlogGrowthSlopeMsPerSec: r.backlogGrowthSlopeMsPerSec,
    realtimeFactor: r.realtimeFactor,
    wallDurationMs: r.wallDurationMs,
    melHistoryRetained: r.melHistoryRetained,
    encoderMelPeakRetainedFrames: r.encoderMelPeakRetainedFrames,
    encoderMelRetainedBytes: r.encoderMelRetainedBytes,
    decoderKvAllocatedBytes: r.decoderKvAllocatedBytes,
    modelLoadedVramBytes: r.modelLoadedVramBytes,
    streamIdleVramBytes: r.streamIdleVramBytes,
    afterFinishVramBytes: r.afterFinishVramBytes,
    afterDestroyVramBytes: r.afterDestroyVramBytes,
    finishFrontendMs: r.finishFrontendMs,
    finishEncoderMs: r.finishEncoderMs,
    finishDecoderMs: r.finishDecoderMs,
  };
}

function assertCorrectness(result, label) {
  if (result.state !== "completed" || result.finishStatus !== "ok") {
    throw new Error(`${label}: stream did not complete (${result.state}/${result.finishStatus})`);
  }
  if (result.encoderTransformerFramesComputed !== result.encoderUniqueFrames) {
    throw new Error(`${label}: real encoder work replayed`);
  }
  if (result.encoderMaxAbsDeltaVsBatch > 1e-5 || result.melMaxAbsDeltaVsBatch > 1e-5) {
    throw new Error(`${label}: tensor parity failed (encoder=${result.encoderMaxAbsDeltaVsBatch}, mel=${result.melMaxAbsDeltaVsBatch})`);
  }
}

async function runCase({ phase, logical, physical, audioPath, mode = "full", realtimeMs = 0, skipParity = true, planName }) {
  const result = await runStreamSession({
    config,
    planName: planName ?? `${phase}-${logical}-${physical}-${realtimeMs || mode}`,
    mode,
    realtimeMs,
    audioPath,
    maxTokens: 1,
    skipParity,
    env: caseEnv(logical, physical),
    timeoutMs: realtimeMs > 0 ? 420_000 : 300_000,
  });
  const compact = compactResult(result, logical, physical, phase, realtimeMs);
  summary.cases.push(compact);
  assertCorrectness(result, `${phase} ${logical}/${physical}`);
  console.log(JSON.stringify({
    phase, logical, physical, paceMs: realtimeMs,
    residenceP95Ms: compact.encoderResidenceP95Ms,
    adapterP95Ms: compact.adapterGroupResidenceP95Ms,
    backlogMaxMs: compact.backlogMaxMs,
    paddingRatio: compact.encoderPhysicalOverheadRatio,
    rtf: compact.realtimeFactor,
  }));
  return result;
}

function assertEquivalent(results, label) {
  if (results.length < 2) return;
  const reference = results[0];
  for (const result of results.slice(1)) {
    if (result.encoderSha256 !== reference.encoderSha256) {
      throw new Error(`${label}: encoder tensor diverged across scheduler candidates`);
    }
    if (JSON.stringify(result.tokens) !== JSON.stringify(reference.tokens)) {
      throw new Error(`${label}: token sequence diverged across scheduler candidates`);
    }
    if (result.text !== reference.text) {
      throw new Error(`${label}: transcript diverged across scheduler candidates`);
    }
  }
}

async function putRemoteWav(name, wav) {
  const remotePath = `${config.remoteRepo}/.encoder-scheduler-${name}.wav`;
  await runRemote(`cat > ${shellQuote(remotePath)}`, { config, input: wav, timeoutMs: 60_000 });
  return remotePath;
}

async function main() {
  await checkRemoteConnection({ config });

  // Safety evidence is collected before any transfer. `.git/info/exclude` is
  // local-only, so source rsync has an explicit independent fixture exclusion.
  const ignore = await runProcess("git", ["check-ignore", "-v", longFixture], {
    cwd: config.localRepo,
    rejectOnNonZero: false,
    timeoutMs: 10_000,
  });
  summary.fixtureIgnored = ignore.exitCode === 0;
  const tracked = await runProcess("git", ["ls-files", "--error-unmatch", longFixture], {
    cwd: config.localRepo,
    rejectOnNonZero: false,
    timeoutMs: 10_000,
  });
  summary.fixtureTracked = tracked.exitCode === 0;
  if (summary.fixtureTracked) throw new Error("private fixture is tracked");

  let longAudioPath = null;
  try {
    await fs.access(longFixture);
    const transfer = await syncFixture(longFixture, { config });
    summary.fixture = await normalizeFixtureOnGpu({ config });
    if (transfer.localSha256 !== summary.fixture.sourceSha256 ||
        transfer.remoteSha256 !== summary.fixture.sourceSha256) {
      throw new Error("fixture SHA-256 changed between local source, transfer and normalization");
    }
    summary.fixture.localPath = longFixture;
    summary.fixture.localBytes = (await fs.stat(longFixture)).size;
    summary.fixture.localSourceSha256 = transfer.localSha256;
    summary.fixture.remoteSourceSha256 = transfer.remoteSha256;
    summary.fixture.transferWallMs = transfer.wallMs;
    longAudioPath = `${summary.fixture.wavPath}`;
  } catch (error) {
    if (process.env.VOXTRAL_LONG_AUDIO) throw error;
    summary.fixture = { skipped: true, reason: `optional fixture unavailable: ${error.message}` };
  }

  await syncSources({ config });
  await buildRemoteVulkan({ config });

  // Sweep A: every supported logical size that fits each fixed physical shape
  // on the short spoken fixture. One correctness run per shape is intentional.
  const shortAudio = config.remoteSmokeAudio;
  for (const physical of [32, 64, 128]) {
    for (const logical of [1, 2, 4, 8, 16, 32, 64, 128].filter((value) => value <= physical)) {
      await runCase({
        phase: "short-compute",
        logical, physical, audioPath: shortAudio, mode: "full", skipParity: false,
      });
    }
  }

  // Sweep B: shortlisted candidates on a deterministic 30-second signal. The
  // synthetic signal exercises ring rollover/backlog mechanics without adding
  // any binary fixture to Git.
  const synthetic = await putRemoteWav("30s", silenceWav(30 * 16_000));
  const syntheticResults = [];
  for (const [logical, physical] of [[4, 32], [8, 32], [16, 32], [32, 32], [128, 128]]) {
    syntheticResults.push(await runCase({ phase: "synthetic-30s-compute", logical, physical, audioPath: synthetic, mode: "full" }));
  }
  assertEquivalent(syntheticResults, "synthetic 30 s");
  await runRemote(`rm -f ${shellQuote(synthetic)}`, { config, timeoutMs: 30_000 });

  if (longAudioPath) {
    // Sweep C: required candidates on the real spoken fixture. Compute-only
    // establishes cost; paced 80/160 ms runs establish residence/backlog.
    const computeResults = [];
    for (const [logical, physical] of [[4, 32], [8, 32], [16, 32], [32, 32], [128, 128]]) {
      computeResults.push(await runCase({ phase: "spoken-120s-compute", logical, physical, audioPath: longAudioPath, mode: "full" }));
    }
    assertEquivalent(computeResults, "spoken compute");
    const paced80Results = [];
    for (const [logical, physical] of [[4, 32], [8, 32], [16, 32], [32, 32], [128, 128]]) {
      paced80Results.push(await runCase({ phase: "spoken-120s-paced-80ms", logical, physical, audioPath: longAudioPath, realtimeMs: 80 }));
    }
    assertEquivalent(paced80Results, "spoken paced 80 ms");
    const paced160Results = [];
    for (const [logical, physical] of [[4, 32], [8, 32], [16, 32], [32, 32], [128, 128]]) {
      paced160Results.push(await runCase({ phase: "spoken-120s-paced-160ms", logical, physical, audioPath: longAudioPath, realtimeMs: 160 }));
    }
    assertEquivalent(paced160Results, "spoken paced 160 ms");
  }

  // Ranking is intentionally transparent: correctness/work/backlog are hard
  // gates; residence p95/p99/max dominate the secondary padding/RTF terms.
  const pacedRankingSource = summary.cases.filter((r) => r.phase === "spoken-120s-paced-80ms");
  const rankingSource = pacedRankingSource.length > 0
    ? pacedRankingSource
    : summary.cases.filter((r) => r.phase === "synthetic-30s-compute");
  const ranked = rankingSource
    .map((r) => {
      const hardFail = r.encoderTransformerFramesComputed !== r.encoderUniqueFrames
        || (r.parityChecked && r.encoderMaxAbsDeltaVsBatch > 1e-5)
        || r.finalBacklogMs > 160
        || r.backlogGrowthSlopeMsPerSec > 1;
      const score = hardFail ? Number.POSITIVE_INFINITY
        : r.encoderResidenceP95Ms * 10
          + r.encoderResidenceP99Ms * 3
          + r.encoderResidenceMaxMs
          + r.adapterGroupResidenceP95Ms * 5
          + r.encoderPhysicalOverheadRatio * 2
          + Math.max(0, r.backlogMaxMs) * 4;
      return { ...r, hardFail, score };
    })
    .sort((a, b) => a.score - b.score);
  summary.ranking = ranked;
  console.log("[encoder-scheduler] ranking (correctness/backlog hard gates, then residence):");
  for (const [index, item] of ranked.entries()) {
    console.log(JSON.stringify({
      rank: index + 1,
      phase: item.phase,
      logical: item.logical,
      physical: item.physical,
      hardFail: item.hardFail,
      score: item.score,
      residenceP95Ms: item.encoderResidenceP95Ms,
      residenceP99Ms: item.encoderResidenceP99Ms,
      residenceMaxMs: item.encoderResidenceMaxMs,
      backlogMaxMs: item.backlogMaxMs,
      paddingRatio: item.encoderPhysicalOverheadRatio,
      realtimeFactor: item.realtimeFactor,
    }));
  }
  summary.finishedAt = new Date().toISOString();
  summary.exitCode = 0;
}

try {
  await main();
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt ??= new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "encoder-scheduler-benchmark",
    backend: "Vulkan",
    command: "npm run benchmark:encoder-scheduler",
    audioMetadata: summary.fixture,
    result: summary,
    textArtifacts: { "summary.csv": casesCsv(summary.cases) },
  });
  console.log(`[encoder-scheduler] summary: ${artifact.directory}`);
}

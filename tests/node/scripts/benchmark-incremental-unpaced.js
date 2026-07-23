// Stage 5 — unpaced maximum-throughput benchmark.
//
// Realtime pacing hides the compute ceiling behind sleeps that wait for the
// next audio chunk. This benchmark drives the SAME incremental production
// scheduler (identical 4-encoder-frame / adapter-group cadence, identical
// precision) with the whole clip immediately available and no 80 ms sleep, so
// it measures how fast the pipeline can actually run. For the 2/4-minute
// fixtures it also runs the paced production path and reports the paced-vs-
// unpaced delta: if unpaced RTF ~= paced compute ratio the pipeline is GPU-
// bound; a large gap would indicate pacing-induced serialization to chase.

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { canonicalWav, profileRun, summarizeRepeats } from "../helpers/benchmark-run.js";

const config = loadEnvironment();
const repeats = Math.max(1, Number(process.env.VOXTRAL_UNPACED_REPEATS || 3));
const runSoak = process.env.VOXTRAL_UNPACED_SOAK === "1"; // 30-min synthetic unpaced (~21 min wall)

const workloads = [
  { name: "short", audio: config.remoteSmokeAudio, paced: false },
  { name: "2min", audio: canonicalWav(config.remoteFixture2min), paced: false },
  { name: "4min", audio: canonicalWav(config.remoteFixture4min), paced: false },
  { name: "2min", audio: canonicalWav(config.remoteFixture2min), paced: true },
  { name: "4min", audio: canonicalWav(config.remoteFixture4min), paced: true },
];
if (runSoak) workloads.push({ name: "synthetic30min", syntheticSeconds: 1800, paced: false });

function throughput(agg, run) {
  const wallSec = (run.wallDurationMs || 0) / 1000;
  const encExec = run.profileStages?.encoder_graph_execute?.count || 0;
  return {
    audioSecPerWallSec: run.realtimeFactor > 0 ? Number((1 / run.realtimeFactor).toFixed(4)) : 0,
    pipelineRtf: agg.derived.pipelineRtf,
    wallRtf: agg.derived.wallRtf,
    computeHeadroomRatio: agg.derived.computeHeadroomRatio,
    decoderStepsPerSec: wallSec > 0 ? Number(((run.decoderSteps || 0) / wallSec).toFixed(2)) : 0,
    encoderFramesPerSec: wallSec > 0 ? Number(((encExec * 4) / wallSec).toFixed(2)) : 0,
    adapterGroupsPerSec: wallSec > 0 ? Number(((run.adapterGroupsCommitted || 0) / wallSec).toFixed(2)) : 0,
    gpuBusyMean: agg.derived.gpu?.busyMean,
    sclkMean: agg.derived.gpu?.sclkMean,
    backendSyncCount: agg.derived.backendSyncCount,
    commandSubmitCount: agg.derived.commandSubmitCount,
  };
}

const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run benchmark:incremental-unpaced",
  repeats,
  results: [],
};

try {
  for (const w of workloads) {
    const reps = w.syntheticSeconds ? 1 : repeats;
    const runs = [];
    for (let i = 0; i < reps; i++) {
      const { run, agg } = await profileRun({
        config,
        label: `unpaced-${w.name}-${w.paced ? "paced" : "unpaced"}-r${i}`,
        audioPath: w.audio || null,
        syntheticSeconds: w.syntheticSeconds || 0,
        paced: w.paced,
        timeoutMs: w.syntheticSeconds ? 1_800_000 : 900_000,
      });
      if (run.state !== "completed") throw new Error(`${w.name} ${w.paced ? "paced" : "unpaced"} r${i}: state=${run.state}`);
      runs.push({ run, agg, tp: throughput(agg, run) });
      const tp = runs.at(-1).tp;
      console.log(`[unpaced] ${w.name} ${w.paced ? "paced " : "unpaced"} r${i}: RTF=${tp.pipelineRtf} wallRTF=${tp.wallRtf} tok/s=${tp.decoderStepsPerSec} busy=${tp.gpuBusyMean}% sclk=${tp.sclkMean}`);
    }
    summary.results.push({
      workload: w.name,
      mode: w.paced ? "paced" : "unpaced",
      tokens: runs[0].run.tokens?.length ?? null,
      pipelineRtf: summarizeRepeats(runs.map((r) => r.tp.pipelineRtf)),
      wallRtf: summarizeRepeats(runs.map((r) => r.tp.wallRtf)),
      decoderStepsPerSec: summarizeRepeats(runs.map((r) => r.tp.decoderStepsPerSec)),
      gpuBusyMean: summarizeRepeats(runs.map((r) => r.tp.gpuBusyMean)),
      sclkMean: summarizeRepeats(runs.map((r) => r.tp.sclkMean)),
      representative: runs[Math.floor(runs.length / 2)].tp,
    });
  }

  // paced-vs-unpaced contrast per fixture
  summary.pacedVsUnpaced = [];
  for (const name of ["2min", "4min"]) {
    const u = summary.results.find((r) => r.workload === name && r.mode === "unpaced");
    const p = summary.results.find((r) => r.workload === name && r.mode === "paced");
    if (u && p) {
      summary.pacedVsUnpaced.push({
        fixture: name,
        unpacedRtf: u.pipelineRtf.median,
        pacedRtf: p.pipelineRtf.median,
        unpacedBusy: u.gpuBusyMean.median,
        pacedBusy: p.gpuBusyMean.median,
        unpacedSclk: u.sclkMean.median,
        pacedSclk: p.sclkMean.median,
      });
    }
  }
  console.log("[unpaced] paced-vs-unpaced:", JSON.stringify(summary.pacedVsUnpaced, null, 2));
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[unpaced] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config, testName: "incremental-unpaced-throughput", backend: "Vulkan",
    command: summary.command, result: summary,
  });
  console.log(`[unpaced] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
}

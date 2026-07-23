// Stage 11 — RX 6600 hardware-ceiling sweep.
//
// Runs the production incremental pipeline (variant C: encoder KV F32 +
// decoder KV FP16, 4/4) unpaced on the 2-minute fixture under two amdgpu DPM
// policies — the box default `auto` and forced `high` (pins the top sclk/mclk
// DPM state) — and compares full-pipeline RTF against GPU busy%, clocks, power
// and temperature. This isolates how much of the gap to the compute ceiling is
// the conservative DPM policy versus the graph structure itself.
//
// The original power level is always restored, even on error. Token output must
// be identical across power states (power cannot change compute results); the
// script asserts that as a correctness guard.

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { canonicalWav, profileRun, summarizeRepeats } from "../helpers/benchmark-run.js";
import { readPowerLevel, setPowerLevel, readClockTables, VALID_LEVELS } from "../helpers/gpu-power.js";

const config = loadEnvironment();
const fixture = process.env.VOXTRAL_CEILING_FIXTURE || canonicalWav(config.remoteFixture2min);
const repeats = Math.max(1, Number(process.env.VOXTRAL_CEILING_REPEATS || 3));
const levels = (process.env.VOXTRAL_CEILING_LEVELS || "auto,high")
  .split(",").map((s) => s.trim()).filter((s) => VALID_LEVELS.has(s));

const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run benchmark:hardware-ceiling",
  fixture,
  repeats,
  levels,
  perLevel: [],
};

let originalLevel = null;
try {
  originalLevel = await readPowerLevel({ config });
  summary.originalPowerLevel = originalLevel;
  let oracleTokens = null;

  for (const level of levels) {
    await setPowerLevel(level, { config });
    const clockTable = await readClockTables({ config });
    const runs = [];
    for (let i = 0; i < repeats; i++) {
      const cold = i === 0;
      const { run, agg } = await profileRun({
        config,
        label: `ceiling-${level}-r${i}`,
        audioPath: fixture,
        paced: false,
        warmup: !cold, // first repeat cold (no graph warmup), rest warm
        env: cold ? { MESA_SHADER_CACHE_DISABLE: "true" } : {},
      });
      if (run.state !== "completed") throw new Error(`ceiling ${level} r${i}: state=${run.state}`);
      if (oracleTokens === null) oracleTokens = run.tokens;
      else if (JSON.stringify(run.tokens) !== JSON.stringify(oracleTokens)) {
        throw new Error(`ceiling ${level} r${i}: token output changed across power state (correctness violation)`);
      }
      runs.push({ cold, agg: agg.derived });
      console.log(`[ceiling] ${level} r${i}${cold ? " cold" : " warm"}: RTF=${agg.derived.pipelineRtf} sclkMean=${agg.derived.gpu?.sclkMean} busy=${agg.derived.gpu?.busyMean} power=${agg.derived.gpu?.powerMeanW}W temp=${agg.derived.gpu?.tempMaxC}C`);
    }
    const warm = runs.filter((r) => !r.cold);
    const pool = warm.length ? warm : runs;
    summary.perLevel.push({
      level,
      clockTable,
      rtf: summarizeRepeats(pool.map((r) => r.agg.pipelineRtf)),
      sclkMean: summarizeRepeats(pool.map((r) => r.agg.gpu?.sclkMean)),
      sclkMax: summarizeRepeats(pool.map((r) => r.agg.gpu?.sclkMax)),
      busyMean: summarizeRepeats(pool.map((r) => r.agg.gpu?.busyMean)),
      memBusyMean: summarizeRepeats(pool.map((r) => r.agg.gpu?.memBusyMean)),
      mclkMean: summarizeRepeats(pool.map((r) => r.agg.gpu?.mclkMean)),
      powerMeanW: summarizeRepeats(pool.map((r) => r.agg.gpu?.powerMeanW)),
      tempMaxC: summarizeRepeats(pool.map((r) => r.agg.gpu?.tempMaxC)),
      encoderExecMeanMs: summarizeRepeats(pool.map((r) => r.agg.encoderExecMeanMs)),
      decoderStepMeanMs: summarizeRepeats(pool.map((r) => r.agg.decoderStepMeanMs)),
      runs,
    });
  }

  // Ceiling delta: how much forcing high buys over the box default.
  const byLevel = Object.fromEntries(summary.perLevel.map((l) => [l.level, l]));
  if (byLevel.auto && byLevel.high) {
    const a = byLevel.auto.rtf.median, h = byLevel.high.rtf.median;
    summary.highVsAuto = {
      rtfAuto: a, rtfHigh: h,
      rtfImprovementPct: a > 0 ? Number((100 * (a - h) / a).toFixed(2)) : 0,
      sclkAuto: byLevel.auto.sclkMean.median, sclkHigh: byLevel.high.sclkMean.median,
    };
    console.log(`[ceiling] high vs auto: RTF ${a} -> ${h} (${summary.highVsAuto.rtfImprovementPct}% faster), sclk ${summary.highVsAuto.sclkAuto} -> ${summary.highVsAuto.sclkHigh} MHz`);
  }
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[ceiling] error: ${error.message}`);
} finally {
  if (originalLevel) {
    try {
      const restored = await setPowerLevel(originalLevel, { config });
      summary.restoredPowerLevel = restored;
      console.log(`[ceiling] restored power level to '${restored}'`);
    } catch (e) {
      summary.restoredPowerLevel = `FAILED: ${e.message}`;
      console.error(`[ceiling] CRITICAL: failed to restore power level to '${originalLevel}': ${e.message}`);
    }
  }
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "rx6600-hardware-ceiling",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
  });
  console.log(`[ceiling] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
}

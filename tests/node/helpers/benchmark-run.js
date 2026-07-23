import path from "node:path";

import { loadEnvironment } from "../config/environment.js";
import { runStreamSession } from "./stream.js";
import { aggregateProfile } from "./profile-report.js";

/**
 * Map a private M4A fixture path to the canonical normalized WAV produced once
 * on the GPU host by normalizeFixtureOnGpu(). Perf/profile runs consume this
 * deterministic 16 kHz mono PCM WAV, never the M4A directly.
 */
export function canonicalWav(m4aPath) {
  const dir = path.posix.dirname(m4aPath);
  const stem = path.posix.basename(m4aPath).replace(/\.m4a$/u, "");
  return `${dir}/${stem}-16k-mono-pcm16.wav`;
}

/**
 * One profiling run of the incremental production pipeline. Perf-oriented:
 * warm graphs, oracle parity recompute skipped (correctness is covered by the
 * acceptance suites), full stage profile + amdgpu telemetry + memory peak.
 * `paced:false` feeds the plan back-to-back (max throughput); `paced:true`
 * paces chunk arrival at paceMs (the production realtime cadence).
 */
export async function profileRun({
  config = loadEnvironment(),
  label,
  audioPath = null,
  syntheticSeconds = 0,
  paced = false,
  paceMs = 80,
  mode = "80ms",
  env = {},
  telemetryIntervalMs = 100,
  warmup = true,
  timeoutMs = 900_000,
} = {}) {
  const run = await runStreamSession({
    config,
    planName: label,
    audioPath,
    syntheticSeconds,
    mode,
    realtimeMs: paced ? paceMs : 0,
    warmup,
    skipParity: true,
    monitorMemory: true,
    gpuTelemetry: true,
    telemetryIntervalMs,
    env: { VOXTRAL_PROFILE: "1", ...env },
    timeoutMs,
  });
  return { label, paced, run, agg: aggregateProfile(run) };
}

/** median of a numeric array (returns 0 for empty). */
export function median(values) {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (xs.length === 0) return 0;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

export function coefficientOfVariation(values) {
  const xs = values.filter((v) => Number.isFinite(v));
  if (xs.length < 2) return 0;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  if (mean === 0) return 0;
  const variance = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(variance) / mean;
}

export function summarizeRepeats(values) {
  const xs = values.filter((v) => Number.isFinite(v));
  return {
    n: xs.length,
    median: median(xs),
    min: xs.length ? Math.min(...xs) : 0,
    max: xs.length ? Math.max(...xs) : 0,
    cv: coefficientOfVariation(xs),
  };
}

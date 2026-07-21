import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { prepareStreamingAudio } from "../helpers/audio.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { createChunkPlan } from "../helpers/chunks.js";
import { runBatchInference } from "../helpers/inference.js";
import { checkRemoteConnection, runRemote, shellQuote } from "../helpers/remote.js";
import { planCounts, runStreamSession } from "../helpers/stream.js";

const enabled = process.env.VOXTRAL_TEST_STREAM === "1";

// Hard numerical gate for incremental-vs-batch ENCODER tensor parity. The design
// is bit-exact by construction (same graph, RoPE-offset-invariant, prefix-stable),
// so 0 is expected; the gate matches the Mel frontend's.
const ENC_DELTA_GATE = 1e-5;
// Encoder architectural window: one batch chunk = 3000 Mel frames. The incremental
// encoder must never retain or run more than this per chunk.
const CHUNK_MEL = 3000;

// Feed plans exercised against the same clip. All must yield identical encoder
// output (SHA + delta), identical tokens and text, regardless of chunk boundaries.
const PLAN_SPECS = [
  { name: "full", options: { strategy: "full" } },
  { name: "80ms", options: { strategy: "80ms" } },
  { name: "160ms", options: { strategy: "160ms" } },
  { name: "320ms", options: { strategy: "320ms" } },
  { name: "480ms", options: { strategy: "480ms" } },
  { name: "1000ms", options: { strategy: "1000ms" } },
  { name: "single-sample", options: { strategy: "single-sample" } },
  { name: "seeded-random", options: { strategy: "seeded-random", seed: 20260721, maxSamples: 9_000 } },
  { name: "zero-mixed-80ms", options: { strategy: "80ms", zeroLengthAt: [0, 1_280, 5_000] } },
];

// Build a mono/16 kHz/PCM16 WAV of `samples` silent frames (for work-bound runs:
// encoder work depends on frame count, not content).
function silenceWav(samples) {
  const dataBytes = samples * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(16_000, 24);
  buf.writeUInt32LE(16_000 * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(dataBytes, 40);
  return buf;
}

describe.skipIf(!enabled).sequential("RX 6600 incremental causal encoder acceptance", () => {
  const config = loadEnvironment();

  test("incremental encoder: batch-parity tensor, chunk-invariant, during-feed, bounded", async () => {
    await checkRemoteConnection({ config });
    const build = await buildRemoteVulkan({ config });
    expect(build.binaryPath).toBe(config.remoteBinary);

    // Reference: the existing batch CLI over the same fixture.
    const reference = await runBatchInference({
      config,
      testName: "incremental-encoder-reference",
      timeoutMs: 240_000,
    });
    expect(reference.exitCode).toBe(0);
    expect(reference.transcript).toBeTruthy();
    expect(reference.transcript).not.toBe("[no-transcript]");

    const prepared = await prepareStreamingAudio(config.localSmokeAudio);
    const totalSamples = prepared.pcm.length / 2;

    const runs = [];
    for (const spec of PLAN_SPECS) {
      const plan = createChunkPlan(prepared.pcm, spec.options);
      const counts = planCounts(plan);
      expect(counts.reduce((a, b) => a + b, 0)).toBe(totalSamples);
      const result = await runStreamSession({ config, planName: `enc-${spec.name}`, counts });
      runs.push({ spec: spec.name, dataFeeds: counts.filter((c) => c > 0).length, result });
    }

    const base = runs[0].result;

    // --- Per-run invariants: incremental encoder, batch parity, during-feed, bounded ---
    for (const { spec, result } of runs) {
      expect.soft(result.state, `${spec}: state`).toBe("completed");
      expect.soft(result.finishStatus, `${spec}: finishStatus`).toBe("ok");
      expect.soft(result.inferenceRuns, `${spec}: inference runs once`).toBe(1);

      // The encoder is genuinely incremental.
      expect.soft(result.incrementalEncoder, `${spec}: incremental encoder`).toBe(true);
      expect.soft(result.encoderStrategy, `${spec}: strategy`).toBe("bounded-window-recompute");

      // Numerical parity of the incremental encoder output against the batch encoder.
      expect.soft(result.encoderMaxAbsDeltaVsBatch, `${spec}: encoder delta vs batch`)
        .toBeLessThanOrEqual(ENC_DELTA_GATE);

      // New encoder frames are produced DURING feed (not deferred to finish), and
      // the split adds up with no lost/duplicated frame.
      expect.soft(result.encoderFramesBeforeFinish, `${spec}: frames before finish`).toBeGreaterThan(0);
      expect.soft(
        result.encoderFramesBeforeFinish + result.encoderFramesFlushedAtFinish,
        `${spec}: frame split adds up`,
      ).toBe(result.encoderFrames);

      // finish() never re-encodes the whole Mel; it runs at most the last 1-2 chunks.
      expect.soft(result.fullMelReencodedAtFinish, `${spec}: no full re-encode at finish`).toBe(false);

      // Bounded encoder context: never exceeds one architectural chunk, and the Mel
      // window is released at finish.
      expect.soft(result.encoderPeakContextFrames, `${spec}: peak context bounded`).toBeLessThanOrEqual(CHUNK_MEL);
      expect.soft(result.encoderMaxWindowFrames, `${spec}: max window bounded`).toBeLessThanOrEqual(CHUNK_MEL);
      expect.soft(result.encoderContextFramesRetained, `${spec}: context released at finish`).toBe(0);

      // Exactly one FINAL_TEXT followed by one COMPLETED.
      expect.soft(result.events.map((e) => e.type), `${spec}: events`).toEqual(["final_text", "completed"]);
    }

    // --- Chunk invariance: every plan is identical to the single-feed run ---
    for (const { spec, result } of runs) {
      expect.soft(result.encoderSha256, `${spec}: encoder SHA-256`).toBe(base.encoderSha256);
      expect.soft(result.encoderFrames, `${spec}: encoder frame count`).toBe(base.encoderFrames);
      expect.soft(result.melSha256, `${spec}: Mel SHA-256 (regression)`).toBe(base.melSha256);
      expect.soft(result.tokens, `${spec}: tokens`).toEqual(base.tokens);
      expect.soft(result.text, `${spec}: transcript`).toBe(base.text);
    }

    // --- Batch parity: incremental stream output matches the batch CLI ---
    expect(base.text.trim()).toBe(reference.transcript);
    if (reference.tokens.length > 0) {
      expect(base.tokens).toEqual(reference.tokens);
    }

    // --- Vulkan / RX 6600 evidence ---
    expect(base.evidence.vulkanEnabled).toBe(true);
    expect(base.evidence.rx6600Detected).toBe(true);
    expect(base.evidence.cpuOnlyFallbackDetected).toBe(false);

    // --- Persist an evidence bundle ---
    const invariance = runs.map(({ spec, dataFeeds, result }) => ({
      plan: spec,
      dataFeeds,
      encoderSha256: result.encoderSha256,
      encoderFrames: result.encoderFrames,
      encoderFramesBeforeFinish: result.encoderFramesBeforeFinish,
      encoderFramesFlushedAtFinish: result.encoderFramesFlushedAtFinish,
      encoderExecutions: result.encoderExecutions,
      encoderInputFramesProcessed: result.encoderInputFramesProcessed,
      encoderFramesRecomputed: result.encoderFramesRecomputed,
      encoderPeakContextFrames: result.encoderPeakContextFrames,
      encoderMaxAbsDeltaVsBatch: result.encoderMaxAbsDeltaVsBatch,
      tokens: result.tokens.length,
    }));
    const artifact = await writeArtifactBundle({
      config,
      testName: "incremental-encoder-acceptance",
      backend: "Vulkan",
      command: runs.map((r) => r.result.commandLine).join("\n"),
      result: {
        exitCode: 0,
        backend: "Vulkan",
        totalSamples,
        reference: { transcript: reference.transcript, tokens: reference.tokens },
        invariance,
        evidence: base.evidence,
      },
    });
    expect(artifact.directory).toContain(config.artifactDir);
  }, 900_000);

  test("incremental encoder: work per second of audio does not grow with duration", async () => {
    await checkRemoteConnection({ config });
    await buildRemoteVulkan({ config });

    // 5 s, 30 s and 2 min of silence. Encoder work depends on frame count, not
    // content; decode is capped (max-tokens 1) so only the encoder is exercised.
    const durations = [5, 30, 120];
    const points = [];
    for (const sec of durations) {
      const wav = silenceWav(sec * 16_000);
      const remotePath = `${config.remoteRepo}/.enc-workbound-${sec}s.wav`;
      await runRemote(`cat > ${shellQuote(remotePath)}`, { config, input: wav, timeoutMs: 60_000 });
      const result = await runStreamSession({
        config,
        planName: `enc-work-${sec}s`,
        mode: "1000ms",
        audioPath: remotePath,
        maxTokens: 1,
        timeoutMs: 600_000,
      });
      await runRemote(`rm -f ${shellQuote(remotePath)}`, { config, timeoutMs: 30_000 });

      expect.soft(result.state, `${sec}s: state`).toBe("completed");
      expect.soft(result.encoderMaxAbsDeltaVsBatch, `${sec}s: delta`).toBeLessThanOrEqual(ENC_DELTA_GATE);
      // Bounded context: one architectural chunk plus at most a single feed's
      // newly-stable frames that land before the next compaction (transient-then-
      // compact, like the Mel frontend). Bounded and independent of duration.
      expect.soft(result.encoderPeakContextFrames, `${sec}s: bounded context`).toBeLessThan(CHUNK_MEL + 1500);

      const workRatio = result.encoderInputFramesProcessed / Math.max(1, result.melFrames);
      points.push({
        seconds: sec,
        melFrames: result.melFrames,
        encoderFrames: result.encoderFrames,
        encoderExecutions: result.encoderExecutions,
        encoderInputFramesProcessed: result.encoderInputFramesProcessed,
        encoderFramesRecomputed: result.encoderFramesRecomputed,
        encoderPeakContextFrames: result.encoderPeakContextFrames,
        workRatio,
      });
    }

    // (a) Bounded: encoder work is always a small constant multiple of the unique
    // Mel — never the forbidden O(duration^2) blow-up.
    for (const p of points) {
      expect.soft(p.workRatio, `${p.seconds}s: work ratio bounded`).toBeLessThan(8);
    }
    // (b) Does not grow with duration: the chunk-0 recompute is a one-time bounded
    // transient, so the longer streams amortize DOWN (2 min ratio <= 30 s ratio).
    const p30 = points.find((p) => p.seconds === 30);
    const p120 = points.find((p) => p.seconds === 120);
    expect(p120.workRatio).toBeLessThanOrEqual(p30.workRatio + 0.01);
    // (c) Retained context is bounded and does NOT grow with duration.
    expect(p120.encoderPeakContextFrames).toBeLessThanOrEqual(p30.encoderPeakContextFrames);

    const artifact = await writeArtifactBundle({
      config,
      testName: "incremental-encoder-workbound",
      backend: "Vulkan",
      command: "npm run acceptance:incremental-encoder (work-bound)",
      result: { exitCode: 0, backend: "Vulkan", points },
    });
    expect(artifact.directory).toContain(config.artifactDir);
  }, 1_200_000);
});

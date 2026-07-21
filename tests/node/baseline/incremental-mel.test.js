import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { prepareStreamingAudio } from "../helpers/audio.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { createChunkPlan } from "../helpers/chunks.js";
import { runBatchInference } from "../helpers/inference.js";
import { checkRemoteConnection } from "../helpers/remote.js";
import { planCounts, runStreamSession } from "../helpers/stream.js";

const enabled = process.env.VOXTRAL_TEST_STREAM === "1";

// One full STFT window; the incremental frontend must never retain more.
const N_FFT = 400;
// Hard numerical gate for incremental-vs-batch Mel parity (bitwise expected).
const MEL_DELTA_GATE = 1e-5;

// Feed plans exercised against the same clip. All must yield identical canonical
// PCM, identical incremental Mel (SHA + frame count), identical tokens and text.
const PLAN_SPECS = [
  { name: "full", options: { strategy: "full" } },
  { name: "80ms", options: { strategy: "80ms" } },
  { name: "160ms", options: { strategy: "160ms" } },
  { name: "320ms", options: { strategy: "320ms" } },
  { name: "480ms", options: { strategy: "480ms" } },
  { name: "1000ms", options: { strategy: "1000ms" } },
  { name: "seeded-random", options: { strategy: "seeded-random", seed: 20260721, maxSamples: 9_000 } },
  { name: "zero-mixed-80ms", options: { strategy: "80ms", zeroLengthAt: [0, 1_280, 5_000] } },
];

describe.skipIf(!enabled).sequential("RX 6600 incremental Mel acceptance", () => {
  const config = loadEnvironment();

  test("incremental STFT/log-Mel: chunk-invariant, one-shot frames, bounded PCM, batch parity", async () => {
    await checkRemoteConnection({ config });
    const build = await buildRemoteVulkan({ config });
    expect(build.binaryPath).toBe(config.remoteBinary);

    // Reference: the existing batch CLI over the same fixture.
    const reference = await runBatchInference({
      config,
      testName: "incremental-mel-reference",
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
      const result = await runStreamSession({ config, planName: spec.name, counts });
      runs.push({ spec: spec.name, dataFeeds: counts.filter((c) => c > 0).length, result });
    }

    const base = runs[0].result;

    // --- Per-run invariants: incremental Mel, one-shot frames, bounded PCM ---
    for (const { spec, result } of runs) {
      expect.soft(result.state, `${spec}: state`).toBe("completed");
      expect.soft(result.finishStatus, `${spec}: finishStatus`).toBe("ok");
      expect.soft(result.inferenceRuns, `${spec}: inference runs once`).toBe(1);
      expect.soft(result.samplesReceived, `${spec}: samples received`).toBe(totalSamples);
      expect.soft(result.samplesConsumed, `${spec}: samples consumed`).toBe(totalSamples);

      // The frontend is genuinely incremental and never buffers the whole PCM.
      expect.soft(result.incrementalMel, `${spec}: incremental Mel`).toBe(true);
      expect.soft(result.fullPcmBufferedAtFinish, `${spec}: no full PCM at finish`).toBe(false);

      // Every Mel frame is computed exactly once (no recomputation of the prefix).
      expect.soft(result.dftFramesComputed, `${spec}: DFT == Mel frames`).toBe(result.melFrames);
      expect.soft(
        result.melFramesBeforeFinish + result.melFramesFlushedAtFinish,
        `${spec}: frames split adds up`,
      ).toBe(result.melFrames);

      // For multi-second audio most frames appear during feed; finish only flushes
      // the final right-padded window.
      expect.soft(result.melFramesBeforeFinish, `${spec}: frames before finish`).toBeGreaterThan(0);
      expect.soft(result.melFramesFlushedAtFinish, `${spec}: small final flush`).toBeLessThanOrEqual(4);

      // Bounded PCM retention independent of stream duration: the peak rolling tail
      // is below one window, and the tail is released at finish.
      expect.soft(result.pcmPeakRetainedSamples, `${spec}: peak retained bounded`).toBeLessThan(N_FFT);
      expect.soft(result.pcmRetainedSamples, `${spec}: tail released at finish`).toBe(0);

      // Numerical parity of the incremental Mel against the batch Mel.
      expect.soft(result.melMaxAbsDeltaVsBatch, `${spec}: Mel delta vs batch`).toBeLessThanOrEqual(MEL_DELTA_GATE);

      // Exactly one FINAL_TEXT followed by one COMPLETED.
      expect.soft(result.events.map((e) => e.type), `${spec}: events`).toEqual(["final_text", "completed"]);
    }

    // --- Chunk invariance: every plan is identical to the single-feed run ---
    for (const { spec, result } of runs) {
      expect.soft(result.pcmSha256, `${spec}: PCM SHA-256`).toBe(base.pcmSha256);
      expect.soft(result.melSha256, `${spec}: Mel SHA-256`).toBe(base.melSha256);
      expect.soft(result.melFrames, `${spec}: Mel frame count`).toBe(base.melFrames);
      expect.soft(result.dftFramesComputed, `${spec}: DFT count`).toBe(base.dftFramesComputed);
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
      samples: result.samplesReceived,
      pcmSha256: result.pcmSha256,
      melSha256: result.melSha256,
      melFrames: result.melFrames,
      melFramesBeforeFinish: result.melFramesBeforeFinish,
      melFramesFlushedAtFinish: result.melFramesFlushedAtFinish,
      dftFramesComputed: result.dftFramesComputed,
      melMaxAbsDeltaVsBatch: result.melMaxAbsDeltaVsBatch,
      pcmPeakRetainedSamples: result.pcmPeakRetainedSamples,
      tokens: result.tokens.length,
    }));
    const artifact = await writeArtifactBundle({
      config,
      testName: "incremental-mel-acceptance",
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
});

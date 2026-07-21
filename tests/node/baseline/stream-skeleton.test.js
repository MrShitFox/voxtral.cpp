import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { prepareStreamingAudio } from "../helpers/audio.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { createChunkPlan } from "../helpers/chunks.js";
import { runBatchInference } from "../helpers/inference.js";
import { checkRemoteConnection } from "../helpers/remote.js";
import { StreamingEventCollector } from "../helpers/streaming-transport.js";
import { planCounts, runStreamSession } from "../helpers/stream.js";

const enabled = process.env.VOXTRAL_TEST_STREAM === "1";

// Feed plans exercised against the same clip. All must yield an identical
// canonical PCM stream, identical tokens and identical transcript.
const PLAN_SPECS = [
  { name: "full", options: { strategy: "full" } },
  { name: "80ms", options: { strategy: "80ms" } },
  { name: "160ms", options: { strategy: "160ms" } },
  { name: "480ms", options: { strategy: "480ms" } },
  { name: "seeded-random", options: { strategy: "seeded-random", seed: 1337, maxSamples: 9_000 } },
  { name: "zero-mixed-80ms", options: { strategy: "80ms", zeroLengthAt: [0, 1_280, 5_000] } },
];

describe.skipIf(!enabled).sequential("RX 6600 streaming skeleton acceptance", () => {
  const config = loadEnvironment();

  test("chunk-invariant PCM + batch parity through the compatibility finish", async () => {
    await checkRemoteConnection({ config });
    // Ensure every target, including voxtral-stream-test, is built on the host.
    const build = await buildRemoteVulkan({ config });
    expect(build.binaryPath).toBe(config.remoteBinary);

    // Reference: the existing batch CLI over the same fixture.
    const reference = await runBatchInference({
      config,
      testName: "stream-skeleton-reference",
      timeoutMs: 240_000,
    });
    expect(reference.exitCode).toBe(0);
    expect(reference.transcript).toBeTruthy();
    expect(reference.transcript).not.toBe("[no-transcript]");

    // Canonical PCM from the local copy of the same WAV; drives the feed plans.
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

    // --- Per-run lifecycle invariants ---
    for (const { spec, result } of runs) {
      expect.soft(result.state, `${spec}: state`).toBe("completed");
      expect.soft(result.finishStatus, `${spec}: finishStatus`).toBe("ok");
      expect.soft(result.inferenceRuns, `${spec}: inference runs once`).toBe(1);
      expect.soft(result.wavSamples, `${spec}: wav samples`).toBe(totalSamples);
      expect.soft(result.samplesReceived, `${spec}: samples received`).toBe(totalSamples);
      expect.soft(result.samplesConsumed, `${spec}: samples consumed`).toBe(totalSamples);

      // Exactly one FINAL_TEXT followed by one COMPLETED; no duplicate finals.
      expect.soft(result.events.map((e) => e.type), `${spec}: events`).toEqual(["final_text", "completed"]);

      // Reuse the harness event collector to assert ordering independently.
      const collector = new StreamingEventCollector();
      for (const event of result.events) {
        collector.add({ type: event.type === "final_text" ? "final" : event.type });
      }
      expect.soft(collector.events.at(-1)?.type, `${spec}: terminal event`).toBe("completed");
    }

    // --- Chunk invariance: every plan is bit-identical to the single-feed run ---
    for (const { spec, result } of runs) {
      expect.soft(result.pcmSha256, `${spec}: PCM SHA-256`).toBe(base.pcmSha256);
      expect.soft(result.pcmFloats, `${spec}: PCM float count`).toBe(base.pcmFloats);
      expect.soft(result.tokens, `${spec}: tokens`).toEqual(base.tokens);
      expect.soft(result.text, `${spec}: transcript`).toBe(base.text);
    }

    // --- Batch parity: stream output matches the batch CLI ---
    // Raw-text invariance across chunk plans is asserted above (untrimmed).
    // parseInferenceOutput trims each CLI stdout line, so the batch reference is
    // already trimmed; the model's detokenized transcript begins with a leading
    // space identically in both paths. Compare on the trimmed transcript.
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
      tokens: result.tokens.length,
      inferenceRuns: result.inferenceRuns,
      transcript: result.text,
    }));
    const artifact = await writeArtifactBundle({
      config,
      testName: "stream-skeleton-acceptance",
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

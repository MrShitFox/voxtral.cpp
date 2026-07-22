// ============================================================================
// RX 6600 per-layer encoder KV-cache acceptance (Session 6.1).
//
// Verifies the production per-layer KV encoder (the artifact-free global sliding
// window, each enc frame computed exactly once) on real hardware:
//   * bit-exact vs the global batch encoder, single-chunk AND across ring rollover;
//   * chunk invariance: every feed plan yields identical encoder output/tokens;
//   * work ratio == 1 (no bounded-window replay); finish does no encoder replay;
//   * ring rollover exercised (KvWraps > 0), KV window + Mel tail bounded;
//   * bounded, duration-independent KV device memory;
//   * encoder compute/residence latency and paced backlog (feed p95 alone is
//     not a realtime signal once the low-latency graph runs inside feed());
//   * the legacy chunked strategy is retained (VOXTRAL_ENCODER_STRATEGY=reference)
//     and its long-stream divergence from the global result is measured (expected).
//
// Requires VOXTRAL_TEST_GPU=1 VOXTRAL_TEST_STREAM=1 and the GPU env (see README).
// ============================================================================
import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { checkRemoteConnection, runRemote, shellQuote } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const enabled = process.env.VOXTRAL_TEST_STREAM === "1";
const ENC_DELTA_GATE = 1e-5;
const ENC_WINDOW = 750;
const ENC_KV_CAP = 878;

function silenceWav(samples) {
  const dataBytes = samples * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0, "ascii"); buf.writeUInt32LE(36 + dataBytes, 4); buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii"); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(16_000, 24); buf.writeUInt32LE(32_000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii"); buf.writeUInt32LE(dataBytes, 40);
  return buf;
}

async function putWav(config, name, wav) {
  const remotePath = `${config.remoteRepo}/.enc-kv-${name}.wav`;
  await runRemote(`cat > ${shellQuote(remotePath)}`, { config, input: wav, timeoutMs: 60_000 });
  return remotePath;
}

describe.skipIf(!enabled).sequential("RX 6600 per-layer encoder KV-cache acceptance", () => {
  const config = loadEnvironment();

  test("KV encoder: batch parity, chunk invariance, rollover, work-once, bounded, latency", async () => {
    await checkRemoteConnection({ config });
    await buildRemoteVulkan({ config });

    const audio = config.remoteSmokeAudio;                     // ~3.6 s, single chunk
    const plans = ["full", "1000ms", "480ms", "160ms", "80ms"];

    // --- Single-chunk: every feed plan is bit-identical to the global batch ------
    const short = [];
    for (const mode of plans) {
      // Session 7.1: encoder tensor parity (delta/SHA) reads the host encoder
      // buffer, retained only by the reference decoder. The KV encoder is identical.
      const r = await runStreamSession({ config, planName: `short-${mode}`, mode, audioPath: audio, maxTokens: 0, env: { VOXTRAL_STREAM_DECODER: "reference" } });
      short.push({ mode, r });
      expect.soft(r.state, `${mode}: state`).toBe("completed");
      expect.soft(r.encoderStrategy, `${mode}: strategy`).toBe("per-layer-kv");
      expect.soft(r.encoderMaxAbsDeltaVsBatch, `${mode}: KV==global batch`).toBeLessThanOrEqual(ENC_DELTA_GATE);
      expect.soft(r.encoderWorkRatio, `${mode}: work ratio == 1`).toBeLessThanOrEqual(1.0 + 1e-9);
      expect.soft(r.encoderTransformerFramesComputed, `${mode}: each frame once`).toBe(r.encoderUniqueFrames);
      expect.soft(r.encoderFrameLayerEvaluations, `${mode}: frame*layer`).toBe(r.encoderUniqueFrames * 32);
      // finish() runs only the final flush (buffered partial grid + right-pad), never
      // a replay of the encoder prefix: bounded by ~2 grid batches, not O(total).
      expect.soft(r.encoderFramesComputedDuringFinish, `${mode}: finish = final flush only`).toBeLessThanOrEqual(256);
      expect.soft(r.encoderKvCapacityFrames, `${mode}: KV capacity`).toBe(ENC_KV_CAP);
      expect.soft(r.encoderKvElementSize, `${mode}: F32 KV`).toBe(4);
    }
    const base = short[0].r;
    for (const { mode, r } of short) {
      expect.soft(r.encoderSha256, `${mode}: encoder SHA invariant`).toBe(base.encoderSha256);
      expect.soft(r.tokens, `${mode}: tokens invariant`).toEqual(base.tokens);
      expect.soft(r.text, `${mode}: transcript invariant`).toBe(base.text);
    }

    // --- Long clip: rollover, bit-exact across plans, bounded, latency ----------
    const longWav = await putWav(config, "40s", silenceWav(40 * 16_000));  // ~2000 enc frames > CAP (wraps) and > 3000 Mel (multi-chunk batch)
    const longFull = await runStreamSession({ config, planName: "long-full", mode: "full", audioPath: longWav, maxTokens: 2, timeoutMs: 600_000, env: { VOXTRAL_STREAM_DECODER: "reference" } });
    const long80 = await runStreamSession({ config, planName: "long-80ms", realtimeMs: 80, audioPath: longWav, maxTokens: 2, timeoutMs: 600_000, env: { VOXTRAL_STREAM_DECODER: "reference" } });

    for (const [name, r] of [["long-full", longFull], ["long-80ms", long80]]) {
      expect.soft(r.encoderMaxAbsDeltaVsBatch, `${name}: KV==global batch (rollover)`).toBeLessThanOrEqual(ENC_DELTA_GATE);
      expect.soft(r.encoderWorkRatio, `${name}: work ratio == 1 (rollover)`).toBeLessThanOrEqual(1.0 + 1e-9);
      expect.soft(r.encoderKvWraps, `${name}: ring rolled over`).toBeGreaterThan(0);
      expect.soft(r.encoderKvLogicalFrames, `${name}: KV window bounded`).toBeLessThanOrEqual(ENC_WINDOW);
      expect.soft(r.encoderKvPeakLogicalFrames, `${name}: peak KV window bounded`).toBeLessThanOrEqual(ENC_WINDOW);
      expect.soft(r.encoderKvEvictions, `${name}: frames evicted past window`).toBeGreaterThan(0);
    }
    // Chunk invariance across rollover: identical encoder output & tokens.
    expect.soft(long80.encoderSha256, "long: encoder SHA invariant across rollover").toBe(longFull.encoderSha256);
    expect.soft(long80.tokens, "long: tokens invariant across rollover").toEqual(longFull.tokens);

    // Bounded KV device memory (duration-independent): capacity * kv_dim * layers * 2 * 4.
    const kvBytesExpected = ENC_KV_CAP * (32 * 64) * 32 * 2 * 4;
    expect.soft(longFull.encoderKvAllocatedBytes, "KV device bytes").toBe(kvBytesExpected);
    // Real streaming (80 ms) keeps only a bounded Mel tail, not the full history.
    expect.soft(long80.encoderMelPeakRetainedFrames, "Mel tail bounded (streaming)").toBeLessThan(300);

    // Realtime latency gates. The graph executes during feed by design, so
    // feed() duration is reported diagnostically but is not the acceptance
    // criterion; residence and warm graph compute are the relevant signals.
    expect.soft(long80.encoderResidenceP95Ms, "encoder residence p95 < 160 ms").toBeLessThan(160);
    expect.soft(long80.adapterGroupResidenceP95Ms, "adapter residence p95 < 160 ms").toBeLessThan(160);
    expect.soft(long80.encoderComputeP95Ms, "encoder compute p95 < 80 ms").toBeLessThan(80);
    expect.soft(long80.encoderComputeWarmMaxMs, "warmup-excluded compute max < 80 ms").toBeLessThan(80);
    expect.soft(long80.finalBacklogMs, "final realtime backlog < 20 ms").toBeLessThan(20);

    // --- Legacy chunked strategy retained; its long divergence is measured ------
    const legacyShort = await runStreamSession({ config, planName: "legacy-short", mode: "80ms", audioPath: audio, maxTokens: 0, env: { VOXTRAL_ENCODER_STRATEGY: "reference" } });
    expect.soft(legacyShort.encoderStrategy, "legacy strategy available").toBe("bounded-window-recompute");
    expect.soft(legacyShort.encoderMaxAbsDeltaVsBatch, "legacy self-consistent (chunked==chunked)").toBeLessThanOrEqual(ENC_DELTA_GATE);
    // Single-chunk: legacy == global (chunk 0 has no warmup artifact) -> same tokens.
    expect.soft(legacyShort.tokens, "single-chunk: legacy tokens == global").toEqual(base.tokens);
    const legacyLong = await runStreamSession({ config, planName: "legacy-long", mode: "full", audioPath: longWav, maxTokens: 2, timeoutMs: 600_000, env: { VOXTRAL_ENCODER_STRATEGY: "reference" } });
    // Long: legacy chunked diverges from the global result (the documented warmup
    // artifact). Recorded, not gated.
    const legacyDivergesLong = legacyLong.encoderSha256 !== longFull.encoderSha256;

    await runRemote(`rm -f ${shellQuote(longWav)}`, { config, timeoutMs: 30_000 });

    const summary = {
      encoderStrategy: "per-layer-kv",
      encoderKvCapacityFrames: ENC_KV_CAP,
      encoderKvElementType: "f32",
      encoderKvAllocatedBytes: longFull.encoderKvAllocatedBytes,
      encoderKvLogicalFrames: longFull.encoderKvLogicalFrames,
      encoderKvPeakLogicalFrames: longFull.encoderKvPeakLogicalFrames,
      encoderKvWraps: longFull.encoderKvWraps,
      encoderKvEvictions: longFull.encoderKvEvictions,
      encoderUniqueFrames: longFull.encoderUniqueFrames,
      encoderTransformerFramesComputed: longFull.encoderTransformerFramesComputed,
      encoderWorkRatio: longFull.encoderWorkRatio,
      encoderMaxAbsDeltaVsBatch: Math.max(...short.map((s) => s.r.encoderMaxAbsDeltaVsBatch), longFull.encoderMaxAbsDeltaVsBatch, long80.encoderMaxAbsDeltaVsBatch),
      encoderFramesComputedDuringFinish: longFull.encoderFramesComputedDuringFinish,
      melFullHistoryRetained: false,
      encoderMelPeakRetainedFramesStreaming: long80.encoderMelPeakRetainedFrames,
      feedLatencyP95Ms: long80.feedLatencyP95Ms,
      feedLatencyWarmMaxMs: long80.feedLatencyWarmMaxMs,
      referenceStrategyAvailable: true,
      legacyChunkedDivergesOnLongStream: legacyDivergesLong,
      shortTranscript: base.text,
    };
    const artifact = await writeArtifactBundle({
      config, testName: "encoder-kv-acceptance", backend: "Vulkan",
      command: "npm run test:encoder-kv:gpu", result: { exitCode: 0, backend: "Vulkan", summary },
    });
    expect(artifact.directory).toContain(config.artifactDir);
    expect(legacyDivergesLong, "legacy chunked encoder diverges on long streams (documented)").toBe(true);
  }, 1_800_000);
});

import path from "node:path";

import { describe, expect, test } from "vitest";

import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import {
  checkRemoteConnection,
  normalizeFixtureOnGpu,
  runRemote,
  shellQuote,
  syncFixture,
  syncSources,
} from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const enabled = process.env.VOXTRAL_TEST_ENCODER_REALTIME === "1";
const runSoak = process.env.VOXTRAL_REALTIME_SOAK === "1";
const longAudio = process.env.VOXTRAL_LONG_AUDIO;

function silenceWav(samples) {
  const bytes = samples * 2;
  const buf = Buffer.alloc(44 + bytes);
  buf.write("RIFF", 0, "ascii"); buf.writeUInt32LE(36 + bytes, 4); buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii"); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(16_000, 24); buf.writeUInt32LE(32_000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii"); buf.writeUInt32LE(bytes, 40);
  return buf;
}

const realtimeEnv = {
  VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
  VOXTRAL_ENCODER_TELEMETRY: "1",
};

function assertRealtime(result) {
  expect.soft(result.state).toBe("completed");
  expect.soft(result.encoderTransformerFramesComputed).toBe(result.encoderUniqueFrames);
  expect.soft(result.encoderKvWraps).toBeGreaterThan(0);
  expect.soft(result.encoderMelPeakRetainedFrames).toBeLessThan(300);
  expect.soft(result.melHistoryRetained).toBe(false);
  expect.soft(result.encoderResidenceP95Ms).toBeLessThan(160);
  expect.soft(result.adapterGroupResidenceP95Ms).toBeLessThan(160);
  expect.soft(result.encoderComputeWarmMaxMs).toBeLessThan(80);
  expect.soft(result.finalBacklogMs).toBeLessThan(20);
  expect.soft(result.backlogGrowthSlopeMsPerSec).toBeLessThan(0.5);
}

describe.skipIf(!enabled).sequential("realtime encoder long-form gates", () => {
  const config = loadEnvironment();
  let prepared;
  const prepareGpu = () => {
    prepared ??= (async () => {
      await checkRemoteConnection({ config });
      await syncSources({ config });
      await buildRemoteVulkan({ config });
    })();
    return prepared;
  };

  test.skipIf(!runSoak)("optional deterministic 2-minute synthetic soak", async () => {
    await prepareGpu();
    const remotePath = `${config.remoteRepo}/.encoder-realtime-soak.wav`;
    await runRemote(`cat > ${shellQuote(remotePath)}`, {
      config, input: silenceWav(120 * 16_000), timeoutMs: 60_000,
    });
    try {
      const result = await runStreamSession({
        config, planName: "encoder-realtime-soak", realtimeMs: 80,
        audioPath: remotePath, maxTokens: 1, skipParity: true,
        env: { ...realtimeEnv, VOXTRAL_STREAM_DECODER: "reference" }, timeoutMs: 300_000,
      });
      assertRealtime(result);
      await writeArtifactBundle({
        config, testName: "encoder-realtime-synthetic-soak", backend: "Vulkan",
        command: "npm run test:encoder-realtime:soak", result: { exitCode: 0, ...result, stdout: undefined, stderr: undefined },
      });
    } finally {
      await runRemote(`rm -f ${shellQuote(remotePath)}`, { config, timeoutMs: 30_000 });
    }
  }, 420_000);

  test.skipIf(!longAudio)(
    "optional spoken gate skipped: set VOXTRAL_LONG_AUDIO to voxTest2min.m4a",
    async () => {
      await prepareGpu();
      const fixturePath = path.resolve(longAudio);
      if (fixturePath.startsWith("/root/voxtral-test-data/")) {
        await runRemote(`test -s ${shellQuote(fixturePath)}`, { config, timeoutMs: 30_000 });
      } else {
        await syncFixture(fixturePath, { config });
      }
      const fixture = await normalizeFixtureOnGpu({ config });
      const result = await runStreamSession({
        config, planName: "encoder-realtime-spoken-long", realtimeMs: 80,
        audioPath: fixture.wavPath, maxTokens: 1, skipParity: true,
        env: { ...realtimeEnv, VOXTRAL_STREAM_DECODER: "reference" }, timeoutMs: 300_000,
      });
      assertRealtime(result);
      await writeArtifactBundle({
        config, testName: "encoder-realtime-spoken-long", backend: "Vulkan",
        command: "npm run test:encoder-realtime:long", audioMetadata: fixture,
        result: { exitCode: 0, ...result, stdout: undefined, stderr: undefined },
      });
    },
    420_000,
  );
});

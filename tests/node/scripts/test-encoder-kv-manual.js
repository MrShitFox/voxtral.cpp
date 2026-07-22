// Opt-in flash-attention vs explicit softmax(QK^T+mask)V oracle.
import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { buildRemoteVulkan } from "../helpers/build.js";
import { checkRemoteConnection, runRemote, shellQuote, syncSources } from "../helpers/remote.js";
import { runStreamSession } from "../helpers/stream.js";

const config = loadEnvironment();
const summary = { startedAt: new Date().toISOString(), cases: [] };

function silenceWav(samples) {
  const bytes = samples * 2;
  const buf = Buffer.alloc(44 + bytes);
  buf.write("RIFF", 0, "ascii"); buf.writeUInt32LE(36 + bytes, 4); buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii"); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(16_000, 24); buf.writeUInt32LE(32_000, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii"); buf.writeUInt32LE(bytes, 40);
  return buf;
}

const schedulerEnv = (logical) => ({
  VOXTRAL_ENC_KV_LOGICAL_BATCH: String(logical),
  VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
});

function assertOracle(result, label) {
  if (!result.manualOracleChecked) throw new Error(`${label}: manual oracle was not executed`);
  if (result.encoderMaxAbsDeltaVsBatch > 1e-5) throw new Error(`${label}: production batch parity failed`);
  const normalizedRms = result.encoderManualRmsDelta / Math.max(result.encoderManualReferenceRms, Number.EPSILON);
  // Manual softmax and fused Vulkan flash use different reduction/exp kernels;
  // the strict gates therefore combine an outlier bound with distribution-wide
  // RMS/cosine bounds. The latter prevent a permissive max-only comparison.
  if (result.encoderMaxAbsDeltaVsManual > 0.25) throw new Error(`${label}: manual max delta ${result.encoderMaxAbsDeltaVsManual}`);
  if (result.encoderManualMeanAbsDelta > 0.006) throw new Error(`${label}: manual mean delta ${result.encoderManualMeanAbsDelta}`);
  // The 20 s post-rollover oracle measures 0.012256 on RX 6600/RADV; keep a
  // narrow ~6% envelope above that observed reduction-order difference while
  // retaining the independent max/mean/cosine guards above and below.
  if (normalizedRms > 0.013) throw new Error(`${label}: manual normalized RMS ${normalizedRms}`);
  if (result.encoderManualCosineSimilarity < 0.9999) throw new Error(`${label}: manual cosine ${result.encoderManualCosineSimilarity}`);
  return normalizedRms;
}

function compact(result, label, normalizedRms, production, manual) {
  return {
    label,
    encoderFrames: result.encoderFrames,
    encoderKvWraps: result.encoderKvWraps,
    encoderSha256: result.encoderSha256,
    encoderManualSha256: result.encoderManualSha256,
    encoderMaxAbsDeltaVsBatch: result.encoderMaxAbsDeltaVsBatch,
    encoderMaxAbsDeltaVsManual: result.encoderMaxAbsDeltaVsManual,
    encoderManualMeanAbsDelta: result.encoderManualMeanAbsDelta,
    encoderManualRmsDelta: result.encoderManualRmsDelta,
    encoderManualReferenceRms: result.encoderManualReferenceRms,
    encoderManualNormalizedRms: normalizedRms,
    encoderManualCosineSimilarity: result.encoderManualCosineSimilarity,
    productionWallDurationMs: production.wallDurationMs,
    manualWallDurationMs: manual.wallDurationMs,
    manualRuntimeRatio: production.wallDurationMs > 0
      ? manual.wallDurationMs / production.wallDurationMs : 0,
    tokens: result.tokens,
    text: result.text,
  };
}

async function main() {
  await checkRemoteConnection({ config });
  await syncSources({ config });
  await buildRemoteVulkan({ config });

  for (const logical of [4, 8]) {
    const production = await runStreamSession({
      config,
      planName: `flash-production-short-${logical}-32`,
      mode: "80ms",
      maxTokens: 0,
      env: schedulerEnv(logical),
      timeoutMs: 300_000,
    });
    const flash = await runStreamSession({
      config,
      planName: `manual-oracle-short-${logical}-32`,
      mode: "80ms",
      maxTokens: 0,
      manualOracle: true,
      env: schedulerEnv(logical),
      timeoutMs: 300_000,
    });
    const normalizedRms = assertOracle(flash, `short ${logical}/32`);
    const manual = await runStreamSession({
      config,
      planName: `manual-stream-short-${logical}-32`,
      mode: "80ms",
      maxTokens: 0,
      env: { ...schedulerEnv(logical), VOXTRAL_ENC_KV_MANUAL: "1" },
      timeoutMs: 300_000,
    });
    if (manual.encoderSha256 !== flash.encoderManualSha256) {
      throw new Error(`short ${logical}/32: direct manual tensor differs from oracle tensor`);
    }
    if (production.encoderSha256 !== flash.encoderSha256) {
      throw new Error(`short ${logical}/32: production flash tensor differs from oracle primary tensor`);
    }
    if (JSON.stringify(manual.tokens) !== JSON.stringify(flash.tokens) || manual.text !== flash.text) {
      throw new Error(`short ${logical}/32: manual token/transcript divergence`);
    }
    summary.cases.push(compact(flash, `short-${logical}-32`, normalizedRms, production, manual));
  }

  // Cross the 878-frame physical ring capacity and repeat the independent
  // comparison after rollover. Silence is deterministic and generated in-memory;
  // no binary fixture is committed.
  const rolloverPath = `${config.remoteRepo}/.manual-oracle-rollover.wav`;
  await runRemote(`cat > ${shellQuote(rolloverPath)}`, {
    config, input: silenceWav(20 * 16_000), timeoutMs: 60_000,
  });
  try {
    const production = await runStreamSession({
      config,
      planName: "flash-production-rollover-4-32",
      mode: "160ms",
      audioPath: rolloverPath,
      maxTokens: 0,
      env: schedulerEnv(4),
      timeoutMs: 600_000,
    });
    const rollover = await runStreamSession({
      config,
      planName: "manual-oracle-rollover-4-32",
      mode: "160ms",
      audioPath: rolloverPath,
      maxTokens: 0,
      manualOracle: true,
      env: schedulerEnv(4),
      timeoutMs: 600_000,
    });
    if (rollover.encoderKvWraps <= 0) throw new Error("rollover oracle did not wrap the ring");
    const normalizedRms = assertOracle(rollover, "rollover 4/32");
    const manual = await runStreamSession({
      config,
      planName: "manual-stream-rollover-4-32",
      mode: "160ms",
      audioPath: rolloverPath,
      maxTokens: 0,
      env: { ...schedulerEnv(4), VOXTRAL_ENC_KV_MANUAL: "1" },
      timeoutMs: 600_000,
    });
    if (production.encoderSha256 !== rollover.encoderSha256 ||
        manual.encoderSha256 !== rollover.encoderManualSha256) {
      throw new Error("rollover 4/32: direct flash/manual tensor differs from oracle tensors");
    }
    if (JSON.stringify(manual.tokens) !== JSON.stringify(production.tokens) ||
        manual.text !== production.text) {
      throw new Error("rollover 4/32: manual token/transcript divergence");
    }
    summary.cases.push(compact(rollover, "rollover-4-32", normalizedRms, production, manual));
  } finally {
    await runRemote(`rm -f ${shellQuote(rolloverPath)}`, { config, timeoutMs: 30_000 });
  }
  summary.exitCode = 0;
}

try {
  await main();
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "encoder-kv-manual-oracle",
    backend: "Vulkan",
    command: "npm run test:encoder-kv-manual",
    result: summary,
  });
  console.log(`[encoder-kv-manual] summary: ${artifact.directory}`);
}

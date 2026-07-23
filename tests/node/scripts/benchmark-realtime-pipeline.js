import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  exactTokens,
  gate,
  prepareSession8,
  session8PrecisionEnvironment,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run benchmark:realtime-pipeline",
  steps: [],
  variants: [],
};

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

const variants = [
  {
    // F32 storage control on the post-ring/reusable-graph code. The immutable
    // 68d002f pre-optimization baseline is a separate preserved artifact.
    name: "post-ring-f32-control-4x32",
    encoderShape: "4/32",
    encoderKv: "F32",
    decoderKv: "F32",
    env: {
      VOXTRAL_ENCODER_KV_TYPE: "f32",
      VOXTRAL_DECODER_KV_TYPE: "f32",
      VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
    },
  },
  {
    name: "decoder-fp16-4x32",
    encoderShape: "4/32",
    encoderKv: "F32",
    decoderKv: "F16",
    env: {
      VOXTRAL_ENCODER_KV_TYPE: "f32",
      VOXTRAL_DECODER_KV_TYPE: "f16",
      VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
    },
  },
  {
    name: "dual-fp16-4x32",
    encoderShape: "4/32",
    encoderKv: "F16",
    decoderKv: "F16",
    env: {
      VOXTRAL_ENCODER_KV_TYPE: "f16",
      VOXTRAL_DECODER_KV_TYPE: "f16",
      VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: "32",
    },
  },
  {
    name: "dual-fp16-4x16",
    encoderShape: "4/16",
    encoderKv: "F16",
    decoderKv: "F16",
    env: {
      VOXTRAL_ENCODER_KV_TYPE: "f16",
      VOXTRAL_DECODER_KV_TYPE: "f16",
      VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: "16",
    },
  },
  {
    name: "dual-fp16-4x8",
    encoderShape: "4/8",
    encoderKv: "F16",
    decoderKv: "F16",
    env: {
      VOXTRAL_ENCODER_KV_TYPE: "f16",
      VOXTRAL_DECODER_KV_TYPE: "f16",
      VOXTRAL_ENC_KV_LOGICAL_BATCH: "4",
      VOXTRAL_ENC_KV_PHYSICAL_ROWS: "8",
    },
  },
  {
    name: "production-selected-4x4",
    encoderShape: "4/4",
    encoderKv: null,
    decoderKv: null,
    env: null,
  },
];

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  const selected = matrix.result.productionDecision.selected;
  gate(selected, "precision matrix has no selected production variant");
  const selectedVariant = matrix.result.variants[selected];
  const productionEnv = session8PrecisionEnvironment(selected);
  Object.assign(variants.at(-1), {
    name: `production-selected-${selected.toLowerCase()}-4x4`,
    encoderKv: selectedVariant.encoderKv,
    decoderKv: selectedVariant.decoderKv,
    env: productionEnv,
  });
  summary.precisionMatrixArtifact = matrix.directory;
  summary.selectedPrecision = selected;
  await prepareSession8(summary, { config });
  let oracleTokens = null;
  let oracleTranscript = null;
  for (const variant of variants) {
    const run = await runStreamSession({
      config,
      planName: variant.name,
      syntheticSeconds: 30,
      mode: "80ms",
      warmup: true,
      skipParity: true,
      monitorMemory: true,
      env: {
        VOXTRAL_ENCODER_FINISH_PHYSICAL: "8",
        VOXTRAL_ENCODER_TELEMETRY: "1",
        VOXTRAL_PROFILE: "1",
        ...variant.env,
      },
      timeoutMs: 600_000,
    });
    gate(run.state === "completed", `${variant.name}: state=${run.state}`);
    if (oracleTokens === null) {
      oracleTokens = run.tokens;
      oracleTranscript = run.text;
    } else {
      gate(exactTokens(run.tokens, oracleTokens), `${variant.name}: token parity failed`);
      gate(run.text === oracleTranscript, `${variant.name}: transcript parity failed`);
    }
    summary.variants.push({
      name: variant.name,
      encoderShape: variant.encoderShape,
      encoderKv: variant.encoderKv,
      decoderKv: variant.decoderKv,
      runtime: "warm graphs",
      duration: "short-30s-compute",
      run: summarizeRun(run),
      result: "PASS",
    });
    console.log(`[realtime-pipeline] ${variant.name}: RTF=${run.pipelineRtf.toFixed(4)} peakVRAM=${run.peakVramBytes}`);
  }

  const cold = await runStreamSession({
    config,
    planName: "benchmark-production-cold",
    mode: "80ms",
    realtimeMs: 80,
    skipParity: true,
    env: { ...productionEnv, MESA_SHADER_CACHE_DISABLE: "true" },
    timeoutMs: 180_000,
  });
  const warmGraphs = await runStreamSession({
    config,
    planName: "benchmark-production-warm-graphs",
    mode: "80ms",
    realtimeMs: 80,
    warmup: true,
    skipParity: true,
    env: productionEnv,
    timeoutMs: 180_000,
  });
  summary.runtimeMatrix = {
    coldShaderDiskCacheDisabled: true,
    coldProcess: summarizeRun(cold, { includeTokens: false }).latency,
    warmModel: {
      firstDecoderStepMs: cold.firstDecoderStepMs,
      firstTokenMs: cold.firstTokenMs,
      firstVisibleTextMs: cold.firstVisibleTextMs,
      finishMs: cold.finishLatencyMs,
    },
    warmGraphs: summarizeRun(warmGraphs, { includeTokens: false }).latency,
  };
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const csv = [
    "variant,encoder_shape,encoder_kv,decoder_kv,runtime,duration,rtf,backlog_slope,backlog_final,first_token_warm,first_token_cold,finish,peak_vram,tokens,transcript,result",
    ...summary.variants.map((variant) => {
      const r = variant.run;
      return [
        variant.name,
        variant.encoderShape,
        variant.encoderKv,
        variant.decoderKv,
        variant.runtime,
        variant.duration,
        r.pipelineRtf,
        r.backlog.slopeMsPerSec,
        r.backlog.finalMs,
        r.latency.firstTokenMs,
        r.latency.coldFirstTokenMs,
        r.latency.finishMs,
        r.memory.peakVramBytes,
        r.nTokens,
        JSON.stringify(r.transcript),
        variant.result,
      ].map(csvCell).join(",");
    }),
  ].join("\n");
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8-realtime-pipeline-benchmark",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    textArtifacts: { "matrix.csv": csv },
  });
  console.log(`[realtime-pipeline] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[realtime-pipeline] error: ${summary.error}`);
}

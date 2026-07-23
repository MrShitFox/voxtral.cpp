import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { runStreamSession } from "../helpers/stream.js";
import {
  SESSION8_PRODUCTION_ENV,
  exactTokens,
  gate,
  gateKvMemory,
  prepareSession8,
  summarizeRun,
} from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:kv-fp16",
  steps: [],
};

try {
  await prepareSession8(summary, { config });
  const result = await runStreamSession({
    config,
    planName: "session8-kv-fp16",
    mode: "80ms",
    kvParity: true,
    env: SESSION8_PRODUCTION_ENV,
    timeoutMs: 600_000,
  });

  const production = result.production;
  const fp16Reference = result.fp16Reference;
  const f32SameShape = result.f32SameShape;
  const f32Production = result.f32Production;
  const f32Reference = result.f32Reference;
  gate(result.tokenParity === true, "FP16 production/reference token parity failed");
  gate(result.transcriptParity === true, "FP16 production/reference transcript parity failed");
  gate(exactTokens(production.tokens, fp16Reference.tokens) &&
       exactTokens(production.tokens, f32SameShape.tokens) &&
       exactTokens(production.tokens, f32Production.tokens) &&
       exactTokens(production.tokens, f32Reference.tokens),
  "token arrays are not byte-for-byte identical");
  gate(production.text === fp16Reference.text &&
       production.text === f32SameShape.text &&
       production.text === f32Production.text &&
       production.text === f32Reference.text,
    "transcripts are not byte-for-byte identical");
  gateKvMemory(production, "FP16 production");
  gate(fp16Reference.decoderKvElementSize === 2 && fp16Reference.encoderKvElementSize === 2,
    "FP16 tensor reference did not use FP16 KV");
  gate(f32Reference.decoderKvElementSize === 4 && f32Reference.encoderKvElementSize === 4,
    "F32 oracle did not use F32 KV");
  for (const [name, comparison] of Object.entries(result.numerical ?? {})) {
    gate(comparison.available === true, `${name}: numerical comparison unavailable`);
    gate(Number.isFinite(comparison.maxAbsDelta) && Number.isFinite(comparison.normalizedRms),
      `${name}: non-finite numerical delta`);
    gate(Number.isFinite(comparison.cosineSimilarity), `${name}: non-finite cosine similarity`);
  }

  summary.production = summarizeRun(production);
  summary.fp16Reference = summarizeRun(fp16Reference);
  summary.f32SameShape = summarizeRun(f32SameShape);
  summary.f32Production = summarizeRun(f32Production);
  summary.f32Reference = summarizeRun(f32Reference);
  summary.numerical = result.numerical;
  summary.memoryTable = [
    {
      cache: "decoder",
      f32Bytes: f32Reference.decoderKvAllocatedBytes,
      fp16Bytes: production.decoderKvAllocatedBytes,
      savingBytes: f32Reference.decoderKvAllocatedBytes - production.decoderKvAllocatedBytes,
    },
    {
      cache: "encoder",
      f32Bytes: f32Reference.encoderKvAllocatedBytes,
      fp16Bytes: production.encoderKvAllocatedBytes,
      savingBytes: f32Reference.encoderKvAllocatedBytes - production.encoderKvAllocatedBytes,
    },
  ];
  summary.tokenParity = true;
  summary.transcriptParity = true;
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8-kv-fp16",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    textArtifacts: summary.production ? {
      "transcript.txt": summary.production.transcript,
      "token-ids.txt": summary.production.tokens.join("\n"),
    } : {},
  });
  console.log(`[kv-fp16] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[kv-fp16] error: ${summary.error}`);
}

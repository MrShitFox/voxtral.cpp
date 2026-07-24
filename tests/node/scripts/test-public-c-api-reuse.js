import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import {
  gateChecks,
  preparePublicApi,
  publicRunChecks,
  runPublicApi,
} from "../helpers/public-api.js";

const config = loadEnvironment();
const iterations = Math.max(
  100,
  Number(process.env.VOXTRAL_PUBLIC_REUSE_STREAMS ?? 100),
);
const VRAM_TAIL_LIMIT_BYTES = 64 * 1024 * 1024;
const RSS_TAIL_LIMIT_KIB = 64 * 1024;
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run test:public-c-api-reuse",
  requestedIterations: iterations,
};

try {
  await preparePublicApi({ config });
  const run = await runPublicApi({
    config,
    iterations,
    timeoutMs: 1_500_000,
  });
  const baseChecks = publicRunChecks(run);
  const reuseChecks = {
    exactlyRequestedIterations: run.iterations === iterations,
    tokenConsistency: run.tokenConsistency === true,
    transcriptConsistency: run.transcriptConsistency === true,
    resetPristine: run.resetPristine === true,
    vramPlateau:
      run.vramTailRangeBytes <= VRAM_TAIL_LIMIT_BYTES,
    rssPlateau:
      run.rssTailRangeKiB <= RSS_TAIL_LIMIT_KIB,
  };
  gateChecks(baseChecks, "public reuse base");
  gateChecks(reuseChecks, "public reuse");

  summary.baseChecks = baseChecks;
  summary.reuseChecks = reuseChecks;
  summary.iterations = run.iterations;
  summary.tokensPerStream = run.tokenCount;
  summary.vramTailRangeBytes = run.vramTailRangeBytes;
  summary.rssTailRangeKiB = run.rssTailRangeKiB;
  summary.transcript = run.transcript;
  summary.exitCode = 0;
  console.log(
    `[public-c-api-reuse] PASS iterations=${run.iterations} ` +
    `tokens/stream=${run.tokenCount} VRAM-tail=${run.vramTailRangeBytes}B ` +
    `RSS-tail=${run.rssTailRangeKiB}KiB`,
  );
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-c-api-reuse] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-c-api-reuse",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
  });
  console.log(
    `[public-c-api-reuse] ` +
    `${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`,
  );
}

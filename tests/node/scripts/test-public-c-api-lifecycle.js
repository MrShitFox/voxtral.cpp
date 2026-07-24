import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import {
  gateChecks,
  preparePublicApi,
  publicRunChecks,
  runPublicApi,
} from "../helpers/public-api.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run test:public-c-api-lifecycle",
};

try {
  await preparePublicApi({ config });
  const run = await runPublicApi({
    config,
    lifecycle: true,
    timeoutMs: 420_000,
  });
  const baseChecks = publicRunChecks(run);
  const lifecycleChecks = {
    secondStreamRejected:
      run.lifecycle?.secondStreamRejected === true,
    cancelIdempotent:
      run.lifecycle?.cancelIdempotent === true,
    oversizeFeedStructured:
      run.lifecycle?.oversizeFeedStructured === true,
    resetActiveRejected:
      run.lifecycle?.resetActiveRejected === true,
    resetFromCreated:
      run.lifecycle?.resetFromCreated === true,
    cancelEmitsNoFinal:
      run.lifecycle?.cancelEmitsNoFinal === true,
    destroyNullSafe: run.destroy?.nullSafe === true,
    destroyCreated: run.destroy?.created === true,
    destroyActive: run.destroy?.active === true,
    destroyCancelled: run.destroy?.cancelled === true,
    destroyCompleted: run.destroy?.completed === true,
    destroyUnderBackpressure:
      run.destroy?.underBackpressure === true,
    finishUnderBackpressure:
      run.destroy?.finishUnderBackpressure === true,
    cancelUnderBackpressure:
      run.destroy?.cancelUnderBackpressure === true,
    contextLeaseReleased:
      run.destroy?.leaseReleased === true,
  };
  gateChecks(baseChecks, "public lifecycle base");
  gateChecks(lifecycleChecks, "public lifecycle edges");

  summary.baseChecks = baseChecks;
  summary.lifecycleChecks = lifecycleChecks;
  summary.tokens = run.tokenCount;
  summary.transcript = run.transcript;
  summary.exitCode = 0;
  console.log(
    `[public-c-api-lifecycle] PASS tokens=${run.tokenCount} ` +
    `edges=${Object.keys(lifecycleChecks).length}`,
  );
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-c-api-lifecycle] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-c-api-lifecycle",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
  });
  console.log(
    `[public-c-api-lifecycle] ` +
    `${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`,
  );
}

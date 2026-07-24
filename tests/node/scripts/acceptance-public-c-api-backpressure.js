import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import {
  exactTokens,
  gateChecks,
  preparePublicApi,
  publicRunChecks,
  runPublicApi,
} from "../helpers/public-api.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:public-c-api-backpressure",
};

try {
  const prepared = await preparePublicApi({ config });
  summary.buildWallMs = prepared.build.wallMs;

  const normal = await runPublicApi({
    config,
    chunkSamples: 1280,
    timeoutMs: 360_000,
  });
  const backpressure = await runPublicApi({
    config,
    chunkSamples: 1280,
    eventCapacity: 8,
    delayedConsumer: true,
    timeoutMs: 360_000,
  });

  const normalChecks = publicRunChecks(normal);
  const backpressureChecks = publicRunChecks(
    backpressure,
    { requireBackpressure: true },
  );
  gateChecks(normalChecks, "normal public C API");
  gateChecks(backpressureChecks, "backpressured public C API");

  const parity = {
    tokensExact: exactTokens(backpressure.tokens, normal.tokens),
    transcriptExact: backpressure.transcript === normal.transcript,
    eventSequenceExact:
      backpressure.eventSignature === normal.eventSignature,
    tokenEventsExact:
      backpressure.tokenEvents === normal.tokenEvents,
    partialEventsExact:
      backpressure.partialEvents === normal.partialEvents,
    terminalEventsExact:
      backpressure.finalEvents === 1 &&
      backpressure.completedEvents === 1,
  };
  gateChecks(parity, "backpressure parity");

  summary.normalChecks = normalChecks;
  summary.backpressureChecks = backpressureChecks;
  summary.parity = parity;
  summary.backpressure = {
    offeredSamples: backpressure.samplesOffered,
    consumedSamples: backpressure.samplesConsumed,
    retriedSamples: backpressure.samplesRetried,
    queueFullReturns: backpressure.queueFullReturns,
    droppedAudio: backpressure.samplesOffered - backpressure.samplesConsumed,
    droppedMandatoryEvents:
      normal.tokenEvents + normal.finalEvents + normal.completedEvents -
      backpressure.tokenEvents - backpressure.finalEvents -
      backpressure.completedEvents,
    tokenEvents: backpressure.tokenEvents,
    partialEvents: backpressure.partialEvents,
    finalEvents: backpressure.finalEvents,
    completedEvents: backpressure.completedEvents,
    transcript: backpressure.transcript,
  };
  summary.exitCode = 0;
  console.log(
    `[public-c-api-backpressure] PASS offered=${backpressure.samplesOffered} ` +
    `consumed=${backpressure.samplesConsumed} ` +
    `queue-full=${backpressure.queueFullReturns}`,
  );
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.message;
  process.exitCode = 1;
  console.error(`[public-c-api-backpressure] error: ${error.message}`);
} finally {
  summary.finishedAt = new Date().toISOString();
  const artifact = await writeArtifactBundle({
    config,
    testName: "public-c-api-backpressure",
    backend: "Vulkan",
    command: summary.command,
    result: summary,
    textArtifacts: summary.backpressure ? {
      "transcript.txt": summary.backpressure.transcript,
    } : {},
  });
  console.log(
    `[public-c-api-backpressure] ` +
    `${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`,
  );
}

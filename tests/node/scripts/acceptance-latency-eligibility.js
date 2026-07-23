import { loadEnvironment } from "../config/environment.js";
import { writeArtifactBundle } from "../helpers/artifacts.js";
import { loadLatestPrecisionMatrix } from "../helpers/precision-cache.js";
import { gate, gateLatency } from "../helpers/session8.js";

const config = loadEnvironment();
const summary = {
  startedAt: new Date().toISOString(),
  command: "npm run acceptance:latency-eligibility",
  gates: {
    firstDecoderStepOverheadMs: 200,
    firstTokenOverheadMs: 250,
    firstPartialOverheadMs: 350,
  },
  rows: [],
};

try {
  const matrix = await loadLatestPrecisionMatrix(config);
  summary.sourceArtifact = matrix.directory;
  const selected = matrix.result.productionDecision.selected;
  gate(selected, "precision matrix has no selected production variant");
  summary.selected = selected;
  for (const [fixtureId, fixture] of Object.entries(matrix.result.fixtureResults)) {
    const variant = fixture.variants[selected];
    const paced80 = variant.plans.find((plan) => plan.name === "paced-80ms");
    gate(paced80, `${fixtureId}-${selected}: paced-80ms latency run missing`);
    const latencyRun = {
      ...paced80.latency,
      firstDecoderStepMs: paced80.latency.firstDecoderStepMs,
      firstDecoderStepEligibilityMs: paced80.latency.firstDecoderStepEligibilityMs,
      firstDecoderStepOverheadMs: paced80.latency.firstDecoderStepOverheadMs,
      firstTokenMs: paced80.latency.firstTokenMs,
      firstTokenEligibilityMs: paced80.latency.firstTokenEligibilityMs,
      firstTokenOverheadMs: paced80.latency.firstTokenOverheadMs,
      firstVisibleTextMs: paced80.latency.firstVisibleTextMs,
      firstPartialEligibilityMs: paced80.latency.firstPartialEligibilityMs,
      firstPartialOverheadMs: paced80.latency.firstPartialOverheadMs,
    };
    gateLatency(latencyRun, `${fixtureId}-${selected}`);
    summary.rows.push({
      fixture: fixtureId,
      lifecycle: {
        modelLoadMs: paced80.latency.modelLoadMs,
        contextCreationMs: paced80.latency.contextCreationMs,
        vulkanWarmupMs: paced80.latency.warmupMs,
        streamStartMs: paced80.latency.streamStartMs,
      },
      firstDecoderStep: {
        absoluteMs: paced80.latency.firstDecoderStepMs,
        eligibilityMs: paced80.latency.firstDecoderStepEligibilityMs,
        overheadMs: paced80.latency.firstDecoderStepOverheadMs,
        gateMs: 200,
        result: "PASS",
      },
      firstLexicalToken: {
        absoluteMs: paced80.latency.firstTokenMs,
        eligibilityMs: paced80.latency.firstTokenEligibilityMs,
        overheadMs: paced80.latency.firstTokenOverheadMs,
        gateMs: 250,
        result: "PASS",
      },
      firstPartial: {
        absoluteMs: paced80.latency.firstVisibleTextMs,
        eligibilityMs: paced80.latency.firstPartialEligibilityMs,
        overheadMs: paced80.latency.firstPartialOverheadMs,
        gateMs: 350,
        result: "PASS",
      },
    });
  }
  summary.absoluteTimesAreDiagnosticOnly = true;
  summary.exitCode = 0;
} catch (error) {
  summary.exitCode = 1;
  summary.error = error.stack ?? error.message;
  process.exitCode = 1;
} finally {
  summary.finishedAt = new Date().toISOString();
  const csv = [
    "fixture,metric,absolute_ms,eligibility_ms,overhead_ms,gate_ms,result",
    ...summary.rows.flatMap((row) => [
      `${row.fixture},first_decoder_step,${row.firstDecoderStep.absoluteMs},${row.firstDecoderStep.eligibilityMs},${row.firstDecoderStep.overheadMs},${row.firstDecoderStep.gateMs},${row.firstDecoderStep.result}`,
      `${row.fixture},first_lexical_token,${row.firstLexicalToken.absoluteMs},${row.firstLexicalToken.eligibilityMs},${row.firstLexicalToken.overheadMs},${row.firstLexicalToken.gateMs},${row.firstLexicalToken.result}`,
      `${row.fixture},first_partial,${row.firstPartial.absoluteMs},${row.firstPartial.eligibilityMs},${row.firstPartial.overheadMs},${row.firstPartial.gateMs},${row.firstPartial.result}`,
    ]),
  ].join("\n");
  const artifact = await writeArtifactBundle({
    config,
    testName: "session8.1-latency-eligibility",
    backend: "Vulkan/RADV RX 6600",
    command: summary.command,
    result: summary,
    textArtifacts: { "latency.csv": csv },
  });
  console.log(`[latency-eligibility] ${summary.exitCode === 0 ? "PASS" : "FAIL"} summary: ${artifact.directory}`);
  if (summary.error) console.error(`[latency-eligibility] error: ${summary.error}`);
}
